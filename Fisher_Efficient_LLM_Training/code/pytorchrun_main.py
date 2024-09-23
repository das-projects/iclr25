import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data

import transformers
from transformers import AutoConfig, AutoTokenizer

import datasets
import datasets.distributed

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

transformers.logging.set_verbosity_error()

torch_compile_options = {
    "epilogue_fusion": False,
    "max_autotune": False,
    "shape_padding": True,
    "trace.enabled": False,  # Output Triton kernel outputs!
    "triton.cudagraphs": False,
}


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_restarts"],
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for. "
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=1_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)

    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=True, action="store_true")

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


class C4DataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None
        self.seed_for_shuffle = args.seed

    def prepare_data(self):
        # Download datasets
        pass  # Streaming datasets do not need to download

    def setup(self, stage=None):
        # Set process rank and world size
        if hasattr(self.trainer, "global_rank"):
            self.process_rank = self.trainer.global_rank
            self.world_size = self.trainer.world_size
        else:
            self.process_rank = 0
            self.world_size = 1

        if stage == "fit" or stage is None:
            data = datasets.load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            data = data.shuffle(seed=self.seed_for_shuffle)
            self.train_data = PreprocessedIterableDataset(
                data=data,
                tokenizer=self.tokenizer,
                batch_size=self.args.batch_size,
                max_length=self.args.max_length,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )
        if stage == "validate" or stage is None:
            val_data = datasets.load_dataset(
                "allenai/c4",
                "en",
                split="validation",
                streaming=True,
                trust_remote_code=True,
            )
            val_data = val_data.shuffle(seed=self.seed_for_shuffle)
            self.val_data = val_data.map(
                self.preprocess_batched,
                batched=True,
                remove_columns=["text", "timestamp", "url"],
            )

    def preprocess_batched(self, batch):
        batch = self.tokenizer(
            batch["text"],
            max_length=self.args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=None,
            num_workers=self.args.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
        )


class LlamaLightningModule(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.args = args
        self.tokenizer = tokenizer

        model_config = AutoConfig.from_pretrained(
            args.model_config, trust_remote_code=True
        )
        self.model = LlamaForCausalLM(model_config)
        self.model.generation_config.pad_token_id = tokenizer.pad_token_id

        if args.activation_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.pad_idx = tokenizer.eos_token_id

        # For logging
        self.tokens_seen = 0

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_idx] = -100
        outputs = self.model(**batch, labels=labels)
        loss = outputs.loss

        self.tokens_seen += (
            batch["input_ids"] != self.pad_idx
        ).sum().item() * self.trainer.world_size

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.log(
            "tokens_seen",
            self.tokens_seen,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_idx] = -100
        outputs = self.model(**batch, labels=labels)
        loss = outputs.loss

        tokens = (
            batch["input_ids"] != self.pad_idx
        ).sum().item() * self.trainer.world_size

        self.log("val_loss", loss, on_step=False, prog_bar=True, logger=True)
        self.log("val_tokens", tokens, on_step=False, prog_bar=True, logger=True)

        # Early stopping condition
        if tokens >= self.args.target_eval_tokens:
            self.trainer.should_stop = True

        return {"loss": loss, "tokens": tokens}

    def configure_optimizers(self):
        args = self.args

        # Create parameter groups
        if "galore" in args.optimizer.lower():
            galore_params, regular_params = self.get_galore_params()
            param_groups = [
                {"params": regular_params},
                {
                    "params": galore_params,
                    "rank": args.rank,
                    "update_proj_gap": args.update_proj_gap,
                    "scale": args.galore_scale,
                    "proj_type": args.proj_type,
                },
            ]
        else:
            param_groups = [{"params": self.model.parameters()}]

        # Select optimizer
        optimizer = self.create_optimizer(param_groups)

        # Scheduler
        scheduler = self.create_scheduler(optimizer)

        return [optimizer], [scheduler]

    def get_galore_params(self):
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(
                target_key in module_name for target_key in target_modules_list
            ):
                galore_params.append(module.weight)
        id_galore_params = {id(p) for p in galore_params}
        regular_params = [
            p for p in self.model.parameters() if id(p) not in id_galore_params
        ]
        return galore_params, regular_params

    def create_optimizer(self, param_groups):
        args = self.args
        optimizer = None
        optimizer_name = args.optimizer.lower()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                param_groups, lr=args.lr, weight_decay=args.weight_decay
            )
        elif optimizer_name == "galore_adamw":
            optimizer = GaLoreAdamW(
                param_groups, lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.beta1,
            )
        elif args.optimizer.lower() == "adafactor":
            args.beta1 = None if args.beta1 == 0.0 else args.beta1
            optimizer = transformers.optimization.Adafactor(
                trainable_params,
                lr=args.lr,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=args.beta1,
                weight_decay=args.weight_decay,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        elif args.optimizer.lower() == "galore_adafactor":
            args.beta1 = None if args.beta1 == 0.0 else args.beta1
            optimizer = GaLoreAdafactor(
                param_groups,
                lr=args.lr,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=args.beta1,
                weight_decay=args.weight_decay,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        elif args.optimizer.lower() == "adam8bit":
            optimizer = bnb.optim.Adam8bit(
                trainable_params, lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.optimizer.lower() == "galore_adamw8bit":
            optimizer = GaLoreAdamW8bit(
                param_groups, lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported")
        return optimizer

    def create_scheduler(self, optimizer):
        args = self.args
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return scheduler


class CustomCheckpointCallback(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save additional information
        checkpoint["tokens_seen"] = pl_module.tokens_seen
        # Add any other custom information

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Load additional information
        pl_module.tokens_seen = checkpoint.get("tokens_seen", 0)
        # Load any other custom information


def main(args):
    pl.seed_everything(args.seed)

    num_gpus = 1 if args.single_gpu else torch.cuda.device_count()
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % num_gpus == 0
            ), "total_batch_size must be divisible by number of GPUs"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * num_gpus
            )
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * num_gpus == args.total_batch_size
    ), "gradient_accumulation * batch_size * num_gpus must be equal to total_batch_size"

    tokenizer = AutoTokenizer.from_pretrained(
        "google-t5/t5-base",
        model_max_length=args.max_length,
        trust_remote_code=True,
        clean_up_tokenization_spaces=True,
        pad_token_id=0,
    )

    model = LlamaLightningModule(args, tokenizer)
    data_module = C4DataModule(args, tokenizer)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project="galore-c4", name=args.name)
    wandb_logger.log_hyperparams(vars(args))

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        every_n_train_steps=args.save_every,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    custom_checkpoint = CustomCheckpointCallback()

    # Define the trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy="ddp" if not args.single_gpu else None,
        max_steps=args.num_training_steps,
        accumulate_grad_batches=args.gradient_accumulation,
        precision="bf16" if args.dtype in ["bf16", "bfloat16"] else 32,
        callbacks=[checkpoint_callback, lr_monitor, custom_checkpoint],
        logger=wandb_logger,
        val_check_interval=args.eval_every,
        gradient_clip_val=args.grad_clipping if args.grad_clipping > 0.0 else None,
        gradient_clip_algorithm="norm",
        enable_checkpointing=True,
        # Other trainer arguments as needed
        enable_model_summary=False,
        compile_model=True,
        compile_options=torch_compile_options,
        deterministic=True,
        benchmark=True,
        # auto_scale_batch_size='power',
        auto_lr_find=True,
    )

    trainer.tune(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
