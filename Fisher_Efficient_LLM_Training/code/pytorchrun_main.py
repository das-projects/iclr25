import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=1_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
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
        self.seed_for_shuffle = 42

    def prepare_data(self):
        # Download datasets
        pass  # Streaming datasets do not need to download

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
            data = data.shuffle(seed=self.seed_for_shuffle)
            if not self.args.single_gpu:
                data = datasets.distributed.split_dataset_by_node(
                    data, rank=self.trainer.global_rank, world_size=self.trainer.world_size,
                )
            self.train_data = PreprocessedIterableDataset(
                data, self.tokenizer, batch_size=self.args.batch_size, max_length=self.args.max_length
            )
        if stage == 'validate' or stage is None:
            val_data = datasets.load_dataset(
                "allenai/c4",
                "en",
                split="validation",
                streaming=True,
                trust_remote_code=True
                )
            val_data = val_data.shuffle(seed=42)
            if not self.args.single_gpu:
                val_data = datasets.distributed.split_dataset_by_node(
                    val_data, rank=self.trainer.global_rank, world_size=self.trainer.world_size
                )
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

class MyLightningModule(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        model_config = AutoConfig.from_pretrained(
            args.model_config,
            trust_remote_code=True
        )
        if args.use_hf_model:
            self.model = AutoModelForCausalLM.from_config(model_config)
        else:
            self.model = LlamaForCausalLM(model_config)
        self.model.generation_config.pad_token_id = tokenizer.pad_token_id

        if args.activation_checkpointing:
            self.model.gradient_checkpointing_enable()
        # Handle dtype
        if args.dtype in ["bf16", "bfloat16"]:
            self.model = self.model.to(dtype=torch.bfloat16)

        self.pad_idx = tokenizer.eos_token_id

        # For logging
        self.tokens_seen = 0
        self.save_hyperparameters()

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_idx] = -100
        outputs = self.model(**batch, labels=labels)
        loss = outputs.loss

        self.tokens_seen += (batch["input_ids"] != self.pad_idx).sum().item() * self.trainer.world_size

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_idx] = -100
        outputs = self.model(**batch, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        args = self.args
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = None

        if 'galore' in args.optimizer.lower():
            # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
            galore_params = []
            target_modules_list = ["attn", "mlp"]
            for module_name, module in self.model.named_modules():
                if not isinstance(module, nn.Linear):
                    continue

                if not any(target_key in module_name for target_key in target_modules_list):
                    continue
                
                print('enable GaLore for weights in module: ', module_name)
                galore_params.append(module.weight)
            id_galore_params = [id(p) for p in galore_params]
            # make parameters without "rank" to another group
            regular_params = [p for p in self.model.parameters() if id(p) not in id_galore_params]
            # then call galore_adamw
            param_groups = [{'params': regular_params}, 
                            {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}]
            
        logger.info(f"\n{self.model}\n")
        logger.info(f"Total params: {sum(p.numel() for p in self.model.parameters()) / 1_000_000:.2f}M")
        logger.info(f"Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
        if 'galore' in args.optimizer.lower():
            logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
        logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")
        
        if args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "galore_adamw":
            optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
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
            optimizer = bnb.optim.Adam8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "galore_adamw8bit":
            optimizer = GaLoreAdamW8bit(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported")

        # Scheduler
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

def main(args):
    pl.seed_everything(args.seed)
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % torch.cuda.device_count() == 0, "total_batch_size must be divisible by number of GPUs"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * torch.cuda.device_count())
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * torch.cuda.device_count() == args.total_batch_size, \
        "gradient_accumulation * batch_size * num_gpus must be equal to total_batch_size"

    tokenizer = AutoTokenizer.from_pretrained(
        "google-t5/t5-base",
        model_max_length=args.max_length,
        trust_remote_code=True,
        clean_up_tokenization_spaces=True,
        pad_token_id=0,
    )
    
    model = MyLightningModule(args, tokenizer)
    data_module = C4DataModule(args, tokenizer)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project="galore-c4", name=args.name)
    wandb_logger.log_hyperparams(vars(args))

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        every_n_train_steps=args.save_every,
        save_top_k=-1,
    )

    # Define the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1 if args.single_gpu else torch.cuda.device_count(),
        strategy='ddp' if not args.single_gpu else None,
        max_steps=args.num_training_steps,
        accumulate_grad_batches=args.gradient_accumulation,
        precision=16 if args.dtype in ['bf16', 'bfloat16'] else 32,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        val_check_interval=args.eval_every,
        gradient_clip_val=args.grad_clipping if args.grad_clipping > 0.0 else None,
        enable_checkpointing=True,
        # Other trainer arguments as needed
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
