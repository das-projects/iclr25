import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import datasets
import wandb

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from peft_pretraining.modeling_llama import LlamaForCausalLM
from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

class LlamaModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google-t5/t5-base",
            model_max_length=self.hparams.max_length,
            trust_remote_code=True,
            clean_up_tokenization_spaces=True,
            pad_token_id=0,
        )

        model_config = AutoConfig.from_pretrained(self.hparams.model_config, trust_remote_code=True)
        self.model = LlamaForCausalLM(model_config)

        if self.hparams.activation_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.loss_fn = nn.CrossEntropyLoss()

    def preprocess_batched(self, batch):
        return self.tokenizer(
            batch["text"],
            max_length=self.hparams.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def forward(self, batch):
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.model(**batch, labels=labels)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.num_training_steps, eta_min=self.hparams.lr * self.hparams.min_lr_ratio)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = datasets.load_dataset(
            "allenai/c4", 
            "en", 
            split="train", 
            streaming=True,
            trust_remote_code=True
            )
        dataset = dataset.shuffle(seed=42)
        dataset = PreprocessedIterableDataset(dataset, self.tokenizer, batch_size=self.hparams.batch_size, max_length=self.hparams.max_length)
        return DataLoader(dataset, batch_size=None, num_workers=self.hparams.workers, pin_memory=True)

    def val_dataloader(self):
        dataset = datasets.load_dataset(
            "allenai/c4", 
            "en", 
            split="validation", 
            streaming=True,
            trust_remote_code=True
            )
        dataset = dataset.shuffle(seed=42)
        dataset_mapped = dataset.map(
            lambda batch: self.tokenizer(
            batch["text"],
            max_length=self.hparams.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ),
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        dataset_mapped.batch = lambda batch_size: training_utils.batch_fn(dataset_mapped, batch_size)
        return DataLoader(dataset_mapped.batch(batch_size=self.hparams.batch_size), batch_size=None, num_workers=self.hparams.workers, pin_memory=True)
    


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


def main():
    args = parse_args()

    seed_everything(args.seed)

    model = LlamaModel(args)

    wandb_logger = pl.loggers.WandbLogger(project="galore-c4")

    trainer = pl.Trainer(
        max_steps=args.num_training_steps,
        logger=wandb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=args.eval_every,
        gradient_clip_val=args.grad_clipping,
        precision=16 if args.dtype == 'bfloat16' else 32,
        accumulate_grad_batches=args.gradient_accumulation,
        devices="auto",
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
