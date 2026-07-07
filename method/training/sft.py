"""Train the pathway SFT LoRA adapter.

The original migrated script was DDP-only and contained historical GPFS paths.
This maintained entry point keeps the same SFT objective but can run either as a
normal single-process command or under ``torchrun`` for distributed training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_TRAIN = "/root/autodl-tmp/data/train_11_species_dataset.csv"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/qwen3_8b_sft"


@dataclass
class SFTConfig:
    base_model: str
    train_path: str
    save_dir: str
    resume_adapter: str | None
    batch_size: int
    gradient_accumulation_steps: int
    lr: float
    epochs: int
    max_length: int
    text_column: str
    limit: int | None
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    gradient_checkpointing: bool
    device: str


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, text_column: str, max_length: int, limit: int | None = None):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length
        df = pd.read_csv(path, engine="python", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
        if limit is not None:
            df = df.head(limit)
        self.records = df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[index]
        question = str(row.get("question", ""))
        answer = str(row.get(self.text_column, row.get("answer", row.get("formatted_answer_no_phenotype", ""))))
        prompt_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        answer_text = f"{answer}<|im_end|>"
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_collate_fn(pad_id: int):
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != pad_id).long(),
        }

    return collate


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train", dest="train_path", default=DEFAULT_TRAIN)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE)
    parser.add_argument("--resume-adapter")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return SFTConfig(
        base_model=args.base_model,
        train_path=args.train_path,
        save_dir=args.save_dir,
        resume_adapter=args.resume_adapter,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        text_column=args.text_column,
        limit=args.limit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        device=args.device,
    )


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[bool, int, str]:
    distributed = is_distributed()
    if not distributed:
        return False, 0, ""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return True, local_rank, f"cuda:{local_rank}"


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def train() -> None:
    cfg = parse_args()
    distributed, local_rank, ddp_device = setup_distributed()
    device = ddp_device or cfg.device
    is_main = local_rank == 0
    save_root = Path(cfg.save_dir)
    if is_main:
        save_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    if cfg.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    if cfg.resume_adapter:
        model = PeftModel.from_pretrained(base_model, cfg.resume_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=[item.strip() for item in cfg.target_modules.split(",") if item.strip()],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.to(device)
    if is_main and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    dataset = SFTDataset(cfg.train_path, tokenizer, cfg.text_column, cfg.max_length, cfg.limit)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )
    optimizer = optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=cfg.lr)

    history: list[dict[str, float | int]] = []
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        steps = 0
        progress = tqdm(loader, desc=f"SFT epoch {epoch + 1}", disable=not is_main)
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss
            (loss / cfg.gradient_accumulation_steps).backward()
            total_loss += float(loss.item())
            steps += 1
            if is_main:
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(loader) % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if is_main:
            avg_loss = total_loss / max(steps, 1)
            history.append({"epoch": epoch + 1, "loss": avg_loss})
            checkpoint_dir = save_root / f"checkpoint_epoch_{epoch + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            unwrap_model(model).save_pretrained(checkpoint_dir)
            with (save_root / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            with (save_root / "run_config.json").open("w", encoding="utf-8") as handle:
                json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)
                handle.write("\n")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    train()
