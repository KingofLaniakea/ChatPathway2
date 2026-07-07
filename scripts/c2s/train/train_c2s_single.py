"""Train the Qwen Cell2Sentence transfer adapter.

This maintained entry point preserves the legacy C2S JSONL contract while
making the training run configurable from the experiment matrix.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_INIT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_stage3_sft_hnn/checkpoint_epoch_5"
DEFAULT_TRAIN_JSONL = "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small.jsonl"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_small"


@dataclass
class C2STrainConfig:
    base_model: str
    init_adapter: str | None
    train_jsonl: str
    save_dir: str
    batch_size: int
    gradient_accumulation_steps: int
    lr: float
    epochs: int
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    gradient_checkpointing: bool
    device: str
    limit: int | None


def safe_encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False, allowed_special="none")
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=False)


class C2SPairingDataset(Dataset):
    def __init__(self, path: str, tokenizer: Any, max_length: int, limit: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: list[dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.data.append(json.loads(line))
                if limit is not None and len(self.data) >= limit:
                    break

        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>") or 151644
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") or 151645
        self.user_ids = tokenizer.encode("user\n", add_special_tokens=False)
        self.assistant_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.data[index]
        instruction = str(item["instruction"])
        output = str(item["output"])

        raw_prompt_ids = safe_encode(self.tokenizer, instruction)
        raw_answer_ids = safe_encode(self.tokenizer, output)
        prompt_ids = (
            [self.im_start_id]
            + self.user_ids
            + raw_prompt_ids
            + [self.im_end_id, 10]
            + [self.im_start_id]
            + self.assistant_ids
        )
        answer_ids = raw_answer_ids + [self.im_end_id]
        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]

        if len(input_ids) == 0 or all(label == -100 for label in labels):
            input_ids = [self.im_start_id] + self.user_ids + [151643] + [self.im_end_id]
            labels = [-100, -100, 151643, -100]

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


def parse_args() -> C2STrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--init-adapter", default=DEFAULT_INIT_ADAPTER)
    parser.add_argument("--no-init-adapter", action="store_true")
    parser.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1648)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    return C2STrainConfig(
        base_model=args.base_model,
        init_adapter=None if args.no_init_adapter else args.init_adapter,
        train_jsonl=args.train_jsonl,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        device=args.device,
        limit=args.limit,
    )


def build_model(cfg: C2STrainConfig) -> Any:
    dtype = torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map={"": cfg.device},
    )
    base_model.config.use_cache = False
    if cfg.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    if cfg.init_adapter:
        model = PeftModel.from_pretrained(base_model, cfg.init_adapter, is_trainable=True)
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
        return model

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=[item.strip() for item in cfg.target_modules.split(",") if item.strip()],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base_model, lora_config)


def train() -> None:
    cfg = parse_args()
    save_root = Path(cfg.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(save_root)

    model = build_model(cfg)
    model.enable_input_require_grads()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    dataset = C2SPairingDataset(cfg.train_jsonl, tokenizer, cfg.max_length, cfg.limit)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )
    optimizer = optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=cfg.lr)

    history: list[dict[str, float | int]] = []
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        progress = tqdm(loader, desc=f"C2S epoch {epoch + 1}")
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            loss = outputs.loss
            (loss / cfg.gradient_accumulation_steps).backward()
            total_loss += float(loss.item())
            steps += 1
            progress.set_postfix({"loss": f"{loss.item():.6f}"})

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(loader) % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(steps, 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        checkpoint_dir = save_root / f"checkpoint_epoch_{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        with (save_root / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    with (save_root / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    train()
