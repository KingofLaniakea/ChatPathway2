"""Jointly train a LoRA adapter and a latent dynamics regularizer.

This generalizes FrameworkA beyond the hand-written TDHNN module. The model
loads a trainable SFT LoRA adapter, a frozen AE projection, and a trainable
latent dynamics module from ``method.dynamics.latent_teacher``. The dynamics
rollout is decoded back into hidden-space velocity and aligned to real
answer-token hidden-state velocity while the standard SFT loss is optimized.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.dynamics.latent_teacher import build_dynamics, checkpoint_payload, load_projection, rollout


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_SFT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"
DEFAULT_AE = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"
DEFAULT_TRAIN = "/root/autodl-tmp/data/train_11_species_dataset.csv"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/joint_lora_dynamics"


@dataclass
class JointDynamicsConfig:
    variant: str
    base_model: str
    sft_adapter: str
    ae_ckpt: str
    train_path: str
    save_dir: str
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    lr: float
    dynamics_lr: float
    latent_dim: int
    max_length: int
    max_steps: int
    lambda_align: float
    lambda_reg: float
    text_column: str
    device: str
    limit: int | None
    gradient_checkpointing: bool


class JointSFTDataset(Dataset):
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
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        target = f"{answer}<|im_end|>"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        input_ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_length]
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


def parse_args() -> JointDynamicsConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--variant",
        choices=("neural_ode", "latent_ode", "gradient_flow", "generic", "koopman", "sindy"),
        required=True,
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft-adapter", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--ae-ckpt", default=DEFAULT_AE)
    parser.add_argument("--train", dest="train_path", default=DEFAULT_TRAIN)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dynamics-lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--lambda-align", type=float, default=0.5)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return JointDynamicsConfig(**vars(parser.parse_args()))


def velocity_alignment_loss(
    h_real: torch.Tensor,
    z_all: torch.Tensor,
    labels: torch.Tensor,
    projection: torch.nn.Module,
    dynamics: torch.nn.Module,
    variant: str,
    max_steps: int,
) -> torch.Tensor:
    answer_mask = labels != -100
    answer_lengths = answer_mask.sum(dim=1)
    if not (answer_lengths > 0).any():
        return h_real.new_tensor(0.0)

    first_answer_idx = answer_mask.to(torch.int64).argmax(dim=1)
    last_prompt_idx = (first_answer_idx - 1).clamp(min=0)
    losses: list[torch.Tensor] = []
    for i in range(h_real.size(0)):
        ans_indices = torch.where(answer_mask[i])[0][:max_steps]
        if ans_indices.numel() == 0:
            continue
        positions = torch.cat([last_prompt_idx[i].unsqueeze(0), ans_indices])
        if positions.numel() < 2:
            continue

        prompt_end = max(int(first_answer_idx[i].item()), 1)
        control = z_all[i, :prompt_end].mean(dim=0)
        z0 = z_all[i, positions[0]]
        predicted_z = rollout(dynamics, variant, z0.float(), control.float(), positions.numel() - 1)
        predicted_velocity = predicted_z[1:] - predicted_z[:-1]
        predicted_high = projection.up(predicted_velocity.float())

        real_hidden = h_real[i, positions].float()
        real_velocity = real_hidden[1:] - real_hidden[:-1]
        min_len = min(real_velocity.size(0), predicted_high.size(0))
        if min_len == 0:
            continue
        cosine = F.cosine_similarity(predicted_high[:min_len], real_velocity[:min_len], dim=-1)
        losses.append(1.0 - cosine.mean())

    if not losses:
        return h_real.new_tensor(0.0)
    return torch.stack(losses).mean()


def train() -> None:
    cfg = parse_args()
    save_root = Path(cfg.save_dir) / cfg.variant
    save_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        device_map={"": cfg.device},
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    if cfg.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(base_model, cfg.sft_adapter, is_trainable=True)
    model.enable_input_require_grads()

    projection = load_projection(cfg.ae_ckpt, base_model.config.hidden_size, cfg.latent_dim, cfg.device)
    dynamics = build_dynamics(cfg.variant, cfg.latent_dim, cfg.latent_dim).to(cfg.device).float()
    optimizer = optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr},
            {"params": dynamics.parameters(), "lr": cfg.dynamics_lr},
        ]
    )

    dataset = JointSFTDataset(cfg.train_path, tokenizer, cfg.text_column, cfg.max_length, cfg.limit)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )

    history: list[dict[str, float | int]] = []
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):
        model.train()
        dynamics.train()
        totals = {"sft": 0.0, "align": 0.0, "reg": 0.0, "total": 0.0}
        steps = 0
        progress = tqdm(loader, desc=f"joint {cfg.variant} epoch {epoch + 1}")
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                use_cache=False,
            )
            h_real = outputs.hidden_states[-1].float()
            z_all, _ = projection(h_real)
            loss_sft = outputs.loss
            loss_align = velocity_alignment_loss(
                h_real,
                z_all,
                labels,
                projection,
                dynamics,
                cfg.variant,
                cfg.max_steps,
            )
            loss_reg = dynamics.regularization_loss()
            loss = loss_sft + cfg.lambda_align * loss_align + cfg.lambda_reg * loss_reg
            (loss / cfg.gradient_accumulation_steps).backward()

            totals["sft"] += float(loss_sft.item())
            totals["align"] += float(loss_align.item())
            totals["reg"] += float(loss_reg.item())
            totals["total"] += float(loss.item())
            steps += 1
            progress.set_postfix({"sft": f"{loss_sft.item():.3f}", "align": f"{loss_align.item():.3f}"})

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(loader) % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        row = {"epoch": epoch + 1, **{key: value / max(steps, 1) for key, value in totals.items()}}
        history.append(row)
        checkpoint_dir = save_root / f"checkpoint_epoch_{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        torch.save(checkpoint_payload(dynamics, cfg.variant, cfg), checkpoint_dir / "dynamics_func.pt")
        with (save_root / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    with (save_root / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    train()
