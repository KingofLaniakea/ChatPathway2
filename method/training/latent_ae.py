"""Train the shared reconstruction AE bridge with deterministic model selection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.training.common import (
    EarlyStopping,
    artifact_sha256,
    base_model_identity,
    accumulation_divisor,
    append_jsonl,
    configure_logger,
    ensure_disjoint_groups,
    ensure_new_output_dir,
    file_sha256,
    git_commit,
    seed_everything,
    stable_group_split,
    write_json,
)
from method.training.sequence import encode_supervised


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_SFT = "/root/autodl-tmp/checkpoints/shared/pathway_sft/checkpoint_best"
DEFAULT_TRAIN = "/root/autodl-tmp/data/train_kegg_pathway_record_balanced_0p1pct.csv"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/shared/pathway_reconstruction_ae"


@dataclass
class AEConfig:
    base_model: str
    sft_lora: str
    train_path: str
    validation_path: str | None
    save_dir: str
    batch_size: int
    gradient_accumulation_steps: int
    lr: float
    epochs: int
    max_length: int
    answer_budget_fraction: float
    latent_dim: int
    cosine_weight: float
    validation_fraction: float
    validation_group_column: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    seed: int
    deterministic: bool
    limit: int | None
    hash_inputs: bool
    device: str


def parse_args() -> AEConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft-lora", default=DEFAULT_SFT)
    parser.add_argument("--train", dest="train_path", default=DEFAULT_TRAIN)
    parser.add_argument("--validation", dest="validation_path")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--answer-budget-fraction", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--cosine-weight", type=float, default=2.0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-group-column", default="source_json")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hash-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 < args.answer_budget_fraction < 1:
        parser.error("--answer-budget-fraction must be between 0 and 1")
    return AEConfig(**vars(args))


class CascadeProjection(nn.Module):
    def __init__(self, high_dim: int = 4096, mid_dim: int = 1024, latent_dim: int = 128):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(high_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.LayerNorm(mid_dim // 2),
            nn.SiLU(),
            nn.Linear(mid_dim // 2, latent_dim),
        )
        self.up = nn.Sequential(
            nn.Linear(latent_dim, mid_dim // 2),
            nn.SiLU(),
            nn.Linear(mid_dim // 2, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, high_dim),
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.down(hidden)
        return latent, self.up(latent)


class AEDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
        answer_budget_fraction: float,
    ):
        self.records = frame.to_dict(orient="records")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_budget_fraction = answer_budget_fraction

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[index]
        question = str(row.get("question", ""))
        answer = str(row.get("answer", row.get("formatted_answer_no_phenotype", "")))
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        encoded = encode_supervised(
            self.tokenizer,
            prompt,
            answer,
            max_length=self.max_length,
            answer_budget_fraction=self.answer_budget_fraction,
        )
        loss_mask = [int(label != -100) for label in encoded.labels]
        first_answer = next((index for index, active in enumerate(loss_mask) if active), None)
        if first_answer is not None and first_answer > 0:
            # The Hamiltonian rollout starts from the final prompt token. Train
            # the AE on that anchor as well as answer states to avoid an OOD z0.
            loss_mask[first_answer - 1] = 1
        return {
            "input_ids": torch.tensor(encoded.input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
            "prompt_tokens_dropped": encoded.prompt_tokens_dropped,
            "answer_tokens_dropped": encoded.answer_tokens_dropped,
        }


def make_collate_fn(pad_id: int):
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id
        )
        loss_mask = torch.nn.utils.rnn.pad_sequence(
            [item["loss_mask"] for item in batch], batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(item["input_ids"]) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "prompt_tokens_dropped": sum(int(item["prompt_tokens_dropped"]) for item in batch),
            "answer_tokens_dropped": sum(int(item["answer_tokens_dropped"]) for item in batch),
        }

    return collate


def load_frames(cfg: AEConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(cfg.train_path, engine="c", quoting=csv.QUOTE_MINIMAL, on_bad_lines="error")
    if cfg.limit is not None:
        frame = frame.head(cfg.limit)
    if frame.empty:
        raise ValueError("AE training CSV contains no rows")
    if cfg.validation_path:
        validation = pd.read_csv(
            cfg.validation_path, engine="c", quoting=csv.QUOTE_MINIMAL, on_bad_lines="error"
        )
        if validation.empty:
            raise ValueError("explicit AE validation CSV contains no rows")
        ensure_disjoint_groups(
            frame,
            validation,
            group_column=cfg.validation_group_column,
        )
        return frame.reset_index(drop=True), validation.reset_index(drop=True)
    return stable_group_split(
        frame,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
        group_column=cfg.validation_group_column,
    )


def reconstruction_losses(
    *,
    backbone: nn.Module,
    projection: CascadeProjection,
    batch: dict[str, torch.Tensor],
    device: str,
    cosine_weight: float,
) -> dict[str, torch.Tensor] | None:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    with torch.no_grad():
        hidden = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[-1].float()
    _, reconstructed = projection(hidden)
    active = loss_mask.bool()
    if not active.any():
        return None
    target = hidden[active]
    prediction = reconstructed[active]
    mse = nn.functional.mse_loss(prediction, target)
    cosine = 1.0 - nn.functional.cosine_similarity(prediction, target, dim=-1).mean()
    return {"total": mse + cosine_weight * cosine, "mse": mse, "cosine": cosine}


def evaluate(
    backbone: nn.Module,
    projection: CascadeProjection,
    loader: DataLoader,
    cfg: AEConfig,
) -> dict[str, float | int]:
    projection.eval()
    sums = {"total": 0.0, "mse": 0.0, "cosine": 0.0}
    steps = 0
    prompt_dropped = answer_dropped = 0
    for batch in loader:
        losses = reconstruction_losses(
            backbone=backbone,
            projection=projection,
            batch=batch,
            device=cfg.device,
            cosine_weight=cfg.cosine_weight,
        )
        if losses is None:
            continue
        for key in sums:
            sums[key] += float(losses[key].item())
        steps += 1
        prompt_dropped += int(batch["prompt_tokens_dropped"])
        answer_dropped += int(batch["answer_tokens_dropped"])
    return {
        **{key: value / max(steps, 1) for key, value in sums.items()},
        "prompt_tokens_dropped": prompt_dropped,
        "answer_tokens_dropped": answer_dropped,
    }


def save_checkpoint(destination: Path, projection: CascadeProjection, epoch: int, metrics: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(projection.state_dict(), destination / "ae_proj.pt")
    write_json(destination / "checkpoint_metrics.json", {"epoch": epoch, **metrics})


def train() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    save_root = ensure_new_output_dir(cfg.save_dir)
    logger = configure_logger(save_root / "train.log", "pathway_reconstruction_ae")
    write_json(save_root / "run_config.json", asdict(cfg))
    train_frame, validation_frame = load_frames(cfg)
    manifest: dict[str, Any] = {
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "train_path": str(Path(cfg.train_path).resolve()),
        "validation_path": str(Path(cfg.validation_path).resolve()) if cfg.validation_path else "deterministic_group_split",
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "seed": cfg.seed,
        "base_model_identity": base_model_identity(cfg.base_model),
        "sft_adapter_sha256": artifact_sha256(cfg.sft_lora),
    }
    if cfg.hash_inputs:
        manifest["train_sha256"] = file_sha256(cfg.train_path)
        if cfg.validation_path:
            manifest["validation_sha256"] = file_sha256(cfg.validation_path)
    write_json(save_root / "run_manifest.json", manifest)
    logger.info("train_rows=%d validation_rows=%d seed=%d", len(train_frame), len(validation_frame), cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": cfg.device},
    )
    backbone = PeftModel.from_pretrained(base_model, cfg.sft_lora).eval()
    backbone.requires_grad_(False)
    projection = CascadeProjection(
        high_dim=base_model.config.hidden_size,
        latent_dim=cfg.latent_dim,
    ).to(cfg.device).float()
    optimizer = optim.AdamW(projection.parameters(), lr=cfg.lr)
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        AEDataset(train_frame, tokenizer, cfg.max_length, cfg.answer_budget_fraction),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        AEDataset(validation_frame, tokenizer, cfg.max_length, cfg.answer_budget_fraction),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    early_stopping = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_min_delta)
    history: list[dict[str, Any]] = []
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        projection.train()
        sums = {"total": 0.0, "mse": 0.0, "cosine": 0.0}
        steps = 0
        prompt_dropped = answer_dropped = 0
        progress = tqdm(train_loader, desc=f"AE epoch {epoch}")
        for step, batch in enumerate(progress):
            losses = reconstruction_losses(
                backbone=backbone,
                projection=projection,
                batch=batch,
                device=cfg.device,
                cosine_weight=cfg.cosine_weight,
            )
            if losses is None:
                continue
            divisor = accumulation_divisor(step, len(train_loader), cfg.gradient_accumulation_steps)
            (losses["total"] / divisor).backward()
            for key in sums:
                sums[key] += float(losses[key].item())
            steps += 1
            prompt_dropped += int(batch["prompt_tokens_dropped"])
            answer_dropped += int(batch["answer_tokens_dropped"])
            if (step + 1) % cfg.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(total=f"{losses['total'].item():.4f}", mse=f"{losses['mse'].item():.4f}")

        train_metrics: dict[str, float | int] = {
            **{key: value / max(steps, 1) for key, value in sums.items()},
            "prompt_tokens_dropped": prompt_dropped,
            "answer_tokens_dropped": answer_dropped,
        }
        validation_metrics = evaluate(backbone, projection, validation_loader, cfg)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "lr": cfg.lr}
        history.append(record)
        append_jsonl(save_root / "metrics.jsonl", record)
        write_json(save_root / "history.json", history)
        logger.info(
            "epoch=%d train_total=%.6f val_total=%.6f val_mse=%.6f val_cosine=%.6f",
            epoch,
            train_metrics["total"],
            validation_metrics["total"],
            validation_metrics["mse"],
            validation_metrics["cosine"],
        )
        save_checkpoint(save_root / f"checkpoint_epoch_{epoch}", projection, epoch, record)
        improved, should_stop = early_stopping.update(validation_metrics["total"], epoch)
        if improved:
            save_checkpoint(save_root / "checkpoint_best", projection, epoch, record)
            write_json(
                save_root / "best_checkpoint.json",
                {
                    "epoch": epoch,
                    "monitor": "validation.total",
                    "value": validation_metrics["total"],
                    "path": str(save_root / "checkpoint_best"),
                },
            )
        if should_stop:
            stopped_early = True
            logger.info(
                "early_stop epoch=%d best_epoch=%d best_validation_total=%.6f",
                epoch,
                early_stopping.best_epoch,
                early_stopping.best,
            )
            break

    write_json(
        save_root / "run_complete.json",
        {
            "status": "completed",
            "completed_epochs": len(history),
            "best_epoch": early_stopping.best_epoch,
            "best_validation_total": early_stopping.best,
            "early_stopped": stopped_early,
        },
    )


if __name__ == "__main__":
    train()
