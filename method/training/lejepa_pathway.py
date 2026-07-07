"""LeJEPA-style latent prediction for pathway language.

This is an exploratory method-layer training script. It does not replace SFT or
FrameworkA. It freezes a backbone language model, embeds question and answer
texts, and trains a small JEPA-style predictor:

    prompt embedding -> latent predictor -> answer latent

The objective is latent prediction plus SIGReg-style distribution regularizers.
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
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"
DEFAULT_TRAIN = "/root/autodl-tmp/data/train_11_species_dataset.csv"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/pathway_lejepa_sentence"


@dataclass
class LeJEPAConfig:
    base_model: str
    adapter: str | None
    train_path: str
    save_path: str
    batch_size: int
    epochs: int
    lr: float
    latent_dim: int
    max_length: int
    lambda_sigreg: float
    device: str
    limit: int | None


class PathwayPairDataset(Dataset):
    def __init__(self, path: str, limit: int | None = None):
        self.df = pd.read_csv(path, engine="python", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
        if limit is not None:
            self.df = self.df.head(limit)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.df.iloc[index]
        question = "" if pd.isna(row.get("question", "")) else str(row.get("question", ""))
        answer_value = row.get("answer", row.get("formatted_answer_no_phenotype", ""))
        answer = "" if pd.isna(answer_value) else str(answer_value)
        return {"question": question, "answer": answer}


class PathwayTextJEPA(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, prompt_embedding: torch.Tensor, answer_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_latent = self.encoder(prompt_embedding)
        target_latent = self.encoder(answer_embedding)
        predicted_target = self.predictor(context_latent)
        return predicted_target, target_latent, context_latent


def parse_args() -> LeJEPAConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Optional LoRA adapter used only for frozen embedding extraction.")
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--save", default=DEFAULT_SAVE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lambda-sigreg", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    return LeJEPAConfig(
        base_model=args.base_model,
        adapter=None if args.no_adapter else args.adapter,
        train_path=args.train,
        save_path=args.save,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        max_length=args.max_length,
        lambda_sigreg=args.lambda_sigreg,
        device=args.device,
        limit=args.limit,
    )


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def embed_texts(texts: list[str], tokenizer: Any, model: Any, cfg: LeJEPAConfig) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
    ).to(cfg.device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
        pooled = mean_pool(outputs.hidden_states[-1], encoded["attention_mask"])
    return pooled.float()


def sigreg_style_loss(values: torch.Tensor) -> torch.Tensor:
    """Small SIGReg-style anti-collapse regularizer for sentence latents.

    This keeps each latent dimension roughly centered with nonzero variance and
    penalizes cross-dimension correlation. It is intentionally lightweight so it
    can be used as a first pathway-language JEPA probe.
    """

    if values.size(0) < 2:
        return values.new_tensor(0.0)
    centered = values - values.mean(dim=0, keepdim=True)
    std = centered.std(dim=0).clamp(min=1e-4)
    normalized = centered / std
    variance_loss = (std - 1.0).pow(2).mean()
    covariance = normalized.t() @ normalized / max(values.size(0) - 1, 1)
    off_diag = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = off_diag.pow(2).mean()
    mean_loss = values.mean(dim=0).pow(2).mean()
    return mean_loss + variance_loss + covariance_loss


def train() -> None:
    cfg = parse_args()
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if cfg.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(cfg.device)
    if cfg.adapter:
        backbone = PeftModel.from_pretrained(backbone, cfg.adapter)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    probe = PathwayTextJEPA(backbone.config.hidden_size, cfg.latent_dim).to(cfg.device)
    optimizer = optim.AdamW(probe.parameters(), lr=cfg.lr)
    loader = DataLoader(PathwayPairDataset(cfg.train_path, cfg.limit), batch_size=cfg.batch_size, shuffle=True)

    history: list[dict[str, float | int]] = []
    for epoch in range(cfg.epochs):
        probe.train()
        running = {"pred": 0.0, "sigreg": 0.0, "total": 0.0}
        steps = 0
        for batch in tqdm(loader, desc=f"LeJEPA epoch {epoch + 1}"):
            prompt_texts = [f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n" for text in batch["question"]]
            answer_texts = [f"{text}<|im_end|>" for text in batch["answer"]]
            prompt_emb = embed_texts(prompt_texts, tokenizer, backbone, cfg)
            answer_emb = embed_texts(answer_texts, tokenizer, backbone, cfg)

            pred, target, context = probe(prompt_emb, answer_emb)
            loss_pred = nn.functional.smooth_l1_loss(pred, target.detach())
            loss_sigreg = sigreg_style_loss(pred) + sigreg_style_loss(target) + sigreg_style_loss(context)
            loss = loss_pred + cfg.lambda_sigreg * loss_sigreg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            running["pred"] += float(loss_pred.item())
            running["sigreg"] += float(loss_sigreg.item())
            running["total"] += float(loss.item())
            steps += 1

        row = {"epoch": epoch + 1, **{key: value / max(steps, 1) for key, value in running.items()}}
        history.append(row)
        torch.save(
            {
                "model_state_dict": probe.state_dict(),
                "config": asdict(cfg),
                "hidden_size": backbone.config.hidden_size,
                "latent_dim": cfg.latent_dim,
            },
            save_path / f"lejepa_epoch_{epoch + 1}.pt",
        )
        with (save_path / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    with (save_path / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    train()
