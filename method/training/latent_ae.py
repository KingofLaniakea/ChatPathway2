"""Train the shared reconstruction AE bridge with deterministic model selection."""

from __future__ import annotations

import argparse
import json
import time
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
from method.training.csv_io import read_training_csv
from method.training.prefix_sampling import (
    EpochPrefixView,
    PREFIX_POLICIES,
    PREFIX_SAMPLING_MODES,
)


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_SFT = "/root/autodl-tmp/checkpoints/shared/pathway_sft/checkpoint_best"
DEFAULT_TRAIN = "/root/autodl-tmp/data/pathway_v4_full/train_pathway_continuation_v4.csv"
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
    prefix_sampling: str
    prefix_policy: str
    latent_dim: int
    cosine_weight: float
    predictive_weight: float
    latent_mean_weight: float
    latent_variance_weight: float
    latent_covariance_weight: float
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--answer-budget-fraction", type=float, default=0.5)
    parser.add_argument(
        "--prefix-sampling",
        choices=PREFIX_SAMPLING_MODES,
        default="one_per_record",
    )
    parser.add_argument(
        "--prefix-policy",
        choices=tuple(PREFIX_POLICIES),
        default="balanced_cycle",
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument(
        "--cosine-weight",
        type=float,
        default=0.0,
        help="Optional reconstruction-direction ablation; the maintained B1 baseline is pure MSE.",
    )
    parser.add_argument(
        "--predictive-weight",
        type=float,
        default=0.0,
        help="Optional B2 one-layer latent prediction objective.",
    )
    parser.add_argument("--latent-mean-weight", type=float, default=0.0)
    parser.add_argument("--latent-variance-weight", type=float, default=0.0)
    parser.add_argument("--latent-covariance-weight", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-group-column", default="pathway_family_id")
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
    for name in (
        "cosine_weight",
        "predictive_weight",
        "latent_mean_weight",
        "latent_variance_weight",
        "latent_covariance_weight",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
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


class LatentTransitionPredictor(nn.Module):
    """Small discarded-after-training head used by the registered B2 AE arm."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class AEDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
        answer_budget_fraction: float,
        prefix_sampling: str,
        prefix_policy: str,
        seed: int,
    ):
        self.prefix_view = EpochPrefixView(
            frame.to_dict(orient="records"),
            sampling_mode=prefix_sampling,
            policy=prefix_policy,
            seed=seed,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_budget_fraction = answer_budget_fraction

    def __len__(self) -> int:
        return len(self.prefix_view)

    def set_epoch(self, epoch: int) -> None:
        self.prefix_view.set_epoch(epoch)

    def selection_summary(self) -> dict[str, int]:
        return self.prefix_view.selection_summary()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.prefix_view.row(index)
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
            "answer_mask": torch.tensor(
                [int(label != -100) for label in encoded.labels],
                dtype=torch.long,
            ),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
            "step_span_groups": [
                torch.tensor(group, dtype=torch.long).reshape(-1, 2)
                for group in encoded.step_span_groups
            ],
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
        answer_mask = torch.nn.utils.rnn.pad_sequence(
            [item["answer_mask"] for item in batch], batch_first=True, padding_value=0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(item["input_ids"]) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_mask": answer_mask,
            "loss_mask": loss_mask,
            "step_span_groups": [item["step_span_groups"] for item in batch],
            "prompt_tokens_dropped": sum(int(item["prompt_tokens_dropped"]) for item in batch),
            "answer_tokens_dropped": sum(int(item["answer_tokens_dropped"]) for item in batch),
        }

    return collate


def load_frames(cfg: AEConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_training_csv(cfg.train_path)
    if cfg.limit is not None:
        frame = frame.head(cfg.limit)
    if frame.empty:
        raise ValueError("AE training CSV contains no rows")
    if cfg.prefix_sampling == "one_per_record" and "record_id" not in frame.columns:
        raise ValueError("one_per_record prefix sampling requires record_id")
    if cfg.validation_path:
        validation = read_training_csv(cfg.validation_path)
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


def latent_geometry_losses(latent: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return scale/conditioning diagnostics without assigning q/p semantics."""

    if latent.ndim != 2:
        raise ValueError("latent geometry expects [tokens, latent_dim]")
    if latent.size(0) < 2:
        zero = latent.new_zeros(())
        return {"latent_mean": zero, "latent_variance": zero, "latent_covariance": zero}
    centered = latent - latent.mean(dim=0, keepdim=True)
    variance = centered.pow(2).mean(dim=0)
    covariance = centered.transpose(0, 1) @ centered / float(latent.size(0) - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return {
        "latent_mean": latent.mean(dim=0).pow(2).mean(),
        "latent_variance": (variance - 1.0).pow(2).mean(),
        "latent_covariance": off_diagonal.pow(2).mean(),
    }


def _predictive_loss(
    latent: torch.Tensor,
    batch: dict[str, Any],
    predictor: LatentTransitionPredictor | None,
) -> torch.Tensor:
    if predictor is None:
        return latent.new_zeros(())
    answer_mask = batch["answer_mask"].to(latent.device).bool()
    sample_losses: list[torch.Tensor] = []
    for sample_index, groups in enumerate(batch["step_span_groups"]):
        if not groups or not bool(answer_mask[sample_index].any()):
            continue
        first_answer = int(answer_mask[sample_index].to(torch.int64).argmax().item())
        anchor = latent[sample_index, max(first_answer - 1, 0)]
        layer_states = torch.stack([
            torch.cat([
                latent[sample_index, int(start) : int(end)]
                for start, end in group.to(latent.device).tolist()
            ]).mean(dim=0)
            for group in groups
        ])
        sequence = torch.cat([anchor.unsqueeze(0), layer_states], dim=0)
        predicted = predictor(sequence[:-1])
        sample_losses.append(nn.functional.smooth_l1_loss(predicted, sequence[1:].detach()))
    if not sample_losses:
        return latent.new_zeros(())
    return torch.stack(sample_losses).mean()


def reconstruction_losses(
    *,
    backbone: nn.Module,
    projection: CascadeProjection,
    batch: dict[str, torch.Tensor],
    device: str,
    cosine_weight: float,
    predictive_weight: float,
    latent_mean_weight: float,
    latent_variance_weight: float,
    latent_covariance_weight: float,
    predictor: LatentTransitionPredictor | None = None,
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
    latent, reconstructed = projection(hidden)
    active = loss_mask.bool()
    if not active.any():
        return None
    target = hidden[active]
    prediction = reconstructed[active]
    mse = nn.functional.mse_loss(prediction, target)
    cosine = 1.0 - nn.functional.cosine_similarity(prediction, target, dim=-1).mean()
    geometry = latent_geometry_losses(latent[active])
    predictive = _predictive_loss(latent, batch, predictor)
    total = (
        mse
        + cosine_weight * cosine
        + predictive_weight * predictive
        + latent_mean_weight * geometry["latent_mean"]
        + latent_variance_weight * geometry["latent_variance"]
        + latent_covariance_weight * geometry["latent_covariance"]
    )
    return {
        "total": total,
        "mse": mse,
        "cosine": cosine,
        "predictive": predictive,
        **geometry,
    }


def evaluate(
    backbone: nn.Module,
    projection: CascadeProjection,
    predictor: LatentTransitionPredictor | None,
    loader: DataLoader,
    cfg: AEConfig,
) -> dict[str, float | int]:
    projection.eval()
    if predictor is not None:
        predictor.eval()
    sums = {
        "total": 0.0,
        "mse": 0.0,
        "cosine": 0.0,
        "predictive": 0.0,
        "latent_mean": 0.0,
        "latent_variance": 0.0,
        "latent_covariance": 0.0,
    }
    steps = 0
    prompt_dropped = answer_dropped = 0
    for batch in loader:
        with torch.no_grad():
            losses = reconstruction_losses(
                backbone=backbone,
                projection=projection,
                batch=batch,
                device=cfg.device,
                cosine_weight=cfg.cosine_weight,
                predictive_weight=cfg.predictive_weight,
                latent_mean_weight=cfg.latent_mean_weight,
                latent_variance_weight=cfg.latent_variance_weight,
                latent_covariance_weight=cfg.latent_covariance_weight,
                predictor=predictor,
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


def save_checkpoint(
    destination: Path,
    projection: CascadeProjection,
    predictor: LatentTransitionPredictor | None,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(projection.state_dict(), destination / "ae_proj.pt")
    if predictor is not None:
        torch.save(predictor.state_dict(), destination / "transition_predictor.pt")
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
        "train_eligible_prefix_rows": len(train_frame),
        "train_records": (
            int(train_frame["record_id"].nunique())
            if "record_id" in train_frame.columns
            else len(train_frame)
        ),
        "train_samples_per_epoch": (
            int(train_frame["record_id"].nunique())
            if cfg.prefix_sampling == "one_per_record"
            else len(train_frame)
        ),
        "validation_prefix_rows": len(validation_frame),
        "validation_records": (
            int(validation_frame["record_id"].nunique())
            if "record_id" in validation_frame.columns
            else len(validation_frame)
        ),
        "validation_samples": (
            int(validation_frame["record_id"].nunique())
            if "record_id" in validation_frame.columns
            else len(validation_frame)
        ),
        "validation_prefix_sampling": (
            "one_per_record"
            if "record_id" in validation_frame.columns
            else "all_rows"
        ),
        "validation_prefix_policy": "balanced_cycle",
        "prefix_sampling": cfg.prefix_sampling,
        "prefix_policy": cfg.prefix_policy,
        "seed": cfg.seed,
        "base_model_identity": base_model_identity(cfg.base_model),
        "sft_adapter_sha256": artifact_sha256(cfg.sft_lora),
    }
    if cfg.hash_inputs:
        manifest["train_sha256"] = file_sha256(cfg.train_path)
        if cfg.validation_path:
            manifest["validation_sha256"] = file_sha256(cfg.validation_path)
    write_json(save_root / "run_manifest.json", manifest)
    logger.info(
        "train_eligible_prefix_rows=%d train_records=%d train_samples_per_epoch=%d "
        "validation_prefix_rows=%d validation_records=%d validation_samples=%d seed=%d",
        manifest["train_eligible_prefix_rows"],
        manifest["train_records"],
        manifest["train_samples_per_epoch"],
        manifest["validation_prefix_rows"],
        manifest["validation_records"],
        manifest["validation_samples"],
        cfg.seed,
    )

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
    predictor = (
        LatentTransitionPredictor(cfg.latent_dim).to(cfg.device).float()
        if cfg.predictive_weight > 0
        else None
    )
    trainable_parameters = list(projection.parameters())
    if predictor is not None:
        trainable_parameters.extend(predictor.parameters())
    optimizer = optim.AdamW(trainable_parameters, lr=cfg.lr)
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_dataset = AEDataset(
        train_frame,
        tokenizer,
        cfg.max_length,
        cfg.answer_budget_fraction,
        cfg.prefix_sampling,
        cfg.prefix_policy,
        cfg.seed,
    )
    validation_dataset = AEDataset(
        validation_frame,
        tokenizer,
        cfg.max_length,
        cfg.answer_budget_fraction,
        "one_per_record" if "record_id" in validation_frame.columns else "all_rows",
        "balanced_cycle",
        cfg.seed,
    )
    manifest["validation_prefix_selection"] = validation_dataset.selection_summary()
    write_json(save_root / "run_manifest.json", manifest)
    logger.info(
        "validation_prefix_selection=%s",
        json.dumps(manifest["validation_prefix_selection"], sort_keys=True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    early_stopping = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_min_delta)
    history: list[dict[str, Any]] = []
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        train_dataset.set_epoch(epoch)
        prefix_selection = train_dataset.selection_summary()
        logger.info(
            "epoch=%d train_samples=%d prefix_selection=%s",
            epoch,
            len(train_dataset),
            prefix_selection,
        )
        projection.train()
        if predictor is not None:
            predictor.train()
        train_started = time.perf_counter()
        sums = {
            "total": 0.0,
            "mse": 0.0,
            "cosine": 0.0,
            "predictive": 0.0,
            "latent_mean": 0.0,
            "latent_variance": 0.0,
            "latent_covariance": 0.0,
        }
        steps = 0
        prompt_dropped = answer_dropped = 0
        input_tokens = reconstruction_tokens = 0
        progress = tqdm(train_loader, desc=f"AE epoch {epoch}")
        for step, batch in enumerate(progress):
            input_tokens += int(batch["attention_mask"].sum().item())
            reconstruction_tokens += int(batch["loss_mask"].sum().item())
            losses = reconstruction_losses(
                backbone=backbone,
                projection=projection,
                batch=batch,
                device=cfg.device,
                cosine_weight=cfg.cosine_weight,
                predictive_weight=cfg.predictive_weight,
                latent_mean_weight=cfg.latent_mean_weight,
                latent_variance_weight=cfg.latent_variance_weight,
                latent_covariance_weight=cfg.latent_covariance_weight,
                predictor=predictor,
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
                nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(total=f"{losses['total'].item():.4f}", mse=f"{losses['mse'].item():.4f}")

        train_metrics: dict[str, float | int] = {
            **{key: value / max(steps, 1) for key, value in sums.items()},
            "prompt_tokens_dropped": prompt_dropped,
            "answer_tokens_dropped": answer_dropped,
            "input_tokens": input_tokens,
            "reconstruction_tokens": reconstruction_tokens,
            "seconds": time.perf_counter() - train_started,
        }
        train_metrics["input_tokens_per_second"] = input_tokens / max(
            float(train_metrics["seconds"]), 1e-9
        )
        train_metrics["reconstruction_tokens_per_second"] = reconstruction_tokens / max(
            float(train_metrics["seconds"]), 1e-9
        )
        validation_started = time.perf_counter()
        validation_metrics = evaluate(backbone, projection, predictor, validation_loader, cfg)
        validation_metrics["seconds"] = time.perf_counter() - validation_started
        record = {
            "epoch": epoch,
            "prefix_selection": prefix_selection,
            "train": train_metrics,
            "validation": validation_metrics,
            "lr": cfg.lr,
        }
        history.append(record)
        append_jsonl(save_root / "metrics.jsonl", record)
        write_json(save_root / "history.json", history)
        logger.info(
            "epoch=%d train_total=%.6f val_total=%.6f val_mse=%.6f val_cosine=%.6f train_seconds=%.1f validation_seconds=%.1f reconstruction_tokens_per_second=%.1f",
            epoch,
            train_metrics["total"],
            validation_metrics["total"],
            validation_metrics["mse"],
            validation_metrics["cosine"],
            train_metrics["seconds"],
            validation_metrics["seconds"],
            train_metrics["reconstruction_tokens_per_second"],
        )
        save_checkpoint(save_root / f"checkpoint_epoch_{epoch}", projection, predictor, epoch, record)
        improved, should_stop = early_stopping.update(validation_metrics["total"], epoch)
        if improved:
            save_checkpoint(save_root / "checkpoint_best", projection, predictor, epoch, record)
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
