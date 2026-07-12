"""Train a LoRA adapter with HNN or forced/damped-HNN latent regularization."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.dynamics.hamiltonian import DAMPING_MODES, LatentHamiltonianDynamics, STRUCTURE_MODES, VARIANTS
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
DEFAULT_AE = "/root/autodl-tmp/checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"
DEFAULT_TRAIN = "/root/autodl-tmp/data/train_kegg_pathway_record_balanced_0p1pct.csv"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/hamiltonian_joint"


@dataclass
class TrainConfig:
    base_model: str
    sft_lora: str
    ae_checkpoint: str
    train_path: str
    validation_path: str | None
    save_dir: str
    variant: str
    structure_mode: str
    damping_mode: str
    hidden_dim: int
    latent_dim: int
    batch_size: int
    gradient_accumulation_steps: int
    lr: float
    dynamics_lr: float
    epochs: int
    max_length: int
    answer_budget_fraction: float
    max_dynamics_steps: int
    dynamics_dt: float
    lambda_align: float
    lambda_state: float
    lambda_latent_state: float
    lambda_structure: float
    lambda_force: float
    lambda_damping: float
    validation_fraction: float
    validation_group_column: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    seed: int
    deterministic: bool
    limit: int | None
    hash_inputs: bool
    gradient_checkpointing: bool
    device: str


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft-lora", default=DEFAULT_SFT)
    parser.add_argument("--ae-ckpt", dest="ae_checkpoint", default=DEFAULT_AE)
    parser.add_argument("--train", dest="train_path", default=DEFAULT_TRAIN)
    parser.add_argument("--validation", dest="validation_path")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE)
    parser.add_argument("--variant", choices=VARIANTS, default="forced_damped_hnn")
    parser.add_argument("--structure-mode", choices=STRUCTURE_MODES, default="orthogonal_poisson")
    parser.add_argument("--damping-mode", choices=DAMPING_MODES, default="isotropic")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dynamics-lr", "--hnn-lr", dest="dynamics_lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--answer-budget-fraction", type=float, default=0.5)
    parser.add_argument(
        "--max-dynamics-steps",
        type=int,
        default=128,
        help="Maximum ordered graph-layer transitions; longer targets are reported and truncated.",
    )
    parser.add_argument(
        "--dynamics-dt",
        type=float,
        default=1.0 / 128.0,
        help="Fixed batch-independent surrogate-time increment per graph layer.",
    )
    parser.add_argument("--lambda-align", type=float, default=0.5)
    parser.add_argument("--lambda-state", type=float, default=0.5)
    parser.add_argument("--lambda-latent-state", type=float, default=0.1)
    parser.add_argument("--lambda-structure", type=float, default=1e-4)
    parser.add_argument("--lambda-force", type=float, default=1e-4)
    parser.add_argument("--lambda-damping", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-group-column", default="source_json")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hash-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_dynamics_steps < 1:
        parser.error("--max-dynamics-steps must be positive")
    if args.dynamics_dt <= 0:
        parser.error("--dynamics-dt must be positive")
    if not 0 < args.answer_budget_fraction < 1:
        parser.error("--answer-budget-fraction must be between 0 and 1")
    values = vars(args)
    values["gradient_checkpointing"] = not values.pop("no_gradient_checkpointing")
    return TrainConfig(**values)


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


def unwrap_state_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "projection"):
            if isinstance(raw.get(key), dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise ValueError("AE checkpoint does not contain a state dictionary")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in raw.items()}


class CSVPathwayDataset(Dataset):
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
        prompt = f"<|im_start|>user\n{row.get('question', '')}<|im_end|>\n<|im_start|>assistant\n"
        answer = str(row.get("answer", row.get("formatted_answer_no_phenotype", "")))
        encoded = encode_supervised(
            self.tokenizer,
            prompt,
            answer,
            max_length=self.max_length,
            answer_budget_fraction=self.answer_budget_fraction,
        )
        return {
            "input_ids": torch.tensor(encoded.input_ids, dtype=torch.long),
            "labels": torch.tensor(encoded.labels, dtype=torch.long),
            "step_span_groups": [
                torch.tensor(group, dtype=torch.long).reshape(-1, 2)
                for group in encoded.step_span_groups
            ],
            "prompt_tokens_dropped": encoded.prompt_tokens_dropped,
            "answer_tokens_dropped": encoded.answer_tokens_dropped,
            "substeps_total": encoded.substeps_total,
            "substeps_retained": encoded.substeps_retained,
            "semantic_steps_total": encoded.semantic_steps_total,
            "semantic_steps_retained": encoded.semantic_steps_retained,
        }


def make_collate_fn(pad_id: int):
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(item["input_ids"]) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "step_span_groups": [item["step_span_groups"] for item in batch],
            "prompt_tokens_dropped": sum(int(item["prompt_tokens_dropped"]) for item in batch),
            "answer_tokens_dropped": sum(int(item["answer_tokens_dropped"]) for item in batch),
            "substeps_total": sum(int(item["substeps_total"]) for item in batch),
            "substeps_retained": sum(int(item["substeps_retained"]) for item in batch),
            "semantic_steps_total": sum(int(item["semantic_steps_total"]) for item in batch),
            "semantic_steps_retained": sum(int(item["semantic_steps_retained"]) for item in batch),
        }

    return collate


def load_frames(cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = pd.read_csv(cfg.train_path, engine="c", quoting=csv.QUOTE_MINIMAL, on_bad_lines="error")
    if cfg.limit is not None:
        train_frame = train_frame.head(cfg.limit)
    required = {"question"}
    missing = required - set(train_frame.columns)
    if missing:
        raise ValueError(f"training CSV missing columns: {', '.join(sorted(missing))}")
    if cfg.validation_path:
        validation_frame = pd.read_csv(
            cfg.validation_path, engine="c", quoting=csv.QUOTE_MINIMAL, on_bad_lines="error"
        )
        if train_frame.empty or validation_frame.empty:
            raise ValueError("training and explicit validation CSVs must both contain rows")
        ensure_disjoint_groups(
            train_frame,
            validation_frame,
            group_column=cfg.validation_group_column,
        )
        return train_frame.reset_index(drop=True), validation_frame.reset_index(drop=True)
    return stable_group_split(
        train_frame,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
        group_column=cfg.validation_group_column,
    )


def fixed_time_grid(cfg: TrainConfig, device: str) -> torch.Tensor:
    return torch.arange(
        cfg.max_dynamics_steps + 1,
        device=device,
        dtype=torch.float32,
    ) * cfg.dynamics_dt


def batch_losses(
    *,
    model: nn.Module,
    projection: CascadeProjection,
    dynamics: LatentHamiltonianDynamics,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: str,
    training: bool,
    time_grid: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    inputs = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    model_context = nullcontext() if training else torch.no_grad()
    with model_context:
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )
        loss_sft = outputs.loss
        hidden = outputs.hidden_states[-1]
        latent, _ = projection(hidden.float())

    answer_mask = labels != -100
    first_answer = answer_mask.to(torch.int64).argmax(dim=1)
    last_prompt = (first_answer - 1).clamp(min=0)
    valid_indices = [
        index
        for index, groups in enumerate(batch["step_span_groups"])
        if bool(answer_mask[index].any()) and len(groups) > 0
    ]
    if not valid_indices:
        zero = loss_sft.new_zeros(())
        return {
            "total": loss_sft,
            "sft": loss_sft,
            "align": zero,
            "state": zero,
            "latent_state": zero,
            "regularization": zero,
            "dynamics_truncated_samples": 0,
            "dynamics_truncated_semantic_steps": 0,
            "text_truncated_substeps": int(batch["substeps_total"]) - int(batch["substeps_retained"]),
            "text_truncated_semantic_steps": int(batch["semantic_steps_total"]) - int(batch["semantic_steps_retained"]),
            "prompt_tokens_dropped": int(batch["prompt_tokens_dropped"]),
            "answer_tokens_dropped": int(batch["answer_tokens_dropped"]),
        }

    valid_tensor = torch.tensor(valid_indices, dtype=torch.long, device=hidden.device)
    z0 = latent[valid_tensor, last_prompt[valid_tensor]]
    maximum_used_steps = max(
        min(len(batch["step_span_groups"][index]), cfg.max_dynamics_steps)
        for index in valid_indices
    )
    trajectory = odeint(
        dynamics,
        z0.float(),
        time_grid[: maximum_used_steps + 1],
        method="rk4",
    ).transpose(0, 1)
    predicted_hidden = projection.up(trajectory)

    velocity_similarities: list[torch.Tensor] = []
    state_similarities: list[torch.Tensor] = []
    latent_state_losses: list[torch.Tensor] = []
    dynamics_truncated_samples = 0
    dynamics_truncated_semantic_steps = 0
    for local_index, batch_index in enumerate(valid_indices):
        groups = batch["step_span_groups"][batch_index]
        raw_length = len(groups)
        used_length = min(raw_length, cfg.max_dynamics_steps)
        if raw_length > used_length:
            dynamics_truncated_samples += 1
            dynamics_truncated_semantic_steps += raw_length - used_length
        # Targets come from the stage-1-initialized language representation.
        # Stop gradients through this branch so the auxiliary loss cannot make
        # its own target move toward the dynamics prediction in the same step;
        # LoRA is updated through CE and the prompt-anchor rollout path.
        target_states = torch.stack([
            torch.cat([
                hidden[batch_index, int(start) : int(end)].float()
                for start, end in group.to(hidden.device).tolist()
            ], dim=0).mean(dim=0).detach()
            for group in groups[:used_length]
        ])
        generated_states = predicted_hidden[local_index, 1 : used_length + 1]
        generated_velocity = (
            predicted_hidden[local_index, 1 : used_length + 1]
            - predicted_hidden[local_index, :used_length]
        )
        target_sequence = torch.cat(
            [hidden[batch_index, last_prompt[batch_index]].float().unsqueeze(0), target_states],
            dim=0,
        )
        target_velocity = target_sequence[1:] - target_sequence[:-1]
        velocity_similarities.append(
            nn.functional.cosine_similarity(generated_velocity, target_velocity, dim=-1).mean()
        )
        state_similarities.append(
            nn.functional.cosine_similarity(generated_states, target_states, dim=-1).mean()
        )
        target_latent_states = projection.down(target_states)
        latent_state_losses.append(
            nn.functional.smooth_l1_loss(
                trajectory[local_index, 1 : used_length + 1],
                target_latent_states,
            )
        )

    loss_align = 1.0 - torch.stack(velocity_similarities).mean()
    loss_state = 1.0 - torch.stack(state_similarities).mean()
    loss_latent_state = torch.stack(latent_state_losses).mean()
    loss_regularization = dynamics.regularization_loss(
        time_grid[: maximum_used_steps + 1],
        z0,
        lambda_structure=cfg.lambda_structure,
        lambda_force=cfg.lambda_force,
        lambda_damping=cfg.lambda_damping,
    )
    total = (
        loss_sft
        + cfg.lambda_align * loss_align
        + cfg.lambda_state * loss_state
        + cfg.lambda_latent_state * loss_latent_state
        + loss_regularization
    )
    return {
        "total": total,
        "sft": loss_sft,
        "align": loss_align,
        "state": loss_state,
        "latent_state": loss_latent_state,
        "regularization": loss_regularization,
        "dynamics_truncated_samples": dynamics_truncated_samples,
        "dynamics_truncated_semantic_steps": dynamics_truncated_semantic_steps,
        "text_truncated_substeps": int(batch["substeps_total"]) - int(batch["substeps_retained"]),
        "text_truncated_semantic_steps": int(batch["semantic_steps_total"]) - int(batch["semantic_steps_retained"]),
        "prompt_tokens_dropped": int(batch["prompt_tokens_dropped"]),
        "answer_tokens_dropped": int(batch["answer_tokens_dropped"]),
    }


def evaluate(
    *,
    model: nn.Module,
    projection: CascadeProjection,
    dynamics: LatentHamiltonianDynamics,
    loader: DataLoader,
    cfg: TrainConfig,
    device: str,
    time_grid: torch.Tensor,
) -> dict[str, float | int]:
    model.eval()
    dynamics.eval()
    totals = {
        "total": 0.0,
        "sft": 0.0,
        "align": 0.0,
        "state": 0.0,
        "latent_state": 0.0,
        "regularization": 0.0,
    }
    counters = {
        "dynamics_truncated_samples": 0,
        "dynamics_truncated_semantic_steps": 0,
        "text_truncated_substeps": 0,
        "text_truncated_semantic_steps": 0,
        "prompt_tokens_dropped": 0,
        "answer_tokens_dropped": 0,
    }
    steps = 0
    for batch in loader:
        losses = batch_losses(
            model=model,
            projection=projection,
            dynamics=dynamics,
            batch=batch,
            cfg=cfg,
            device=device,
            training=False,
            time_grid=time_grid,
        )
        for key in totals:
            totals[key] += float(losses[key].detach().item())  # type: ignore[union-attr]
        for key in counters:
            counters[key] += int(losses[key])
        steps += 1
    return {
        **{key: value / max(steps, 1) for key, value in totals.items()},
        "steps": steps,
        **counters,
    }


def save_checkpoint(
    *,
    destination: Path,
    model: nn.Module,
    dynamics: LatentHamiltonianDynamics,
    cfg: TrainConfig,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "dynamics_config": dynamics.export_config(),
        "model_state_dict": dynamics.state_dict(),
        "metrics": metrics,
        "training_config": asdict(cfg),
    }
    torch.save(payload, destination / "hamiltonian_dynamics.pt")
    write_json(destination / "checkpoint_metrics.json", metrics)


def train(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or parse_args()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    save_root = ensure_new_output_dir(cfg.save_dir)
    logger = configure_logger(save_root / "train.log", f"framework_a.{cfg.variant}")
    write_json(save_root / "run_config.json", asdict(cfg))

    train_frame, validation_frame = load_frames(cfg)
    manifest = {
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "train_path": str(Path(cfg.train_path).resolve()),
        "validation_path": str(Path(cfg.validation_path).resolve()) if cfg.validation_path else "deterministic_group_split",
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "seed": cfg.seed,
        "validation_group_column": cfg.validation_group_column,
        "base_model_identity": base_model_identity(cfg.base_model),
        "sft_adapter_sha256": artifact_sha256(cfg.sft_lora),
        "ae_checkpoint_sha256": artifact_sha256(cfg.ae_checkpoint),
    }
    if cfg.hash_inputs:
        manifest["train_sha256"] = file_sha256(cfg.train_path)
        if cfg.validation_path:
            manifest["validation_sha256"] = file_sha256(cfg.validation_path)
    write_json(save_root / "run_manifest.json", manifest)
    logger.info("variant=%s structure=%s train_rows=%d validation_rows=%d", cfg.variant, cfg.structure_mode, len(train_frame), len(validation_frame))

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
    model = PeftModel.from_pretrained(base_model, cfg.sft_lora, is_trainable=True)
    model.enable_input_require_grads()

    projection = CascadeProjection(
        high_dim=base_model.config.hidden_size,
        latent_dim=cfg.latent_dim,
    ).to(cfg.device).float()
    projection.load_state_dict(unwrap_state_dict(torch.load(cfg.ae_checkpoint, map_location=cfg.device)))
    projection.requires_grad_(False)
    projection.eval()

    dynamics = LatentHamiltonianDynamics(
        cfg.latent_dim,
        variant=cfg.variant,
        hidden_dim=cfg.hidden_dim,
        structure_mode=cfg.structure_mode,
        damping_mode=cfg.damping_mode,
    ).to(cfg.device).float()
    optimizer = optim.AdamW(
        [
            {"params": [parameter for parameter in model.parameters() if parameter.requires_grad], "lr": cfg.lr},
            {"params": dynamics.parameters(), "lr": cfg.dynamics_lr},
        ]
    )

    # Dynamics variants allocate different parameter counts. Reset stochastic
    # state after construction so LoRA dropout and batch order remain matched.
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    generator = torch.Generator().manual_seed(cfg.seed)
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        CSVPathwayDataset(train_frame, tokenizer, cfg.max_length, cfg.answer_budget_fraction),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        CSVPathwayDataset(validation_frame, tokenizer, cfg.max_length, cfg.answer_budget_fraction),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    time_grid = fixed_time_grid(cfg, cfg.device)
    early_stopping = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_min_delta)
    history: list[dict[str, Any]] = []
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        dynamics.train()
        sums = {
            "total": 0.0,
            "sft": 0.0,
            "align": 0.0,
            "state": 0.0,
            "latent_state": 0.0,
            "regularization": 0.0,
        }
        counters = {
            "dynamics_truncated_samples": 0,
            "dynamics_truncated_semantic_steps": 0,
            "text_truncated_substeps": 0,
            "text_truncated_semantic_steps": 0,
            "prompt_tokens_dropped": 0,
            "answer_tokens_dropped": 0,
        }
        progress = tqdm(train_loader, desc=f"{cfg.variant} epoch {epoch}")
        for step, batch in enumerate(progress):
            losses = batch_losses(
                model=model,
                projection=projection,
                dynamics=dynamics,
                batch=batch,
                cfg=cfg,
                device=cfg.device,
                training=True,
                time_grid=time_grid,
            )
            divisor = accumulation_divisor(step, len(train_loader), cfg.gradient_accumulation_steps)
            (losses["total"] / divisor).backward()  # type: ignore[operator]
            for key in sums:
                sums[key] += float(losses[key].detach().item())  # type: ignore[union-attr]
            for key in counters:
                counters[key] += int(losses[key])

            group_end = (step + 1) % cfg.gradient_accumulation_steps == 0 or step + 1 == len(train_loader)
            if group_end:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(
                sft=f"{float(losses['sft'].detach().item()):.3f}",  # type: ignore[union-attr]
                align=f"{float(losses['align'].detach().item()):.3f}",  # type: ignore[union-attr]
            )

        train_metrics = {key: value / max(len(train_loader), 1) for key, value in sums.items()}
        train_metrics.update(counters)
        validation_metrics = evaluate(
            model=model,
            projection=projection,
            dynamics=dynamics,
            loader=validation_loader,
            cfg=cfg,
            device=cfg.device,
            time_grid=time_grid,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "dynamics_learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        append_jsonl(save_root / "metrics.jsonl", record)
        write_json(save_root / "history.json", history)
        logger.info(
            "epoch=%d train_total=%.6f val_total=%.6f val_sft=%.6f val_velocity=%.6f val_state=%.6f dynamics_truncated_semantic_steps=%d text_truncated_substeps=%d",
            epoch,
            train_metrics["total"],
            validation_metrics["total"],
            validation_metrics["sft"],
            validation_metrics["align"],
            validation_metrics["state"],
            counters["dynamics_truncated_semantic_steps"],
            counters["text_truncated_substeps"],
        )

        epoch_dir = save_root / f"checkpoint_epoch_{epoch}"
        save_checkpoint(
            destination=epoch_dir,
            model=model,
            dynamics=dynamics,
            cfg=cfg,
            epoch=epoch,
            metrics=record,
        )
        improved, should_stop = early_stopping.update(float(validation_metrics["total"]), epoch)
        if improved:
            save_checkpoint(
                destination=save_root / "checkpoint_best",
                model=model,
                dynamics=dynamics,
                cfg=cfg,
                epoch=epoch,
                metrics=record,
            )
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
            "variant": cfg.variant,
            "structure_mode": cfg.structure_mode,
        },
    )


if __name__ == "__main__":
    train()
