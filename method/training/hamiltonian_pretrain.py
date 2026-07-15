"""Pretrain HNN/FDHNN in a fixed stage-1 SFT plus frozen-AE latent space."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.dynamics.hamiltonian import (
    DAMPING_MODES,
    STRUCTURE_MODES,
    VARIANTS,
    LatentHamiltonianDynamics,
)
from method.training.common import (
    accumulation_divisor,
    append_jsonl,
    artifact_sha256,
    base_model_identity,
    configure_logger,
    ensure_new_output_dir,
    file_sha256,
    git_commit,
    seed_everything,
    write_json,
)
from method.training.framework_a import (
    CSVPathwayDataset,
    CascadeProjection,
    batch_losses,
    fixed_time_grid,
    load_frames,
    make_collate_fn,
    unwrap_state_dict,
)
from method.training.prefix_sampling import PREFIX_POLICIES, PREFIX_SAMPLING_MODES
from method.training.staged_objectives import DYNAMICS_RESOLUTIONS, finite_metric_record


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_SFT = "/root/autodl-tmp/checkpoints/shared/pathway_sft/checkpoint_best"
DEFAULT_AE = "/root/autodl-tmp/checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"
DEFAULT_TRAIN = "/root/autodl-tmp/data/pathway_v4_full/train_pathway_continuation_v4.csv"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/hamiltonian_pretrain"


@dataclass
class PretrainConfig:
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
    dynamics_lr: float
    epochs: int
    max_length: int
    answer_budget_fraction: float
    prefix_sampling: str
    prefix_policy: str
    max_dynamics_steps: int
    dynamics_dt: float
    dynamics_resolution: str
    substep_dt: float
    lambda_align: float
    lambda_state: float
    lambda_latent_state: float
    lambda_structure: float
    lambda_force: float
    lambda_damping: float
    validation_fraction: float
    validation_group_column: str
    stability_min_epochs: int
    stability_min_coverage: float
    stability_min_relative_improvement: float
    stability_max_relative_regression: float
    seed: int
    deterministic: bool
    limit: int | None
    hash_inputs: bool
    device: str
    # The shared loss implementation reads these stage-2 fields.
    kl_weight: float = 0.0


def parse_args() -> PretrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft-lora", default=DEFAULT_SFT)
    parser.add_argument("--ae-ckpt", dest="ae_checkpoint", default=DEFAULT_AE)
    parser.add_argument("--train", dest="train_path", default=DEFAULT_TRAIN)
    parser.add_argument("--validation", dest="validation_path")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--structure-mode", choices=STRUCTURE_MODES, default="orthogonal_poisson")
    parser.add_argument("--damping-mode", choices=DAMPING_MODES, default="isotropic")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=12)
    parser.add_argument("--dynamics-lr", "--hnn-lr", dest="dynamics_lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--answer-budget-fraction", type=float, default=0.5)
    parser.add_argument("--prefix-sampling", choices=PREFIX_SAMPLING_MODES, default="one_per_record")
    parser.add_argument("--prefix-policy", choices=tuple(PREFIX_POLICIES), default="dynamics_cycle")
    parser.add_argument("--max-dynamics-steps", type=int, default=512)
    parser.add_argument("--dynamics-dt", type=float, default=1.0 / 128.0)
    parser.add_argument("--dynamics-resolution", choices=DYNAMICS_RESOLUTIONS, default="substep_multiscale")
    parser.add_argument("--substep-dt", type=float, default=1.0 / 512.0)
    parser.add_argument("--lambda-align", type=float, default=0.5)
    parser.add_argument("--lambda-state", type=float, default=0.5)
    parser.add_argument("--lambda-latent-state", type=float, default=0.1)
    parser.add_argument("--lambda-structure", type=float, default=1e-4)
    parser.add_argument("--lambda-force", type=float, default=1e-4)
    parser.add_argument("--lambda-damping", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-group-column", default="pathway_family_id")
    parser.add_argument("--stability-min-epochs", type=int, default=2)
    parser.add_argument("--stability-min-coverage", type=float, default=0.95)
    parser.add_argument("--stability-min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--stability-max-relative-regression", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hash-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 3:
        parser.error("dynamics pretraining must use between 1 and 3 epochs")
    if not 1 <= args.stability_min_epochs <= args.epochs:
        parser.error("--stability-min-epochs must be within the epoch budget")
    if args.gradient_accumulation_steps < 1 or args.max_dynamics_steps < 1:
        parser.error("accumulation and dynamics-step counts must be positive")
    if args.substep_dt <= 0 or args.substep_dt >= args.dynamics_dt:
        parser.error("--substep-dt must be positive and smaller than --dynamics-dt")
    for name in (
        "stability_min_coverage",
        "stability_min_relative_improvement",
        "stability_max_relative_regression",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return PretrainConfig(**vars(args))


PRETRAIN_LOSS_KEYS = ("dynamics", "align", "state", "latent_state", "regularization")
PRETRAIN_COUNTER_KEYS = (
    "dynamics_valid_samples",
    "dynamics_layer_boundaries",
    "dynamics_truncated_substeps",
    "dynamics_truncated_samples",
    "dynamics_truncated_semantic_steps",
    "text_truncated_substeps",
    "text_truncated_semantic_steps",
    "prompt_tokens_dropped",
    "answer_tokens_dropped",
)


def stability_decision(
    validation_history: list[dict[str, float]],
    *,
    minimum_epochs: int,
    minimum_coverage: float,
    minimum_relative_improvement: float,
    maximum_relative_regression: float,
) -> tuple[bool, str]:
    if len(validation_history) < minimum_epochs:
        return False, "minimum_epochs_not_reached"
    if not all(
        finite_metric_record(row.get(key))
        for row in validation_history
        for key in ("dynamics", "coverage")
    ):
        return False, "non_finite_validation_metric"
    current = validation_history[-1]
    if current["coverage"] < minimum_coverage:
        return False, "insufficient_trajectory_coverage"
    initial = validation_history[0]["dynamics"]
    best = min(row["dynamics"] for row in validation_history)
    denominator = max(abs(initial), 1e-12)
    relative_improvement = (initial - best) / denominator
    if relative_improvement < minimum_relative_improvement:
        return False, "insufficient_validation_improvement"
    allowed = best * (1.0 + maximum_relative_regression) + 1e-12
    if current["dynamics"] > allowed:
        return False, "validation_regressed_from_best"
    return True, "finite_covered_improved_and_non_regressing"


def _dataset(frame: Any, tokenizer: Any, cfg: PretrainConfig, *, training: bool) -> CSVPathwayDataset:
    return CSVPathwayDataset(
        frame,
        tokenizer,
        cfg.max_length,
        cfg.answer_budget_fraction,
        cfg.prefix_sampling if training else "one_per_record",
        cfg.prefix_policy if training else "balanced_cycle",
        cfg.seed,
    )


def _metrics(
    model: nn.Module,
    projection: CascadeProjection,
    dynamics: LatentHamiltonianDynamics,
    loader: DataLoader,
    cfg: PretrainConfig,
    time_grid: torch.Tensor,
) -> dict[str, float | int]:
    model.eval()
    dynamics.eval()
    sums = {key: 0.0 for key in PRETRAIN_LOSS_KEYS}
    counters = {key: 0 for key in PRETRAIN_COUNTER_KEYS}
    examples = 0
    for batch in loader:
        losses = batch_losses(
            model=model,
            projection=projection,
            dynamics=dynamics,
            batch=batch,
            cfg=cfg,  # type: ignore[arg-type]
            device=cfg.device,
            training=False,
            time_grid=time_grid,
            dynamics_to_lora_scale=0.0,
        )
        batch_size = int(batch["input_ids"].shape[0])
        for key in sums:
            sums[key] += float(losses[key].detach().item()) * batch_size  # type: ignore[union-attr]
        for key in counters:
            counters[key] += int(losses[key])
        examples += batch_size
    values: dict[str, float | int] = {
        **{key: value / max(examples, 1) for key, value in sums.items()},
        **counters,
        "examples": examples,
    }
    values["coverage"] = counters["dynamics_valid_samples"] / max(examples, 1)
    return values


def _save_checkpoint(
    destination: Path,
    dynamics: LatentHamiltonianDynamics,
    cfg: PretrainConfig,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "epoch": epoch,
            "dynamics_config": dynamics.export_config(),
            "model_state_dict": dynamics.state_dict(),
            "metrics": metrics,
            "training_config": asdict(cfg),
        },
        destination / "hamiltonian_dynamics.pt",
    )
    write_json(destination / "checkpoint_metrics.json", metrics)


def train(cfg: PretrainConfig | None = None) -> None:
    cfg = cfg or parse_args()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    save_root = ensure_new_output_dir(cfg.save_dir)
    logger = configure_logger(save_root / "train.log", f"hamiltonian_pretrain.{cfg.variant}")
    write_json(save_root / "run_config.json", asdict(cfg))
    train_frame, validation_frame = load_frames(cfg)  # type: ignore[arg-type]
    manifest: dict[str, Any] = {
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "training_phase": "dynamics_only_fixed_sft_ae",
        "train_path": str(Path(cfg.train_path).resolve()),
        "validation_path": str(Path(cfg.validation_path).resolve()) if cfg.validation_path else "deterministic_group_split",
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "base_model_identity": base_model_identity(cfg.base_model),
        "sft_adapter_sha256": artifact_sha256(cfg.sft_lora),
        "ae_checkpoint_sha256": artifact_sha256(cfg.ae_checkpoint),
        "dynamics_resolution": cfg.dynamics_resolution,
        "seed": cfg.seed,
    }
    if cfg.hash_inputs:
        manifest["train_sha256"] = file_sha256(cfg.train_path)
        if cfg.validation_path:
            manifest["validation_sha256"] = file_sha256(cfg.validation_path)
    write_json(save_root / "run_manifest.json", manifest)

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
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, cfg.sft_lora).eval()
    model.requires_grad_(False)
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
    optimizer = optim.AdamW(dynamics.parameters(), lr=cfg.dynamics_lr)

    collate = make_collate_fn(tokenizer.pad_token_id)
    train_dataset = _dataset(train_frame, tokenizer, cfg, training=True)
    validation_dataset = _dataset(validation_frame, tokenizer, cfg, training=False)
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
    time_grid = fixed_time_grid(cfg, cfg.device)  # type: ignore[arg-type]
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, float]] = []
    best_value = math.inf
    best_epoch = 0
    stability_passed = False
    stability_reason = "minimum_epochs_not_reached"

    for epoch in range(1, cfg.epochs + 1):
        train_dataset.set_epoch(epoch)
        dynamics.train()
        started = time.perf_counter()
        sums = {key: 0.0 for key in PRETRAIN_LOSS_KEYS}
        counters = {key: 0 for key in PRETRAIN_COUNTER_KEYS}
        examples = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"{cfg.variant} fixed-latent epoch {epoch}")
        for step, batch in enumerate(progress):
            losses = batch_losses(
                model=model,
                projection=projection,
                dynamics=dynamics,
                batch=batch,
                cfg=cfg,  # type: ignore[arg-type]
                device=cfg.device,
                training=False,
                time_grid=time_grid,
                dynamics_to_lora_scale=0.0,
            )
            divisor = accumulation_divisor(step, len(train_loader), cfg.gradient_accumulation_steps)
            dynamics_loss = losses["dynamics"]
            if dynamics_loss.requires_grad:  # type: ignore[union-attr]
                (dynamics_loss / divisor).backward()  # type: ignore[operator]
            batch_size = int(batch["input_ids"].shape[0])
            for key in sums:
                sums[key] += float(losses[key].detach().item()) * batch_size  # type: ignore[union-attr]
            for key in counters:
                counters[key] += int(losses[key])
            examples += batch_size
            if (step + 1) % cfg.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                if any(parameter.grad is not None for parameter in dynamics.parameters()):
                    nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(dynamics=f"{float(losses['dynamics'].detach().item()):.4f}")  # type: ignore[union-attr]

        train_metrics: dict[str, float | int] = {
            **{key: value / max(examples, 1) for key, value in sums.items()},
            **counters,
            "examples": examples,
            "coverage": counters["dynamics_valid_samples"] / max(examples, 1),
            "seconds": time.perf_counter() - started,
        }
        validation = _metrics(model, projection, dynamics, validation_loader, cfg, time_grid)
        validation_history.append({
            "dynamics": float(validation["dynamics"]),
            "coverage": float(validation["coverage"]),
        })
        stability_passed, stability_reason = stability_decision(
            validation_history,
            minimum_epochs=cfg.stability_min_epochs,
            minimum_coverage=cfg.stability_min_coverage,
            minimum_relative_improvement=cfg.stability_min_relative_improvement,
            maximum_relative_regression=cfg.stability_max_relative_regression,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "stability_passed": stability_passed,
            "stability_reason": stability_reason,
            "learning_rate": cfg.dynamics_lr,
        }
        history.append(record)
        append_jsonl(save_root / "metrics.jsonl", record)
        write_json(save_root / "history.json", history)
        _save_checkpoint(save_root / f"checkpoint_epoch_{epoch}", dynamics, cfg, epoch, record)
        value = float(validation["dynamics"])
        if value < best_value:
            best_value = value
            best_epoch = epoch
            _save_checkpoint(save_root / "checkpoint_best", dynamics, cfg, epoch, record)
            write_json(
                save_root / "best_checkpoint.json",
                {
                    "epoch": epoch,
                    "monitor": "validation.dynamics",
                    "value": value,
                    "path": str(save_root / "checkpoint_best"),
                },
            )
        _save_checkpoint(save_root / "checkpoint_last", dynamics, cfg, epoch, record)
        logger.info(
            "epoch=%d train_dynamics=%.6f val_dynamics=%.6f val_coverage=%.4f stability_passed=%s reason=%s seconds=%.1f",
            epoch,
            train_metrics["dynamics"],
            validation["dynamics"],
            validation["coverage"],
            stability_passed,
            stability_reason,
            train_metrics["seconds"],
        )
        if stability_passed:
            break

    write_json(
        save_root / "run_complete.json",
        {
            "status": "completed" if stability_passed else "max_epochs_without_stability",
            "completed_epochs": len(history),
            "stability_passed": stability_passed,
            "stability_reason": stability_reason,
            "best_epoch": best_epoch,
            "best_validation_dynamics": best_value,
            "variant": cfg.variant,
            "dynamics_resolution": cfg.dynamics_resolution,
        },
    )


if __name__ == "__main__":
    train()
