"""Train a LoRA adapter with HNN or forced/damped-HNN latent regularization."""

from __future__ import annotations

import argparse
import json
import time
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
from method.training.csv_io import read_training_csv
from method.training.prefix_sampling import (
    EpochPrefixView,
    PREFIX_POLICIES,
    PREFIX_SAMPLING_MODES,
)
from method.training.staged_objectives import (
    DYNAMICS_RESOLUTIONS,
    flatten_span_groups,
    multiscale_time_points,
    scale_gradient,
)


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_SFT = "/root/autodl-tmp/checkpoints/shared/pathway_sft/checkpoint_best"
DEFAULT_AE = "/root/autodl-tmp/checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"
DEFAULT_TRAIN = "/root/autodl-tmp/data/pathway_v4_full/train_pathway_continuation_v4.csv"
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
    dynamics_init_checkpoint: str | None
    dynamics_init_run_complete: str | None
    require_pretrained_dynamics: bool
    dynamics_to_lora_warmup_fraction: float
    kl_weight: float
    kl_max_tokens: int
    gradient_conflict_interval: int
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dynamics-lr", "--hnn-lr", dest="dynamics_lr", type=float, default=2e-4)
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
        default="dynamics_cycle",
    )
    parser.add_argument(
        "--max-dynamics-steps",
        type=int,
        default=512,
        help="Maximum complete event-object advances; longer targets are reported and truncated.",
    )
    parser.add_argument(
        "--dynamics-dt",
        type=float,
        default=1.0 / 128.0,
        help="Fixed batch-independent surrogate-time increment per graph layer.",
    )
    parser.add_argument(
        "--dynamics-resolution",
        choices=DYNAMICS_RESOLUTIONS,
        default="substep_multiscale",
    )
    parser.add_argument(
        "--substep-dt",
        type=float,
        default=1.0 / 512.0,
        help="Fast within-layer increment for substep_multiscale trajectories.",
    )
    parser.add_argument("--lambda-align", type=float, default=0.5)
    parser.add_argument("--lambda-state", type=float, default=0.5)
    parser.add_argument("--lambda-latent-state", type=float, default=0.1)
    parser.add_argument("--lambda-structure", type=float, default=1e-4)
    parser.add_argument("--lambda-force", type=float, default=1e-4)
    parser.add_argument("--lambda-damping", type=float, default=1e-3)
    parser.add_argument("--dynamics-init-checkpoint")
    parser.add_argument("--dynamics-init-run-complete")
    parser.add_argument("--require-pretrained-dynamics", action="store_true")
    parser.add_argument(
        "--dynamics-to-lora-warmup-fraction",
        type=float,
        default=0.0,
        help="Warm only the dynamics gradient routed into LoRA; dynamics itself receives full gradients.",
    )
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-max-tokens", type=int, default=256)
    parser.add_argument("--gradient-conflict-interval", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-group-column", default="pathway_family_id")
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
    if args.substep_dt <= 0 or args.substep_dt >= args.dynamics_dt:
        parser.error("--substep-dt must be positive and smaller than --dynamics-dt")
    if not 0.0 <= args.dynamics_to_lora_warmup_fraction <= 1.0:
        parser.error("--dynamics-to-lora-warmup-fraction must be in [0, 1]")
    if args.kl_weight < 0:
        parser.error("--kl-weight must be non-negative")
    if args.kl_max_tokens < 1:
        parser.error("--kl-max-tokens must be positive")
    if args.gradient_conflict_interval < 0:
        parser.error("--gradient-conflict-interval must be non-negative")
    if args.require_pretrained_dynamics and (
        not args.dynamics_init_checkpoint or not args.dynamics_init_run_complete
    ):
        parser.error(
            "--require-pretrained-dynamics needs both --dynamics-init-checkpoint "
            "and --dynamics-init-run-complete"
        )
    if args.dynamics_init_run_complete and not args.dynamics_init_checkpoint:
        parser.error("--dynamics-init-run-complete needs --dynamics-init-checkpoint")
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
    train_frame = read_training_csv(cfg.train_path)
    if cfg.limit is not None:
        train_frame = train_frame.head(cfg.limit)
    required = {"question"}
    missing = required - set(train_frame.columns)
    if missing:
        raise ValueError(f"training CSV missing columns: {', '.join(sorted(missing))}")
    if cfg.prefix_sampling == "one_per_record" and "record_id" not in train_frame.columns:
        raise ValueError("one_per_record prefix sampling requires record_id")
    if cfg.validation_path:
        validation_frame = read_training_csv(cfg.validation_path)
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


def supervised_logit_positions(labels: torch.Tensor, maximum: int) -> torch.Tensor:
    """Select logits that predict supervised assistant tokens.

    Causal-LM logits at position ``t`` predict the label at ``t+1``.  The
    returned ``[batch, position]`` indices therefore shift every supervised
    label position left by one and sample deterministically when capped.
    """

    if maximum < 1:
        raise ValueError("maximum supervised KL positions must be positive")
    label_positions = torch.nonzero(labels != -100, as_tuple=False)
    label_positions = label_positions[label_positions[:, 1] > 0]
    if label_positions.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=labels.device)
    positions = label_positions.clone()
    positions[:, 1] -= 1
    if positions.size(0) <= maximum:
        return positions
    selected = torch.linspace(
        0,
        positions.size(0) - 1,
        steps=maximum,
        device=positions.device,
    ).round().long()
    return positions[selected]


def select_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if positions.ndim != 2 or positions.size(-1) != 2:
        raise ValueError("logit positions must have shape [N, 2]")
    if positions.numel() == 0:
        return logits.new_empty((0, logits.size(-1)))
    return logits[positions[:, 0], positions[:, 1]]


def stage1_reference_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor | None,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    if reference_logits is None or positions is None or positions.numel() == 0:
        return current_logits.new_zeros(())
    current = select_logits(current_logits, positions).float()
    reference = reference_logits.to(device=current.device, dtype=torch.float32)
    if current.shape != reference.shape:
        raise ValueError("current/reference KL logits have different shapes")
    reference_probability = nn.functional.softmax(reference, dim=-1)
    return nn.functional.kl_div(
        nn.functional.log_softmax(current, dim=-1),
        reference_probability,
        reduction="batchmean",
    )


def load_pretrained_dynamics(
    dynamics: LatentHamiltonianDynamics,
    checkpoint: str,
    *,
    run_complete: str | None,
    require_stability: bool,
) -> dict[str, Any]:
    path = Path(checkpoint)
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict) or not isinstance(raw.get("model_state_dict"), dict):
        raise ValueError("pretrained dynamics checkpoint has no model_state_dict")
    config = raw.get("dynamics_config", {})
    expected = dynamics.export_config()
    for key in ("latent_dim", "variant", "structure_mode", "damping_mode"):
        if config.get(key) != expected.get(key):
            raise ValueError(
                f"pretrained dynamics {key}={config.get(key)!r} does not match {expected.get(key)!r}"
            )
    status_path = Path(run_complete) if run_complete else path.parent.parent / "run_complete.json"
    if status_path.resolve().parent != path.resolve().parent.parent:
        raise ValueError("dynamics checkpoint and run_complete marker are not from the same run")
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if require_stability and (
        status.get("status") != "completed" or status.get("stability_passed") is not True
    ):
        raise RuntimeError(
            "dynamics pretraining did not pass the registered stability gate: "
            f"{status_path}"
        )
    dynamics.load_state_dict(unwrap_state_dict(raw["model_state_dict"]))
    return {
        "checkpoint": str(path.resolve()),
        "run_complete": str(status_path.resolve()),
        "pretrain_status": status,
    }


def batch_losses(
    *,
    model: nn.Module,
    projection: CascadeProjection,
    dynamics: LatentHamiltonianDynamics,
    batch: dict[str, Any],
    cfg: TrainConfig,
    device: str,
    training: bool,
    time_grid: torch.Tensor,
    dynamics_to_lora_scale: float = 1.0,
    reference_logits: torch.Tensor | None = None,
    kl_positions: torch.Tensor | None = None,
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
        loss_kl = stage1_reference_kl(outputs.logits, reference_logits, kl_positions)
        hidden = outputs.hidden_states[-1].float()
        # Forward values are identical for every scale.  Only the auxiliary
        # gradient returning to LoRA is warmed; dynamics parameters always see
        # the full objective from the first optimizer step.
        dynamics_hidden = scale_gradient(hidden, dynamics_to_lora_scale)
        latent, _ = projection(dynamics_hidden)

    loss_sft_regularized = loss_sft + cfg.kl_weight * loss_kl

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
            "total": loss_sft_regularized,
            "sft": loss_sft,
            "kl": loss_kl,
            "sft_regularized": loss_sft_regularized,
            "dynamics": zero,
            "align": zero,
            "state": zero,
            "latent_state": zero,
            "regularization": zero,
            "dynamics_valid_samples": 0,
            "dynamics_layer_boundaries": 0,
            "dynamics_truncated_substeps": 0,
            "dynamics_truncated_samples": 0,
            "dynamics_truncated_semantic_steps": 0,
            "text_truncated_substeps": int(batch["substeps_total"]) - int(batch["substeps_retained"]),
            "text_truncated_semantic_steps": int(batch["semantic_steps_total"]) - int(batch["semantic_steps_retained"]),
            "prompt_tokens_dropped": int(batch["prompt_tokens_dropped"]),
            "answer_tokens_dropped": int(batch["answer_tokens_dropped"]),
        }

    velocity_similarities: list[torch.Tensor] = []
    state_similarities: list[torch.Tensor] = []
    latent_state_losses: list[torch.Tensor] = []
    dynamics_truncated_samples = 0
    dynamics_truncated_semantic_steps = 0
    dynamics_truncated_substeps = 0
    dynamics_layer_boundaries = 0
    regularization_times: list[torch.Tensor] = []
    regularization_states: list[torch.Tensor] = []

    if cfg.dynamics_resolution == "graph_layer":
        valid_tensor = torch.tensor(valid_indices, dtype=torch.long, device=hidden.device)
        z0_batch = latent[valid_tensor, last_prompt[valid_tensor]]
        maximum_used_steps = max(
            min(len(batch["step_span_groups"][index]), cfg.max_dynamics_steps)
            for index in valid_indices
        )
        batch_trajectory = odeint(
            dynamics,
            z0_batch.float(),
            time_grid[: maximum_used_steps + 1],
            method="rk4",
        ).transpose(0, 1)
        batch_predicted_hidden = projection.up(batch_trajectory)
        regularization_times.append(time_grid[: maximum_used_steps + 1])
        regularization_states.append(z0_batch)

    for local_index, batch_index in enumerate(valid_indices):
        groups = batch["step_span_groups"][batch_index]
        if cfg.dynamics_resolution == "graph_layer":
            raw_length = len(groups)
            used_length = min(raw_length, cfg.max_dynamics_steps)
            if raw_length > used_length:
                dynamics_truncated_samples += 1
                dynamics_truncated_semantic_steps += raw_length - used_length
            target_states = torch.stack([
                torch.cat([
                    hidden[batch_index, int(start) : int(end)]
                    for start, end in group.to(hidden.device).tolist()
                ], dim=0).mean(dim=0).detach()
                for group in groups[:used_length]
            ])
            trajectory = batch_trajectory[local_index, : used_length + 1]
            generated_states = batch_predicted_hidden[local_index, 1 : used_length + 1]
            dynamics_layer_boundaries += used_length
        else:
            spans, layer_indices, raw_length = flatten_span_groups(
                groups,
                maximum_events=cfg.max_dynamics_steps,
            )
            used_length = len(spans)
            if not spans:
                continue
            if raw_length > used_length:
                dynamics_truncated_samples += 1
                dynamics_truncated_substeps += raw_length - used_length
            target_states = torch.stack([
                hidden[batch_index, start:end].mean(dim=0).detach()
                for start, end in spans
            ])
            sample_times = multiscale_time_points(
                layer_indices,
                layer_dt=cfg.dynamics_dt,
                substep_dt=cfg.substep_dt,
                device=hidden.device,
            )
            sample_z0 = latent[batch_index, last_prompt[batch_index]].unsqueeze(0)
            trajectory = odeint(
                dynamics,
                sample_z0.float(),
                sample_times,
                method="rk4",
            ).squeeze(1)
            generated_states = projection.up(trajectory[1:])
            regularization_times.append(sample_times)
            regularization_states.append(sample_z0)
            dynamics_layer_boundaries += sum(
                1
                for position, layer_index in enumerate(layer_indices)
                if position == 0 or layer_index != layer_indices[position - 1]
            )

        generated_velocity = (
            projection.up(trajectory[1 : used_length + 1])
            - projection.up(trajectory[:used_length])
        )
        target_sequence = torch.cat(
            [hidden[batch_index, last_prompt[batch_index]].detach().unsqueeze(0), target_states],
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
                trajectory[1 : used_length + 1],
                target_latent_states,
            )
        )

    if not velocity_similarities:
        zero = loss_sft.new_zeros(())
        return {
            "total": loss_sft_regularized,
            "sft": loss_sft,
            "kl": loss_kl,
            "sft_regularized": loss_sft_regularized,
            "dynamics": zero,
            "align": zero,
            "state": zero,
            "latent_state": zero,
            "regularization": zero,
            "dynamics_valid_samples": 0,
            "dynamics_layer_boundaries": 0,
            "dynamics_truncated_substeps": dynamics_truncated_substeps,
            "dynamics_truncated_samples": dynamics_truncated_samples,
            "dynamics_truncated_semantic_steps": dynamics_truncated_semantic_steps,
            "text_truncated_substeps": int(batch["substeps_total"]) - int(batch["substeps_retained"]),
            "text_truncated_semantic_steps": int(batch["semantic_steps_total"]) - int(batch["semantic_steps_retained"]),
            "prompt_tokens_dropped": int(batch["prompt_tokens_dropped"]),
            "answer_tokens_dropped": int(batch["answer_tokens_dropped"]),
        }

    loss_align = 1.0 - torch.stack(velocity_similarities).mean()
    loss_state = 1.0 - torch.stack(state_similarities).mean()
    loss_latent_state = torch.stack(latent_state_losses).mean()
    loss_regularization = dynamics.regularization_loss(
        torch.cat(regularization_times),
        torch.cat(regularization_states, dim=0),
        lambda_structure=cfg.lambda_structure,
        lambda_force=cfg.lambda_force,
        lambda_damping=cfg.lambda_damping,
    )
    loss_dynamics = (
        cfg.lambda_align * loss_align
        + cfg.lambda_state * loss_state
        + cfg.lambda_latent_state * loss_latent_state
        + loss_regularization
    )
    total = loss_sft_regularized + loss_dynamics
    return {
        "total": total,
        "sft": loss_sft,
        "kl": loss_kl,
        "sft_regularized": loss_sft_regularized,
        "dynamics": loss_dynamics,
        "align": loss_align,
        "state": loss_state,
        "latent_state": loss_latent_state,
        "regularization": loss_regularization,
        "dynamics_valid_samples": len(velocity_similarities),
        "dynamics_layer_boundaries": dynamics_layer_boundaries,
        "dynamics_truncated_substeps": dynamics_truncated_substeps,
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
    if (
        cfg.dynamics_init_checkpoint
        or cfg.require_pretrained_dynamics
        or cfg.dynamics_to_lora_warmup_fraction > 0
        or cfg.kl_weight > 0
        or cfg.gradient_conflict_interval > 0
        or cfg.dynamics_resolution != "graph_layer"
    ):
        # Keep one implementation of the staged objective and checkpoint
        # contract.  The distributed entry point also handles world_size=1,
        # which is the maintained task-parallel configuration for HNN/FDHNN.
        from method.training.framework_a_ddp import train as train_staged

        train_staged(cfg)
        return
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    save_root = ensure_new_output_dir(cfg.save_dir)
    logger = configure_logger(save_root / "train.log", f"framework_a.{cfg.variant}")
    write_json(save_root / "run_config.json", asdict(cfg))

    train_frame, validation_frame = load_frames(cfg)
    manifest = {
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
    logger.info(
        "variant=%s structure=%s train_eligible_prefix_rows=%d train_records=%d "
        "train_samples_per_epoch=%d validation_prefix_rows=%d validation_records=%d "
        "validation_samples=%d",
        cfg.variant,
        cfg.structure_mode,
        manifest["train_eligible_prefix_rows"],
        manifest["train_records"],
        manifest["train_samples_per_epoch"],
        manifest["validation_prefix_rows"],
        manifest["validation_records"],
        manifest["validation_samples"],
    )

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
    train_dataset = CSVPathwayDataset(
        train_frame,
        tokenizer,
        cfg.max_length,
        cfg.answer_budget_fraction,
        cfg.prefix_sampling,
        cfg.prefix_policy,
        cfg.seed,
    )
    validation_dataset = CSVPathwayDataset(
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
        generator=generator,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
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
        train_dataset.set_epoch(epoch)
        prefix_selection = train_dataset.selection_summary()
        logger.info(
            "epoch=%d train_samples=%d prefix_selection=%s",
            epoch,
            len(train_dataset),
            json.dumps(prefix_selection, sort_keys=True),
        )
        model.train()
        dynamics.train()
        train_started = time.perf_counter()
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
            "input_tokens": 0,
            "supervised_tokens": 0,
        }
        progress = tqdm(train_loader, desc=f"{cfg.variant} epoch {epoch}")
        for step, batch in enumerate(progress):
            counters["input_tokens"] += int(batch["attention_mask"].sum().item())
            counters["supervised_tokens"] += int((batch["labels"] != -100).sum().item())
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
                if key in losses:
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
        train_metrics["seconds"] = time.perf_counter() - train_started
        train_metrics["input_tokens_per_second"] = counters["input_tokens"] / max(
            float(train_metrics["seconds"]), 1e-9
        )
        train_metrics["supervised_tokens_per_second"] = counters["supervised_tokens"] / max(
            float(train_metrics["seconds"]), 1e-9
        )
        validation_started = time.perf_counter()
        validation_metrics = evaluate(
            model=model,
            projection=projection,
            dynamics=dynamics,
            loader=validation_loader,
            cfg=cfg,
            device=cfg.device,
            time_grid=time_grid,
        )
        validation_metrics["seconds"] = time.perf_counter() - validation_started
        record = {
            "epoch": epoch,
            "prefix_selection": prefix_selection,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "dynamics_learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        append_jsonl(save_root / "metrics.jsonl", record)
        write_json(save_root / "history.json", history)
        logger.info(
            "epoch=%d train_total=%.6f val_total=%.6f val_sft=%.6f val_velocity=%.6f val_state=%.6f dynamics_truncated_semantic_steps=%d text_truncated_substeps=%d train_seconds=%.1f validation_seconds=%.1f supervised_tokens_per_second=%.1f",
            epoch,
            train_metrics["total"],
            validation_metrics["total"],
            validation_metrics["sft"],
            validation_metrics["align"],
            validation_metrics["state"],
            counters["dynamics_truncated_semantic_steps"],
            counters["text_truncated_substeps"],
            train_metrics["seconds"],
            validation_metrics["seconds"],
            train_metrics["supervised_tokens_per_second"],
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
