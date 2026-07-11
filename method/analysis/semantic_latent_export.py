#!/usr/bin/env python3
"""Export held-out semantic-layer latent trajectories for Tasks 0 and 2.

Each trajectory is constructed exactly from ``method.training.sequence``:
the first point is the final prompt-token state and each later point pools all
retained substep spans belonging to one graph layer.  Substeps in the same
layer are therefore a set-valued observation, never consecutive ODE steps.

The emitted NPZ is a superset of both downstream contracts.
``observed_latents`` (Task 0) contains the prompt anchor followed by gold
layers. ``target_latents`` and optional ``predicted_latents`` (Task 2) contain
answer layers only, so an identical prompt point cannot artificially lower
PCTE. A Hamiltonian checkpoint additionally produces ``rollout_latents`` from
each gold prompt anchor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from method.training.sequence import EncodedSupervision, encode_supervised


POINT_CONSTRUCTION_VERSION = "prompt_anchor_plus_layer_set_mean_v1"
DEFAULT_SEED = 20260711


@dataclass(frozen=True)
class PointSpec:
    """Token locations for one prompt anchor plus ordered graph layers."""

    anchor: int
    layer_span_groups: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def length(self) -> int:
        return 1 + len(self.layer_span_groups)


@dataclass(frozen=True)
class EncodedRole:
    sample_index: int
    role: str
    input_ids: tuple[int, ...]
    point_spec: PointSpec
    prompt_tokens_dropped: int
    answer_tokens_dropped: int
    semantic_steps_total: int
    semantic_steps_retained: int
    substeps_total: int
    substeps_retained: int


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    source_row: int
    roles: tuple[EncodedRole, ...]


@dataclass(frozen=True)
class RoleOutput:
    latents: np.ndarray
    hidden_states: np.ndarray | None = None
    reconstructed_states: np.ndarray | None = None


class RowExportError(ValueError):
    pass


def prompt_for(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def point_spec(encoded: EncodedSupervision, max_steps: int) -> PointSpec:
    """Build the exact layer-set points used by maintained Hamiltonian training."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    answer_positions = [index for index, label in enumerate(encoded.labels) if label != -100]
    if not answer_positions:
        raise RowExportError("answer has no retained tokens")
    groups = encoded.step_span_groups[:max_steps]
    if not groups:
        raise RowExportError("answer has no fully retained semantic graph layer")
    anchor = max(answer_positions[0] - 1, 0)
    return PointSpec(anchor=anchor, layer_span_groups=groups)


def encode_role(
    tokenizer: Any,
    *,
    question: str,
    answer: str,
    sample_index: int,
    role: str,
    max_length: int,
    answer_budget_fraction: float,
    max_steps: int,
) -> EncodedRole:
    if not question.strip():
        raise RowExportError("question is empty")
    if not answer.strip():
        raise RowExportError(f"{role} answer is empty")
    encoded = encode_supervised(
        tokenizer,
        prompt_for(question),
        answer,
        max_length=max_length,
        answer_budget_fraction=answer_budget_fraction,
    )
    spec = point_spec(encoded, max_steps)
    return EncodedRole(
        sample_index=sample_index,
        role=role,
        input_ids=tuple(encoded.input_ids),
        point_spec=spec,
        prompt_tokens_dropped=encoded.prompt_tokens_dropped,
        answer_tokens_dropped=encoded.answer_tokens_dropped,
        semantic_steps_total=encoded.semantic_steps_total,
        semantic_steps_retained=encoded.semantic_steps_retained,
        substeps_total=encoded.substeps_total,
        substeps_retained=encoded.substeps_retained,
    )


def stable_sample_id(row: dict[str, str], row_index: int) -> str:
    existing = str(row.get("sample_id", "")).strip()
    if existing:
        return existing
    material = "\n".join(
        (
            str(row.get("record_id", "")),
            str(row.get("source_json", "")),
            str(row.get("pathway_block", "")),
            str(row.get("prefix_step_count", "")),
            str(row.get("question", "")),
            str(row_index),
        )
    )
    return f"semantic-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def read_rows(
    path: Path,
    *,
    question_column: str,
    answer_column: str,
    predicted_column: str,
    limit: int | None,
) -> tuple[list[dict[str, str]], bool]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {question_column, answer_column} - fields
        if missing:
            raise ValueError(f"input CSV missing columns: {', '.join(sorted(missing))}")
        has_prediction = predicted_column in fields
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({str(key): "" if value is None else str(value) for key, value in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("input CSV contains no selected rows")
    return rows, has_prediction


def prepare_samples(
    rows: Sequence[dict[str, str]],
    tokenizer: Any,
    *,
    question_column: str,
    answer_column: str,
    predicted_column: str,
    has_prediction: bool,
    max_length: int,
    answer_budget_fraction: float,
    max_steps: int,
    strict: bool,
) -> tuple[list[PreparedSample], list[dict[str, Any]]]:
    samples: list[PreparedSample] = []
    skipped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows, start=2):
        sample_id = stable_sample_id(row, row_index)
        try:
            if sample_id in seen_ids:
                raise RowExportError(f"duplicate sample_id: {sample_id}")
            question = row.get(question_column, "")
            roles = [
                encode_role(
                    tokenizer,
                    question=question,
                    answer=row.get(answer_column, ""),
                    sample_index=len(samples),
                    role="gold",
                    max_length=max_length,
                    answer_budget_fraction=answer_budget_fraction,
                    max_steps=max_steps,
                )
            ]
            if has_prediction:
                roles.append(
                    encode_role(
                        tokenizer,
                        question=question,
                        answer=row.get(predicted_column, ""),
                        sample_index=len(samples),
                        role="predicted",
                        max_length=max_length,
                        answer_budget_fraction=answer_budget_fraction,
                        max_steps=max_steps,
                    )
                )
        except (RowExportError, ValueError, json.JSONDecodeError) as exc:
            if strict:
                raise RowExportError(f"CSV row {row_index} ({sample_id}): {exc}") from exc
            skipped.append({"csv_row": row_index, "sample_id": sample_id, "reason": str(exc)})
            continue
        seen_ids.add(sample_id)
        samples.append(PreparedSample(sample_id, row_index, tuple(roles)))
    if not samples:
        raise ValueError("no rows could be exported")
    return samples, skipped


def pad_trajectories(values: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise ValueError("cannot pad an empty trajectory collection")
    dimensions = {tuple(value.shape[1:]) for value in values}
    if len(dimensions) != 1 or any(value.ndim != 2 or value.shape[0] < 1 for value in values):
        raise ValueError("trajectories must be non-empty [points, dim] arrays with a shared dimension")
    lengths = np.asarray([value.shape[0] for value in values], dtype=np.int64)
    output = np.zeros((len(values), int(lengths.max()), values[0].shape[1]), dtype=np.float32)
    for index, value in enumerate(values):
        output[index, : value.shape[0]] = value.astype(np.float32, copy=False)
    return output, lengths


def _unwrap_ae_state(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "projection"):
            if isinstance(raw.get(key), dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise ValueError("AE checkpoint does not contain a state dictionary")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in raw.items()}


def load_projection(checkpoint: str, hidden_size: int, device: str) -> tuple[Any, int]:
    import torch

    from method.dynamics.latent_teacher import CascadeProjection

    state = _unwrap_ae_state(torch.load(checkpoint, map_location="cpu"))
    down_first = state.get("down.0.weight")
    down_last = state.get("down.6.weight")
    if down_first is None or down_last is None or down_first.ndim != 2 or down_last.ndim != 2:
        raise ValueError("AE checkpoint is missing down.0.weight or down.6.weight")
    checkpoint_hidden = int(down_first.shape[1])
    if checkpoint_hidden != hidden_size:
        raise ValueError(
            f"AE hidden dimension {checkpoint_hidden} does not match backbone hidden size {hidden_size}"
        )
    projection = CascadeProjection(
        high_dim=hidden_size,
        mid_dim=int(down_first.shape[0]),
        latent_dim=int(down_last.shape[0]),
    ).to(device=device, dtype=torch.float32)
    projection.load_state_dict(state, strict=True)
    projection.requires_grad_(False)
    projection.eval()
    return projection, int(down_last.shape[0])


def _decoder_body(model: Any) -> Any:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    body = getattr(base, "model", None)
    return body if body is not None else None


def _forward_hidden(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    """Avoid materializing vocabulary logits for the Qwen causal-LM backbone."""

    body = _decoder_body(model)
    if body is not None:
        output = body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return output.hidden_states[-1]


def export_role_outputs(
    encoded_roles: Sequence[EncodedRole],
    *,
    model: Any,
    projection: Any,
    pad_token_id: int,
    batch_size: int,
    device: str,
    include_reconstruction_states: bool,
) -> dict[tuple[int, str], RoleOutput]:
    import torch

    outputs: dict[tuple[int, str], RoleOutput] = {}
    for start in range(0, len(encoded_roles), batch_size):
        batch = encoded_roles[start : start + batch_size]
        maximum = max(len(item.input_ids) for item in batch)
        ids = torch.full((len(batch), maximum), pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        for index, item in enumerate(batch):
            length = len(item.input_ids)
            ids[index, :length] = torch.as_tensor(item.input_ids, dtype=torch.long, device=device)
            mask[index, :length] = 1
        with torch.no_grad():
            hidden_batch = _forward_hidden(model, ids, mask)
        for local_index, item in enumerate(batch):
            hidden = hidden_batch[local_index, : len(item.input_ids)].float()
            states = [hidden[item.point_spec.anchor]]
            for group in item.point_spec.layer_span_groups:
                tokens = torch.cat([hidden[start:end] for start, end in group], dim=0)
                if not tokens.numel():
                    raise RuntimeError("empty semantic-layer token span reached model export")
                states.append(tokens.mean(dim=0))
            semantic_hidden = torch.stack(states)
            with torch.no_grad():
                latent, reconstructed = projection(semantic_hidden)
            outputs[(item.sample_index, item.role)] = RoleOutput(
                latents=latent.detach().cpu().float().numpy(),
                hidden_states=(
                    semantic_hidden.detach().cpu().float().numpy()
                    if include_reconstruction_states
                    else None
                ),
                reconstructed_states=(
                    reconstructed.detach().cpu().float().numpy()
                    if include_reconstruction_states
                    else None
                ),
            )
        del hidden_batch, ids, mask
    return outputs


def load_dynamics(checkpoint: str, latent_dim: int, device: str) -> tuple[Any, dict[str, Any]]:
    import torch

    from method.dynamics.hamiltonian import LatentHamiltonianDynamics

    raw = torch.load(checkpoint, map_location="cpu")
    if not isinstance(raw, dict) or not isinstance(raw.get("dynamics_config"), dict):
        raise ValueError("Hamiltonian checkpoint must contain dynamics_config and model_state_dict")
    state = raw.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Hamiltonian checkpoint model_state_dict is invalid")
    config = dict(raw["dynamics_config"])
    accepted = set(inspect.signature(LatentHamiltonianDynamics.__init__).parameters) - {"self"}
    constructor = {key: value for key, value in config.items() if key in accepted}
    if int(constructor.get("latent_dim", -1)) != latent_dim:
        raise ValueError(
            f"Hamiltonian latent dimension {constructor.get('latent_dim')} does not match AE {latent_dim}"
        )
    dynamics = LatentHamiltonianDynamics(**constructor).to(device=device, dtype=torch.float32)
    dynamics.load_state_dict(state, strict=True)
    dynamics.eval()
    return dynamics, raw


def rk4_rollout(dynamics: Any, initial: np.ndarray, points: int, dt: float, device: str) -> np.ndarray:
    import torch

    if dt <= 0:
        raise ValueError("dynamics_dt must be positive")
    z = torch.as_tensor(initial, dtype=torch.float32, device=device)
    trajectory = [z.detach().cpu().numpy()]
    for index in range(1, points):
        time = (index - 1) * dt

        def field(offset: float, value: Any) -> Any:
            with torch.enable_grad():
                return dynamics(time + offset, value).detach()

        k1 = field(0.0, z)
        k2 = field(dt / 2.0, z + dt * k1 / 2.0)
        k3 = field(dt / 2.0, z + dt * k2 / 2.0)
        k4 = field(dt, z + dt * k3)
        z = (z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0).detach()
        trajectory.append(z.cpu().numpy())
    return np.stack(trajectory, axis=1).astype(np.float32, copy=False)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _shape_manifest(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


def _stats(encoded_roles: Iterable[EncodedRole]) -> dict[str, int]:
    values = list(encoded_roles)
    return {
        "prompt_tokens_dropped": sum(item.prompt_tokens_dropped for item in values),
        "answer_tokens_dropped": sum(item.answer_tokens_dropped for item in values),
        "semantic_steps_total": sum(item.semantic_steps_total for item in values),
        "semantic_steps_retained_by_token_budget": sum(item.semantic_steps_retained for item in values),
        "semantic_steps_exported_after_max_steps": sum(
            len(item.point_spec.layer_span_groups) for item in values
        ),
        "substeps_total": sum(item.substeps_total for item in values),
        "substeps_retained_by_token_budget": sum(item.substeps_retained for item in values),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Prepared validation/test CSV.")
    parser.add_argument("--output", required=True, help="Destination .npz artifact.")
    parser.add_argument("--manifest-output", help="Defaults to OUTPUT with .manifest.json suffix.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="SFT or stage-2 LoRA used for both gold and prediction.")
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--dynamics-checkpoint", help="Optional maintained Hamiltonian checkpoint; triggers RK4 export.")
    parser.add_argument("--dynamics-dt", type=float, help="Defaults to the dynamics training config, then 1/128.")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--predicted-column", default="predicted_answer")
    parser.add_argument("--dataset-id", help="Defaults to input filename plus its SHA256 prefix.")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--answer-budget-fraction", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=128, help="Maximum graph layers; prompt anchor is additional.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-reconstruction-states", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.output).suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_steps < 1 or args.batch_size < 1 or args.max_length < 2:
        parser.error("--max-steps and --batch-size must be positive; --max-length must be at least 2")
    if not 0 < args.answer_budget_fraction < 1:
        parser.error("--answer-budget-fraction must lie in (0, 1)")
    if args.dynamics_dt is not None and args.dynamics_dt <= 0:
        parser.error("--dynamics-dt must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch

    from method.dynamics.latent_teacher import load_backbone
    from method.training.common import (
        artifact_sha256,
        base_model_identity,
        file_sha256,
        git_commit,
        seed_everything,
        write_json,
    )

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = (
        Path(args.manifest_output).expanduser().resolve()
        if args.manifest_output
        else output_path.with_suffix(".manifest.json")
    )
    for destination in (output_path, manifest_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite")
    device = args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, deterministic=args.deterministic)
    rows, has_prediction = read_rows(
        input_path,
        question_column=args.question_column,
        answer_column=args.answer_column,
        predicted_column=args.predicted_column,
        limit=args.limit,
    )
    tokenizer, model = load_backbone(args.base_model, args.adapter, device)
    samples, skipped = prepare_samples(
        rows,
        tokenizer,
        question_column=args.question_column,
        answer_column=args.answer_column,
        predicted_column=args.predicted_column,
        has_prediction=has_prediction,
        max_length=args.max_length,
        answer_budget_fraction=args.answer_budget_fraction,
        max_steps=args.max_steps,
        strict=args.strict,
    )
    projection, latent_dim = load_projection(args.ae_checkpoint, int(model.config.hidden_size), device)
    encoded_roles = [role for sample in samples for role in sample.roles]
    role_outputs = export_role_outputs(
        encoded_roles,
        model=model,
        projection=projection,
        pad_token_id=int(tokenizer.pad_token_id),
        batch_size=args.batch_size,
        device=device,
        include_reconstruction_states=args.include_reconstruction_states,
    )

    gold = [role_outputs[(index, "gold")] for index in range(len(samples))]
    observed_latents, observed_lengths = pad_trajectories([item.latents for item in gold])
    # Task 2 compares answer trajectories only. Including the identical prompt
    # anchor would mechanically lower PCTE, so target/predicted arrays omit it;
    # Task 0 keeps the anchor in observed_latents as the rollout initial state.
    target_latents, target_lengths = pad_trajectories([item.latents[1:] for item in gold])
    arrays: dict[str, np.ndarray] = {
        "sample_ids": np.asarray([sample.sample_id for sample in samples], dtype=np.str_),
        "lengths": observed_lengths,
        "target_lengths": target_lengths.copy(),
        "observed_latents": observed_latents,
        "target_latents": target_latents,
    }
    if has_prediction:
        predicted = [role_outputs[(index, "predicted")].latents for index in range(len(samples))]
        arrays["predicted_latents"], arrays["predicted_lengths"] = pad_trajectories(
            [item[1:] for item in predicted]
        )
    if args.include_reconstruction_states:
        arrays["hidden_states"], hidden_lengths = pad_trajectories(
            [item.hidden_states for item in gold if item.hidden_states is not None]
        )
        arrays["reconstructed_states"], reconstructed_lengths = pad_trajectories(
            [item.reconstructed_states for item in gold if item.reconstructed_states is not None]
        )
        if not np.array_equal(hidden_lengths, observed_lengths) or not np.array_equal(
            reconstructed_lengths, observed_lengths
        ):
            raise RuntimeError("reconstruction state lengths diverged from gold latent lengths")

    dynamics_raw: dict[str, Any] | None = None
    dynamics_dt = args.dynamics_dt
    if args.dynamics_checkpoint:
        dynamics, dynamics_raw = load_dynamics(args.dynamics_checkpoint, latent_dim, device)
        training_config = dynamics_raw.get("training_config", {})
        checkpoint_dt = float(training_config.get("dynamics_dt", 1.0 / 128.0))
        if dynamics_dt is not None and not math.isclose(
            dynamics_dt,
            checkpoint_dt,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"--dynamics-dt={dynamics_dt} disagrees with checkpoint dt={checkpoint_dt}"
            )
        dynamics_dt = checkpoint_dt
        rollout = rk4_rollout(
            dynamics,
            observed_latents[:, 0],
            observed_latents.shape[1],
            float(dynamics_dt),
            device,
        )
        for index, length in enumerate(observed_lengths):
            rollout[index, int(length) :] = 0.0
        arrays["rollout_latents"] = rollout
    if dynamics_dt is None:
        dynamics_dt = 1.0 / 128.0

    input_hash = file_sha256(input_path)
    base_identity = base_model_identity(args.base_model)
    base_id = json.dumps(base_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    adapter_id = f"sha256:{artifact_sha256(args.adapter)}"
    ae_id = f"sha256:{artifact_sha256(args.ae_checkpoint)}"
    dynamics_id = (
        f"sha256:{artifact_sha256(args.dynamics_checkpoint)}" if args.dynamics_checkpoint else "not_provided"
    )
    dataset_id = args.dataset_id or f"{input_path.stem}:sha256:{input_hash[:16]}"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "semantic_layer_latents",
        "dataset_id": dataset_id,
        "split": args.split,
        "granularity": "graph_layer",
        "point_construction_version": POINT_CONSTRUCTION_VERSION,
        "dynamics_dt": float(dynamics_dt),
        "checkpoints": {"base": base_id, "sft": adapter_id, "ae": ae_id, "dynamics": dynamics_id},
        "representation": {
            "base_checkpoint": base_id,
            "adapter_checkpoint": adapter_id,
            "ae_checkpoint": ae_id,
        },
        "checkpoint_paths": {
            "base": str(Path(args.base_model).expanduser().resolve()),
            "sft": str(Path(args.adapter).expanduser().resolve()),
            "ae": str(Path(args.ae_checkpoint).expanduser().resolve()),
            "dynamics": (
                str(Path(args.dynamics_checkpoint).expanduser().resolve())
                if args.dynamics_checkpoint
                else None
            ),
        },
        "input": {"path": str(input_path), "sha256": input_hash, "selected_rows": len(rows)},
        "exported_samples": len(samples),
        "skipped_samples": len(skipped),
        "skipped_rows": skipped[:1000],
        "skipped_rows_truncated": max(0, len(skipped) - 1000),
        "prediction_column_present": has_prediction,
        "strict": args.strict,
        "seed": args.seed,
        "max_length": args.max_length,
        "answer_budget_fraction": args.answer_budget_fraction,
        "max_graph_layer_steps": args.max_steps,
        "latent_dim": latent_dim,
        "arrays": _shape_manifest(arrays),
        "span_budget_statistics": _stats(encoded_roles),
        "layer_set_semantics": {
            "observed_and_rollout_trajectory": "final_prompt_token_then_ordered_graph_layers",
            "target_and_predicted_trajectory": "ordered_graph_layers_without_prompt_anchor",
            "within_layer": "all retained substep token spans pooled into one mean state",
            "within_layer_order_interpreted_as_causal": False,
            "time_semantics": "reasoning-index; not biological time",
        },
        "dynamics_config": dynamics_raw.get("dynamics_config") if dynamics_raw else None,
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
    }
    if skipped:
        manifest["selection_bias_warning"] = (
            "Non-strict export skipped invalid rows; do not compare aggregate metrics without reporting this cohort change."
        )
    _atomic_npz(output_path, arrays)
    write_json(manifest_path, manifest)
    print(
        f"Wrote {len(samples)} semantic-layer trajectories to {output_path} "
        f"and {manifest_path} (skipped={len(skipped)})"
    )


if __name__ == "__main__":
    main()
