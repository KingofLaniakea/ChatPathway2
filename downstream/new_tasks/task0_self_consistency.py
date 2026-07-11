#!/usr/bin/env python3
"""Task 0: AE reconstruction and Hamiltonian-rollout self-consistency.

This task is a held-out diagnostic, not a biological accuracy benchmark.  It
separately measures (a) whether the AE reconstructs backbone hidden states and
(b) whether an autonomous HNN/forced-damped-HNN rollout follows observed latent
states.  A good score does not by itself prove causal or phenotype validity.

The portable input is an NPZ with:

``hidden_states`` / ``reconstructed_states``
    Optional matching arrays shaped ``[samples, points, hidden_dim]``.
``observed_latents`` / ``rollout_latents``
    Matching arrays shaped ``[samples, points, latent_dim]``.  Instead of
    ``rollout_latents``, pass ``--dynamics-checkpoint`` to perform an RK4
    rollout from the first observed point.
``lengths``
    Optional valid point counts for right-padded arrays.

``points`` may be tokens, graph layers, or independently ordered causal
substeps, but the granularity is recorded explicitly and must not be described
as biological time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from downstream.common.io import write_json, write_rows
from downstream.new_tasks.schemas import (
    SchemaError,
    load_json_object,
    require_choice,
    require_fields,
    require_integer,
    require_mapping,
    require_number,
    require_text,
    validate_finite_array,
)


DEFAULT_HORIZONS = (1, 5, 10, 20)


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    require_fields(
        value,
        (
            "schema_version",
            "dataset_id",
            "split",
            "granularity",
            "point_construction_version",
            "dynamics_dt",
            "checkpoints",
        ),
        "manifest",
    )
    if require_integer(value["schema_version"], "manifest.schema_version") != 1:
        raise SchemaError("manifest.schema_version must be 1.")
    require_text(value["dataset_id"], "manifest.dataset_id")
    require_choice(value["split"], ("validation", "test"), "manifest.split")
    granularity = require_choice(
        value["granularity"], ("token", "graph_layer", "causal_substep"), "manifest.granularity"
    )
    if granularity == "causal_substep":
        provenance = require_mapping(value.get("ordering_provenance"), "manifest.ordering_provenance")
        require_fields(provenance, ("source", "version"), "manifest.ordering_provenance")
        require_text(provenance["source"], "manifest.ordering_provenance.source")
        require_text(provenance["version"], "manifest.ordering_provenance.version")
    require_text(value["point_construction_version"], "manifest.point_construction_version")
    if require_number(value["dynamics_dt"], "manifest.dynamics_dt") <= 0:
        raise SchemaError("manifest.dynamics_dt must be positive.")
    checkpoints = require_mapping(value["checkpoints"], "manifest.checkpoints")
    require_fields(checkpoints, ("base", "sft", "ae", "dynamics"), "manifest.checkpoints")
    for name in ("base", "sft", "ae", "dynamics"):
        require_text(checkpoints[name], f"manifest.checkpoints.{name}")
    return dict(value)


def _lengths(value: Any, samples: int, points: int) -> np.ndarray:
    if value is None:
        return np.full(samples, points, dtype=int)
    raw = np.asarray(value)
    if raw.shape != (samples,) or not np.issubdtype(raw.dtype, np.integer):
        raise SchemaError(f"lengths must be an integer array shaped [{samples}].")
    result = raw.astype(int)
    if np.any(result < 1) or np.any(result > points):
        raise SchemaError(f"lengths must lie in [1, {points}].")
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return float(np.allclose(left, right, atol=1e-12))
    return float(np.dot(left, right) / denominator)


def _point_metrics(
    expected: np.ndarray,
    actual: np.ndarray,
    lengths: np.ndarray,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    if expected.shape != actual.shape or expected.ndim != 3:
        raise SchemaError("paired arrays must have identical [samples, points, dim] shape.")
    rows: list[dict[str, float | int]] = []
    for sample_index, length in enumerate(lengths):
        for point_index in range(int(length)):
            target = expected[sample_index, point_index]
            prediction = actual[sample_index, point_index]
            rows.append({
                "sample_index": sample_index,
                "point_index": point_index,
                "mse": float(np.mean((prediction - target) ** 2)),
                "cosine_similarity": _cosine(prediction, target),
            })
    return rows, {
        "num_points": len(rows),
        "mse": float(np.mean([row["mse"] for row in rows])) if rows else 0.0,
        "cosine_similarity": float(np.mean([row["cosine_similarity"] for row in rows])) if rows else 0.0,
    }


def evaluate_reconstruction(
    hidden_states: np.ndarray,
    reconstructed_states: np.ndarray,
    lengths: np.ndarray | None = None,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    validate_finite_array(hidden_states, "hidden_states", ndim=3)
    validate_finite_array(reconstructed_states, "reconstructed_states", ndim=3)
    if hidden_states.shape != reconstructed_states.shape:
        raise SchemaError("hidden_states and reconstructed_states must have identical shape.")
    valid_lengths = _lengths(lengths, hidden_states.shape[0], hidden_states.shape[1])
    return _point_metrics(hidden_states, reconstructed_states, valid_lengths)


def evaluate_rollout(
    observed_latents: np.ndarray,
    rollout_latents: np.ndarray,
    lengths: np.ndarray | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    validate_finite_array(observed_latents, "observed_latents", ndim=3)
    validate_finite_array(rollout_latents, "rollout_latents", ndim=3)
    if observed_latents.shape != rollout_latents.shape:
        raise SchemaError("observed_latents and rollout_latents must have identical shape.")
    valid_lengths = _lengths(lengths, observed_latents.shape[0], observed_latents.shape[1])
    rows, aggregate = _point_metrics(observed_latents, rollout_latents, valid_lengths)
    horizon_rows = []
    for horizon in sorted(set(int(value) for value in horizons)):
        if horizon < 1:
            raise SchemaError("all rollout horizons must be positive integers.")
        eligible = np.flatnonzero(valid_lengths > horizon)
        if not len(eligible):
            horizon_rows.append({"horizon": horizon, "num_samples": 0, "mse": None, "cosine_similarity": None})
            continue
        mse_values, cosine_values = [], []
        for index in eligible:
            target = observed_latents[index, horizon]
            prediction = rollout_latents[index, horizon]
            mse_values.append(float(np.mean((prediction - target) ** 2)))
            cosine_values.append(_cosine(prediction, target))
        horizon_rows.append({
            "horizon": horizon,
            "num_samples": len(eligible),
            "mse": float(np.mean(mse_values)),
            "cosine_similarity": float(np.mean(cosine_values)),
        })
    return rows, {**aggregate, "horizons": horizon_rows}


def rollout_from_checkpoint(
    observed_latents: np.ndarray,
    checkpoint: str,
    dt: float,
    device_name: str = "auto",
) -> np.ndarray:
    """RK4-roll out the maintained dynamics checkpoint from each observed z0."""

    if dt <= 0:
        raise SchemaError("dt must be positive.")
    validate_finite_array(observed_latents, "observed_latents", ndim=3)
    import torch

    from method.dynamics.hamiltonian import LatentHamiltonianDynamics

    device = torch.device(device_name if device_name != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    raw = torch.load(checkpoint, map_location="cpu")
    if not isinstance(raw, dict) or not isinstance(raw.get("dynamics_config"), dict):
        raise SchemaError("dynamics checkpoint must contain dynamics_config and model_state_dict.")
    state = raw.get("model_state_dict")
    if not isinstance(state, dict):
        raise SchemaError("dynamics checkpoint model_state_dict is missing or invalid.")
    config = raw["dynamics_config"]
    allowed = {
        "latent_dim",
        "variant",
        "hidden_dim",
        "structure_mode",
        "initial_damping",
        "structure_reflections",
        "damping_mode",
    }
    dynamics = LatentHamiltonianDynamics(**{key: value for key, value in config.items() if key in allowed})
    dynamics.load_state_dict(state, strict=True)
    dynamics = dynamics.to(device=device, dtype=torch.float32).eval()
    z = torch.as_tensor(observed_latents[:, 0], device=device, dtype=torch.float32)
    trajectory = [z.detach().cpu().numpy()]
    for point_index in range(1, observed_latents.shape[1]):
        time = float(point_index - 1) * dt

        def field(offset: float, value: Any) -> Any:
            return dynamics(time + offset, value).detach()

        k1 = field(0.0, z)
        k2 = field(dt / 2, z + dt * k1 / 2)
        k3 = field(dt / 2, z + dt * k2 / 2)
        k4 = field(dt, z + dt * k3)
        z = (z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6).detach()
        trajectory.append(z.cpu().numpy())
    return np.stack(trajectory, axis=1)


def evaluate_npz(
    arrays: Any,
    manifest: dict[str, Any],
    *,
    rollout_checkpoint: str | None = None,
    device: str = "auto",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    metadata = validate_manifest(manifest)
    dt = float(metadata["dynamics_dt"])
    available = set(arrays.files if hasattr(arrays, "files") else arrays.keys())
    lengths = arrays["lengths"] if "lengths" in available else None
    result: dict[str, Any] = {
        "task": "task0_self_consistency",
        "manifest": metadata,
        "granularity": metadata["granularity"],
        "time_semantics": "reasoning-index; not biological time",
    }
    tables: dict[str, list[dict[str, Any]]] = {}
    has_hidden = "hidden_states" in available or "reconstructed_states" in available
    if has_hidden:
        if not {"hidden_states", "reconstructed_states"}.issubset(available):
            raise SchemaError("AE evaluation requires both hidden_states and reconstructed_states.")
        recon_rows, recon_summary = evaluate_reconstruction(
            arrays["hidden_states"], arrays["reconstructed_states"], lengths
        )
        result["ae_reconstruction"] = recon_summary
        tables["reconstruction_points"] = recon_rows

    if "observed_latents" not in available:
        raise SchemaError("Task 0 requires observed_latents.")
    observed = arrays["observed_latents"]
    if "rollout_latents" in available:
        rollout = arrays["rollout_latents"]
        result["rollout_source"] = "precomputed"
    elif rollout_checkpoint:
        rollout = rollout_from_checkpoint(observed, rollout_checkpoint, dt, device)
        result["rollout_source"] = "dynamics_checkpoint"
    else:
        raise SchemaError("Provide rollout_latents in the NPZ or --dynamics-checkpoint.")
    rollout_rows, rollout_summary = evaluate_rollout(observed, rollout, lengths, horizons)
    result["dynamics_rollout"] = rollout_summary
    result["complete_self_consistency"] = bool(has_hidden)
    result["interpretation_warning"] = (
        "AE/HNN agreement is a representation-fit diagnostic; it is not evidence of biological causality, "
        "phenotype accuracy, or real-time dynamics."
    )
    tables["rollout_points"] = rollout_rows
    return result, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="NPZ artifact following the Task 0 contract.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dynamics-checkpoint", help="Maintained hamiltonian_dynamics.pt; used when rollout_latents is absent.")
    parser.add_argument("--manifest", required=True, help="Held-out split and checkpoint provenance JSON.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    args = parser.parse_args()
    with np.load(args.input) as arrays:
        summary, tables = evaluate_npz(
            arrays,
            load_json_object(args.manifest),
            rollout_checkpoint=args.dynamics_checkpoint,
            device=args.device,
            horizons=args.horizons,
        )
    output_dir = Path(args.output_dir)
    write_json(output_dir / "summary_metrics.json", summary)
    for name, rows in tables.items():
        write_rows(output_dir / f"{name}.csv", rows)
    print(summary)


if __name__ == "__main__":
    main()
