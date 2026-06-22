#!/usr/bin/env python3
"""Task III: Physics-Consistency Trajectory Error (PCTE).

Two execution modes are supported:

* ``--pred-latents`` + ``--target-latents`` evaluates already extracted latent
  trajectories. This is the fast and reproducible mode for large experiments.
* ``--input --base-model --ae-ckpt`` extracts answer-span hidden states for
  ``predicted_answer`` and ``answer``, projects both with the trained AE, then
  applies DTW. ``--adapter`` optionally evaluates the same LoRA used to
  generate predictions.

PCTE is a predicted-vs-gold trajectory metric. It is intentionally distinct
from HNN vector-field self-consistency, which asks a different question.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from downstream.io import load_records, mean, write_json, write_rows


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return max(0.0, 1.0 - float(np.dot(left, right)) / denominator)


def dtw_distance(left: np.ndarray, right: np.ndarray, metric: str, max_length: int) -> tuple[float, int]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Each trajectory must be [steps, latent_dim] with matching latent_dim.")
    if left.shape[0] > max_length:
        left = left[np.linspace(0, left.shape[0] - 1, max_length).round().astype(int)]
    if right.shape[0] > max_length:
        right = right[np.linspace(0, right.shape[0] - 1, max_length).round().astype(int)]
    costs = np.full((left.shape[0] + 1, right.shape[0] + 1), np.inf)
    lengths = np.zeros((left.shape[0] + 1, right.shape[0] + 1), dtype=int)
    costs[0, 0] = 0.0
    for i in range(1, left.shape[0] + 1):
        for j in range(1, right.shape[0] + 1):
            distance = cosine_distance(left[i - 1], right[j - 1]) if metric == "cosine" else float(np.linalg.norm(left[i - 1] - right[j - 1]))
            predecessor = min(((costs[i - 1, j], lengths[i - 1, j]), (costs[i, j - 1], lengths[i, j - 1]),
                               (costs[i - 1, j - 1], lengths[i - 1, j - 1])), key=lambda item: item[0])
            costs[i, j] = distance + predecessor[0]
            lengths[i, j] = predecessor[1] + 1
    return float(costs[-1, -1] / max(lengths[-1, -1], 1)), int(lengths[-1, -1])


def as_trajectory_list(value: np.ndarray) -> list[np.ndarray]:
    if value.ndim == 2:
        return [value]
    if value.ndim != 3:
        raise ValueError(f"Expected [samples, steps, dim] or [steps, dim], got {value.shape}.")
    return [value[index] for index in range(value.shape[0])]


def load_latents(path: str) -> list[np.ndarray]:
    source = Path(path)
    if source.suffix == ".npy":
        return as_trajectory_list(np.load(source))
    with np.load(source, allow_pickle=True) as arrays:
        for key in ("latents", "z", "trajectories"):
            if key in arrays:
                return as_trajectory_list(arrays[key])
        if len(arrays.files) == 1:
            return as_trajectory_list(arrays[arrays.files[0]])
    raise ValueError(f"Could not find a latent array in {source}.")


def remove_padding(trajectory: np.ndarray) -> np.ndarray:
    """Drop all-zero right padding produced by fixed-width trajectory exports."""
    valid = np.flatnonzero(np.any(np.abs(trajectory) > 0, axis=1))
    return trajectory[: valid[-1] + 1] if valid.size else trajectory[:0]


def evaluate(predicted: list[np.ndarray], target: list[np.ndarray], metric: str, max_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(predicted) != len(target):
        raise ValueError("Predicted and target inputs must contain the same number of trajectories.")
    rows = []
    for index, (prediction, truth) in enumerate(zip(predicted, target)):
        prediction, truth = remove_padding(prediction), remove_padding(truth)
        if not len(prediction) or not len(truth):
            continue
        distance, path_length = dtw_distance(prediction, truth, metric, max_length)
        rows.append({
            "sample_id": index, "predicted_steps": len(prediction), "target_steps": len(truth),
            "pcte": distance, "dtw_path_length": path_length,
        })
    return rows, {"num_samples": len(rows), "metric": metric, "mean_pcte": mean([float(row["pcte"]) for row in rows])}


def unwrap_state(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "projection"):
            if isinstance(raw.get(key), dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise ValueError("AE checkpoint does not contain a state dictionary.")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in raw.items()}


def load_projection(checkpoint: str, device: Any, dtype: Any) -> Any:
    import torch
    import torch.nn as nn

    state = unwrap_state(torch.load(checkpoint, map_location="cpu"))
    if "encoder.0.weight" in state:
        high_dim, mid_dim, latent_dim = state["encoder.0.weight"].shape[1], state["encoder.0.weight"].shape[0], state["encoder.3.weight"].shape[0]
        projection = nn.Sequential(nn.Linear(high_dim, mid_dim), nn.LayerNorm(mid_dim), nn.Tanh(), nn.Linear(mid_dim, latent_dim))
        projection.load_state_dict({key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")})
    elif "down.6.weight" in state:
        high_dim, mid_dim, latent_dim = state["down.0.weight"].shape[1], state["down.0.weight"].shape[0], state["down.6.weight"].shape[0]
        projection = nn.Sequential(nn.Linear(high_dim, mid_dim), nn.LayerNorm(mid_dim), nn.SiLU(), nn.Linear(mid_dim, mid_dim // 2), nn.LayerNorm(mid_dim // 2), nn.SiLU(), nn.Linear(mid_dim // 2, latent_dim))
        projection.load_state_dict({key.removeprefix("down."): value for key, value in state.items() if key.startswith("down.")})
    elif "down.3.weight" in state:
        high_dim, mid_dim, latent_dim = state["down.0.weight"].shape[1], state["down.0.weight"].shape[0], state["down.3.weight"].shape[0]
        projection = nn.Sequential(nn.Linear(high_dim, mid_dim), nn.LayerNorm(mid_dim), nn.Tanh(), nn.Linear(mid_dim, latent_dim))
        projection.load_state_dict({key.removeprefix("down."): value for key, value in state.items() if key.startswith("down.")})
    else:
        raise ValueError("Unsupported AE state layout; expected encoder.* or down.* weights.")
    return projection.to(device=device, dtype=dtype).eval()


def extract_latents(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    projection = load_projection(args.ae_ckpt, device, dtype)
    predicted, target = [], []
    for record in records[: args.limit]:
        prompt = f"<|im_start|>user\n{record.get(args.question_column, '')}<|im_end|>\n<|im_start|>assistant\n"
        prefix_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        for collection, column in ((predicted, args.predicted_column), (target, args.target_column)):
            text = f"{record.get(column, '')}<|im_end|>"
            ids = tokenizer.encode(prompt + text, add_special_tokens=False, truncation=True, max_length=args.max_length)
            input_ids = torch.tensor(ids, device=device).unsqueeze(0)
            with torch.no_grad():
                hidden = model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1][0]
                latent = projection(hidden.to(dtype))[prefix_length:].float().cpu().numpy()
            collection.append(latent)
    return predicted, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pred-latents", help="NPY/NPZ predicted latent trajectories; pair with --target-latents.")
    source.add_argument("--input", help="Prediction CSV/JSON/JSONL for online hidden-state extraction.")
    parser.add_argument("--target-latents", help="NPY/NPZ target latent trajectories.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    parser.add_argument("--dtw-max-length", type=int, default=256)
    parser.add_argument("--base-model")
    parser.add_argument("--adapter")
    parser.add_argument("--ae-ckpt")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--predicted-column", default="predicted_answer")
    parser.add_argument("--target-column", default="answer")
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.pred_latents:
        if not args.target_latents:
            parser.error("--target-latents is required with --pred-latents.")
        predicted, target = load_latents(args.pred_latents), load_latents(args.target_latents)
        extraction_mode = "precomputed_latents"
    else:
        if not args.base_model or not args.ae_ckpt:
            parser.error("--base-model and --ae-ckpt are required with --input.")
        records = load_records(args.input)
        if args.limit is None:
            args.limit = len(records)
        predicted, target = extract_latents(records, args)
        extraction_mode = "online_hidden_state_projection"
    rows, summary = evaluate(predicted, target, args.metric, args.dtw_max_length)
    summary["extraction_mode"] = extraction_mode
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
