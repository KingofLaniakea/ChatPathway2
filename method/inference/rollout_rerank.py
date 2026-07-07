"""Rerank generated pathway-answer candidates with a latent dynamics teacher."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from method.dynamics.latent_teacher import (
    build_dynamics,
    extract_latent_trajectories,
    load_backbone,
    load_projection,
    rollout,
    trajectory_losses,
)


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"
DEFAULT_AE = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/checkpoints/latent_dynamics_teachers/neural_ode/neural_ode_epoch_3.pt"
DEFAULT_INPUT = "/root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv"
DEFAULT_OUTPUT = "/root/autodl-tmp/runs/latent_dynamics_rerank/reranked_candidates.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--ae-ckpt", default=DEFAULT_AE)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-columns", default="predicted_answer", help="Comma-separated columns containing candidate answers.")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--batch-size", type=int, default=1, help="Kept for future batching; reranking is candidate-wise.")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def candidate_columns(args: argparse.Namespace, df: pd.DataFrame) -> list[str]:
    columns = [value.strip() for value in args.candidate_columns.split(",") if value.strip()]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Candidate columns not found: {', '.join(missing)}")
    return columns


def score_candidate(
    question: str,
    candidate: str,
    sample_id: int,
    tokenizer: Any,
    backbone: Any,
    projection: Any,
    dynamics: Any,
    variant: str,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    records = [{args.question_column: question, "__candidate__": candidate}]
    trajectories = extract_latent_trajectories(
        records,
        tokenizer,
        backbone,
        projection,
        args.device,
        "__candidate__",
        args.max_length,
        args.max_steps,
        start_sample_id=sample_id,
    )
    if not trajectories:
        return {"rollout_loss": float("inf"), "velocity_loss": float("inf"), "rollout_cosine": -1.0, "steps": 0}
    item = trajectories[0]
    target = item.z.to(args.device)
    predicted = rollout(dynamics, variant, target[0], item.control.to(args.device), target.size(0) - 1)
    losses = trajectory_losses(predicted, target)
    return {
        "rollout_loss": float(losses["rollout"].item()),
        "velocity_loss": float(losses["velocity"].item()),
        "rollout_cosine": float(losses["cosine"].item()),
        "steps": int(target.size(0)),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, engine="python", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
    if args.limit is not None:
        df = df.head(args.limit)
    if args.question_column not in df.columns:
        raise ValueError(f"Input must contain question column '{args.question_column}'.")
    columns = candidate_columns(args, df)

    raw = torch.load(args.checkpoint, map_location=args.device)
    variant = raw["variant"]
    cfg = raw.get("config", {})
    latent_dim = int(cfg.get("latent_dim", args.latent_dim))
    tokenizer, backbone = load_backbone(args.base_model, None if args.no_adapter else args.adapter, args.device)
    projection = load_projection(args.ae_ckpt, backbone.config.hidden_size, latent_dim, args.device)
    dynamics = build_dynamics(variant, latent_dim, latent_dim).to(args.device).float()
    dynamics.load_state_dict(raw["model_state_dict"])
    dynamics.eval()

    rows = []
    detail_rows = []
    for index, record in tqdm(list(df.iterrows()), desc=f"{variant} candidate rerank"):
        question = str(record[args.question_column])
        scored = []
        for column in columns:
            candidate = "" if pd.isna(record[column]) else str(record[column])
            score = score_candidate(question, candidate, int(index), tokenizer, backbone, projection, dynamics, variant, args)
            scored.append((column, candidate, score))
            detail_rows.append({
                "sample_id": int(index),
                "candidate_column": column,
                "rollout_loss": score["rollout_loss"],
                "velocity_loss": score["velocity_loss"],
                "rollout_cosine": score["rollout_cosine"],
                "steps": score["steps"],
            })
        best_column, best_candidate, best_score = min(scored, key=lambda item: float(item[2]["rollout_loss"]))
        output_record = record.to_dict()
        output_record["reranked_answer"] = best_candidate
        output_record["reranked_source_column"] = best_column
        output_record["rerank_rollout_loss"] = best_score["rollout_loss"]
        output_record["rerank_velocity_loss"] = best_score["velocity_loss"]
        output_record["rerank_rollout_cosine"] = best_score["rollout_cosine"]
        output_record["rerank_teacher_variant"] = variant
        rows.append(output_record)

    pd.DataFrame(rows).to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    with output_path.with_suffix(".details.jsonl").open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output_path.with_suffix(".run.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote reranked CSV: {output_path}")


if __name__ == "__main__":
    main()
