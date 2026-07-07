"""Score latent dynamics teacher rollouts on pathway text trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from method.dynamics.latent_teacher import (
    build_dynamics,
    extract_latent_trajectories,
    load_backbone,
    load_projection,
    read_records,
    rollout,
    trajectory_losses,
)


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_ADAPTER = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"
DEFAULT_AE = "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"
DEFAULT_INPUT = "/root/autodl-tmp/data/test_7_species_dataset.csv"
DEFAULT_OUTPUT = "/root/autodl-tmp/runs/latent_dynamics_rollout/scores.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--ae-ckpt", default=DEFAULT_AE)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = torch.load(args.checkpoint, map_location=args.device)
    variant = raw["variant"]
    cfg = raw.get("config", {})
    latent_dim = int(cfg.get("latent_dim", args.latent_dim))

    tokenizer, backbone = load_backbone(args.base_model, None if args.no_adapter else args.adapter, args.device)
    projection = load_projection(args.ae_ckpt, backbone.config.hidden_size, latent_dim, args.device)
    dynamics = build_dynamics(variant, latent_dim, latent_dim).to(args.device).float()
    dynamics.load_state_dict(raw["model_state_dict"])
    dynamics.eval()

    records = read_records(args.input, args.limit)
    metrics = []
    with output_path.open("w", encoding="utf-8") as handle:
        for start in tqdm(range(0, len(records), args.batch_size), desc=f"{variant} rollout scoring"):
            batch_records = records[start : start + args.batch_size]
            trajectories = extract_latent_trajectories(
                batch_records,
                tokenizer,
                backbone,
                projection,
                args.device,
                args.text_column,
                args.max_length,
                args.max_steps,
                start_sample_id=start,
            )
            for item in trajectories:
                target = item.z.to(args.device)
                predicted = rollout(dynamics, variant, target[0], item.control.to(args.device), target.size(0) - 1)
                losses = trajectory_losses(predicted, target)
                mse = torch.mean((predicted[: target.size(0)] - target) ** 2)
                row = {
                    "sample_id": item.sample_id,
                    "variant": variant,
                    "text_column": args.text_column,
                    "steps": int(target.size(0)),
                    "rollout_loss": float(losses["rollout"].item()),
                    "velocity_loss": float(losses["velocity"].item()),
                    "rollout_cosine": float(losses["cosine"].item()),
                    "rollout_mse": float(mse.item()),
                }
                metrics.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "variant": variant,
        "checkpoint": args.checkpoint,
        "num_samples": len(metrics),
        "mean_rollout_loss": sum(row["rollout_loss"] for row in metrics) / max(len(metrics), 1),
        "mean_velocity_loss": sum(row["velocity_loss"] for row in metrics) / max(len(metrics), 1),
        "mean_rollout_cosine": sum(row["rollout_cosine"] for row in metrics) / max(len(metrics), 1),
        "mean_rollout_mse": sum(row["rollout_mse"] for row in metrics) / max(len(metrics), 1),
    }
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with output_path.with_suffix(".run.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(summary)


if __name__ == "__main__":
    main()
