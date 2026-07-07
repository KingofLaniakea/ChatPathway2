"""Train latent dynamics teachers on ChatPathway answer trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from method.dynamics.latent_teacher import (
    build_dynamics,
    checkpoint_payload,
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
DEFAULT_TRAIN = "/root/autodl-tmp/data/train_11_species_dataset.csv"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/latent_dynamics_teachers"


@dataclass
class TrainConfig:
    variant: str
    base_model: str
    adapter: str | None
    ae_ckpt: str
    train_path: str
    save_dir: str
    batch_size: int
    epochs: int
    lr: float
    latent_dim: int
    max_length: int
    max_steps: int
    lambda_velocity: float
    lambda_reg: float
    text_column: str
    device: str
    limit: int | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--variant",
        choices=("neural_ode", "latent_ode", "gradient_flow", "generic", "koopman", "sindy"),
        required=True,
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--ae-ckpt", default=DEFAULT_AE)
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--lambda-velocity", type=float, default=0.5)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    return TrainConfig(
        variant=args.variant,
        base_model=args.base_model,
        adapter=None if args.no_adapter else args.adapter,
        ae_ckpt=args.ae_ckpt,
        train_path=args.train,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        max_length=args.max_length,
        max_steps=args.max_steps,
        lambda_velocity=args.lambda_velocity,
        lambda_reg=args.lambda_reg,
        text_column=args.text_column,
        device=args.device,
        limit=args.limit,
    )


def train() -> None:
    cfg = parse_args()
    save_dir = Path(cfg.save_dir) / cfg.variant
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, backbone = load_backbone(cfg.base_model, cfg.adapter, cfg.device)
    projection = load_projection(cfg.ae_ckpt, backbone.config.hidden_size, cfg.latent_dim, cfg.device)
    dynamics = build_dynamics(cfg.variant, cfg.latent_dim, cfg.latent_dim).to(cfg.device).float()
    optimizer = optim.AdamW(dynamics.parameters(), lr=cfg.lr)
    records = read_records(cfg.train_path, cfg.limit)

    history: list[dict[str, float | int]] = []
    for epoch in range(cfg.epochs):
        dynamics.train()
        totals = {"rollout": 0.0, "velocity": 0.0, "reg": 0.0, "total": 0.0}
        steps = 0
        for start in tqdm(range(0, len(records), cfg.batch_size), desc=f"{cfg.variant} epoch {epoch + 1}"):
            batch_records = records[start : start + cfg.batch_size]
            trajectories = extract_latent_trajectories(
                batch_records,
                tokenizer,
                backbone,
                projection,
                cfg.device,
                cfg.text_column,
                cfg.max_length,
                cfg.max_steps,
                start_sample_id=start,
            )
            if not trajectories:
                continue

            loss = torch.tensor(0.0, device=cfg.device)
            rollout_total = torch.tensor(0.0, device=cfg.device)
            velocity_total = torch.tensor(0.0, device=cfg.device)
            for item in trajectories:
                target = item.z.to(cfg.device)
                predicted = rollout(dynamics, cfg.variant, target[0], item.control.to(cfg.device), target.size(0) - 1)
                losses = trajectory_losses(predicted, target)
                rollout_total = rollout_total + losses["rollout"]
                velocity_total = velocity_total + losses["velocity"]

            count = max(len(trajectories), 1)
            rollout_loss = rollout_total / count
            velocity_loss = velocity_total / count
            reg_loss = dynamics.regularization_loss()
            loss = rollout_loss + cfg.lambda_velocity * velocity_loss + cfg.lambda_reg * reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
            optimizer.step()

            totals["rollout"] += float(rollout_loss.item())
            totals["velocity"] += float(velocity_loss.item())
            totals["reg"] += float(reg_loss.item())
            totals["total"] += float(loss.item())
            steps += 1

        row = {"epoch": epoch + 1, **{key: value / max(steps, 1) for key, value in totals.items()}}
        history.append(row)
        torch.save(checkpoint_payload(dynamics, cfg.variant, cfg), save_dir / f"{cfg.variant}_epoch_{epoch + 1}.pt")
        with (save_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    with (save_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    train()
