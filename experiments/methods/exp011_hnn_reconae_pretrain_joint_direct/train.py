"""Warm-start joint stage-2 training from the stable E010 HNN checkpoint."""

import os

from experiments._launch import asset_path, controlled_training_budget_args, run_torchrun_module, seeded_asset_path


if __name__ == "__main__":
    os.environ.setdefault("CHATPATHWAY_NPROC_PER_NODE", "1")
    run_torchrun_module(
        "method.training.framework_a_ddp",
        [
            *controlled_training_budget_args(),
            "--variant", "hnn",
            "--structure-mode", "orthogonal_poisson",
            "--damping-mode", "isotropic",
            "--dynamics-resolution", "substep_multiscale",
            "--max-dynamics-steps", "512",
            "--dynamics-init-checkpoint", seeded_asset_path("checkpoints/experiments/exp010_hnn_reconae_dynamics_only/dynamics_pretrain/checkpoint_best/hamiltonian_dynamics.pt"),
            "--dynamics-init-run-complete", seeded_asset_path("checkpoints/experiments/exp010_hnn_reconae_dynamics_only/dynamics_pretrain/run_complete.json"),
            "--require-pretrained-dynamics",
            "--dynamics-to-lora-warmup-fraction", "0.1",
            "--lr", "1e-5",
            "--dynamics-lr", "2e-4",
            "--epochs", "3",
            "--kl-weight", "0.02",
            "--gradient-conflict-interval", "100",
            "--base-model", asset_path("models/qwen3_8B"),
            "--sft-lora", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
            "--ae-ckpt", seeded_asset_path("checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"),
            "--train", asset_path("data/pathway_v4_full/train_pathway_continuation_v4.csv"),
            "--validation", asset_path("data/pathway_v4_full/validation_pathway_continuation_v4.csv"),
            "--save-dir", seeded_asset_path("checkpoints/experiments/exp011_hnn_reconae_pretrain_joint_direct/final_lora"),
        ],
    )
