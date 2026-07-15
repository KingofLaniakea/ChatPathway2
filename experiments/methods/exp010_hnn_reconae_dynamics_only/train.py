"""Freeze stage-1 SFT/AE and pretrain the pure HNN for at most three epochs."""

from experiments._launch import asset_path, controlled_training_budget_args, run_module, seeded_asset_path


if __name__ == "__main__":
    run_module(
        "method.training.hamiltonian_pretrain",
        [
            *controlled_training_budget_args(),
            "--variant", "hnn",
            "--structure-mode", "orthogonal_poisson",
            "--damping-mode", "isotropic",
            "--dynamics-resolution", "substep_multiscale",
            "--max-dynamics-steps", "512",
            "--epochs", "3",
            "--base-model", asset_path("models/qwen3_8B"),
            "--sft-lora", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
            "--ae-ckpt", seeded_asset_path("checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"),
            "--train", asset_path("data/pathway_v4_full/train_pathway_continuation_v4.csv"),
            "--validation", asset_path("data/pathway_v4_full/validation_pathway_continuation_v4.csv"),
            "--save-dir", seeded_asset_path("checkpoints/experiments/exp010_hnn_reconae_dynamics_only/dynamics_pretrain"),
        ],
    )
