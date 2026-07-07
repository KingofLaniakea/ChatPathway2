"""Score GENERIC-style teacher rollouts on latent pathway trajectories."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "method.inference.latent_dynamics_rollout",
        [
            "--checkpoint",
            asset_path("checkpoints/latent_dynamics_teachers/generic/generic_epoch_3.pt"),
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--adapter",
            asset_path("checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"),
            "--ae-ckpt",
            asset_path("checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"),
            "--input",
            asset_path("data/test_7_species_dataset.csv"),
            "--output",
            asset_path("runs/latent_dynamics_rollout/generic_scores.jsonl"),
        ],
    )
