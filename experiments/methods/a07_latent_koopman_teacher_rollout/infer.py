"""Score Koopman teacher rollouts on latent pathway trajectories."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.inference.latent_dynamics_rollout",
        [
            "--checkpoint",
            "/root/autodl-tmp/checkpoints/latent_dynamics_teachers/koopman/koopman_epoch_3.pt",
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5",
            "--ae-ckpt",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt",
            "--input",
            "/root/autodl-tmp/data/test_7_species_dataset.csv",
            "--output",
            "/root/autodl-tmp/runs/latent_dynamics_rollout/koopman_scores.jsonl",
        ],
    )
