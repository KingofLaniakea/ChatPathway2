"""Rerank generated candidates with a Neural ODE rollout teacher."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "method.inference.rollout_rerank",
        [
            "--checkpoint",
            asset_path("checkpoints/latent_dynamics_teachers/neural_ode/neural_ode_epoch_3.pt"),
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--adapter",
            asset_path("checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"),
            "--ae-ckpt",
            asset_path("checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"),
            "--input",
            asset_path("runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv"),
            "--output",
            asset_path("runs/latent_dynamics_rerank/neural_ode_reranked_candidates.csv"),
        ],
    )
