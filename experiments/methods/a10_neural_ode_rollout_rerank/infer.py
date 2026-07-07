"""Rerank generated candidates with a Neural ODE rollout teacher."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.inference.rollout_rerank",
        [
            "--checkpoint",
            "/root/autodl-tmp/checkpoints/latent_dynamics_teachers/neural_ode/neural_ode_epoch_3.pt",
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5",
            "--ae-ckpt",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt",
            "--input",
            "/root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv",
            "--output",
            "/root/autodl-tmp/runs/latent_dynamics_rerank/neural_ode_reranked_candidates.csv",
        ],
    )
