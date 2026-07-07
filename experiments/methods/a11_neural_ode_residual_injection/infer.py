"""Generate with Neural ODE latent-rollout residual injection."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.inference.rollout_residual_injection",
        [
            "--checkpoint",
            "/root/autodl-tmp/checkpoints/latent_dynamics_teachers/neural_ode/neural_ode_epoch_3.pt",
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_4",
            "--ae-ckpt",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt",
            "--input",
            "/root/autodl-tmp/data/test_7_species_dataset.csv",
            "--output",
            "/root/autodl-tmp/runs/latent_dynamics_injection/neural_ode_residual_generation.csv",
        ],
    )
