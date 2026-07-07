"""Train a LoRA adapter with a frozen Neural ODE dynamics teacher."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.training.dynamics_distilled_lora",
        [
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--sft-adapter",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5",
            "--teacher-checkpoint",
            "/root/autodl-tmp/checkpoints/latent_dynamics_teachers/neural_ode/neural_ode_epoch_3.pt",
            "--ae-ckpt",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt",
            "--train",
            "/root/autodl-tmp/data/train_11_species_dataset.csv",
            "--save-dir",
            "/root/autodl-tmp/checkpoints/dynamics_distilled_lora/neural_ode",
        ],
    )
