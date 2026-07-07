"""Train LoRA and Neural ODE dynamics jointly."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.training.joint_lora_dynamics",
        [
            "--variant",
            "neural_ode",
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--sft-adapter",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5",
            "--ae-ckpt",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt",
            "--train",
            "/root/autodl-tmp/data/train_11_species_dataset.csv",
            "--save-dir",
            "/root/autodl-tmp/checkpoints/joint_lora_dynamics",
        ],
    )
