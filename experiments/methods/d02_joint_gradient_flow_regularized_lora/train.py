"""Train LoRA and gradient-flow dynamics jointly."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "method.training.joint_lora_dynamics",
        [
            "--variant",
            "gradient_flow",
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--sft-adapter",
            asset_path("checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"),
            "--ae-ckpt",
            asset_path("checkpoints/legacy/qwen3_8b_ae_latent_128_cos/ae_epoch_5/ae_proj.pt"),
            "--train",
            asset_path("data/train_11_species_dataset.csv"),
            "--save-dir",
            asset_path("checkpoints/joint_lora_dynamics"),
        ],
    )
