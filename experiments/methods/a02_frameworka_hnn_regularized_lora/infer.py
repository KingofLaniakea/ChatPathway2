"""Run direct LoRA inference for a FrameworkA HNN-regularized adapter."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "method.inference.pathway",
        [
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--adapter",
            asset_path("checkpoints/qwen3_8b_FrameworkA_ae_cos/checkpoint_epoch_4"),
            "--input",
            asset_path("data/test_7_species_dataset.csv"),
            "--output",
            asset_path("runs/inference/frameworka_ae_cos/test_7_species_frameworka_ae_cos_epoch4.csv"),
        ],
    )
