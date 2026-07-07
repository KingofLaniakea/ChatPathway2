"""Run direct generation with the jointly trained Neural ODE-regularized adapter."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "method.inference.pathway",
        [
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--adapter",
            asset_path("checkpoints/joint_lora_dynamics/neural_ode/checkpoint_epoch_3"),
            "--input",
            asset_path("data/test_7_species_dataset.csv"),
            "--output",
            asset_path("runs/inference/joint_lora_dynamics/neural_ode_epoch3.csv"),
        ],
    )
