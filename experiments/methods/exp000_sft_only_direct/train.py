"""Run the full training pipeline for this experiment."""

from experiments._launch import asset_path, run_steps


if __name__ == "__main__":
    run_steps([
        ('method.training.sft', ['--base-model', asset_path('models/qwen3_8B'), '--train', asset_path('data/train_11_species_dataset.csv'), '--save-dir', asset_path('checkpoints/experiments/exp000_sft_only_direct/sft')]),
    ])
