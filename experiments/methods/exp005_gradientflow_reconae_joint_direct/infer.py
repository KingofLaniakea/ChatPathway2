"""Run inference/evaluation for this experiment."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        'method.inference.pathway',
        ['--base-model', asset_path('models/qwen3_8B'), '--adapter', asset_path('checkpoints/experiments/exp005_gradientflow_reconae_joint_direct/final_lora/gradient_flow/checkpoint_epoch_3'), '--input', asset_path('data/test_7_species_dataset.csv'), '--output', asset_path('runs/experiments/exp005_gradientflow_reconae_joint_direct/direct.csv')],
    )
