"""Run inference/evaluation for this experiment."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        'method.inference.lejepa_pathway',
        ['--base-model', asset_path('models/qwen3_8B'), '--adapter', asset_path('checkpoints/experiments/exp000_sft_only_direct/sft/checkpoint_epoch_5'), '--checkpoint', asset_path('checkpoints/experiments/z001_lejepa_speculative_probe/lejepa_epoch_3.pt'), '--input', asset_path('data/test_7_species_dataset.csv'), '--output', asset_path('runs/experiments/z001_lejepa_speculative_probe/scores.jsonl')],
    )
