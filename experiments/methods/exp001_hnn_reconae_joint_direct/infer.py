"""Run inference/evaluation for this experiment."""

from experiments._launch import asset_path, run_module, seeded_asset_path


if __name__ == "__main__":
    run_module(
        'method.inference.pathway',
        ['--base-model', asset_path('models/qwen3_8B'), '--adapter', seeded_asset_path('checkpoints/experiments/exp001_hnn_reconae_joint_direct/final_lora/checkpoint_best'), '--input', asset_path('data/test_kegg_pathway_eval.csv'), '--output', seeded_asset_path('runs/experiments/exp001_hnn_reconae_joint_direct/direct.csv')],
    )
