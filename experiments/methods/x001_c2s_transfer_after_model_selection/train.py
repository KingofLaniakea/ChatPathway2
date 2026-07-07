"""Run the full training pipeline for this experiment."""

from experiments._launch import asset_path, run_steps


if __name__ == "__main__":
    run_steps([
        ('scripts.c2s.train.train_c2s_single', ['--base-model', asset_path('models/qwen3_8B'), '--init-adapter', asset_path('checkpoints/selected_pathway_adapter/checkpoint_epoch_best'), '--train-jsonl', asset_path('data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl'), '--save-dir', asset_path('checkpoints/experiments/x001_c2s_transfer_after_model_selection/final_lora')]),
    ])
