"""Train the Qwen C2S transfer adapter."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "scripts.c2s.train.train_c2s_single",
        [
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--init-adapter",
            asset_path("checkpoints/legacy/qwen3_8b_stage3_sft_hnn/checkpoint_epoch_5"),
            "--train-jsonl",
            asset_path("data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl"),
            "--save-dir",
            asset_path("checkpoints/qwen3_8b_pathway_c2s_sft_small_5percent"),
        ],
    )
