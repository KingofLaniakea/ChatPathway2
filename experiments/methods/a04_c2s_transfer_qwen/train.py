"""Train the Qwen C2S transfer adapter."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "scripts.c2s.train.train_c2s_single",
        [
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--init-adapter",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_stage3_sft_hnn/checkpoint_epoch_5",
            "--train-jsonl",
            "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl",
            "--save-dir",
            "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_small_5percent",
        ],
    )
