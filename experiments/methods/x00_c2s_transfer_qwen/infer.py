"""Generate Qwen C2S prediction artifacts for Task VI."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "downstream.tasks.task6_perturbed_cell.generation",
        [
            "--model",
            "qwen_c2s",
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--adapter",
            asset_path("checkpoints/qwen3_8b_pathway_c2s_sft_small_5percent/checkpoint_epoch_5"),
            "--test-jsonl",
            asset_path("data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl"),
            "--output",
            asset_path("runs/c2s/jurkat_ours_results_epoch5_5percent.jsonl"),
        ],
    )
