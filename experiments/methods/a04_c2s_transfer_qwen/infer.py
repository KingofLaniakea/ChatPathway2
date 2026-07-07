"""Generate Qwen C2S prediction artifacts for Task VI."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "downstream.tasks.task6_perturbed_cell.generation",
        [
            "--model",
            "qwen_c2s",
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_small_5percent/checkpoint_epoch_5",
            "--test-jsonl",
            "/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl",
            "--output",
            "/root/autodl-tmp/runs/c2s/jurkat_ours_results_epoch5_5percent.jsonl",
        ],
    )
