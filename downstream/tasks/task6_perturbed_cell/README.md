# Task VI: Perturbed-Cell Transfer Evaluation

This task evaluates whether a pathway-trained initialization transfers into a
single-cell perturbation setting after C2S supervision.

## Scope

This is not a pure zero-shot claim. The current Qwen path loads the Stage-3
pathway adapter as initialization, then uses a C2S LoRA checkpoint trained on
single-cell rank-text data. The task is therefore a transfer/application test:
does the pathway prior provide a useful initialization for limited C2S
fine-tuning, and how does the resulting model compare with a C2S-specific Gemma
baseline under the same text-to-vector scoring space?

## Server Assets

| Label | Path |
| --- | --- |
| C2S train JSONL | `/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl` |
| C2S test JSONL | `/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl` |
| Qwen base | `/root/autodl-tmp/models/qwen3_8B` |
| Qwen C2S adapter | `/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_small_5percent/checkpoint_epoch_5` |
| Gemma baseline | `/root/autodl-tmp/models/C2S-Scale-Gemma-2-2B` |
| Qwen prediction JSONL | `/root/autodl-tmp/runs/c2s/jurkat_ours_results_epoch5.jsonl` |
| Gemma prediction JSONL | `/root/autodl-tmp/runs/c2s/jurkat_test_gemma_predictions_result_5percent_500.jsonl` |

The legacy Qwen script generated 100 rows by default; the legacy Gemma script
generated 500 rows. A reportable comparison should fix the same `--limit` for
both and save the `.run.json` metadata.

## Regenerate Predictions

```bash
python -m downstream.tasks.task6_perturbed_cell.generation \
  --model qwen_c2s \
  --limit 500 \
  --output /root/autodl-tmp/runs/c2s/jurkat_qwen_c2s_epoch5_limit500.jsonl \
  --overwrite

python -m downstream.tasks.task6_perturbed_cell.generation \
  --model gemma \
  --limit 500 \
  --output /root/autodl-tmp/runs/c2s/jurkat_gemma_c2s_limit500.jsonl \
  --overwrite
```

## Score Predictions

```bash
python -m downstream.tasks.task6_perturbed_cell \
  --c2s-predictions /root/autodl-tmp/runs/c2s/jurkat_qwen_c2s_epoch5_limit500.jsonl \
  --c2s-train /root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl \
  --output-dir /root/autodl-tmp/runs/downstream/task6/qwen_c2s_limit500

python -m downstream.tasks.task6_perturbed_cell \
  --c2s-predictions /root/autodl-tmp/runs/c2s/jurkat_gemma_c2s_limit500.jsonl \
  --c2s-train /root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl \
  --output-dir /root/autodl-tmp/runs/downstream/task6/gemma_limit500
```
