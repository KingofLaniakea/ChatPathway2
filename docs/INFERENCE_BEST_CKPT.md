# Current inference checkpoint selection

This file records the checkpoint paths currently selected by the maintained
ChatPathway2 inference code. It records code defaults; it does not independently
re-rank checkpoints.

## Main pathway inference

Entry point:

```bash
python -m method.inference.pathway
```

Current default paths in `method/inference/pathway.py`:

| Item | Path |
| --- | --- |
| Base model | `/root/autodl-tmp/models/qwen3_8B` |
| Selected LoRA adapter | `/root/autodl-tmp/checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_4` |
| Default input CSV | `/root/autodl-tmp/data/test_7_species_dataset.csv` |
| Default output CSV | `/root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv` |

Runtime behavior:

```text
Qwen3-8B base model
  + selected LoRA adapter checkpoint_epoch_4
  -> greedy generation
  -> predicted_answer
```

The script does not load or execute:

| Artifact | Runtime use in `method.inference.pathway` |
| --- | --- |
| `ae_proj.pt` | not loaded |
| `hnn_func.pt` | not loaded |

## Historical or debug inference entries

These files are kept for provenance/debugging and should not override the main
pathway default unless an experiment explicitly says so.

| File | Adapter/model path | Intended status |
| --- | --- | --- |
| `method/inference/pathway_batch.py` | `/root/autodl-tmp/checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_2` | older small-batch/default-small test path |
| `scripts/inference/test_single_generation.py` | `/root/autodl-tmp/checkpoints/legacy/qwen3_8b_stage3_sft_hnn/checkpoint_epoch_5` | single C2S/debug probe for a legacy Stage-3 adapter |
| `scripts/inference/zero_shot_inference.py` | `/root/autodl-tmp/models/C2S-Scale-Gemma-2-2B` | Gemma zero-shot/debug comparison, no Qwen LoRA |

## Practical rule

For pathway hypothesis generation, use the main entry point and the selected
FrameworkA epoch-4 adapter unless the experiment manifest explicitly pins a
different adapter.
