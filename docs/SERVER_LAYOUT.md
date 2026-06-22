# Server storage layout and legacy links

Canonical server root: `/root/autodl-tmp`.

| Location | Contents | Git status |
| --- | --- | --- |
| `ChatPathway2/` | source code and documentation only | Git repository |
| `models/` | immutable base models | not tracked |
| `data/` | datasets and reference material | not tracked |
| `checkpoints/` | LoRA, AE, HNN checkpoints | not tracked |
| `runs/` | generated outputs, logs, evaluation reports | not tracked |

`ChatPathway2` contains no model, dataset, checkpoint, or run-output symlink.
The arrows shown in the server file browser are *root-level compatibility
links*, retained solely for historic scripts that hard-code old paths.

| Compatibility link | Canonical target category |
| --- | --- |
| `qwen3_8B`, `C2S-Scale-Gemma-2-2B` | `models/` |
| `qwen3_8b_sft`, `qwen3_8b_FrameworkA`, `qwen3_8b_FrameworkA_1`, `qwen3_8b_FrameworkA_ae_cos` | `checkpoints/legacy/` |
| `qwen3_8b_ae_latent_128_cos`, `qwen3_8b_ae_latent_256`, `qwen3_8b_ae_new_1`, `qwen3_8b_ae_only` | `checkpoints/legacy/` |
| `qwen3_8b_stage1_hnn_only`, `qwen3_8b_stage3_sft_hnn` | `checkpoints/legacy/` |
| `test_7_species_dataset.csv`, `test_7_species_dataset_small.csv`, `train_11_species_dataset.csv` | `data/` |
| `GSE264667_jurkat_C2S_format.h5ad`, `GSE264667_jurkat_raw_singlecell_01.h5ad` | `data/` |
| `test_7_predictions_*.csv` | `runs/legacy-method/predictions/` |

The links make old scripts runnable but are not the source of truth for new
work. New output must use `runs/<experiment>/...`; new code should receive
paths through CLI arguments or a configuration file.

Every fresh SSH session must run `source /etc/network_turbo` before GitHub or
Hugging Face access. It is only an academic GitHub/Hugging Face accelerator and
can make unrelated access such as package indexes slower.
