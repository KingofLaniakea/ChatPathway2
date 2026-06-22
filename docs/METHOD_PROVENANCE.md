# `method/` provenance and current inference semantics

## Migration boundary

Commit `a419f53` (`Import audited server source snapshot`) imported the server
source snapshot into `ChatPathway2`. The initial 37 Python source files were
byte-checked against the server source before the old root-level copies were
removed. `method/` was therefore migrated, not rewritten.

From that baseline through this document's revision, the only modified legacy
method file is `method/inference.py` in commit `ff2a71a`:

* base model remains `/root/autodl-tmp/qwen3_8B`;
* adapter remains `/root/autodl-tmp/qwen3_8b_FrameworkA_1/checkpoint_epoch_4`;
* default input remains `/root/autodl-tmp/test_7_species_dataset.csv`;
* batch size, maximum input length, maximum new tokens, greedy decoding, and
  prompt template remain the legacy defaults;
* output changed deliberately from the misleading, pre-existing legacy file
  `test_7_predictions_ae_cos.csv` to
  `runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv`.

The change adds command-line overrides, output-directory creation, overwrite
protection, and a `.run.json` record. It does **not** change the trained model
or data format.

## What the current inference run is

`qwen3_8b_FrameworkA_1/checkpoint_epoch_4` contains both a LoRA adapter
(`adapter_model.safetensors`) and a separate `hnn_func.pt`. The inference
script loads only the LoRA adapter. It does not load the HNN state, load the AE
projector, perform ODE rollout, or inject a latent trajectory into decoding.

It is consequently accurate to call the current job **FrameworkA_1 LoRA direct
generation**. If the checkpoint was trained with the FrameworkA script, the
LoRA may have been shaped by its training losses, but this is not HNN runtime
inference. In addition, the audited FrameworkA source detaches the HNN
trajectory before the alignment loss; that alignment loss does not optimize the
HNN parameters. `hnn_func.pt` alone is not evidence of a fully jointly trained
counterfactual dynamics model.
