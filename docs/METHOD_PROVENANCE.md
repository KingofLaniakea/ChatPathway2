# `method/` provenance and current inference semantics

## Migration boundary

Commit `a419f53` (`Import audited server source snapshot`) imported the server
source snapshot into `ChatPathway2`. The initial 37 Python source files were
byte-checked against the server source before the old root-level copies were
removed. `method/` was therefore migrated, not rewritten.

After that import, maintained changes to method code are intentionally narrow:

`method/inference/pathway.py` was made configurable in commit `ff2a71a`:

* base model is now referenced directly as `/root/autodl-tmp/models/qwen3_8B`;
* adapter is now referenced directly as `/root/autodl-tmp/checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_4`;
* default input is now referenced directly as `/root/autodl-tmp/data/test_7_species_dataset.csv`;
* batch size, maximum input length, maximum new tokens, greedy decoding, and
  prompt template remain the legacy defaults;
* output changed deliberately from the misleading, pre-existing legacy file
  `test_7_predictions_ae_cos.csv` to
  `runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv`.

The change adds command-line overrides, output-directory creation, overwrite
protection, and a `.run.json` record. It does **not** change the trained model
or data format.

This revision also fixes two training-script maintenance issues without
changing the checkpoint paths or inference semantics:

* `method/training/framework_a.py` keeps the frozen AE projection in the
  alignment graph so `loss_align` can update HNN as intended;
* `method/training/latent_ae.py` restores the MSE/cosine history counters needed
  for the AE training loop to run.

## What the current inference run is

`checkpoints/legacy/qwen3_8b_FrameworkA_1/checkpoint_epoch_4` contains both a LoRA adapter
(`adapter_model.safetensors`) and a separate `hnn_func.pt`. The inference
script loads only the LoRA adapter. It does not load the HNN state, load the AE
projector, perform ODE rollout, or inject a latent trajectory into decoding.

It is consequently accurate to call the current job **FrameworkA_1 LoRA direct
generation**. If the checkpoint was trained with the FrameworkA script, the
LoRA may have been shaped by its training losses, but this is not HNN runtime
inference.

The migrated FrameworkA source used to detach the HNN trajectory before the
alignment loss. The maintained `method/training/framework_a.py` now keeps that
graph intact so future FrameworkA training can update HNN through
`loss_align`. Existing legacy checkpoints may still come from the pre-fix
script; `hnn_func.pt` alone is not evidence of a fully validated
counterfactual dynamics model without a run manifest.
