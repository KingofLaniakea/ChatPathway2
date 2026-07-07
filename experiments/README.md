# ChatPathway2 experiment matrix

This directory contains experiment wrappers. Source implementations stay under
`method/`, `scripts/`, and `downstream/`.

The design matrix is not a checkpoint inventory. `EXPERIMENT_MATRIX.csv` is a
combination table with one row per recommended experiment and columns for the
experimental layers:

| Column | Meaning |
| --- | --- |
| `a_dynamics` | HNN/PHNN/Neural ODE/etc. middle dynamics choice. |
| `b_ae` | AE/latent-space training choice. |
| `d_training_schedule` | Whether dynamics is trained first or jointly with the second LoRA stage. |
| `c_inference` | Direct LoRA, multi-answer rerank, or FrameworkB-style latent averaging. |

Runtime paths and expected artifacts are separated into `runtime_manifest.json`.
Server-specific roots are configured in `chatpathway.config.json`.

## Current Main Row

The current training + inference process is:

```text
exp001_hnn_reconae_joint_direct
a = a1_force_damped_hnn_current_control
b = b0_reconstruction_ae_frozen_current
d = d2_joint_second_lora_and_dynamics
c = c0_direct_lora
```

Its training wrapper runs, from scratch:

```text
method.training.sft
-> method.training.latent_ae
-> method.training.framework_a
```

Its inference wrapper calls:

```text
method.inference.pathway
```

So HNN affects generation only through the final LoRA adapter in this row; HNN
is not loaded by direct inference.

## Commands

```bash
python -m experiments.run_experiment list
python -m experiments.run_experiment axes
python -m experiments.run_experiment runtime exp001_hnn_reconae_joint_direct
python -m experiments.run_experiment train exp001_hnn_reconae_joint_direct --dry-run
python -m experiments.run_experiment infer exp001_hnn_reconae_joint_direct --dry-run
python -m experiments.run_experiment check-assets --phase both --ids exp001_hnn_reconae_joint_direct --strict
python -m experiments.validate_matrix
python -m experiments.run_experiment audit --phase both --quiet
python -m experiments.run_experiment consistency --phase both --quiet
```

To inspect the full inner training chain without importing model dependencies:

```bash
CHATPATHWAY_LAUNCH_DRY_RUN=1 python -m experiments.methods.exp001_hnn_reconae_joint_direct.train
```

## Implemented Rows

Implemented rows have concrete wrappers under `experiments/methods/<id>/` and a
`settings.json` that records the layer choices.

| ID | Status | a | b | d | c |
| --- | --- | --- | --- | --- | --- |
| `exp000_sft_only_direct` | runnable | none | none | SFT only | direct LoRA |
| `exp001_hnn_reconae_joint_direct` | runnable current main | forced/damped HNN | reconstruction AE frozen | joint second LoRA + dynamics | direct LoRA |
| `exp002_phnn_reconae_joint_direct` | runnable candidate | PHNN | reconstruction AE frozen | joint second LoRA + dynamics | direct LoRA |
| `exp003_neuralode_reconae_teacher_direct` | runnable candidate | Neural ODE | reconstruction AE frozen | train teacher then second LoRA | direct LoRA |
| `exp004_neuralode_reconae_joint_direct` | runnable candidate | Neural ODE | reconstruction AE frozen | joint second LoRA + dynamics | direct LoRA |
| `exp005_gradientflow_reconae_joint_direct` | runnable candidate | gradient flow | reconstruction AE frozen | joint second LoRA + dynamics | direct LoRA |
| `exp010_neuralode_reconae_teacher_rerank_partial` | partial | Neural ODE | reconstruction AE frozen | train teacher first | rerank existing candidates |
| `x001_c2s_transfer_after_model_selection` | optional downstream | selected model | not applicable | C2S transfer LoRA | C2S generation |
| `z001_lejepa_speculative_probe` | lowest priority | not dynamics | not core AE | representation probe | latent scoring |

Planned rows such as HNN-staged training, dynamics-aware AE, joint AE+dynamics,
and FrameworkB latent weighted-average inference are listed in
`EXPERIMENT_MATRIX.csv` with `status=planned_method_missing`.

## Diagnostic Wrappers

`diag001` through `diag006` are not full abcd benchmark rows. They train or
score one latent dynamics teacher against frozen SFT LoRA plus frozen
reconstruction AE, so they are useful for checking rollout fit before deciding
whether a dynamics family deserves a full `exp*` row. Each diagnostic directory
still has `settings.json`, but these rows are intentionally excluded from the
core implemented matrix and `runtime_manifest.json`.
