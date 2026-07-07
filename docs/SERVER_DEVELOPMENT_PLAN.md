# Server development plan

This plan separates short-term AutoDL downstream work from later full training
and benchmarking on a new CFFF server.

## Current GitHub status

The experiment-matrix work is on branch:

```text
agent/organize-experiment-matrix
```

It is pushed to GitHub and tracked by the draft PR. `main` is not changed until
the PR is merged.

## Path convention

Code should use repository-relative paths for source files and
`CHATPATHWAY_ASSET_ROOT` for runtime assets.

Default AutoDL layout:

```bash
export CHATPATHWAY_ASSET_ROOT=/root/autodl-tmp
```

New server layout example:

```bash
export CHATPATHWAY_ASSET_ROOT=/data/chatpathway
```

The expected runtime tree is:

```text
$CHATPATHWAY_ASSET_ROOT/
  models/
  data/
  checkpoints/
  runs/
  artifacts/
```

## Phase 1: AutoDL downstream with existing checkpoints

Goal: use existing trusted checkpoints first, especially the selected inference
adapter and C2S artifacts, to finish Tasks I-VI without retraining.

Recommended steps:

```bash
cd $CHATPATHWAY_ASSET_ROOT/ChatPathway2
git fetch origin
git checkout agent/organize-experiment-matrix
git pull --ff-only
export CHATPATHWAY_ASSET_ROOT=/root/autodl-tmp
python -m experiments.run_experiment check-assets --phase infer --ids a02_frameworka_hnn_regularized_lora,e00_c2s_transfer_qwen --strict
```

Then run downstream tasks from existing inference outputs and C2S outputs. Do
not overwrite previous results unless the command explicitly uses a new
`runs/downstream/...` directory or an `--overwrite` flag.

Recommended storage:

| Output type | Relative location |
| --- | --- |
| Existing pathway generation | `runs/inference/frameworka_1/` or row-specific `runs/inference/...` |
| Task I/II outputs | `runs/downstream/task1_2/<method_or_ckpt>/` |
| Task III PCTE outputs | `runs/downstream/task3/<method_or_ckpt>/` |
| Task IV CSP outputs | `runs/downstream/task4/<method_or_ckpt>/` |
| Task V CKI outputs | `runs/downstream/task5/<method_or_ckpt>/` |
| Task VI C2S outputs | `runs/downstream/task6/<method_or_ckpt>/` |

## Phase 2: CFFF server full retraining and benchmark

Goal: reproduce training from shared stages, then fan out experiments and run
the complete benchmark.

One-time setup:

```bash
git clone https://github.com/KingofLaniakea/ChatPathway2.git
cd ChatPathway2
git checkout agent/organize-experiment-matrix
export CHATPATHWAY_ASSET_ROOT=/data/chatpathway
mkdir -p "$CHATPATHWAY_ASSET_ROOT"/{models,data,checkpoints,runs,artifacts}
```

Asset preflight:

```bash
python -m experiments.run_experiment check-assets --phase both --strict
```

Training order:

| Order | Rows | Purpose |
| --- | --- | --- |
| 1 | `a00` or `a01` | Train/retrain shared SFT LoRA. |
| 2 | AE training script | Train or copy the shared AE projector. |
| 3 | `a02`, `a03` | Train HNN/PHNN-regularized direct-generation adapters. |
| 4 | `b00`-`b06` | Train representation probes and latent dynamics teachers using shared SFT+AE. |
| 5 | `c00`, `c01` | Run dynamics-assisted inference using trained teachers. |
| 6 | `d00`-`d02` | Distill or jointly train dynamics back into LoRA adapters. |
| 7 | `e00` | Train/evaluate C2S transfer application if Task VI is in scope. |
| 8 | downstream Tasks I-VI | Run benchmark metrics against frozen outputs. |

Use command plans before launching long jobs:

```bash
python -m experiments.run_experiment plan --phase train --format shell --output runs/experiment_plans/train_all.sh
python -m experiments.run_experiment plan --phase infer --format shell --output runs/experiment_plans/infer_all.sh
```

## Merge policy

Use the PR branch for development until the matrix is reviewed. Merge to `main`
only after the branch passes local checks and either AutoDL downstream smoke
tests or CFFF runtime preflight.
