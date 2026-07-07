# Server development plan

This plan separates short-term AutoDL downstream work from later full training
and benchmarking on a new CFFF server.

## Current GitHub status

The working branch is direct-pushed to GitHub `main` after local checks. There
is no PR branch requirement for this solo-development workflow.

```text
main
```

## Path convention

Code should use repository-relative paths for source files and
`chatpathway.config.json` for runtime assets. Each server should set the active
profile and that profile's `asset_root`; wrappers then construct model, data,
checkpoint, run, and artifact paths automatically.

Default AutoDL layout:

```json
"active_profile": "autodl"
```

New server layout example:

```json
"active_profile": "cfff"
```

The expected runtime tree is:

```text
<profile asset_root>/
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
cd /root/autodl-tmp/ChatPathway2
git fetch origin
git checkout main
git pull --ff-only
python -m experiments.run_experiment check-assets --profile autodl --phase infer --ids b00_frameworka_force_damped_hnn_regularized_lora,x00_c2s_transfer_qwen --strict
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
git checkout main
mkdir -p /data/chatpathway/{models,data,checkpoints,runs,artifacts}
```

Then edit `chatpathway.config.json` so `active_profile` is `cfff` and the
`cfff.asset_root` is `/data/chatpathway`.

Asset preflight:

```bash
python -m experiments.run_experiment check-assets --profile cfff --phase both --strict
```

Training order:

| Order | Rows | Purpose |
| --- | --- | --- |
| 1 | `a00` or `a01` | Train/retrain shared SFT LoRA. |
| 2 | AE training script | Train or copy the shared AE projector. |
| 3 | `b00`, `b01` | Train current force/damp HNN control and PHNN candidate under the same layer. |
| 4 | `b02`-`b07` | Train latent dynamics teachers using shared SFT+AE. |
| 5 | `c00`, `c01` | Run dynamics-assisted inference using trained teachers. |
| 6 | `d00`-`d02` | Distill or jointly train dynamics back into LoRA adapters. |
| 7 | `x00` | Train/evaluate C2S transfer application if Task VI is in scope. |
| 8 | downstream Tasks I-VI | Run benchmark metrics against frozen outputs. |
| 9 | `z00` | Optional speculative LeJEPA probe only after core benchmark work. |

Use command plans before launching long jobs:

```bash
python -m experiments.run_experiment plan --phase train --format shell --output runs/experiment_plans/train_all.sh
python -m experiments.run_experiment plan --phase infer --format shell --output runs/experiment_plans/infer_all.sh
```

## Sync policy

This repository is currently a solo-development project. After local structure,
wrapper, and dry-run checks pass, commit and push directly to `main`. Use
feature branches only for risky experiments that should not interrupt the main
runtime workflow.
