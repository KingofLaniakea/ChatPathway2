# ChatPathway2 workflow

This file is the operational map for the code that should be used from this
repository. It intentionally does not depend on `PathwayDynamicsLLM`; that tree
is an exploratory refactor and is not an authority for reproducing the current
ChatPathway2 results.

Large assets live outside Git under `/root/autodl-tmp`:

```text
/root/autodl-tmp/models       base models
/root/autodl-tmp/data         datasets and references
/root/autodl-tmp/checkpoints  LoRA, AE, HNN checkpoints
/root/autodl-tmp/runs         inference and downstream outputs
```

## Current pathway model flow

1. SFT LoRA

   Entry point: `method/training/sft.py`

   This trains the first Qwen LoRA adapter on pathway question/answer records.
   The maintained script is argparse-based with `/root/autodl-tmp` defaults. It
   runs as a normal single-process command or under `torchrun` when distributed
   environment variables are present.

2. Latent AE

   Entry point: `method/training/latent_ae.py`

   This freezes the base model plus SFT LoRA, extracts final-layer hidden states,
   and trains a 4096 -> 128 -> 4096 projection. The loss is reconstruction MSE
   plus cosine direction loss. This AE is not yet a dynamics-aware latent-space
   learner.

3. FrameworkA LoRA plus TDHNN

   Entry point: `method/training/framework_a.py`

   This loads the SFT LoRA and frozen AE, initializes `TDHNNFunc`, rolls out the
   latent trajectory from the last prompt token, decodes HNN latent velocity back
   to hidden space, and aligns it to the real answer-token hidden-state velocity
   while also optimizing the SFT objective. The AE is frozen, but the alignment
   graph is intentionally kept so gradients update both the LoRA path and HNN.

   Existing legacy FrameworkA checkpoints may have been produced before this
   maintenance fix. Do not infer their exact gradient behavior from the current
   source without a run manifest.

4. Pathway inference

   Entry point: `method/inference/pathway.py`

   This is direct LoRA generation. It loads the Qwen base model and FrameworkA
   LoRA adapter only. It does not load the AE, HNN, or perform an ODE rollout at
   inference time.

Optional PHNN experiment:

- Entry point: `method/training/framework_a_phnn.py`
- This is copied from the maintained FrameworkA loop and replaces the old
  forced/damped HNN-style dynamics module with a controlled PHNN prototype:
  `(J - R) grad H(z,u,t) + G u`.
- The first implemented `u` is the prompt latent mean, so the existing
  `question, answer` CSV contract still works. See `docs/PHNN_TRAINING_DESIGN.md`.

Optional staged dynamics-distillation experiment:

- Entry point: `method/training/dynamics_distilled_lora.py`
- This assumes a latent dynamics teacher checkpoint already exists, freezes that
  teacher plus the AE, and trains only the LoRA adapter with answer-token CE plus
  decoded teacher-velocity alignment.
- The default inference path for the resulting adapter is still direct
  `method/inference/pathway.py` generation.

Optional generalized joint LoRA plus dynamics experiment:

- Entry point: `method/training/joint_lora_dynamics.py`
- This is a FrameworkA-style training loop with a replaceable middle network
  from `method/dynamics/latent_teacher.py`.
- Implemented matrix rows currently expose Neural ODE and gradient-flow energy
  variants. The script also supports Latent ODE, GENERIC, Koopman, and SINDy for
  future matrix rows.

## Downstream task flow

Run downstream metrics from repository root after generating a prediction CSV:

```bash
python -m downstream.tests.smoke_test
```

Primary current tasks:

- Task I/II: `downstream.tasks.task1_2`
  - Uses inference CSV plus a pathway-to-entity reference.
  - Without `--reference`, PCER is closed-corpus debug only.
- Task III: `downstream.tasks.task3_pcte`
  - Compares generated answer and gold answer trajectories after forwarding both
    through the same model plus AE projection.
  - It does not load or score HNN rollout.
- Task IV: `downstream.tasks.task4_csp`
  - Scores generated step continuations against gold continuations.
  - The natural-language parser must be audited on real outputs.
- Task V: `downstream.tasks.task5_cki`
  - Metric calculator only. It needs supplied WT/KO survival and gate labels.
- Task VI: `downstream.tasks.task6_perturbed_cell`
  - Scores C2S prediction JSONL or aligned expression matrices.
  - `downstream.tasks.task6_perturbed_cell.generation` regenerates Qwen-C2S and
    Gemma C2S prediction artifacts with the same output contract.

Tasks VII-IX are runnable evaluators but are not part of the current reportable
scope without the required corpora and labels in `downstream/DATA_REQUIREMENTS.md`.

## Experiment matrix flow

Use `experiments/run_experiment.py` when comparing training/inference variants:

```bash
python -m experiments.run_experiment list
python -m experiments.run_experiment axes
python -m experiments.run_experiment train a02_frameworka_phnn_prompt_regularized_lora --dry-run
python -m experiments.run_experiment plan --phase train --format shell --output /root/autodl-tmp/runs/experiment_plans/train_all.sh
```

Each implemented row under `experiments/methods/` has a concrete `train.py` and
`infer.py`. Broader design candidates are listed in
`experiments/EXPERIMENT_LAYERS.md` and `experiments/matrix.json` under
`candidate_axes`; they are not treated as runnable until concrete scripts exist.
Runtime prerequisites and expected outputs are recorded in
`experiments/runtime_manifest.json`; inspect one row with:

```bash
python -m experiments.run_experiment runtime a05_latent_neural_ode_teacher_rollout
```

Before launching server jobs, check that required models, datasets, adapters, AE
checkpoints, teacher checkpoints, and output parent directories exist:

```bash
python -m experiments.run_experiment check-assets --phase train --ids a01_frameworka_hnn_regularized_lora --strict
python -m experiments.run_experiment check-assets --phase infer --ids a01_frameworka_hnn_regularized_lora --strict
python -m experiments.run_experiment check-assets --phase both --create-output-dirs
```

Use `--asset-root /path/to/autodl-tmp` or `CHATPATHWAY_ASSET_ROOT` only when
checking a local mirror of the AutoDL asset tree. Inference checks also require
the current row's `infer_artifacts`, because default inference wrappers load the
trained adapter or dynamics teacher produced by that row. `train_outputs` can
contain training-only side artifacts such as `hnn_func.pt`, `phnn_func.pt`, or
`dynamics_func.pt`; direct text generation does not load those files unless the
row's inference script explicitly says so.

The wrapper supports:

- `train`, `infer`, and `pipeline` for one row.
- `run-all` for selected batches with `--ids`, `--exclude`, `--start-at`,
  `--stop-after`, and `--contains`.
- Per-row argument passthrough after `--`, for example
  `python -m experiments.run_experiment train a01_frameworka_hnn_regularized_lora -- --epochs 1 --save-dir /tmp/frameworka_test`.
- `plan` for shell, JSONL, or TSV command manifests.
- `check-assets` for manifest-driven runtime dependency checks.
- `consistency` for checking that wrapper dry-run commands match
  `runtime_manifest.json` path declarations.
- `prepare-smoke` for creating tiny CSV/JSONL pathway and C2S inputs on AutoDL
  before short runtime tests.
- `--log-jsonl` execution logs and `--continue-on-error` for long server runs.
- `CHATPATHWAY_LAUNCH_DRY_RUN=1` on a concrete wrapper module to inspect the
  inner command and default paths without importing model dependencies.
- `python -m experiments.run_experiment audit` to expand and validate every
  implemented train/infer wrapper command without loading models.
- `python -m experiments.run_experiment consistency --phase both --quiet` to
  verify wrapper/manifest path consistency.

The implemented latent-dynamics teacher rows are:

- `a05_latent_neural_ode_teacher_rollout`
- `a06_latent_gradient_flow_teacher_rollout`
- `a07_latent_koopman_teacher_rollout`
- `a08_latent_generic_teacher_rollout`
- `a09_latent_sindy_teacher_rollout`
- `a12_latent_ode_encoder_teacher_rollout`

They train from frozen Qwen+LoRA plus frozen AE latent trajectories and infer by
rollout scoring. They do not generate pathway text directly.

The implemented rollout-assisted inference rows are:

- `a10_neural_ode_rollout_rerank`
- `a11_neural_ode_residual_injection`

These use a trained latent dynamics teacher at inference time. They are kept as
separate prototypes and do not replace `method/inference/pathway.py`.

The implemented staged teacher-to-LoRA row is:

- `a13_neural_ode_distilled_lora_direct`

It uses a trained Neural ODE teacher during training only; inference loads the
resulting adapter directly.

The implemented distributed and generalized joint-training rows are:

- `a14_sft_lora_ddp_direct`
- `a15_joint_neural_ode_regularized_lora`
- `a16_joint_gradient_flow_regularized_lora`

`a14` uses `torch.distributed.run` through the wrapper. `a15` and `a16` jointly
update LoRA plus the selected middle network, then perform direct LoRA
generation at inference time.

## C2S / Task VI flow

The C2S preparation scripts remain in `scripts/c2s/prep/` because they are data
preparation workflows, not downstream metrics:

```text
03_make_c2s_dataset.py          raw GSE264667 h5ad -> C2S-formatted h5ad
04_custom_prompt_formatting.py  C2S-formatted h5ad -> Cell2Sentence dataset
08_train_test_spilt_small.py    Cell2Sentence dataset -> small seen/unseen JSONL
```

The Qwen C2S training scripts remain in `scripts/c2s/train/`. The maintained
Task VI generation wrapper is under `downstream/tasks/task6_perturbed_cell/` so
the comparison artifacts and evaluator live with the task definition.

## Source-of-truth rule

For reproducible experiments, use this repository plus `/root/autodl-tmp`
manifests. Do not import behavior from `PathwayDynamicsLLM` or local notebooks
unless that code is explicitly migrated into `ChatPathway2` and listed in
`docs/CODE_PROVENANCE.md`.
