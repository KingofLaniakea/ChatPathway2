# ChatPathway2 code provenance

This file records where the code in `ChatPathway2` came from and which parts are
maintained source, migrated server code, or newly authored evaluators. It exists
to prevent untracked or exploratory code from being mistaken for the current
project implementation.

## Boundary

`ChatPathway2` is the canonical source tree for the current server-backed work.
`PathwayDynamicsLLM` is not used as a strict implementation source for this
repository. It may be useful for reading ideas, but runnable ChatPathway2
experiments must rely on files in this tree.

## Provenance categories

| Area | Files | Source | Current role |
| --- | --- | --- | --- |
| Method training | `method/training/sft.py` | Migrated from the audited server snapshot file `method/Qwen3_8B_SFT.py`, imported in commit `a419f53` and reorganized in `2044df2`; this revision replaces the historical DDP-only wrapper with a maintained argparse entry point. | Current SFT LoRA training entry point; supports single-process and torchrun/DDP execution. |
| Method training | `method/training/latent_ae.py` | Migrated from the audited server snapshot file `method/Qwen3_8B_Latent_AE_new.py`, imported in `a419f53` and reorganized in `2044df2`. | Current AE projector training entry point; this revision fixes loss-history initialization and tail gradient accumulation. |
| Method training | `method/training/framework_a.py` | Migrated from the audited server snapshot file `method/Qwen3_8B_Method_FrameworkA.py`, imported in `a419f53` and reorganized in `2044df2`. | Current FrameworkA training entry point; this revision restores alignment gradients to HNN and fixes answer-span indexing. |
| Method training prototype | `method/training/framework_a_phnn.py` | Newly copied from maintained `method/training/framework_a.py` and edited in ChatPathway2. It is not a migrated server result. | Experimental PHNN training variant with explicit `J`, positive diagonal `R`, and prompt-latent control input `u`. |
| Method training prototype | `method/training/lejepa_pathway.py` | Newly authored in ChatPathway2, informed by the LeJEPA-style latent prediction paradigm but not copied from an external repository. | Lowest-priority exploratory pathway-language JEPA probe: prompt latent predicts answer latent with anti-collapse regularization. |
| Method inference prototype | `method/inference/lejepa_pathway.py` | Newly authored in ChatPathway2 to pair with `method/training/lejepa_pathway.py`. | Lowest-priority non-generative latent scoring for the pathway LeJEPA probe. |
| Method dynamics prototypes | `method/dynamics/latent_teacher.py`, `method/training/latent_dynamics_teacher.py`, `method/inference/latent_dynamics_rollout.py` | Newly authored in ChatPathway2. | Complete train/inference path for Neural ODE, encoder-conditioned Latent ODE, gradient-flow energy, GENERIC, Koopman, and SINDy latent dynamics teachers. |
| Method training prototype | `method/training/dynamics_distilled_lora.py` | Newly authored in ChatPathway2. | Staged training path that freezes a trained latent dynamics teacher and AE, then updates only LoRA with CE plus decoded-velocity distillation. |
| Method training prototype | `method/training/joint_lora_dynamics.py` | Newly authored in ChatPathway2. | Generalized FrameworkA-style joint LoRA plus dynamics training for Neural ODE, Latent ODE, gradient-flow, GENERIC, Koopman, and SINDy modules. |
| Method dynamics inference prototypes | `method/inference/rollout_rerank.py`, `method/inference/rollout_residual_injection.py` | Newly authored in ChatPathway2. | Rollout-assisted candidate reranking and rollout-to-hidden residual injection prototypes. |
| Experiment wrappers | `experiments/**` | Newly authored in ChatPathway2. | High-level matrix, train/infer wrappers, runtime asset checks, and launch CLI for implemented experiment rows. |
| Legacy method training | `method/training/legacy/hnn_stage1_ddp.py` | Migrated from `method/Qwen3_8B_Method_4_2_stage1.py`, imported in `a419f53`, exact rename in `2044df2`. | Historical HNN-only training reference. |
| Legacy method training | `method/training/legacy/joint_sft_hnn_ddp.py` | Migrated from `method/Qwen3_8B_Method_4_1_2_1.py`, imported in `a419f53`, exact rename in `2044df2`. | Historical joint SFT/HNN reference. |
| Method inference | `method/inference/pathway.py` | Migrated from server `method/inference.py`, then made configurable in commit `ff2a71a` and reorganized in `2044df2`. | Current pathway generation entry point. |
| Method inference | `method/inference/pathway_batch.py` | Migrated from server `method/inference_batch.py`, imported in `a419f53` and reorganized in `2044df2`. | Historical batch inference variant. |
| C2S prep/train/eval scripts | `scripts/c2s/**` | Migrated from server scripts in `a419f53` and grouped under `scripts/` in `2044df2`; `scripts/c2s/train/train_c2s_single.py` is maintained in this revision. | Historical and operational C2S workflows; the maintained single-GPU Qwen C2S trainer is used by optional matrix row `x00`. |
| Data/model/analysis scripts | `scripts/data/**`, `scripts/model/**`, `scripts/analysis/**`, `scripts/inference/**` | Migrated from the audited server snapshot in `a419f53` and grouped in `2044df2`. | Supporting workflows and exploratory analysis. |
| SCGEN baseline | `baselines/scgen/main.py` | Migrated from server `SCGEN/main.py` in `a419f53` and reorganized in `2044df2`. | External baseline code, separate from ChatPathway method code. |
| Downstream common utilities | `downstream/common/**` | Newly authored during the ChatPathway2 downstream setup, then reorganized in `2044df2`. | Shared parser, IO, and sequence-scoring utilities. |
| Downstream Tasks I-V | `downstream/tasks/task1_2/`, `task3_pcte/`, `task4_csp/`, `task5_cki/` | Newly authored evaluators based on downstream task definitions and older script behavior. They are not byte-for-byte copies of legacy scripts. | Current metric implementations for Tasks I-V. |
| Downstream Task VI evaluator | `downstream/tasks/task6_perturbed_cell/main.py` | Newly authored evaluator that reimplements the C2S rank-vector metrics from migrated legacy C2S scripts. | Current Task VI scoring entry point. |
| Downstream Task VI generation | `downstream/tasks/task6_perturbed_cell/generation.py` | Newly authored wrapper around the migrated Qwen-C2S and Gemma server generation paths. | Maintained prediction JSONL generation entry point for Task VI. |
| Downstream Tasks VII-IX | `downstream/tasks/task7_step_shuffling/`, `task8_directional_reranking/`, `task9_counterfactual/` | Newly authored evaluators based on downstream task specifications. | Runnable but not reportable without required corpora and labels. |
| Documentation | `README.md`, `method/README.md`, `downstream/README.md`, `docs/*.md`, `scripts/README.md` | Authored during ChatPathway2 setup and maintenance. | Workflow, storage, and provenance documentation. |
| PHNN paper | `docs/Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent Dynamical Systems.pdf` | User-provided reference document. | Literature reference only; it is not implemented code. |

## Current maintenance and experiment additions

This revision makes targeted maintenance changes and adds experimental method
rows:

- `method/training/sft.py`
  - Replaces the old hard-coded GPFS/DDP-only script with a maintained
    argparse entry point using `/root/autodl-tmp` defaults.
  - Runs as normal single-process training or under `torchrun` when
    `WORLD_SIZE > 1`.
  - Saves LoRA checkpoints, `history.json`, and `run_config.json`.
- `method/training/framework_a.py`
  - Keeps AE parameters frozen.
  - Keeps AE projection and HNN rollout in float32 for stable gradient flow.
  - Adds argparse overrides for model/data/checkpoint paths and core
    hyperparameters while preserving the original training algorithm.
  - Writes `run_config.json` and `history.json` under the save directory.
  - Fixes ODE time-column construction for tensor-valued `torchdiffeq` times.
  - Removes the `detach()` / `torch.no_grad()` break between HNN rollout and
    `loss_align`, so the intended FrameworkA alignment loss can train HNN.
  - Uses the first answer token to find the prompt boundary, avoiding the old
    padding-label bug where short examples could start HNN rollout from a padded
    token.
  - Performs a final optimizer step when the last epoch shard is smaller than
    `gradient_accumulation_steps`.
- `method/training/latent_ae.py`
  - Restores `rec_cos` tracking and initializes `e_mse` / `e_cos` counters.
  - Averages AE reconstruction metrics over valid batches.
  - Performs a final optimizer step when the last epoch shard is smaller than
    `gradient_accumulation_steps`.
- `docs/FRAMEWORK_A_BACKPROP.md`
  - Documents the current FrameworkA training graph, loss routing, and inference
    boundary from maintained ChatPathway2 code.
- `docs/INFERENCE_BEST_CKPT.md`
  - Records the current `method/inference/pathway.py` default base model,
    adapter, input, and output paths.
- `method/training/framework_a_phnn.py`
  - Copies the maintained FrameworkA training loop and replaces the forced/damped
    HNN-style vector field with a controlled PHNN prototype.
  - Uses prompt latent mean as the first external control input `u`; no new CSV
    columns are required for this prototype.
  - Adds argparse overrides, `run_config.json`, `history.json`, and the same
    tensor-safe ODE time-column construction as FrameworkA.
- `docs/PHNN_TRAINING_DESIGN.md`
  - Documents the PHNN formula, `u` choices, current data contract, and training
    chain.
- `method/training/lejepa_pathway.py` and `method/inference/lejepa_pathway.py`
  - Add a LeJEPA-style pathway-language exploration path at the method layer.
  - This is a latent prediction/scoring probe, not a text generation model.
  - This is assigned to the lowest-priority speculative `z` layer, outside the
    first core dynamics benchmark.
- `method/dynamics/latent_teacher.py`, `method/training/latent_dynamics_teacher.py`,
  and `method/inference/latent_dynamics_rollout.py`
  - Add controlled Neural ODE, encoder-conditioned Latent ODE, gradient-flow
    energy, GENERIC, Koopman, and SINDy teacher implementations.
  - Training extracts answer latent trajectories from frozen Qwen+LoRA plus
    frozen AE; inference scores learned rollout behavior.
- `method/training/dynamics_distilled_lora.py`
  - Adds a staged alternative to FrameworkA: the dynamics teacher is trained
    first, then frozen while LoRA is updated with SFT CE plus teacher
    decoded-velocity alignment.
  - Direct inference for the resulting adapter still uses
    `method/inference/pathway.py`; no dynamics module is loaded at generation
    time.
- `method/training/joint_lora_dynamics.py`
  - Adds a generalized joint-training path where LoRA and a selected latent
    dynamics module are updated together against CE plus decoded-velocity
    alignment.
  - The implemented matrix rows currently expose the Neural ODE and
    gradient-flow variants; the script supports additional variants for future
    rows.
- `scripts/c2s/train/train_c2s_single.py`
  - Replaces the hard-coded single-GPU C2S transfer script with an argparse
    entry point preserving the same JSONL contract and Qwen chat formatting.
  - Supports pathway-adapter initialization, fresh LoRA initialization,
    `--limit`, run metadata, loss history, and tail gradient accumulation.
- `experiments/_launch.py`
  - Adds a `torchrun` launcher helper used by the distributed SFT matrix row.
  - Resolves wrapper default model/data/checkpoint paths through
    `chatpathway.config.json`.
  - Adds `CHATPATHWAY_LAUNCH_DRY_RUN=1` support so wrapper defaults can be
    inspected without importing model/runtime dependencies.
- `experiments/run_experiment.py`
  - Adds selection filters, shell/JSONL/TSV command plans, shared argument
    passthrough for selected rows, dry-run execution logs, and
    `--continue-on-error` for long AutoDL batches.
  - Adds an `audit` subcommand that expands every wrapper's inner dry-run
    command without loading model dependencies.
- `experiments/audit_wrappers.py`
  - Audits all implemented train/infer wrapper modules under
    `CHATPATHWAY_LAUNCH_DRY_RUN=1`.
- `experiments/audit_matrix_consistency.py`
  - Expands wrapper dry-run commands and checks that `train_requires`,
    `infer_requires`, `infer_artifacts`, and expected outputs in
    `runtime_manifest.json` are represented in the actual wrapper commands.
- `experiments/check_runtime_assets.py`
  - Adds a torch-free runtime preflight checker for `runtime_manifest.json`.
  - Checks base models, datasets, LoRA/AE/dynamics checkpoints, row dependency
    outputs, trained artifacts needed for inference, and output parent
    directories.
  - Reads `chatpathway.config.json` by default and supports `--profile` or a
    temporary `--asset-root` override for local mirrors of a server asset tree.
- `experiments/prepare_smoke_inputs.py`
  - Adds a dependency-light AutoDL helper that creates tiny pathway CSV and C2S
    JSONL inputs for short training/inference smoke tests.
- `experiments/runtime_manifest.json`
  - Records per-row runtime prerequisites, phase-specific train/infer
    prerequisites, expected training outputs, inference-loaded artifacts,
    expected inference outputs, and dependency notes for AutoDL execution.
  - Records default expanded AutoDL paths; non-AutoDL servers rewrite those paths
    through the configured runtime profile during asset checks and wrapper
    launch.
- `experiments/validate_matrix.py`
  - Validates both the implemented matrix rows and runtime manifest coverage.
- `method/inference/rollout_rerank.py`
  - Adds inference-time candidate reranking using a trained latent dynamics
    teacher. It does not change model weights.
- `method/inference/rollout_residual_injection.py`
  - Adds an isolated prototype that decodes a latent rollout delta back to hidden
    space and injects it into the final prompt embedding before generation.
- `experiments/**`
  - Adds an organized experiment matrix and high-level wrapper for implemented
    training/inference rows.
  - Keeps broader candidate axes separate from runnable rows so the wrapper does
    not point to missing implementations.
- Historical C2S/SCGEN helper scripts
  - Escapes literal `\ctrl100` and `\Delta` strings so repository-wide Python
    syntax checks are clean. This is a string-literal maintenance change, not a
    metric or algorithm change.

These changes affect future training runs only. They do not retroactively prove
how already saved legacy checkpoints were trained.
