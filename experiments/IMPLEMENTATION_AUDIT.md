# Experiment Implementation Audit

This audit records the local code-completion boundary for the experiment
wrappers. GPU training on real assets remains a server validation step.

## Current Evidence

| Requirement | Evidence | Status |
| --- | --- | --- |
| Combination matrix uses abcd layer columns | `experiments/EXPERIMENT_MATRIX.csv`, `experiments/matrix.json` | Implemented |
| Current training+inference pipeline is represented as one row | `exp001_hnn_reconae_joint_direct` | Implemented |
| AE is an explicit layer | `b_ae` column and `b0/b1/b2` choices | Implemented in matrix; only `b0` runnable |
| DDP is not a main comparison | no DDP row in `implemented` | Implemented |
| FrameworkA naming removed from experiment IDs | main row is `exp001_hnn_reconae_joint_direct` | Implemented |
| Inference layer has direct/rerank/FrameworkB options | `c0/c1/c2` in matrix | Implemented in design; `c2` planned |
| Full train wrappers run from scratch | implemented `train.py` wrappers call SFT and later stages in order | Implemented structurally |
| Runtime paths separated from design matrix | `runtime_manifest.json` | Implemented |

## Runnable Rows

| Row | Train chain | Infer chain |
| --- | --- | --- |
| `exp000_sft_only_direct` | `method.training.sft` | `method.inference.pathway` |
| `exp001_hnn_reconae_joint_direct` | `method.training.sft -> method.training.latent_ae -> method.training.framework_a` | `method.inference.pathway` |
| `exp002_phnn_reconae_joint_direct` | `method.training.sft -> method.training.latent_ae -> method.training.framework_a_phnn` | `method.inference.pathway` |
| `exp003_neuralode_reconae_teacher_direct` | `method.training.sft -> method.training.latent_ae -> method.training.latent_dynamics_teacher -> method.training.dynamics_distilled_lora` | `method.inference.pathway` |
| `exp004_neuralode_reconae_joint_direct` | `method.training.sft -> method.training.latent_ae -> method.training.joint_lora_dynamics --variant neural_ode` | `method.inference.pathway` |
| `exp005_gradientflow_reconae_joint_direct` | `method.training.sft -> method.training.latent_ae -> method.training.joint_lora_dynamics --variant gradient_flow` | `method.inference.pathway` |
| `exp010_neuralode_reconae_teacher_rerank_partial` | `method.training.sft -> method.training.latent_ae -> method.training.latent_dynamics_teacher` | `method.inference.rollout_rerank` |
| `x001_c2s_transfer_after_model_selection` | `scripts.c2s.train.train_c2s_single` | `downstream.tasks.task6_perturbed_cell.generation` |
| `z001_lejepa_speculative_probe` | `method.training.lejepa_pathway` | `method.inference.lejepa_pathway` |

`exp010` is intentionally marked partial: the reranker exists, but the
multi-answer candidate generation step required by `c1_multi_answer_rerank` is
not implemented yet.

## Planned Rows

| Row | Missing implementation |
| --- | --- |
| `plan001_hnn_reconae_teacher_direct` | standalone HNN teacher training plus distillation |
| `plan002_phnn_reconae_teacher_direct` | standalone PHNN teacher training plus distillation |
| `plan003_hnn_dynamicsawareae_joint_direct` | dynamics-aware AE objective |
| `plan004_hnn_jointae_joint_direct` | joint AE+dynamics training |
| `plan005_hnn_reconae_joint_rerank` | multi-answer generation and HNN-compatible rerank |
| `plan006_hnn_reconae_joint_frameworkb` | FrameworkB latent weighted-average inference |

## Local Verification

Run from repository root:

```bash
python -m experiments.validate_matrix
python -m experiments.run_experiment audit --phase both --quiet
python -m experiments.run_experiment consistency --phase both --quiet
python -m experiments.run_experiment run-all --phase train --dry-run
python -m experiments.run_experiment run-all --phase infer --dry-run
python -m compileall -q method experiments
git diff --check
```

These checks validate wrapper structure, path consistency, argparse
compatibility, and Python syntax. They do not prove GPU memory, model asset
availability, or real benchmark quality.
