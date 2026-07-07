# Experiment implementation audit

This file records the current evidence for the experiment-matrix goal. The
current local acceptance boundary is code correctness on Mac: structure,
argument compatibility, wrapper/manifest consistency, syntax, and dry-run
launch behavior. Full GPU training is a later runtime validation step, not a
local completion requirement.

## Requirements

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Enumerate worthwhile layer-level candidates | `experiments/EXPERIMENT_LAYERS.md`; `experiments/matrix.json` `candidate_axes` | Implemented as design inventory |
| Build an integrated training/inference matrix | `experiments/matrix.json` with 17 implemented rows | Implemented |
| Every implemented row has a training wrapper | `experiments/methods/<row>/train.py`; checked by `python -m experiments.validate_matrix` | Implemented and structurally verified |
| Every implemented row has an inference wrapper | `experiments/methods/<row>/infer.py`; checked by `python -m experiments.validate_matrix` | Implemented and structurally verified |
| Wrapper commands are launchable without importing heavy model dependencies | `python -m experiments.run_experiment audit` | Verified locally |
| Wrapper commands match manifest paths | `python -m experiments.run_experiment consistency --phase both --quiet` | Verified locally |
| Wrapper-passed target CLI options and literal choices are declared by target scripts | `experiments/audit_matrix_consistency.py` parses target `argparse.add_argument` calls | Verified locally |
| Runtime assets are checkable before server runs | `experiments/check_runtime_assets.py`; `python -m experiments.run_experiment check-assets ...` | Implemented; strict success is a runtime-only server check |
| Smoke inputs are reproducible | `experiments/prepare_smoke_inputs.py`; `python -m experiments.run_experiment prepare-smoke ...` | Implemented; real files are created where the source datasets exist |
| High-level wrapper can launch/list/plan/run batches | `experiments/run_experiment.py` supports `list`, `axes`, `show`, `runtime`, `train`, `infer`, `pipeline`, `run-all`, `plan`, `audit`, `consistency`, `check-assets`, and `prepare-smoke` | Implemented |
| LeJEPA-style pathway-language probe exists | `method/training/lejepa_pathway.py`, `method/inference/lejepa_pathway.py`, row `a03_lejepa_pathway_sentence` | Implemented structurally |

## Implemented Rows

| Row | Train module | Inference module | Inference role |
| --- | --- | --- | --- |
| `a00_sft_lora_direct` | `method.training.sft` | `method.inference.pathway` | Direct LoRA generation |
| `a01_frameworka_hnn_regularized_lora` | `method.training.framework_a` | `method.inference.pathway` | Direct LoRA generation; HNN affects adapter during training |
| `a02_frameworka_phnn_prompt_regularized_lora` | `method.training.framework_a_phnn` | `method.inference.pathway` | Direct LoRA generation; PHNN prototype affects adapter during training |
| `a03_lejepa_pathway_sentence` | `method.training.lejepa_pathway` | `method.inference.lejepa_pathway` | Non-generative latent scoring |
| `a04_c2s_transfer_qwen` | `scripts.c2s.train.train_c2s_single` | `downstream.tasks.task6_perturbed_cell.generation` | C2S generation for Task VI |
| `a05_latent_neural_ode_teacher_rollout` | `method.training.latent_dynamics_teacher --variant neural_ode` | `method.inference.latent_dynamics_rollout` | Rollout scoring |
| `a06_latent_gradient_flow_teacher_rollout` | `method.training.latent_dynamics_teacher --variant gradient_flow` | `method.inference.latent_dynamics_rollout` | Rollout scoring |
| `a07_latent_koopman_teacher_rollout` | `method.training.latent_dynamics_teacher --variant koopman` | `method.inference.latent_dynamics_rollout` | Rollout scoring |
| `a08_latent_generic_teacher_rollout` | `method.training.latent_dynamics_teacher --variant generic` | `method.inference.latent_dynamics_rollout` | Rollout scoring |
| `a09_latent_sindy_teacher_rollout` | `method.training.latent_dynamics_teacher --variant sindy` | `method.inference.latent_dynamics_rollout` | Rollout scoring |
| `a10_neural_ode_rollout_rerank` | `method.training.latent_dynamics_teacher --variant neural_ode` | `method.inference.rollout_rerank` | Rollout-assisted reranking |
| `a11_neural_ode_residual_injection` | `method.training.latent_dynamics_teacher --variant neural_ode` | `method.inference.rollout_residual_injection` | Rollout-assisted generation prototype |
| `a12_latent_ode_encoder_teacher_rollout` | `method.training.latent_dynamics_teacher --variant latent_ode` | `method.inference.latent_dynamics_rollout` | Rollout scoring |
| `a13_neural_ode_distilled_lora_direct` | `method.training.dynamics_distilled_lora` | `method.inference.pathway` | Direct LoRA generation after staged teacher distillation |
| `a14_sft_lora_ddp_direct` | `torch.distributed.run -m method.training.sft` | `method.inference.pathway` | Direct LoRA generation after DDP SFT |
| `a15_joint_neural_ode_regularized_lora` | `method.training.joint_lora_dynamics --variant neural_ode` | `method.inference.pathway` | Direct LoRA generation after joint training |
| `a16_joint_gradient_flow_regularized_lora` | `method.training.joint_lora_dynamics --variant gradient_flow` | `method.inference.pathway` | Direct LoRA generation after joint training |

## Local Verification

Run from repository root:

```bash
python -m experiments.validate_matrix
python -m experiments.run_experiment audit
python -m experiments.run_experiment consistency --phase both --quiet
python -m experiments.run_experiment prepare-smoke --rows 2 --skip-missing
python -m experiments.run_experiment run-all --phase train --dry-run
python -m experiments.run_experiment run-all --phase infer --dry-run
python -m compileall -q method experiments downstream scripts baselines
python -m downstream.tests.smoke_test
git diff --check
```

These checks prove the local code-completion boundary: structure, command
expansion, wrapper/manifest consistency, target argparse compatibility, syntax,
and downstream metric smoke behavior. They intentionally do not prove that Qwen,
PEFT, torchdiffeq, GPU memory, or the real datasets are valid on a server.

## Optional AutoDL Runtime Gates

These are useful before spending GPU time, but they are not required for the Mac
local code-correctness acceptance boundary. Run on
`/root/autodl-tmp/ChatPathway2`:

```bash
python -m experiments.run_experiment check-assets --phase both --strict
python -m experiments.run_experiment check-assets --phase both --create-output-dirs
python -m experiments.run_experiment consistency --phase both --quiet
python -m experiments.run_experiment prepare-smoke --rows 2 --overwrite
```

Then run a small smoke subset before full experiments:

```bash
python -m experiments.run_experiment train a00_sft_lora_direct -- --train /root/autodl-tmp/data/train_11_species_dataset_smoke.csv --epochs 1 --save-dir /root/autodl-tmp/checkpoints/smoke/qwen3_8b_sft
python -m experiments.run_experiment infer a00_sft_lora_direct -- --adapter /root/autodl-tmp/checkpoints/smoke/qwen3_8b_sft/checkpoint_epoch_1 --input /root/autodl-tmp/data/test_7_species_dataset_smoke.csv --output /root/autodl-tmp/runs/smoke/sft_epoch1.csv --max-new-tokens 32 --overwrite
python -m experiments.run_experiment train a05_latent_neural_ode_teacher_rollout -- --train /root/autodl-tmp/data/train_11_species_dataset_smoke.csv --epochs 1 --save-dir /root/autodl-tmp/checkpoints/smoke/latent_dynamics_teachers
python -m experiments.run_experiment infer a05_latent_neural_ode_teacher_rollout -- --checkpoint /root/autodl-tmp/checkpoints/smoke/latent_dynamics_teachers/neural_ode/neural_ode_epoch_1.pt --input /root/autodl-tmp/data/test_7_species_dataset_smoke.csv --output /root/autodl-tmp/runs/smoke/neural_ode_scores.jsonl --limit 2 --overwrite
```

Successful AutoDL runs would validate runtime assets, dependency versions, GPU
memory, and data availability. They are separate from the local implementation
audit above.
