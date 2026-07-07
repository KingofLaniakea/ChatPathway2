# ChatPathway2 experiment matrix

This directory is the high-level training/inference launcher layer. It does not
move source implementations out of `method/`, `scripts/`, or `downstream/`.
Instead, each experiment row has a concrete `train.py` and `infer.py` wrapper.

## Commands

List implemented rows:

```bash
python -m experiments.run_experiment list
```

Show candidate experimental axes:

```bash
python -m experiments.run_experiment axes
```

Show runtime requirements and expected outputs for one row:

```bash
python -m experiments.run_experiment runtime a05_latent_neural_ode_teacher_rollout
```

Check required runtime assets before launching training or inference:

```bash
python -m experiments.run_experiment check-assets --phase train --ids a01_frameworka_hnn_regularized_lora --strict
python -m experiments.run_experiment check-assets --phase infer --ids a01_frameworka_hnn_regularized_lora --strict
```

`runtime_manifest.json` separates `train_requires`, `infer_requires`, and
`infer_artifacts`. This matters for direct-generation rows: inference needs the
trained adapter, but it does not load training-only files such as `hnn_func.pt`
or `dynamics_func.pt`.

For a local mirror of `/root/autodl-tmp`, override the asset root:

```bash
python -m experiments.run_experiment check-assets --phase both --asset-root /path/to/autodl-tmp
```

Dry-run one training command:

```bash
python -m experiments.run_experiment train a02_frameworka_phnn_prompt_regularized_lora --dry-run
```

Run one inference command and pass extra args after `--`:

```bash
python -m experiments.run_experiment infer a01_frameworka_hnn_regularized_lora -- --adapter /path/to/adapter --overwrite
```

Dry-run all implemented inference commands:

```bash
python -m experiments.run_experiment run-all --phase infer --dry-run
```

Dry-run a selected subset and pass the same args to each selected wrapper:

```bash
python -m experiments.run_experiment run-all --phase infer --ids a10_neural_ode_rollout_rerank,a11_neural_ode_residual_injection --dry-run -- --limit 5 --overwrite
```

Render a reproducible shell plan for AutoDL:

```bash
python -m experiments.run_experiment plan --phase train --format shell --output /root/autodl-tmp/runs/experiment_plans/train_all.sh
```

Render a JSONL plan for auditing or job submission tooling:

```bash
python -m experiments.run_experiment plan --phase infer --format jsonl --contains rollout
```

Run a subset and append execution status records:

```bash
python -m experiments.run_experiment run-all --phase train --start-at a05_latent_neural_ode_teacher_rollout --stop-after a09_latent_sindy_teacher_rollout --log-jsonl /root/autodl-tmp/runs/experiment_logs/train_teachers.jsonl
```

Validate that every implemented matrix row has concrete train/infer modules:

```bash
python -m experiments.validate_matrix
```

This also verifies that `experiments/runtime_manifest.json` covers every
implemented row and records non-empty train/infer outputs.

Audit every train/infer wrapper's inner launch command without importing model
dependencies:

```bash
python -m experiments.run_experiment audit
```

Audit that wrapper dry-run commands match the paths declared in
`runtime_manifest.json`:

```bash
python -m experiments.run_experiment consistency --phase both --quiet
```

Create missing output parent directories on the server before a long run:

```bash
python -m experiments.run_experiment check-assets --phase both --create-output-dirs
```

Create tiny pathway/C2S smoke inputs on AutoDL:

```bash
python -m experiments.run_experiment prepare-smoke --rows 2 --overwrite
```

Inspect the inner torchrun command for the distributed SFT row:

```bash
python -m experiments.methods.a14_sft_lora_ddp_direct.train --dry-run --nproc-per-node 1 --limit 2
```

Inspect any wrapper's inner command without importing heavy runtime dependencies:

```bash
CHATPATHWAY_LAUNCH_DRY_RUN=1 python -m experiments.methods.a00_sft_lora_direct.infer
```

See `IMPLEMENTATION_AUDIT.md` for the current goal-level completion audit and
the optional AutoDL runtime validation gates.

## Implemented rows

The runnable rows are defined in `experiments/matrix.json`:

| ID | Training wrapper | Inference wrapper |
| --- | --- | --- |
| `a00_sft_lora_direct` | `experiments/methods/a00_sft_lora_direct/train.py` | `experiments/methods/a00_sft_lora_direct/infer.py` |
| `a01_frameworka_hnn_regularized_lora` | `experiments/methods/a01_frameworka_hnn_regularized_lora/train.py` | `experiments/methods/a01_frameworka_hnn_regularized_lora/infer.py` |
| `a02_frameworka_phnn_prompt_regularized_lora` | `experiments/methods/a02_frameworka_phnn_prompt_regularized_lora/train.py` | `experiments/methods/a02_frameworka_phnn_prompt_regularized_lora/infer.py` |
| `a03_lejepa_pathway_sentence` | `experiments/methods/a03_lejepa_pathway_sentence/train.py` | `experiments/methods/a03_lejepa_pathway_sentence/infer.py` |
| `a04_c2s_transfer_qwen` | `experiments/methods/a04_c2s_transfer_qwen/train.py` | `experiments/methods/a04_c2s_transfer_qwen/infer.py` |
| `a05_latent_neural_ode_teacher_rollout` | `experiments/methods/a05_latent_neural_ode_teacher_rollout/train.py` | `experiments/methods/a05_latent_neural_ode_teacher_rollout/infer.py` |
| `a06_latent_gradient_flow_teacher_rollout` | `experiments/methods/a06_latent_gradient_flow_teacher_rollout/train.py` | `experiments/methods/a06_latent_gradient_flow_teacher_rollout/infer.py` |
| `a07_latent_koopman_teacher_rollout` | `experiments/methods/a07_latent_koopman_teacher_rollout/train.py` | `experiments/methods/a07_latent_koopman_teacher_rollout/infer.py` |
| `a08_latent_generic_teacher_rollout` | `experiments/methods/a08_latent_generic_teacher_rollout/train.py` | `experiments/methods/a08_latent_generic_teacher_rollout/infer.py` |
| `a09_latent_sindy_teacher_rollout` | `experiments/methods/a09_latent_sindy_teacher_rollout/train.py` | `experiments/methods/a09_latent_sindy_teacher_rollout/infer.py` |
| `a10_neural_ode_rollout_rerank` | `experiments/methods/a10_neural_ode_rollout_rerank/train.py` | `experiments/methods/a10_neural_ode_rollout_rerank/infer.py` |
| `a11_neural_ode_residual_injection` | `experiments/methods/a11_neural_ode_residual_injection/train.py` | `experiments/methods/a11_neural_ode_residual_injection/infer.py` |
| `a12_latent_ode_encoder_teacher_rollout` | `experiments/methods/a12_latent_ode_encoder_teacher_rollout/train.py` | `experiments/methods/a12_latent_ode_encoder_teacher_rollout/infer.py` |
| `a13_neural_ode_distilled_lora_direct` | `experiments/methods/a13_neural_ode_distilled_lora_direct/train.py` | `experiments/methods/a13_neural_ode_distilled_lora_direct/infer.py` |
| `a14_sft_lora_ddp_direct` | `experiments/methods/a14_sft_lora_ddp_direct/train.py` | `experiments/methods/a14_sft_lora_ddp_direct/infer.py` |
| `a15_joint_neural_ode_regularized_lora` | `experiments/methods/a15_joint_neural_ode_regularized_lora/train.py` | `experiments/methods/a15_joint_neural_ode_regularized_lora/infer.py` |
| `a16_joint_gradient_flow_regularized_lora` | `experiments/methods/a16_joint_gradient_flow_regularized_lora/train.py` | `experiments/methods/a16_joint_gradient_flow_regularized_lora/infer.py` |

## Design space

Use `EXPERIMENT_LAYERS.md` for the broader layer-by-layer candidate list. Not
every candidate axis is implemented yet; unimplemented axes are intentionally not
included as runnable matrix rows.
