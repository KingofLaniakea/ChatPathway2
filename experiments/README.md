# ChatPathway2 experiment matrix

This directory is the high-level training/inference launcher layer. It does not
move source implementations out of `method/`, `scripts/`, or `downstream/`.
Instead, each experiment row has a concrete `train.py` and `infer.py` wrapper.
The source of truth is `matrix.json`; runtime requirements are in
`runtime_manifest.json`; a spreadsheet-style export is available as
`EXPERIMENT_MATRIX.csv`; and the shared-stage graph is documented in
`TRAINING_MATRIX.md`.

Runtime assets are resolved under `CHATPATHWAY_ASSET_ROOT`. When the environment
variable is unset, wrappers default to `/root/autodl-tmp`.

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
python -m experiments.run_experiment runtime b01_latent_neural_ode_teacher_rollout
```

Check required runtime assets before launching training or inference:

```bash
python -m experiments.run_experiment check-assets --phase train --ids a02_frameworka_hnn_regularized_lora --strict
python -m experiments.run_experiment check-assets --phase infer --ids a02_frameworka_hnn_regularized_lora --strict
```

`runtime_manifest.json` separates `train_requires`, `infer_requires`, and
`infer_artifacts`. This matters for direct-generation rows: inference needs the
trained adapter, but it does not load training-only files such as `hnn_func.pt`
or `dynamics_func.pt`.

For a new server or local mirror, override the asset root used by wrappers and asset checks:

```bash
CHATPATHWAY_ASSET_ROOT=/data/chatpathway python -m experiments.run_experiment check-assets --phase both
```

Dry-run one training command:

```bash
python -m experiments.run_experiment train a03_frameworka_phnn_prompt_regularized_lora --dry-run
```

Run one inference command and pass extra args after `--`:

```bash
python -m experiments.run_experiment infer a02_frameworka_hnn_regularized_lora -- --adapter /path/to/adapter --overwrite
```

Dry-run all implemented inference commands:

```bash
python -m experiments.run_experiment run-all --phase infer --dry-run
```

Dry-run a selected subset and pass the same args to each selected wrapper:

```bash
python -m experiments.run_experiment run-all --phase infer --ids c00_neural_ode_rollout_rerank,c01_neural_ode_residual_injection --dry-run -- --limit 5 --overwrite
```

Render a reproducible shell plan for AutoDL:

```bash
python -m experiments.run_experiment plan --phase train --format shell --output runs/experiment_plans/train_all.sh
```

Render a JSONL plan for auditing or job submission tooling:

```bash
python -m experiments.run_experiment plan --phase infer --format jsonl --contains rollout
```

Run a subset and append execution status records:

```bash
python -m experiments.run_experiment run-all --phase train --start-at b01_latent_neural_ode_teacher_rollout --stop-after b05_latent_sindy_teacher_rollout --log-jsonl runs/experiment_logs/train_teachers.jsonl
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
python -m experiments.methods.a01_sft_lora_ddp_direct.train --dry-run --nproc-per-node 1 --limit 2
```

Inspect any wrapper's inner command without importing heavy runtime dependencies:

```bash
CHATPATHWAY_LAUNCH_DRY_RUN=1 python -m experiments.methods.a00_sft_lora_direct.infer
```

See `IMPLEMENTATION_AUDIT.md` for the current goal-level completion audit and
the optional AutoDL runtime validation gates.

## Implemented rows

The runnable rows are defined in `experiments/matrix.json`:

| ID | Layer | Training wrapper | Inference wrapper |
| --- | --- | --- | --- |
| `a00_sft_lora_direct` | adapter/direct-generation training | `experiments/methods/a00_sft_lora_direct/train.py` | `experiments/methods/a00_sft_lora_direct/infer.py` |
| `a01_sft_lora_ddp_direct` | adapter/direct-generation training | `experiments/methods/a01_sft_lora_ddp_direct/train.py` | `experiments/methods/a01_sft_lora_ddp_direct/infer.py` |
| `a02_frameworka_hnn_regularized_lora` | adapter/direct-generation training | `experiments/methods/a02_frameworka_hnn_regularized_lora/train.py` | `experiments/methods/a02_frameworka_hnn_regularized_lora/infer.py` |
| `a03_frameworka_phnn_prompt_regularized_lora` | adapter/direct-generation training | `experiments/methods/a03_frameworka_phnn_prompt_regularized_lora/train.py` | `experiments/methods/a03_frameworka_phnn_prompt_regularized_lora/infer.py` |
| `b00_lejepa_pathway_sentence` | shared representation and latent-dynamics teachers | `experiments/methods/b00_lejepa_pathway_sentence/train.py` | `experiments/methods/b00_lejepa_pathway_sentence/infer.py` |
| `b01_latent_neural_ode_teacher_rollout` | shared representation and latent-dynamics teachers | `experiments/methods/b01_latent_neural_ode_teacher_rollout/train.py` | `experiments/methods/b01_latent_neural_ode_teacher_rollout/infer.py` |
| `b02_latent_gradient_flow_teacher_rollout` | shared representation and latent-dynamics teachers | `experiments/methods/b02_latent_gradient_flow_teacher_rollout/train.py` | `experiments/methods/b02_latent_gradient_flow_teacher_rollout/infer.py` |
| `b03_latent_koopman_teacher_rollout` | shared representation and latent-dynamics teachers | `experiments/methods/b03_latent_koopman_teacher_rollout/train.py` | `experiments/methods/b03_latent_koopman_teacher_rollout/infer.py` |
| `b04_latent_generic_teacher_rollout` | shared representation and latent-dynamics teachers | `experiments/methods/b04_latent_generic_teacher_rollout/train.py` | `experiments/methods/b04_latent_generic_teacher_rollout/infer.py` |
| `b05_latent_sindy_teacher_rollout` | shared representation and latent-dynamics teachers | `experiments/methods/b05_latent_sindy_teacher_rollout/train.py` | `experiments/methods/b05_latent_sindy_teacher_rollout/infer.py` |
| `b06_latent_ode_encoder_teacher_rollout` | shared representation and latent-dynamics teachers | `experiments/methods/b06_latent_ode_encoder_teacher_rollout/train.py` | `experiments/methods/b06_latent_ode_encoder_teacher_rollout/infer.py` |
| `c00_neural_ode_rollout_rerank` | rollout-assisted inference using trained dynamics | `experiments/methods/c00_neural_ode_rollout_rerank/train.py` | `experiments/methods/c00_neural_ode_rollout_rerank/infer.py` |
| `c01_neural_ode_residual_injection` | rollout-assisted inference using trained dynamics | `experiments/methods/c01_neural_ode_residual_injection/train.py` | `experiments/methods/c01_neural_ode_residual_injection/infer.py` |
| `d00_neural_ode_distilled_lora_direct` | dynamics-to-LoRA distillation or joint regularization | `experiments/methods/d00_neural_ode_distilled_lora_direct/train.py` | `experiments/methods/d00_neural_ode_distilled_lora_direct/infer.py` |
| `d01_joint_neural_ode_regularized_lora` | dynamics-to-LoRA distillation or joint regularization | `experiments/methods/d01_joint_neural_ode_regularized_lora/train.py` | `experiments/methods/d01_joint_neural_ode_regularized_lora/infer.py` |
| `d02_joint_gradient_flow_regularized_lora` | dynamics-to-LoRA distillation or joint regularization | `experiments/methods/d02_joint_gradient_flow_regularized_lora/train.py` | `experiments/methods/d02_joint_gradient_flow_regularized_lora/infer.py` |
| `e00_c2s_transfer_qwen` | downstream transfer/application | `experiments/methods/e00_c2s_transfer_qwen/train.py` | `experiments/methods/e00_c2s_transfer_qwen/infer.py` |
## Design space

Use `EXPERIMENT_LAYERS.md` for the broader layer-by-layer candidate list. Not
every candidate axis is implemented yet; unimplemented axes are intentionally not
included as runnable matrix rows.
