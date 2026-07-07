# Experiment Layers

This file defines the experimental layers used by
`experiments/EXPERIMENT_MATRIX.csv`.

## a: Dynamics Model

| Code | Meaning |
| --- | --- |
| `a0_none` | No latent dynamics. |
| `a1_force_damped_hnn_current_control` | Current temporal forced/damped HNN-style module. |
| `a2_phnn_prompt_control` | Prompt-controlled PHNN. |
| `a3_neural_ode_teacher` | Controlled Neural ODE teacher. |
| `a4_gradient_flow_energy` | Dissipative energy / gradient-flow model. |

## b: AE / Latent Space

| Code | Meaning |
| --- | --- |
| `none` | No AE bridge. |
| `b0_reconstruction_ae_frozen_current` | Current 4096 -> 128 -> 4096 reconstruction/cosine AE, frozen later. |
| `b1_dynamics_aware_ae_pretrain` | Planned AE trained with trajectory/velocity/rollout-aware objective. |
| `b2_joint_ae_and_dynamics` | Planned AE and dynamics jointly trained. |

## d: Training Schedule

| Code | Meaning |
| --- | --- |
| `d0_sft_only` | Only SFT LoRA. |
| `d1_train_dynamics_then_second_lora` | Train dynamics teacher first; then train second LoRA with frozen teacher regularization. |
| `d2_joint_second_lora_and_dynamics` | Train second LoRA and dynamics module together. |

## c: Inference Mode

| Code | Meaning |
| --- | --- |
| `c0_direct_lora` | Direct Qwen + LoRA generation. This is the current main inference mode. |
| `c1_multi_answer_rerank` | Generate multiple answers, then rerank using dynamics/trajectory score. |
| `c2_frameworkb_latent_weighted_average` | FrameworkB: weighted average between HNN rollout and LoRA inference result in latent space. |

## Current Coverage

| Question | Current row |
| --- | --- |
| Plain SFT baseline | `exp000_sft_only_direct` |
| Current full pipeline | `exp001_hnn_reconae_joint_direct` |
| Swap HNN for PHNN under same b/d/c | `exp002_phnn_reconae_joint_direct` |
| Dynamics trained first vs jointly trained | `exp003_neuralode_reconae_teacher_direct` vs `exp004_neuralode_reconae_joint_direct` |
| Alternative dynamics family | `exp005_gradientflow_reconae_joint_direct` |
| Rerank inference prototype | `exp010_neuralode_reconae_teacher_rerank_partial` |

Rows with `status=planned_method_missing` in the CSV are required design
experiments but are not falsely marked runnable.
