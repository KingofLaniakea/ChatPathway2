# Experiment layers

This file separates the experimental design space from the currently runnable
matrix in `experiments/matrix.json`.

## Layer 1: backbone and adapter training

Worth testing:

| Item | Question |
| --- | --- |
| SFT LoRA only | How strong is direct pathway instruction tuning without dynamics? |
| staged SFT -> dynamics -> LoRA | Does a dynamics teacher improve the adapter after plain SFT? |
| joint LoRA + dynamics | Does updating LoRA and the dynamics network together help or destabilize training? |
| dynamics-only teacher | Can the middle network fit latent trajectories when LoRA is frozen? |
| distributed training | Does larger batch / longer sequence stabilize hidden-trajectory losses? |

## Layer 2: latent granularity

Worth testing:

| Item | Meaning |
| --- | --- |
| token-level | Hidden velocity between answer tokens. |
| step-level | Pathway step continuation as the time unit. |
| sentence-level | Whole question or answer sentence latent. |
| pathway-level | One latent per pathway trajectory. |
| masked-span JEPA | Predict held-out pathway spans in latent space. |

## Layer 3: middle network families

Worth testing:

| Item | Formula / role |
| --- | --- |
| none | Baseline direct LoRA. |
| AE only | Learn hidden-to-latent bridge without dynamics. |
| Neural ODE | `dz/dt = f(z,t)` unconstrained vector field. |
| Latent ODE | Encoder-conditioned initial state plus ODE rollout. |
| HNN | `dz/dt = J grad H(z)`. |
| time-dependent HNN | HNN with forcing/damping terms. |
| PHNN | `dz/dt = (J-R)grad H(z,u,t)+Gu`. |
| gradient flow | `dz/dt = -R grad E(z,u)`. |
| GENERIC | reversible plus irreversible dynamics. |
| Koopman | latent dynamics with approximately linear evolution. |
| SINDy | sparse symbolic latent vector field. |
| JEPA | latent prediction without text reconstruction. |

## Layer 4: objective design

Worth testing:

| Item | Meaning |
| --- | --- |
| CE | Standard answer-token cross entropy. |
| reconstruction | AE reconstruction of hidden states. |
| velocity alignment | Match predicted latent/hidden velocity to real hidden velocity. |
| rollout loss | Multi-step latent rollout matches real latent trajectory. |
| energy regularization | Constrain Hamiltonian/energy behavior. |
| dissipation/passivity | PHNN-specific damping and energy balance checks. |
| JEPA latent prediction | Predict target latent from context latent. |
| anti-collapse regularization | SIGReg/VICReg-style latent distribution control. |

## Layer 5: inference mode

Worth testing:

| Item | Meaning |
| --- | --- |
| LoRA-only direct | Dynamics influences generation only through adapter training. |
| downstream-only scoring | Dynamics/AE used only after generation, such as PCTE. |
| rollout reranking | Generate candidates, then rerank by dynamics consistency. |
| rollout residual injection | Use dynamics rollout to perturb hidden states during decoding. |
| non-generative latent inference | JEPA-style scoring and representation probing. |

## Layer 6: downstream validation

Worth testing:

| Item | Role |
| --- | --- |
| Task I/II | NLP/entity consistency. |
| Task III | Predicted-vs-gold latent trajectory DTW. |
| Task IV | Step continuation consistency. |
| Task V | KO/WT causal knowledge interface when labels exist. |
| Task VI | C2S transfer/application comparison. |
| latent energy-field task | New task: inspect learned energy/flow landscape. |

## Current implemented coverage

The runnable matrix currently covers:

| Requirement dimension | Implemented rows |
| --- | --- |
| SFT-only direct baseline | `a00`, `a01` |
| HNN/PHNN-style joint regularization | `a02`, `a03` |
| JEPA-style sentence/prompt latent probe | `b00` |
| C2S transfer/application | `e00` |
| Dynamics-only latent teachers | `b01`-`b05`, `b06` |
| Rollout-assisted inference | `c00`, `c01` |
| Staged dynamics teacher -> LoRA distillation | `d00` |
| Distributed training launcher | `a01` |
| Generalized joint LoRA + ODE/energy dynamics | `d01`, `d02` |

Still-design-only axes include step-level/pathway-level granularity and the
latent energy-field downstream task. They are listed as worthwhile directions
but intentionally not marked as implemented rows until concrete train/infer
entry points exist.
