# Maintained method path

| Path | Role |
| --- | --- |
| `training/sft.py` | four-GPU-capable shared LoRA SFT |
| `training/latent_ae.py` | shared prompt-anchor/answer-state reconstruction AE |
| `training/framework_a.py` | compute-matched stage-2 SFT, HNN, and forced/damped HNN |
| `training/sequence.py` | common head-tail token budget and graph-layer span alignment |
| `dynamics/hamiltonian.py` | `J grad H` and `(J-rI) grad H + F(t)` |
| `inference/pathway.py` | direct greedy JSON generation only |
| `analysis/semantic_latent_export.py` | Task 0/2 held-out graph-layer NPZ export |

The HNN is a stage-2 regularizer and diagnostic, not a per-token decoder. The
active inference command loads only base+LoRA. Historical dynamics prototypes
elsewhere in `method/` are not selectable rows in the maintained matrix.
