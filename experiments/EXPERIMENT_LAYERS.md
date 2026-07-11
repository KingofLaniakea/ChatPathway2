# Experiment axes and boundaries

| Axis | Active values | Boundary |
| --- | --- | --- |
| dynamics | none, HNN, forced/damped HNN | full 128-D latent; learned orthogonal Poisson frame |
| representation | shared stage-1 SFT and shared frozen AE | identical digest within seed |
| stage-2 training | SFT-only, SFT+HNN, SFT+forced/damped HNN | compute-matched loop |
| inference | direct greedy | outputs JSON plus finish/schema diagnostics |
| semantic unit | ordered graph layer | same-layer atomic spans do not consume ODE time |

Deferred values:

- semantic-boundary rollout/mixed inference, pending a tested JSON layer
  controller;
- PHNN, pending an independent observed port/control variable;
- Neural ODE, pending completion of the first Hamiltonian benchmark;
- calibrated HNN-assisted reranking, pending held-out candidate calibration.

`experiments/matrix.json` is authoritative for executable rows.
