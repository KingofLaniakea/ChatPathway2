# Experiment axes and boundaries

| Axis | Active values | Boundary |
| --- | --- | --- |
| dynamics | none, HNN, forced/damped HNN | full 128-D latent; learned orthogonal Poisson frame |
| representation | shared stage-1 SFT and shared frozen pure-MSE AE | identical digest within seed; B2/B3 are registered future ablations |
| stage-2 training | SFT-only; dynamics-only; stable-dynamics then joint; direct-joint ablation | explicit D1/D2/D3/D4 separation |
| inference | direct greedy | outputs JSON plus finish/schema diagnostics |
| semantic unit | complete event object plus graph-layer boundary | within-layer canonical traversal uses a short surrogate step; new layers use a longer step |

Post-current-matrix generation studies (retained, not removed):

- event-by-event generation with layer-dependent step size, pending a tested JSON event controller;
- graph-layer-by-graph-layer generation as a separate boundary-controller comparison;
- token-by-token generation, requiring a separately trained token-resolution
  dynamics objective;
- generation-time multiscale mixing, pending independent validation of both
  controllers and a matched-budget direct-greedy ablation;

Other deferred values:

- PHNN, pending an independent observed port/control variable;
- Neural ODE, pending completion of the first Hamiltonian benchmark;
- calibrated HNN-assisted reranking, pending held-out candidate calibration.

`experiments/matrix.json` is authoritative for executable rows.
