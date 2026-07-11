# ChatPathway2

ChatPathway2 is the maintained pathway-generation and latent-dynamics
experiment repository. Large assets remain outside Git under the selected
runtime profile.

The current controlled benchmark is:

```text
Qwen3-8B
  -> shared stage-1 SFT
  -> shared 4096-128-4096 reconstruction AE
  -> compute-matched stage-2 SFT-only control
     OR stage-2 SFT + HNN
     OR stage-2 SFT + (J-rI) grad H + F(t)
  -> direct greedy JSON generation
```

HNN time advances once per ordered graph layer. Atomic `A relation B` spans in
the same layer form one pooled layer target and are not assigned an invented
within-layer biological order. Token-level Hamiltonian generation is therefore
not an active experiment.

Start with:

- [docs/WORKFLOW.md](docs/WORKFLOW.md) for the exact CFFF preparation, smoke,
  train, inference, and evaluation order;
- [docs/HAMILTONIAN_EXPERIMENTS.md](docs/HAMILTONIAN_EXPERIMENTS.md) for the
  equations and scientific boundaries;
- [experiments/README.md](experiments/README.md) for the five executable matrix
  rows and three-seed artifact layout;
- [dataprocess/README.md](dataprocess/README.md) for schema, identity, split,
  substep, and phenotype policies;
- [downstream/new_tasks/README.md](downstream/new_tasks/README.md) for the
  revised Task 0-6 contracts.

PHNN remains deferred until an independently observed port/control variable is
defined. Neural ODE and semantic-boundary guided inference are not in the first
Hamiltonian benchmark.

Never commit credentials, environment files, model weights, generated CSVs,
checkpoints, runs, or downstream artifacts.
