# ChatPathway2

ChatPathway2 is the maintained pathway-generation and latent-dynamics
experiment repository. Large assets remain outside Git under the selected
runtime profile.

The current controlled benchmark is:

```text
Qwen3-8B
  -> shared stage-1 SFT
  -> shared 4096-128-4096 pure-MSE reconstruction AE
  -> compute-matched stage-2 SFT-only control
     OR fixed-latent HNN pretraining -> stability gate -> stage-2 SFT + HNN
     OR fixed-latent FDHNN pretraining -> stability gate -> stage-2 SFT + (J-rI) grad H + F(t)
     OR direct-joint HNN/FDHNN ablations
  -> direct greedy JSON generation with strict three-attempt repair
```

HNN/FDHNN training advances over complete structured event objects. Canonical
events within one graph layer use a short surrogate increment; crossing a layer
boundary uses a longer one. The traversal is deterministic but is not claimed
to be measured biological time. Token-level Hamiltonian generation is therefore
not an active experiment.

Start with:

- [docs/项目SPEC.md](docs/项目SPEC.md) for the project spirit and the complete
  KEGG-to-dataset-to-model-to-downstream pipeline;
- [docs/实验规划.md](docs/实验规划.md) for the concise A/B/C/D experiment design;
- [docs/实验矩阵.xlsx](docs/实验矩阵.xlsx) for the filterable recommended
  experiment combinations;
- [docs/FROZEN_TASK_SPEC_2026-07-13.md](docs/FROZEN_TASK_SPEC_2026-07-13.md)
  for the frozen Task 0-6 definitions;
- [experiments/README.md](experiments/README.md) for the nine executable matrix
  rows and three-seed artifact layout;
- [dataprocess/README.md](dataprocess/README.md) for schema, identity, split,
  substep, and phenotype policies;
- [downstream/new_tasks/README.md](downstream/new_tasks/README.md) for the
  revised Task 0-6 contracts.

The active dataset target is the audited `pathway_continuation_v4` release built
directly from `processed_graph`. PHNN remains deferred until an independently
observed port/control variable is defined.

Never commit credentials, environment files, model weights, generated CSVs,
checkpoints, runs, or downstream artifacts.
