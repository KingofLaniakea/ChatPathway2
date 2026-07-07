# ChatPathway2

ChatPathway2 is the canonical source repository for server-backed development.
Large assets remain outside Git under `/root/autodl-tmp`.

```text
method/       pathway model training and inference
experiments/  high-level experiment matrix and launch wrappers
downstream/   hypothesis-testing tasks and metrics
scripts/      historical operational workflows grouped by purpose
baselines/    non-ChatPathway baselines
docs/         provenance and server-layout documentation
```

Start with [docs/WORKFLOW.md](docs/WORKFLOW.md) for the current training,
inference, and downstream order. Use [method/README.md](method/README.md) for
pathway generation and [downstream/README.md](downstream/README.md) for
hypothesis testing. The canonical server asset layout is documented in
[docs/SERVER_LAYOUT.md](docs/SERVER_LAYOUT.md), and code origin is tracked in
[docs/CODE_PROVENANCE.md](docs/CODE_PROVENANCE.md). FrameworkA gradient flow is
documented in [docs/FRAMEWORK_A_BACKPROP.md](docs/FRAMEWORK_A_BACKPROP.md), and
the current inference adapter choice is recorded in
[docs/INFERENCE_BEST_CKPT.md](docs/INFERENCE_BEST_CKPT.md). The experimental
PHNN training variant is described in
[docs/PHNN_TRAINING_DESIGN.md](docs/PHNN_TRAINING_DESIGN.md). Experiment rows
and launch/preflight commands live in
[experiments/README.md](experiments/README.md).
