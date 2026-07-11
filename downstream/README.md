# Downstream evaluation

`downstream/new_tasks` is the maintained Task 0-6 suite for the new
multi-step/layer-set experiments. Historical `downstream/tasks` entry points
remain for old artifacts, but do not define the new paper contract.

| Task | Maintained question | Code readiness | External blocker |
| --- | --- | --- | --- |
| 0 | AE reconstruction and HNN rollout self-consistency | evaluator + semantic exporter | trained checkpoints |
| 1 | next-layer atomic event-set CSP and ordered layer sequence | runnable | parser coverage review for legacy CSV |
| 2 | predicted-vs-gold latent DTW PCTE | evaluator + semantic exporter | prediction CSV |
| 3 | direction/shuffle/unrelated candidate reranking | evaluator ready | expert-validated candidates and validation calibration |
| 4 | knockout effect and rescue ranking | strict evaluator ready | real interventions, phenotype evidence, calibrated scorer |
| 5 | perturbed-cell transfer | evaluator ready | aligned cell/gene/perturbation artifacts and controlled baseline |
| 6 | BioMaze external QA | evaluator ready | official dataset, predictions, contamination audit |

Run synthetic contract tests:

```bash
python -m downstream.new_tasks.audit
python -m unittest discover -s downstream/new_tasks/tests -v
python -m downstream.tests.smoke_test
```

Task 1 defaults to `ordering_mode=layer_set`: layers are ordered, events within
one layer are a multiset. A flat causal substep sequence is eligible only with
independent ordering provenance. Task 3 rejects energy-delta monotonicity as a
causal proxy. Task 4 rejects `F(t)` as intervention conditioning. Missing
phenotypes are counted and excluded, never recoded as negatives.

See [new_tasks/README.md](new_tasks/README.md) for schemas and commands, and
[DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) for claim gates.
