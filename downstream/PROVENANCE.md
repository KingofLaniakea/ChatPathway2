# Downstream code provenance

This table distinguishes source migration from new code. “New” means authored
in this repository during the ChatPathway2 setup; it may follow a task idea or
metric definition from an older script, but it is not a file copy.

| Task | Current directory | Origin | Relationship to legacy material |
| --- | --- | --- | --- |
| I PCER + II entities | `tasks/task1_2/`, `common/entities.py` | New | Reimplements the PCER/entity-set objective in the older `ChatpathwayDynamic/downstream/task1_2_eval.py`; legacy code was not copied. Adds external-reference vs closed-corpus distinction and does not reproduce the legacy figure/report bundle exactly. |
| III PCTE | `tasks/task3_pcte/` | New | Uses the old self-consistency evaluator only as an architectural reference for AE checkpoint layouts. PCTE is prediction-vs-gold DTW, whereas the old script evaluates HNN rollout self-consistency. |
| IV CSP | `tasks/task4_csp/` | New | Inspired by the old template-based step evaluator, but uses a new generic JSON/delimited/natural-language parser because no tracked template file exists. |
| V CKI | `tasks/task5_cki/` | New | Replaces the old single hard-coded demonstration with an evaluator over supplied WT/KO survival probabilities and labels. No model inference implementation is copied. |
| VI perturbed cell | `tasks/task6_perturbed_cell/` | Mixed | Reimplements the C2S rank-vector metrics from the legacy C2S evaluation scripts, with an NPZ and C2S-JSONL input contract. The generation entry point preserves the server Qwen-C2S and Gemma comparison paths as configurable defaults. |
| VII step shuffling | `tasks/task7_step_shuffling/`, `common/sequence_scoring.py` | New | New implementation based on the downstream PDF specification. |
| VIII directional reranking | `tasks/task8_directional_reranking/` | New | New implementation based on the downstream PDF specification. |
| IX counterfactual perturbation | `tasks/task9_counterfactual/` | New | New evaluator based on the downstream PDF specification. It does not claim that the current HNN supports interventions. |

`smoke_test.py`, `io.py`, and `README.md` are new supporting code. Synthetic
smoke inputs validate software behavior only; they are not benchmark examples
or experimental evidence.
