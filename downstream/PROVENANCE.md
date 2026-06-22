# Downstream code provenance

This table distinguishes source migration from new code. “New” means authored
in this repository during the ChatPathway2 setup; it may follow a task idea or
metric definition from an older script, but it is not a file copy.

| Task | Current file | Origin | Relationship to legacy material |
| --- | --- | --- | --- |
| I PCER + II entities | `tasks/task1_2.py`, `common/entities.py` | New | Reimplements the PCER/entity-set objective in the older `ChatpathwayDynamic/downstream/task1_2_eval.py`; legacy code was not copied. Adds external-reference vs closed-corpus distinction. |
| III PCTE | `tasks/task3_pcte.py` | New | Uses the old self-consistency evaluator only as an architectural reference for AE checkpoint layouts. PCTE is prediction-vs-gold DTW, whereas the old script evaluates HNN rollout self-consistency. |
| IV CSP | `tasks/task4_csp.py` | New | Inspired by the old template-based step evaluator, but uses a new generic JSON/delimited/natural-language parser because no tracked template file exists. |
| V CKI | `tasks/task5_cki.py` | New | Replaces the old single hard-coded demonstration with an evaluator over supplied WT/KO survival probabilities and labels. No model inference implementation is copied. |
| VI perturbed cell | `tasks/task6_perturbed_cell.py` | New | Reimplements the C2S rank-vector metrics from the legacy C2S evaluation scripts, with an NPZ and C2S-JSONL input contract. |
| VII step shuffling | `tasks/task7_step_shuffling.py`, `common/sequence_scoring.py` | New | New implementation based on the downstream PDF specification. |
| VIII directional reranking | `tasks/task8_directional_reranking.py` | New | New implementation based on the downstream PDF specification. |
| IX counterfactual perturbation | `tasks/task9_counterfactual.py` | New | New evaluator based on the downstream PDF specification. It does not claim that the current HNN supports interventions. |
| X BioSafety-style analysis | `tasks/task10_biosafety.py` | New | New declarative evaluator shell because the PDF provides only a title, not a taxonomy or dataset. |

`smoke_test.py`, `io.py`, and `README.md` are new supporting code. Synthetic
smoke inputs validate software behavior only; they are not benchmark examples
or experimental evidence.
