# Downstream tasks: data requirements and reportability

Synthetic smoke tests only verify code paths. They are never model results.
The table below states the minimum missing asset before each task can support a
scientific claim.

| Task | Code | Required dataset / annotation | Current state |
| --- | --- | --- | --- |
| I PCER | `task1_2.py` | versioned organism-specific KEGG/GO/Enrichr pathway-to-gene library, canonical gene IDs, held-out pathway examples | evaluator works; current 30-row PCER is closed-corpus debug only |
| II entities | `task1_2.py` | entity normalization/synonym map and, ideally, expert entity spans | executable now; heuristic parser is not a biomedical NER gold standard |
| III PCTE | `task3_pcte.py` | prediction/gold pairs, selected AE checkpoint, fixed base/adapter provenance | online one-row server validation passed; needs held-out full-set run |
| IV CSP | `task4_csp.py` | gold continuation boundaries or curated triples for each step | executable now; natural-language fallback parsing needs manual audit |
| V CKI | `task5_cki.py` | pathway graphs, WT/KO/dual-KO interventions, phenotype survival labels, OR/AND/essential/redundant labels, calibrated scorer | no curated CKI dataset or scorer present |
| VI perturbed cell | `task6_perturbed_cell.py` | paired control/perturbed C2S records or aligned expression matrices; split and perturbation labels | existing C2S JSONL outputs can be evaluated; no claim yet about FrameworkA transfer |
| VII shuffle robustness | `task7_step_shuffling.py` | held-out ordered pathways with expert-valid step boundaries, fixed random seed and negative count | candidate generator + scorer implemented; inspect generated boundaries before scoring |
| VIII directional reranking | `task8_directional_reranking.py` | expert-validated candidate groups where negatives differ *only* by direction/mechanism | no valid directional-negative corpus is present; do not auto-reverse prose and report it |
| IX counterfactual perturbation | `task9_counterfactual.py` | paired pre/post-intervention pathway trajectories and an intervention-conditioned generator | current HNN has no intervention input `u`; task is not yet model-runnable |
| X BioSafety analysis | `task10_biosafety.py` | versioned risk taxonomy, evidence corpus/IDs, annotation guide, calibrated severity labels | PDF gives only a title, not a task definition; generic evaluator awaits this specification |

## Required decisions before a benchmark report

1. Choose the organism and frozen database releases for Tasks I/II.
2. Define a non-overlapping pathway-level train/validation/test split. No
   component or near-duplicate leakage across PCER and reranking candidates.
3. Supply structured gold triples for Tasks IV, VII, and VIII; do not infer all
   mechanism labels from the model's own text.
4. Build CKI/Task IX from experimentally grounded perturbation cases and state
   how phenotype survival probabilities are calibrated.
5. Define Task X's risk categories, acceptable evidence, adjudication protocol,
   and severity scale before collecting or scoring examples.
