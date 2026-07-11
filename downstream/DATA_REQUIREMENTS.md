# Downstream claim gates

Synthetic tests verify metrics only. They are not model results.

| Task | Minimum reportable asset |
| --- | --- |
| 0 | held-out semantic export, immutable base/SFT/AE/dynamics IDs, checkpoint `dt` |
| 1 | held-out gold/prediction JSON, frozen parser version, parser coverage; independent provenance for causal substep order |
| 2 | paired predicted/gold exports under the exact same base/adapter/AE manifest |
| 3 | expert-validated positives and direction/shuffle/unrelated negatives; HNN combination fitted on validation only |
| 4 | experimental WT/KO/rescue evidence, available phenotype labels, prompt/conditioning contract, validation-calibrated probability scorer |
| 5 | identical cell/gene order, perturbation IDs, normalization/control matching, held-out split, explicit cell adaptation and baseline |
| 6 | official dataset version/source/license, frozen predictions, chronology and contamination audit |

Current phenotype absence is not a blocker for SFT/AE/HNN trajectory training
or Tasks 0-3. It is a hard eligibility blocker for phenotype accuracy and
knockout/rescue claims. A null prediction on `not_annotated` rows is only format
compliance.
