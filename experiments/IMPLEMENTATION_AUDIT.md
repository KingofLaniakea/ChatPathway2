# Implementation readiness audit

This file distinguishes code readiness from completed scientific results.

| Requirement | Implemented evidence |
| --- | --- |
| stable multi-step JSON and identities | sample/record/source/five-digit pathway-family fields in `dataprocess/schemas.py`, `audit_pathway_csv.py` |
| block-safe phenotype handling | block/file conflict and ambiguity statuses in `build_pathway_csv.py` |
| record-balanced pilot and held-out evals | strict organism-plus-family core split and separately labelled organism-transfer split in `prepare_experiment_data.py` |
| graph-layer atomic-span targets | `method/training/sequence.py`, `framework_a.py` |
| no arbitrary q/p split | orthogonal Poisson `J=Q^T J0 Q` |
| correct forced/damped form | `(J-rI) grad H + F(t)`, zero-init time-only force |
| attribution control | `exp003_stage2_sft_only_direct` |
| reproducibility/model selection | seeds, group validation, early stop, best checkpoint, hashes/logs |
| three-seed isolation | `seeded_asset_path()` and seed-scoped runtime manifest |
| direct inference diagnostics | preserved identities, token/finish reason, JSON/schema validity |
| sequence-budget coverage | 8192-token matrix setting plus `dataprocess.audit_token_budget`; 99.17% layer and 91.07% substep retention on the prepared pilot |
| revised downstream suite | `downstream/new_tasks` Task 0-6 and semantic exporter |

Executable rows are `base000`, `exp000`, `exp003`, `exp001`, and `exp002`.
There is no executable `exp011`-`exp014`: the previous prototype advanced a
graph-layer vector field per token and was removed.

Before reporting any model result, the server must still complete a real
Qwen-tokenizer/GPU smoke, then the three-seed runs. Task 3 needs reviewed hard
negatives; Task 4 needs intervention evidence and a calibrated phenotype
scorer; Task 5 needs aligned cell artifacts; Task 6 needs the official external
benchmark and contamination audit.

The trainer saves per-epoch adapters/dynamics and can warm-start an adapter, but
does not yet provide exact optimizer/scheduler/RNG resume. A killed run should
be treated as a restarted replicate unless exact resume is added.
