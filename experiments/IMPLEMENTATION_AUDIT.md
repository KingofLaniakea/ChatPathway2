# Implementation readiness audit

This file distinguishes code readiness from completed scientific results.

| Requirement | Implemented evidence |
| --- | --- |
| complete v4 graph-event corpus and stable identities | full `processed_graph` canonical index, record/source/five-digit family identities, and immutable release hashes in `dataprocess/index_structured_graphs_v4.py`, `materialize_structured_dataset_v4.py`, and `audit_structured_dataset_v4.py` |
| phenotype isolation | phenotype is excluded from the v4 SFT prompt and target; no graph-level phenotype is copied onto pathway blocks |
| family/source-safe evaluation | family-disjoint train/validation/test, seen-family organism transfer, and strict source-plus-family holdout are materialized from the canonical index |
| complete event-object targets with retained layer boundaries | `method/training/sequence.py`, `framework_a.py`, `staged_objectives.py` |
| no arbitrary q/p split | orthogonal Poisson `J=Q^T J0 Q` |
| correct forced/damped form | `(J-rI) grad H + F(t)`, zero-init time-only force |
| attribution control | `exp003_stage2_sft_only_direct` |
| pure-MSE AE baseline and registered B2/B3 losses | `latent_ae.py`; predictive and latent geometry weights default to zero |
| D2 dynamics-only stability gate | `hamiltonian_pretrain.py`, `exp010`, `exp020` |
| D3 stable-pretrain then joint | `exp011`, `exp021`; low LoRA LR, KL, gradient warm-up and conflict diagnostics |
| D4 direct-joint ablations | `exp001`, `exp002` |
| reproducibility/model selection | seeds, pathway-family-group validation, early stop, triple best checkpoints, hashes/logs |
| three-seed isolation | `seeded_asset_path()` and seed-scoped runtime manifest |
| direct inference diagnostics | preserved identities, token/finish reason, JSON/schema validity |
| sequence-budget coverage | 8192-token fail-closed materialization and trainer checks; actual v4 inclusion and exclusion rates must come from the release `data_audit.json` |
| revised downstream suite | `downstream/new_tasks` Task 0-6 and semantic exporter |

Executable rows are `base000`, `exp000`, `exp003`, `exp010`, `exp011`,
`exp020`, `exp021`, `exp001`, and `exp002`. D2/D3 run one process per method so
the LoRA gradient-angle diagnostic is exact; the CFFF scheduler fills the node
with independent methods/seeds rather than calling this four-process dynamics
training.

Before reporting any model result, the full v4 data audit must pass and the
three-seed runs must complete. Task 3 needs reviewed hard
negatives; Task 4 needs intervention evidence and a calibrated phenotype
scorer; Task 5 needs aligned cell artifacts; Task 6 needs the official external
benchmark and contamination audit.

The trainer saves per-epoch adapters/dynamics and can warm-start an adapter, but
does not yet provide exact optimizer/scheduler/RNG resume. A killed run should
be treated as a restarted replicate unless exact resume is added.
