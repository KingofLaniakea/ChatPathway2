# Controlled experiment matrix

`matrix.json` contains nine executable rows. They are intentionally narrow so
the first result can be attributed.

| Row | Role |
| --- | --- |
| `base000_shared_sft_reconae` | train one shared stage-1 SFT and one shared AE for a seed |
| `exp000_sft_only_direct` | stage-1 direct-generation baseline |
| `exp003_stage2_sft_only_direct` | compute-matched second-SFT control with every dynamics weight and LR zero |
| `exp010_hnn_reconae_dynamics_only` | frozen-SFT/AE HNN pretraining with a validation stability gate |
| `exp011_hnn_reconae_pretrain_joint_direct` | stable HNN then low-LR regularized stage-2 SFT |
| `exp020_forced_damped_hnn_reconae_dynamics_only` | frozen-SFT/AE FDHNN pretraining with a validation stability gate |
| `exp021_forced_damped_hnn_reconae_pretrain_joint_direct` | stable FDHNN then low-LR regularized stage-2 SFT; primary method |
| `exp001_hnn_reconae_joint_direct` | random HNN direct-joint D4 ablation |
| `exp002_forced_damped_hnn_reconae_joint_direct` | random FDHNN direct-joint D4 ablation |

The active release contract is `data/pathway_v4_full/`. The three primary
partitions use disjoint complete five-digit KEGG families on seen source codes;
`test_organism` holds out source codes while reusing train families, and
`test_strict` holds out both source codes and every non-train family. Source holdout is
stratified only by coverage statistics in the canonical data snapshot and does
not claim phylogenetic balance. The scheduler refuses to start unless the
generated read-only `data_audit.json` has passed and all five CSV/record pairs,
source graphs, and declared hashes still match.

All mutable artifacts are seed-scoped:

```text
checkpoints/seeds/<seed>/shared/...
checkpoints/seeds/<seed>/experiments/<row>/...
runs/seeds/<seed>/experiments/<row>/...
```

Within one seed, every D1--D4 row reuses exactly that seed's
shared SFT and AE. Recommended seeds are `20260711`, `20260712`, and
`20260713`.

The B1 AE baseline is pure reconstruction MSE. Optional B2 next-layer
prediction and B3 latent mean/variance/off-diagonal-covariance losses are
implemented but are not silently enabled in these nine rows. Dynamics targets
pool the contextual tokens of each complete event object (participants, action,
mediators, target, and text). Within-layer canonical events use `dt=1/512`; a
new graph layer uses `dt=1/128`. Neither value has a biological time unit.

The controlled matrix pins `max_length=8192` and per-process training
`batch_size=1` for SFT, AE, and every stage-2 arm. Dataset materialization uses
the real tokenizer and excludes any row whose complete prompt plus closed JSON
answer exceeds that budget. Trainers fail if an oversized row slips through;
they never truncate an assistant JSON target.

The formal default is one full-data stage-1 SFT epoch, followed by at most
three AE epochs, one to three dynamics-only epochs, and at most three stage-2
epochs. This prevents the underlying trainers' historical 12-epoch defaults
from silently turning a 515-million-token release into an unintended run.

The formal v4 release contains exactly one seed-fixed prefix per selected
biological record. A global constrained matcher balances the actually eligible
short/middle/long horizons as tightly as mathematically possible after the
8192-token filter. The current trainers therefore see the same registered view
on every epoch; `one_per_record` remains enabled as an identity guard but does
not rotate a one-row group. Alternative epoch-wise horizon schedules are a
separate rematerialization experiment from the complete canonical index, not a
silent change during the controlled matrix.

Direct inference pins per-process `batch_size=1`. The first greedy attempt uses
up to 4096 new tokens. Invalid or unclosed JSON is regenerated with an explicit
repair turn and a larger budget; the third and final attempt may use 8192. If
that output still fails strict schema validation, inference records the failed
history and exits with an error. Four-GPU speedup comes only from disjoint data shards.
Every completed sample is immediately appended to a progress JSONL with its
identity, gold answer, prediction, finish reason, and JSON/schema validity. A
direct one-GPU wrapper writes `direct.progress.jsonl`. The CFFF scheduler writes
four `direct.progress.shard-*-of-*.jsonl` files live, then verifies and merges
them into `direct.progress.jsonl` and the final CSV in original dataset order.

Checkpoint selection uses one dataset-revision-fixed `pathway_family_id`
validation split, not a row or source-only split. The dataset build seed fixes
that family set once; training seeds `20260711/12/13` change model RNG and
artifact paths, not the data split. SFT, AE, and all stage-2 arms reuse the
same validation CSV, whose actual family count and zero-overlap result come
from `data_audit.json`.

## CFFF preparation

```bash
export CHATPATHWAY_PROFILE=cfff
python -m experiments.run_experiment prepare-structured-data-v4 \
  --minimum-train-records 12000 \
  --seed 20260715 \
  --overwrite-release
python -m experiments.run_experiment download-model
python -m experiments.run_experiment check-assets \
  --phase train --ids base000_shared_sft_reconae --profile cfff \
  --create-output-dirs --strict
```

The indexer reconstructs complete rich-action event sets from every
`processed_graph` JSON without sampling or a family cap. Materialization uses a
conservative 515-million-token one-epoch envelope, writes one record JSONL and
one compatibility CSV row per selected record, and creates
`dataset_manifest.json` plus read-only `data_audit.json`. The minimum of 12,000
train records prevents silently releasing another short run. The first measured
v4 packed-training throughput replaces the current runtime estimate.

## One-seed run

```bash
SEED=20260711

python -m experiments.run_experiment train base000_shared_sft_reconae -- --seed "$SEED"
python -m experiments.run_experiment train exp003_stage2_sft_only_direct -- --seed "$SEED"
python -m experiments.run_experiment train exp010_hnn_reconae_dynamics_only -- --seed "$SEED"
python -m experiments.run_experiment train exp011_hnn_reconae_pretrain_joint_direct -- --seed "$SEED"
python -m experiments.run_experiment train exp020_forced_damped_hnn_reconae_dynamics_only -- --seed "$SEED"
python -m experiments.run_experiment train exp021_forced_damped_hnn_reconae_pretrain_joint_direct -- --seed "$SEED"
python -m experiments.run_experiment train exp001_hnn_reconae_joint_direct -- --seed "$SEED"
python -m experiments.run_experiment train exp002_forced_damped_hnn_reconae_joint_direct -- --seed "$SEED"

python -m experiments.run_experiment infer exp000_sft_only_direct -- --seed "$SEED"
python -m experiments.run_experiment infer exp003_stage2_sft_only_direct -- --seed "$SEED"
python -m experiments.run_experiment infer exp011_hnn_reconae_pretrain_joint_direct -- --seed "$SEED"
python -m experiments.run_experiment infer exp021_forced_damped_hnn_reconae_pretrain_joint_direct -- --seed "$SEED"
python -m experiments.run_experiment infer exp001_hnn_reconae_joint_direct -- --seed "$SEED"
python -m experiments.run_experiment infer exp002_forced_damped_hnn_reconae_joint_direct -- --seed "$SEED"
```

## Four-A100 CFFF schedule

The shared SFT is a four-process DDP job. AE, HNN/FDHNN pretraining, D3 joint
training, D4 ablations, and the SFT-only control each use one process so
independent methods and seeds can run concurrently across the four GPUs. This
keeps the exact LoRA gradient-conflict diagnostic well-defined. Direct inference uses four independent model
replicas on mutually exclusive strided input shards; this is data-parallel
evaluation, not model parallelism. Run the dependency-aware scheduler:

```bash
python -m experiments.run_cfff_matrix \
  --seeds 20260711,20260712,20260713 \
  --gpus 0,1,2,3
```

To evaluate only the shared-SFT baseline without launching AE or any stage-2
arm, select the dependency-closed baseline subgraph explicitly:

```bash
python -m experiments.run_cfff_matrix \
  --seeds 20260711 \
  --gpus 0,1,2,3 \
  --only-baseline-inference
```

Across three seeds it runs each shared SFT with all four GPUs, then fills the
node with dependency-ready single-GPU AE/D2/D3/D4 jobs and inference shards.
The merge rejects a
missing or duplicate dataset index, changed source field, mismatched shard run
configuration, or progress hash before creating canonical outputs. Completed
outputs are skipped on restart only when the trainer's atomic
`run_complete.json` marker and required checkpoints all exist. Partial
non-empty trainer directories fail closed instead of being mistaken for a
finished run or overwritten. Inspect the plan without launching with
`--dry-run`; `--inference-shards 1..4` controls evaluation parallelism.

Every trainer refuses a non-empty output directory. Passing a different seed
selects a different artifact tree; it does not change the prepared training rows.

## Smoke and audits

Use at least two `pathway_family_id` groups when passing `--limit`, because
validation is family-group-safe.

```bash
python -m experiments.validate_matrix
python -m experiments.run_experiment audit --quiet
python -m experiments.run_experiment consistency --phase both --quiet
python -m unittest discover -s experiments/tests -v
python -m unittest discover -s method/tests -v
```

Direct greedy inference is the only active inference mode in the current
Hamiltonian matrix. The following generation studies are explicitly retained
for the next research phase; they have not been deleted:

- event-by-event generation with layer-dependent step sizes after a validated
  JSON event-boundary controller exists;
- graph-layer-by-graph-layer generation as a separate controller comparison;
- token-by-token generation after training a separate token-resolution dynamics
  objective; the event/layer checkpoint must not be advanced once per token;
- a generation-time multiscale hybrid of the controllers, ablated against direct greedy
  generation under matched decoding budgets.

All four must compare biological validity, JSON validity, long-horizon error,
and compute. PHNN likewise waits for a real port/control contract.
