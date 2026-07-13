# Controlled experiment matrix

`matrix.json` contains five executable rows. They are intentionally narrow so
the first result can be attributed.

| Row | Role |
| --- | --- |
| `base000_shared_sft_reconae` | train one shared stage-1 SFT and one shared AE for a seed |
| `exp000_sft_only_direct` | stage-1 direct-generation baseline |
| `exp003_stage2_sft_only_direct` | compute-matched second-SFT control with every dynamics weight and LR zero |
| `exp001_hnn_reconae_joint_direct` | stage-2 SFT plus `J grad H` |
| `exp002_forced_damped_hnn_reconae_joint_direct` | primary stage-2 SFT plus `(J-rI) grad H + F(t)` |

The active release is `data/pathway_v3_cap256/`. Its test set is disjoint from
training by both held-out organism and canonical five-digit KEGG pathway
family; validation holds out separate complete families. The scheduler refuses
to start unless the generated read-only `data_audit.json` has passed and still
matches all three CSV hashes.

All mutable artifacts are seed-scoped:

```text
checkpoints/seeds/<seed>/shared/...
checkpoints/seeds/<seed>/experiments/<row>/...
runs/seeds/<seed>/experiments/<row>/...
```

Within one seed, `exp003`, `exp001`, and `exp002` reuse exactly that seed's
shared SFT and AE. Recommended seeds are `20260711`, `20260712`, and
`20260713`.

The controlled matrix pins `max_length=8192` and per-process training
`batch_size=1` for SFT, AE, and every stage-2 arm. Dataset materialization uses
the real tokenizer and excludes any row whose complete prompt plus closed JSON
answer exceeds that budget. Trainers fail if an oversized row slips through;
they never truncate an assistant JSON target.

The materialized rows are eligible prefix views, not the per-epoch training count.
Controlled training uses one deterministic prefix per biological record per
epoch. Across epochs,
each record rotates through short-, middle-, and long-continuation views. SFT
weights short continuation twice per four-record-epochs; HNN/FDHNN weights long
continuation twice so trajectory losses still see multi-step targets. Validation
uses one seed-fixed prefix per record with a balanced short/middle/long policy
and never rotates across epochs. This prevents long pathways from receiving
k-fold weight in either optimization or checkpoint selection and cuts per-epoch
examples without permanently dropping a record.

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
python -m experiments.run_experiment prepare-structured-data \
  --max-records-per-family 256 \
  --minimum-train-records 12000 \
  --seed 20260711 \
  --overwrite
python -m experiments.run_experiment download-model
python -m experiments.run_experiment check-assets \
  --phase train --ids base000_shared_sft_reconae --profile cfff \
  --create-output-dirs --strict
```

The builder reconstructs event sets directly from `processed_graph`, selects a
family-capped diverse release, writes record JSONL plus compatibility CSVs,
and creates `dataset_manifest.json` and read-only `data_audit.json`. The default
minimum of 12,000 train records prevents silently releasing another short
training run. The audit records the actual record/token distribution and a
rough runtime estimate; the first measured v3 SFT run replaces that estimate.

## One-seed run

```bash
SEED=20260711

python -m experiments.run_experiment train base000_shared_sft_reconae -- --seed "$SEED"
python -m experiments.run_experiment train exp003_stage2_sft_only_direct -- --seed "$SEED"
python -m experiments.run_experiment train exp001_hnn_reconae_joint_direct -- --seed "$SEED"
python -m experiments.run_experiment train exp002_forced_damped_hnn_reconae_joint_direct -- --seed "$SEED"

python -m experiments.run_experiment infer exp000_sft_only_direct -- --seed "$SEED"
python -m experiments.run_experiment infer exp003_stage2_sft_only_direct -- --seed "$SEED"
python -m experiments.run_experiment infer exp001_hnn_reconae_joint_direct -- --seed "$SEED"
python -m experiments.run_experiment infer exp002_forced_damped_hnn_reconae_joint_direct -- --seed "$SEED"
```

## Four-A100 CFFF schedule

The shared SFT is a four-process DDP job. The primary forced/damped HNN stage
uses four processes; pure HNN and the stage-2 SFT control use two processes
each and run concurrently. AE uses one GPU while dependency-ready inference
shards can occupy the others. Direct inference uses four independent model
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
node with AE, the four-GPU primary stage, the concurrent two-plus-two control
stages, and inference shards as their dependencies become available. The merge rejects a
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

- graph-layer-by-graph-layer generation after a validated JSON layer-boundary
  controller exists;
- token-by-token generation after training a separate token-resolution dynamics
  objective; the graph-layer checkpoint must not be advanced once per token;
- a multiscale hybrid of the two controllers, ablated against direct greedy
  generation under matched decoding budgets.

All three must compare biological validity, JSON validity, long-horizon error,
and compute. PHNN likewise waits for a real port/control contract.
