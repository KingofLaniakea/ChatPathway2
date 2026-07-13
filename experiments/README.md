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

The default core evaluation is disjoint from training by both held-out organism
and canonical five-digit KEGG pathway family. The separate
`test_kegg_pathway_organism_eval.csv` retains all seven held-out organisms and
allows reported family overlap for cross-species-transfer analysis.

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
`batch_size=1` for SFT, AE, and every stage-2 arm. On the prepared CFFF
record-balanced 0.1% first-round training set,
this retains 99.17% of graph-layer targets and 91.07% of substeps; 31 of 17,416
rows have no complete semantic layer inside the text budget and therefore
contribute CE but no dynamics loss. Truncation counters remain mandatory
reported metrics.

Direct inference pins per-process `batch_size=1` and
`max_new_tokens=1024`. Batch size is an experimental control: an audited
batch-8 attempt changed 6 of the first 40 greedy token trajectories relative
to batch 1, even when some changes were only same-layer substep permutations.
Four-GPU speedup therefore comes only from disjoint data shards. On the 764-row
strict core evaluation, the longest gold answer is 925 tokens (99th percentile
678), so the cap covers every gold target while bounding non-terminating repetition.
Every completed sample is immediately appended to a progress JSONL with its
identity, gold answer, prediction, finish reason, and JSON/schema validity. A
direct one-GPU wrapper writes `direct.progress.jsonl`. The CFFF scheduler writes
four `direct.progress.shard-*-of-*.jsonl` files live, then verifies and merges
them into `direct.progress.jsonl` and the final CSV in original dataset order.

Checkpoint selection uses a deterministic `pathway_family_id` validation split,
not a row or source-only split. For seeds `20260711/12/13`, validation contains
16/18/17 entire families respectively and has zero family overlap with that
seed's optimization rows. SFT, AE, and all stage-2 arms reuse the same split.

## CFFF preparation

```bash
export CHATPATHWAY_PROFILE=cfff
python -m experiments.run_experiment prepare-data --overwrite
python -m experiments.run_experiment download-model
python -m experiments.run_experiment check-assets \
  --phase train --ids base000_shared_sft_reconae --profile cfff \
  --create-output-dirs --strict
```

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

The shared SFT is a four-process DDP job. AE and each stage-2 arm are
single-GPU jobs. Direct inference uses four independent model replicas on
mutually exclusive strided input shards; this is data-parallel evaluation, not
model parallelism. Run the dependency-aware scheduler:

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

It runs the three SFT prerequisites sequentially with all four GPUs, then fills
the GPUs with independent AE, stage-2 control/HNN/forced-damped jobs and
inference shards as their dependencies become available. The merge rejects a
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
