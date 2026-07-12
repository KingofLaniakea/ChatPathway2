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
`batch_size=1` for SFT, AE, and every stage-2 arm. On the prepared CFFF pilot,
this retains 99.17% of graph-layer targets and 91.07% of substeps; 31 of 17,416
rows have no complete semantic layer inside the text budget and therefore
contribute CE but no dynamics loss. Truncation counters remain mandatory
reported metrics.

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

The shared SFT is a four-process DDP job. AE, each stage-2 arm, and each
inference job are single-GPU by design. To keep the machine busy without
claiming false model-parallelism, run the dependency-aware scheduler:

```bash
python -m experiments.run_cfff_matrix \
  --seeds 20260711,20260712,20260713 \
  --gpus 0,1,2,3
```

It runs the three SFT prerequisites sequentially with all four GPUs, then fills
the GPUs with independent AE, stage-2 control/HNN/forced-damped jobs and direct
inference as their dependencies become available. Completed outputs are skipped
on restart; partial non-empty trainer directories still fail closed instead of
being overwritten. Inspect the plan without launching with `--dry-run`.

Every trainer refuses a non-empty output directory. Passing a different seed
selects a different artifact tree; it does not change the prepared pilot rows.

## Smoke and audits

Use at least two `source_json` groups when passing `--limit`, because validation
is group-safe.

```bash
python -m experiments.validate_matrix
python -m experiments.run_experiment audit --quiet
python -m experiments.run_experiment consistency --phase both --quiet
python -m unittest discover -s experiments/tests -v
python -m unittest discover -s method/tests -v
```

Direct greedy inference is the only active inference mode. A graph-layer HNN
cannot be advanced once per generated token: rollout/mixed rows are blocked
until a validated JSON graph-layer boundary controller exists. PHNN likewise
waits for a real port/control contract.
