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

All mutable artifacts are seed-scoped:

```text
checkpoints/seeds/<seed>/shared/...
checkpoints/seeds/<seed>/experiments/<row>/...
runs/seeds/<seed>/experiments/<row>/...
```

Within one seed, `exp003`, `exp001`, and `exp002` reuse exactly that seed's
shared SFT and AE. Recommended seeds are `20260711`, `20260712`, and
`20260713`.

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
