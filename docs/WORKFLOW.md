# Server-to-result workflow

Run from the `ChatPathway2` repository root. On CFFF:

```bash
source /cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui/codex-env.sh
export CHATPATHWAY_PROFILE=cfff
```

The profile resolves assets below
`/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui`; Git contains code only.

## 1. Prepare and audit experiment data

The existing full CSV is the 2026-07-11 legacy aggregate-step export. The
preparation pass scans it once, adds stable record/sample identities, converts
`missing` to `not_annotated`, conservatively parses substeps, and creates:

- `data/train_kegg_pathway_pilot.csv`: stable 0.1% record sample, at most three
  prefixes per record, plus every currently annotated phenotype record;
- `data/test_kegg_pathway_eval.csv`: one last-prefix/next-layer row per record;
- `data/test_kegg_pathway_multistep_eval.csv`: up to three prefixes per record.

```bash
python -m experiments.run_experiment prepare-data --overwrite
```

The command fails if schema/identity checks fail or any `source_json` crosses
train and test. Exact `source_items` are preserved by the new JSON builder; the
already-generated full CSV can only be marked `sentence_parser_v1`.

## 2. Download the pinned base model

```bash
python -m experiments.run_experiment download-model
python -m experiments.run_experiment download-model --verify-only
```

The download resolves the pinned Qwen revision to a full commit SHA, verifies
the tokenizer, tensor index, and every shard, then writes a manifest used by
training provenance.

## 3. Run a minimal end-to-end smoke

Use a disposable seed or explicit disposable output directories. The smoke
must cover one SFT forward/backward, AE save/load, HNN and forced/damped RK4
backward, checkpoint reload, and direct generation. `--limit` must contain at
least two source groups.

```bash
python -m experiments.run_experiment train base000_shared_sft_reconae -- \
  --seed 999001 --epochs 1 --limit 32 --no-hash-inputs
python -m experiments.run_experiment train exp001_hnn_reconae_joint_direct -- \
  --seed 999001 --epochs 1 --limit 32 --no-hash-inputs
python -m experiments.run_experiment train exp002_forced_damped_hnn_reconae_joint_direct -- \
  --seed 999001 --epochs 1 --limit 32 --no-hash-inputs
python -m experiments.run_experiment infer exp002_forced_damped_hnn_reconae_joint_direct -- \
  --seed 999001 --limit 2
```

## 4. Run the controlled matrix

For each of `20260711`, `20260712`, and `20260713`, run shared prerequisites,
the compute-matched stage-2 SFT control, pure HNN, and forced/damped HNN. Compare
direct outputs from stage-1, stage-2-only, HNN, and forced/damped HNN.

Validation is a deterministic `source_json` group split. Checkpoint selection
uses validation loss with early stopping. Every run records config, Git commit,
base revision, SFT/AE digests, data digest, metrics JSONL, and truncation counts.

## 5. Revised downstream tasks

Run structural generation metrics on direct prediction CSVs. Task 0 and Task 2
use the maintained semantic exporter:

```bash
BASE=/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui
python -m method.analysis.semantic_latent_export \
  --input "$BASE/data/test_kegg_pathway_eval.csv" \
  --output "$BASE/artifacts/task0/self_consistency.npz" \
  --manifest-output "$BASE/artifacts/task0/manifest.json" \
  --dataset-id kegg_test_organisms_v1 \
  --base-model "$BASE/models/qwen3_8B" \
  --adapter "$BASE/checkpoints/seeds/20260711/experiments/exp002_forced_damped_hnn_reconae_joint_direct/final_lora/checkpoint_best" \
  --ae-checkpoint "$BASE/checkpoints/seeds/20260711/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt" \
  --dynamics-checkpoint "$BASE/checkpoints/seeds/20260711/experiments/exp002_forced_damped_hnn_reconae_joint_direct/final_lora/checkpoint_best/hamiltonian_dynamics.pt" \
  --include-reconstruction-states
```

Task 4 knockout/rescue is not eligible until real intervention evidence and a
validation-calibrated phenotype scorer exist. Missing phenotype labels are
excluded, not converted to negatives.

## Explicitly deferred

- rollout/mixed generation: training unit is graph layer, while the discarded
  prototype advanced once per token;
- PHNN: no independent observed `u`/port exists;
- phenotype causal claims: current full data contains no available labels;
- Neural ODE expansion: wait until the Hamiltonian benchmark is validated.
