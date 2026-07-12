# Server-to-result workflow

Run from the `ChatPathway2` repository root. On CFFF:

```bash
source /cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui/codex-env.sh
export CHATPATHWAY_PROFILE=cfff
BASE=/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui
```

The profile resolves assets below
`/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui`; Git contains code only.

## 1. Prepare and audit experiment data

The existing full CSV is the 2026-07-11 legacy aggregate-step export. The
preparation pass adds stable record/sample/family identities, converts `missing`
to `not_annotated`, conservatively parses substeps, and creates:

- `data/train_kegg_pathway_pilot.csv`: stable 0.1% record sample after removing
  the held-out five-digit KEGG pathway families, at most three prefixes per
  record, plus every eligible annotated phenotype record;
- `data/test_kegg_pathway_eval.csv`: strict organism-plus-pathway-family-held-out
  core evaluation, one last-prefix/next-layer row per record;
- `data/test_kegg_pathway_multistep_eval.csv`: the same strict records with up
  to three prefixes per record;
- `data/test_kegg_pathway_organism_eval.csv` and its multistep companion: the
  original seven-organism transfer evaluation, where pathway-family overlap is
  intentional and reported rather than mistaken for unseen-pathway evidence.

```bash
python -m experiments.run_experiment prepare-data --overwrite
```

The command fails if schema/identity checks fail or a source, record, sample, or
pathway family crosses the strict train/test boundary. Exact `source_items` are
preserved by the new JSON builder; the already-generated full CSV can only be
marked `sentence_parser_v1`.

Measure the exact semantic-layer/substep retention under the same token budget
used by SFT, AE, and stage 2 before launching the matrix:

```bash
python -m dataprocess.audit_token_budget \
  --input "$BASE/data/train_kegg_pathway_pilot.csv" \
  --base-model "$BASE/models/qwen3_8B" \
  --max-length 8192 \
  --output "$BASE/artifacts/dataset/pilot_token_budget_8192.json"
```

Text-budget truncation is distinct from the 128-step ODE cap and is reported
separately. A row with zero retained semantic layers still contributes SFT, but
not a dynamics alignment target.

The maintained matrix fixes `max_length=8192` and per-process training
`batch_size=1`. This setting was smoke-tested through four-GPU SFT, AE, HNN,
and forced/damped HNN on A100-80GB; do not silently lower it for one arm.

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
least two pathway-family groups.

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

On the four-A100 CFFF node, use `python -m experiments.run_cfff_matrix`. It
uses all four GPUs for each DDP SFT prerequisite, then schedules independent
single-GPU AE, stage-2 arms, and inference jobs across the four devices. Thus
the node is kept busy without pretending that AE/HNN themselves are currently
four-way distributed.

Validation is a deterministic `pathway_family_id` group split, so a family used
for checkpoint selection is absent from that seed's optimization rows. The same
split is reused by SFT, AE, and every stage-2 arm. Checkpoint selection uses
validation loss with early stopping. Every run records config, Git commit, base
revision, SFT/AE digests, data digest, metrics JSONL, and truncation counts.

## 5. Revised downstream tasks

Run structural generation metrics on direct prediction CSVs. Task 0 and Task 2
use the maintained semantic exporter:

```bash
BASE=/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui
python -m method.analysis.semantic_latent_export \
  --input "$BASE/data/test_kegg_pathway_eval.csv" \
  --output "$BASE/artifacts/task0/self_consistency.npz" \
  --manifest-output "$BASE/artifacts/task0/manifest.json" \
  --dataset-id kegg_family_disjoint_core_v2 \
  --max-length 8192 \
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
