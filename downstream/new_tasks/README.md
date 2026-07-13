# Revised downstream task suite

This directory is the strict Task 0-6 contract for new experiments. It does
not replace or mutate historical entry points under `downstream/tasks/`, so old
artifacts remain reproducible.

The detailed Chinese task definition and data-team handoff instructions are
frozen in [FROZEN_TASK_SPEC_2026-07-13.md](../../docs/FROZEN_TASK_SPEC_2026-07-13.md).

The suite follows three non-negotiable rules:

1. Token/substep index is reasoning order, not biological clock time.
2. Under `dz/dt = (J-R) grad(H) + F(t)`, forcing can inject energy. Therefore
   total-energy descent is not a causal-direction label. HNN values in Task 3
   are diagnostics unless calibrated on validation data.
3. `F(t)` is time-only and is not a knockout/control `u`. Task 4 needs real
   interventions and a calibrated phenotype scorer. Missing phenotype labels
   are excluded and counted, never treated as negative.

The machine-readable task/readiness table is [matrix.json](matrix.json).
Implementation provenance and deliberate corrections are in
[PROVENANCE.md](PROVENANCE.md).

## Task definitions

| Task | Scientific question | Required artifact | Reportable output |
| --- | --- | --- | --- |
| 0 AE/HNN self-consistency | Does the AE preserve hidden states, and does the learned ODE follow held-out latent trajectories? | NPZ hidden/reconstruction and observed/rollout latents, or observed latents plus `hamiltonian_dynamics.pt` | reconstruction MSE/cosine and rollout error curves at fixed horizons |
| 1 substep CSP | Given a pathway prefix, are the next graph layer's atomic `A relation B` events and remaining layer sequence correct? | v3 prediction CSV/JSON with structured `remaining_layers/events`; v2 only through an audited fallback | layer-set event metrics by default; ordered-substep metrics only with causal-order provenance |
| 2 PCTE | Are predicted and gold answer trajectories close in the same fixed latent representation? | paired latent NPZ plus representation manifest | DTW PCTE; not HNN self-consistency |
| 3 causal reranking | Does the LLM rank a validated path above direction-reversed, shuffled, and unrelated candidates? | expert-validated candidate JSON/JSONL | LLM Top-1/MRR/rejection; optional validation-calibrated combined score |
| 4 knockout/rescue | Do calibrated phenotype predictions match observed KO effects and rank true rescue interventions? | real intervention cases, test labels, validation-calibrated scorer | Brier/accuracy, KO direction, rescue Hit@1/MRR |
| 5 cell transfer | Does an explicitly cell-adapted checkpoint predict held-out perturbation response? | aligned matrices plus gene/cell/perturbation manifest | expression and delta correlations, controlled-ablation difference |
| 6 BioMaze | Does the frozen checkpoint answer an independent mechanistic QA benchmark? | official benchmark predictions and provenance manifest | option accuracy and validity |

## Atomic substep schema

The preferred Task 1 answer is the same v3 continuation contract used by training:

```json
{
  "schema_version": "pathway_continuation_v3",
  "remaining_layers": [
    {
      "layer_index": 2,
      "events": [
        {
          "source": [{"canonical_id": "hsa:207", "name": "AKT1"}],
          "relation": "phosphorylation",
          "target": [{"canonical_id": "hsa:572", "name": "BAD"}],
          "text": "AKT1 phosphorylates BAD."
        }
      ]
    }
  ]
}
```

The adapter accepts v3 `remaining_layers/events` directly and scores canonical
source/target IDs plus the structured relation without reparsing the event
sentence. Historical v2 `remaining_steps` remains readable; only that fallback
splits sentence/semicolon clauses with exactly one supported relation. Parser
validity and excluded coverage are always emitted alongside accuracy.

Important dataset boundary: the active v3 builder reads canonical relation and
reaction events from `processed_graph`. The 2026-07-11 full server CSV predates
that schema, so `prepare_experiment_data.py` marks its recovered boundaries as
`sentence_parser_v1`. If two legacy source items had no punctuation delimiter,
their boundary cannot be reconstructed losslessly; the fallback parser reports
that provenance and never treats sentence order as an independent causal-order
annotation.

Atomic does not automatically mean sequential. Multiple source items in one
graph layer may be parallel events. The Task 1 manifest therefore requires an
`ordering_mode`:

- `layer_set` (recommended/current data): layers are ordered, but atomic events
  within each layer are scored as a permutation-invariant multiset;
- `causal_substep_sequence`: flat substep order is scored only when
  `ordering_provenance` (`source`, `version`) is supplied and both target and
  prediction use the strict structured schema. Sentence order alone is not
  acceptable provenance.

## Task 0 NPZ

Create a shared Task 0/2 artifact with the exact Framework A layer construction:

```bash
python -m method.analysis.semantic_latent_export \
  --input /path/to/direct_predictions.csv \
  --output /path/to/semantic_latents.npz \
  --manifest-output /path/to/semantic_latents.manifest.json \
  --base-model /path/to/qwen3_8B \
  --adapter /path/to/checkpoint_best \
  --ae-checkpoint /path/to/ae_proj.pt \
  --dynamics-checkpoint /path/to/hamiltonian_dynamics.pt \
  --dataset-id heldout_pathway_v1 \
  --max-length 8192 \
  --include-reconstruction-states
```

`observed_latents` contains prompt anchor plus gold layers for Task 0;
`target_latents` and `predicted_latents` omit the common prompt anchor for Task
2. Invalid prediction JSON fails strict export by default; `--no-strict` skips
and reports affected rows, so coverage changes cannot be silent.

Required:

- `observed_latents`: `[samples, points, latent_dim]`
- `rollout_latents`: same shape, unless `--dynamics-checkpoint` is passed
- optional `lengths`: valid point count per sample

For complete self-consistency also include matching `hidden_states` and
`reconstructed_states`. Use the held-out validation/test split and record
whether points are `token`, `graph_layer`, or `causal_substep`. Use
`graph_layer` when same-layer events are pooled. `causal_substep` requires an
independent `ordering_provenance`; it cannot be inferred from sentence order.
All three are reasoning coordinates, not physical time.

The required manifest records `dataset_id`, `split`, `granularity`,
`point_construction_version`, the exact training/inference `dynamics_dt`, and
immutable IDs for the `base`, `sft`, `ae`, and `dynamics` checkpoints. The
evaluator uses that pinned `dynamics_dt`; it does not choose a new horizon scale.

```bash
python -m downstream.new_tasks.task0_self_consistency \
  --input /path/to/self_consistency.npz \
  --manifest /path/to/self_consistency_manifest.json \
  --dynamics-checkpoint /path/to/checkpoint_best/hamiltonian_dynamics.pt \
  --output-dir /path/to/results/task0
```

## Task 1 command

```bash
python -m downstream.new_tasks.task1_substep_csp \
  --input /path/to/predictions.csv \
  --manifest /path/to/task1_manifest.json \
  --output-dir /path/to/results/task1
```

Natural-language fallback is useful for auditing the current dataset, but a
publishable Task 1 table should freeze parser version, inspect parse failures,
and preferably export explicit structured gold substeps before inference.
The manifest pins `dataset_id`, held-out `split`, immutable model checkpoint,
and parser version. This implementation requires
`parser_version: "atomic_relation_v2"`; use `ordering_mode: "layer_set"` for
the current graph-layer data.

## Task 2 manifest and command

The NPZ must contain `predicted_latents`, `target_latents`,
`predicted_lengths`, and `target_lengths`. The JSON manifest pins the
representation:

```json
{
  "schema_version": 1,
  "dataset_id": "heldout_pathway_v1",
  "split": "test",
  "granularity": "graph_layer",
  "representation": {
    "base_checkpoint": "sha256-or-immutable-id",
    "adapter_checkpoint": "sha256-or-immutable-id",
    "ae_checkpoint": "sha256-or-immutable-id"
  }
}
```

```bash
python -m downstream.new_tasks.task2_pcte \
  --input /path/to/pcte.npz \
  --manifest /path/to/pcte_manifest.json \
  --output-dir /path/to/results/task2
```

## Task 3 candidate and calibration contract

Each case needs `id`, `question`, `expert_validated: true`, an
`annotation_provenance` object (`annotation_id`, `protocol_version`, and
`source_dataset_id`), one positive, and at least one negative. Negative types
are `direction_reversal`, `step_shuffle`, and `unrelated_pathway`. Supply
`llm_score` per candidate or use `--base-model` and optional `--adapter`.

Every negative also carries `negative_provenance`. Direction reversals list
the changed structured relation IDs. Shuffles record original/permuted IDs and
may operate only on `graph_layer` or independently ordered
`causal_substep` units - shuffling parallel events inside one layer is not a
valid negative. Unrelated pathways record their source pathway and matching
protocol so trivial organism/length/vocabulary cues can be controlled.

Optional HNN diagnostics are limited to `hnn_rollout_error`,
`hnn_vector_field_residual`, and `energy_balance_residual`. They do not affect
ranking without a calibration JSON whose `fit_split` is exactly `validation`.
The evaluator rejects raw `energy_delta`/monotonic-energy causal proxies.

```bash
python -m downstream.new_tasks.task3_causal_reranking \
  --input /path/to/expert_candidates.jsonl \
  --base-model /path/to/base_model \
  --adapter /path/to/checkpoint_best \
  --max-length 8192 \
  --calibration /path/to/validation_calibration.json \
  --output-dir /path/to/results/task3
```

## Task 4 knockout/rescue contract

Annotated cases require:

- dataset ID/version, test split, and experimental evidence source;
- a phenotype definition and scorer ID calibrated on validation data;
- immutable base, adapter, and dynamics checkpoint IDs plus the intervention
  prompt template version (or explicit conditioning schema ID);
- exactly one WT state, one or more KO states, and optional rescue candidates;
- explicit interventions (`knockout`, `knockdown`, `inhibition`,
  `overexpression`, `activation`, or `drug`);
- binary gold endpoint plus calibrated predicted probability for every state;
- `dynamics_conditioning` equal to `prompt_initial_condition` or
  `explicit_intervention_conditioned_dynamics`.

Records with `phenotype_available: false` need only `case_id`; they contribute
to missing-label coverage, not to accuracy. `time_only_force`, `F(t)`, or
equivalent conditioning declarations are rejected.

```bash
python -m downstream.new_tasks.task4_knockout_rescue \
  --input /path/to/knockout_rescue.jsonl \
  --output-dir /path/to/results/task4
```

## Task 5 manifest

The NPZ contains same-shaped `[cells, genes]` arrays `control`, `observed`, and
`predicted`, optionally `baseline_predicted`. Its manifest must pin gene order,
cell order, perturbation IDs, normalization, control matching, held-out split,
base checkpoint, cell adapter checkpoint, and cell training data. If a baseline
is present, `controlled_ablation_id` is mandatory.

```bash
python -m downstream.new_tasks.task5_perturbed_cell_transfer \
  --input /path/to/cell_transfer.npz \
  --manifest /path/to/cell_transfer_manifest.json \
  --output-dir /path/to/results/task5
```

## Task 6 BioMaze

BioMaze records contain `id`, `question`, `choices`, `gold_option`, and
`predicted_option`. The manifest pins dataset version/source/license, split,
checkpoint, protocol, and contamination-audit method/status.

```bash
python -m downstream.new_tasks.task6_biomaze \
  --input /path/to/biomaze_predictions.jsonl \
  --manifest /path/to/biomaze_manifest.json \
  --output-dir /path/to/results/task6
```

## Synthetic verification

Synthetic data verify schemas and metric code only. They are never experiment
results.

```bash
python -m downstream.new_tasks.audit
python -m unittest downstream.new_tasks.tests.test_smoke
```
