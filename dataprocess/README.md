# ChatPathway2 KEGG Data Processing

This directory builds ChatPathway2 training CSVs from the newer KEGG pipeline
outputs.

The pipeline PDF describes two useful output layers:

- `processed_graph/<organism>/<pathway>.json`: canonical graph truth layer.
- `processed/<organism>/<pathway>.json`: LLM-facing layered pathway text.

`build_pathway_csv.py` uses `processed/` to create prefix-to-remaining-trajectory
examples, and uses `processed_graph/` only for metadata and phenotype labels when
they exist.

## Expected CFFF Layout

Keep the decompressed KEGG dataset next to the `ChatPathway2` checkout:

```text
chatpathway2_parent/
  ChatPathway2/
  KEGG_all_new/
    processed/
    processed_graph/
  data/
```

The default paths are relative to `ChatPathway2`:

```text
../KEGG_all_new/processed
../KEGG_all_new/processed_graph
../data/train_kegg_pathway_dataset.csv
../data/test_kegg_pathway_dataset.csv
```

## Build CSVs

From the `ChatPathway2` repo root:

```bash
python dataprocess/build_pathway_csv.py --overwrite
```

This writes train/test CSVs. Add `--output ../data/kegg_pathway_dataset.csv` if
you also want a duplicated all-row CSV.

For an organism-held-out split similar to the old setup:

```bash
python dataprocess/build_pathway_csv.py \
  --test-organisms tru,xtr,dre,gga,dmk,dme,cel \
  --overwrite
```

For a quick smoke run:

```bash
python dataprocess/build_pathway_csv.py \
  --max-files 20 \
  --train-output ../data/train_kegg_pathway_dataset.smoke.csv \
  --test-output ../data/test_kegg_pathway_dataset.smoke.csv \
  --overwrite
```

## Example Shape

Each row keeps the legacy `question` and `answer` columns used by the current
training scripts. Extra columns preserve provenance and filtering metadata.

`question` contains the pathway metadata plus observed prefix steps.

`answer` is a JSON string:

```json
{
  "remaining_steps": [
    {
      "layer": "layer 2",
      "step": 2,
      "text": "downstream pathway text"
    }
  ],
  "predicted_phenotype": null
}
```

When a phenotype is available in `processed_graph`, `predicted_phenotype` becomes
an object:

```json
{
  "text": "source phenotype text",
  "status": "available"
}
```

## Phenotype Policy

Only part of the current KEGG-derived dataset has phenotype supervision. The
script does not fabricate missing phenotypes.

- Rows with a source phenotype get `phenotype_status=available`.
- Rows without one get `phenotype_status=missing` and
  `"predicted_phenotype": null` in the answer JSON.
- Use `--require-phenotype` when training or evaluating a phenotype-specific
  objective.
- Use all rows for trajectory planning if the main target is remaining pathway
  rollout; the null phenotype target simply means that phenotype supervision is
  unavailable for that example.

By default, a pathway block with `k` ordered layers generates `k - 1` examples:
prefix lengths `i = 1 ... k - 1`, with all remaining layers in the answer. Add
`--include-empty-prefix` only for ablations that need the `i = 0` no-observation
example.

The current `Step` unit is a graph-layer transition from the pipeline's layered
JSON, not necessarily one atomic biochemical reaction. A layer can summarize
multiple relation/reaction text events that occur at the same graph depth. This
keeps the source graph order intact; splitting those sentences into separate
time steps should be a separate data mode because same-layer events may not have
a true internal temporal order.
