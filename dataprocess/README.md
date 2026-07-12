# KEGG pathway dataset contract

`build_pathway_csv.py` reads ordered text layers from `processed/` and graph
metadata/phenotype evidence from `processed_graph/`. One biological record is
one `(organism, source_json, pathway_id, pathway_block)` identity; every prefix
row has `sample_id=<record_id>:prefix=<count>`.

`pathway_family_id` is the terminal five-digit KEGG reference-map identifier,
so organism-specific IDs such as `hsa04010` and `mmu04010` belong to the same
family. It is a split/audit key, not the sample identity.

This removes exact cross-organism KEGG map-family overlap. It does not prove
that different map IDs have no shared KO sets or homologous subgraphs; claims
about that stronger notion require a versioned KO/graph-similarity clustering
manifest and must be reported separately.

## Maintained answer schema

```json
{
  "remaining_steps": [
    {
      "step": 2,
      "layer": "layer 2",
      "substeps": [
        {
          "substep": 0,
          "text": "AKT1 phosphorylates BAD.",
          "source_item_index": 0
        }
      ]
    }
  ],
  "predicted_phenotype": null
}
```

Layers are ordered. `substeps` preserve original source-item boundaries, but
same-layer item order is serialization provenance, not biological time.

## Phenotype policy

- block-level labels may supervise only their own block;
- file-level labels are inherited only when the file contains one pathway
  block;
- a multi-block file-level label becomes `ambiguous_file_level`, never a label
  copied onto every block;
- disagreeing block/file sources become `conflict`;
- read failures become `source_error`;
- no annotation becomes `not_annotated` with JSON `null`.

Only `available` rows enter phenotype accuracy. Other statuses remain valid for
trajectory training but are excluded from phenotype denominators.

## Existing server corpus and experiment files

The full server CSVs generated on 2026-07-11 contain 32,258,032 train rows and
36,327 test rows. They predate the explicit substep/identity schema. Do not
regenerate 171 GiB merely to start the core benchmark; run:

```bash
export CHATPATHWAY_PROFILE=cfff
python -m experiments.run_experiment prepare-data --overwrite
```

This streaming pass creates:

- a record-balanced training pilot that excludes the deterministically reserved
  KEGG pathway families;
- strict core/multi-step evaluations whose organisms and pathway families are
  both absent from training;
- separate organism-held-out evaluations that intentionally allow and report
  pathway-family overlap, for the distinct cross-species-transfer question.

It upgrades identity/status/family fields, marks recovered boundaries as
`sentence_parser_v1`, and performs strict schema, identity, source, record,
sample, and pathway-family audits. The original organism split is not presented
as unseen-pathway generalization.

For a future exact rebuild from source JSON:

```bash
python dataprocess/build_pathway_csv.py \
  --processed-root ../KEGG_all_new/processed \
  --processed-graph-root ../KEGG_all_new/processed_graph \
  --train-output ../data/train_kegg_pathway_dataset.csv \
  --test-output ../data/test_kegg_pathway_dataset.csv \
  --test-organisms tru,xtr,dre,gga,dmk,dme,cel \
  --overwrite
```

The rebuilt full files above are the organism-transfer source split, so audit
them with `--allow-pathway-family-overlap`, then run `prepare-data` to create
the strict family-disjoint core split. The prepared core train/test pair must
pass `dataprocess/audit_pathway_csv.py --strict` without that allowance.
Trainers use the pandas C parser and reject malformed rows; they do not silently
skip oversized or bad CSV records.
