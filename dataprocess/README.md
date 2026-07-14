# Pathway-continuation v3 dataset

The active builder is `build_structured_dataset.py`. It reads canonical
`processed_graph/<organism>/<pathway>.json` directly; it does not recover
events by splitting the concatenated paragraphs under `processed/`.

## Biological record

Relations and reactions are read only from `processed_graph.relations` and
`processed_graph.reactions`; `processed/` is optional path/text reconciliation,
never event truth. Direction-bearing relation subtypes and reaction direction
form the topology backbone. Binding, dissociation, state-change, modification,
and compound-mediator evidence is retained as context without inventing an
ordering edge. Missing/unknown relations are excluded explicitly. Any invalid
endpoint or inconsistent subtype representation rejects the entire graph
rather than silently deleting one edge and recomputing topology.

The builder condenses backbone cycles with Tarjan SCC and creates one
sink-rooted view per sink SCC. Layers use longest distance in the condensation
DAG and provide ordinal upstream-to-downstream position, not measured time.
Context can attach to a backbone view but cannot expand it. No event is
deduplicated by text.

Identity is explicit:

- `graph_id`: hash of the relative source path together with the canonical graph JSON content hash;
- `view_id`: graph plus sorted sink-node signature;
- `record_id`: graph plus view;
- `base_sample_id`: record plus observed-prefix length;
- `sample_id`: base sample plus prompt profile.

The record JSONL keeps organism, pathway, graph/view/event IDs and source
paths. Those are provenance metadata, not fields the model must generate.

## Model-visible contract

The question shows a complete parseable example of the required JSON shape,
the observed structured layers, and—under the primary profile—the known KEGG
organism code. It contains no pathway name, class, ID, block, title, or
phenotype field.

Three prompt conditions are released:

- `explicit_organism_source_native_ids` (P0): primary training/evaluation;
- `no_explicit_organism_source_native_ids` (P1): exact paired control, but
  native IDs can still reveal species;
- `species_neutral_ids_no_organism` (P2): exact natural-neutral subset using
  already-neutral KO/compound/glycan/reaction/EC IDs. Prefix stripping is never
  treated as a mapping.

The closed target is:

```json
{
  "schema_version": "pathway_continuation_v3",
  "remaining_layers": [
    {
      "layer_index": 2,
      "events": [
        {
          "source": [{"canonical_id": "ko:K00001", "name": "A"}],
          "relation": "activation",
          "target": [{"canonical_id": "ko:K00002", "name": "B"}],
          "text": "A activates B."
        }
      ]
    }
  ]
}
```

Phenotype is currently disabled. `phenotype_status=not_annotated` exists only
in metadata and does not mean a negative phenotype.

## Split and size policy

- Five partitions are fixed: train, validation, strict test, family-only test,
  and organism-only test. Strict test holds out both organism and five-digit
  family; the two diagnostic tests isolate those factors.
- Source graph, graph, view, record, and base-sample identities are disjoint
  across every partition.
- Selection is deterministic and organism-first. At least one train-assigned
  graph per available training organism bypasses fractional sampling, then
  records are added by organism round-robin. Each family remains capped at 256.
- At most three evenly spaced prefix rows are stored per train record; each
  training epoch chooses one deterministically.
- The default release targets 12,000–18,000 accepted train records and at most
  about 36 million input tokens per epoch. The measured four-A100 baseline and
  a 12-epoch ceiling target about 60 hours and reject an estimate above 72
  hours; the first full v3 epoch replaces this estimate with observed speed.

## Token and JSON policy

The real Qwen tokenizer measures the complete chat prompt plus the complete
assistant answer and `<|im_end|>`. A row above 8192 tokens is excluded before
writing the release. Training uses the same check and raises if an oversized
row slips through; assistant JSON is never truncated.

Inference performs at most three attempts. Invalid or unclosed output is
regenerated with an explicit repair turn and larger token budget. The third
failure is written to progress JSONL and raises an error.

## Parallel archive transfer to CFFF

Use the shard-level uploader for the 16 independent `processed_graph`
archives:

```bash
bash dataprocess/upload_cfff_archives_parallel.sh \
  /private/tmp/chatpathway_drive_to_cfff_stage \
  lihaorui@10.193.2.99 \
  /cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui/KEGG_all_new_processed_graph_archives_20260713 \
  4 30456 /private/tmp/chatpathway_cfff_20260713.sock
```

The script transfers only artifacts listed in `SHA256SUMS`. It skips a remote
artifact only after hashing it, resumes `.incoming.<name>` with SFTP `reput`,
and atomically exposes the final name only after remote SHA-256 verification.
Passwords, tokens, private keys, and `rclone.conf` must never enter the script
or its logs.

## Build on CFFF

After `processed_graph` and the pinned Qwen tokenizer are present:

```bash
export CHATPATHWAY_PROFILE=cfff
python -m experiments.run_experiment prepare-structured-data \
  --max-records-per-family 256 \
  --minimum-train-records 12000 \
  --seed 20260711 \
  --overwrite
```

The release is written to `data/pathway_v3_cap256/`:

- train/validation/test CSV compatibility views;
- five primary P0 CSVs and five one-record-per-line JSONLs;
- five exact P1 control CSVs and five strict-natural P2 subset CSVs;
- `source_graph_hashes.jsonl` covering every referenced source artifact;
- `dataset_manifest.json`;
- generated read-only `data_audit.json`.

`data_audit.json` contains row, record, source, family, and per-organism counts;
the full five-way overlap contract; duplicate ID checks; phenotype/parser
status; event, layer, token, truncation, and graph-artifact coverage; all file
and source hashes; exact P0/P1 pairing; and P2 natural-neutral eligibility. The
CFFF scheduler recomputes these checks before allocating GPUs.

## Historical builders

`build_pathway_csv.py`, `prepare_experiment_data.py`, and
`select_training_coverage.py` remain for reproducing the v2 paragraph-based
corpus. Their `sentence_parser_v1` boundaries and phenotype ambiguity policy
must not be presented as the v3 canonical event release.
