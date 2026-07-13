# Pathway-continuation v3 dataset

The active builder is `build_structured_dataset.py`. It reads canonical
`processed_graph/<organism>/<pathway>.json` directly; it does not recover
events by splitting the concatenated paragraphs under `processed/`.

## Biological record

For each graph, every relation and reaction with resolvable endpoints becomes
one stable structured event. Topology is built from all such events, including
events that the historical producer marked `renderable=false`; that flag is
kept for provenance and a deterministic generic renderer supplies text. A
graph with any missing event endpoint is excluded rather than partially
materialized.

The builder condenses cycles with Tarjan SCC and creates one sink-rooted view
per sink SCC. Layers are ordered graph distance from upstream to downstream;
events in one layer are an unordered set, not measured time. No event is
deduplicated by text.

Identity is explicit:

- `graph_id`: hash of the relative source path together with the canonical graph JSON content hash;
- `view_id`: graph plus sorted sink-node signature;
- `record_id`: graph plus view;
- `sample_id`: record plus observed-prefix length.

The record JSONL keeps organism, pathway, graph/view/event IDs and source
paths. Those are provenance metadata, not fields the model must generate.

## Model-visible contract

The question contains only the task instructions, exact JSON shape, and the
observed structured layers. It contains no explicit pathway name, class, ID,
block, title, organism, or phenotype field.

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

- Test uses selected held-out organisms and held-out five-digit KEGG pathway
  families simultaneously.
- Validation holds out separate whole families.
- Train excludes both test and validation families.
- Selection is deterministic, prioritizes distinct organisms and trajectory
  lengths within each family, and defaults to at most 256 records per family.
- At most three evenly spaced prefix rows are stored per train record; each
  training epoch chooses one deterministically.
- The default build fails below 12,000 accepted train records. The first full
  v3 timing, not the old v2 row count, determines the final one-day budget.

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
- one-record-per-line JSONL for all three splits;
- `dataset_manifest.json`;
- generated read-only `data_audit.json`.

`data_audit.json` contains row, record, source and family counts; strict
source/record/sample/family overlap; organism overlap; duplicate ID checks;
phenotype and parser status; event coverage; layer-length and token-length
distributions; truncation exclusions; and graph-artifact coverage. The CFFF
experiment scheduler verifies its pass status, read-only mode, manifest hash,
and all split hashes before allocating GPUs.

## Historical builders

`build_pathway_csv.py`, `prepare_experiment_data.py`, and
`select_training_coverage.py` remain for reproducing the v2 paragraph-based
corpus. Their `sentence_parser_v1` boundaries and phenotype ambiguity policy
must not be presented as the v3 canonical event release.
