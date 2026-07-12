# Dataset status (2026-07-13)

This file separates the three dataset layers that must not be described as one
undifferentiated rebuild.

## 1. Processed JSON corpus

The server-side JSON processing report records:

- 1,368,605 processed JSON files;
- 14,466,491 pathway blocks;
- 32,294,359 generated prefix examples before CSV split/filtering;
- 8,846,196 blocks without phenotype;
- 5,620,295 short blocks skipped;
- 214,343 files without pathway blocks.

These JSON files remain the source of truth for a future exact rebuild. The
maintained builder now preserves original `source_items`, block identity, source
identity, and conservative phenotype provenance.

## 2. Full source CSVs

The current full CSVs were generated before the final builder corrections and
were **not regenerated from JSON in the 2026-07-12/13 final preparation**:

- `train_kegg_pathway_dataset.csv`: 32,258,032 prefix rows from 8,836,173
  biological records, approximately 171 GiB;
- `test_kegg_pathway_dataset.csv`: 36,327 prefix rows from 10,023 biological
  records.

They retain the old aggregate step text. Exact original `source_item`
boundaries cannot always be recovered from these CSVs, so the derived tables
mark recovered boundaries as `sentence_parser_v1`. A future JSON-to-CSV rebuild
will use the exact `layer_set_v1` boundaries.

## 3. Rebuilt experiment tables

The experiment tables **were regenerated with overwrite** from the full source
CSVs. They add stable identity fields, canonical pathway family, structured
substeps, explicit phenotype status, and strict split policies.

The first-round formal training file is:

`data/train_kegg_pathway_record_balanced_0p1pct.csv`

It contains:

- 17,416 prefix rows;
- 7,914 selected biological records;
- 325 pathway families;
- a deterministic 0.1% record-hash sample after removing strict held-out
  families;
- at most three evenly spaced prefixes per biological record.

The current server file is a byte-identical clearly named copy of the originally
generated derived file. SHA-256:

`94082cc155a3bc494d0f462f47528268b05849d516b81fbfbf11a86c0d6d46e8`

It was renamed for terminology clarity; no rows or labels were changed during
the rename.

Evaluation tables:

| Table | Rows/records | Family policy |
| --- | ---: | --- |
| strict core | 764 | 16 held-out families and held-out organisms |
| strict multistep | 1,641 rows / 764 records | same 16 held-out families |
| organism transfer | 10,023 | all seven held-out organisms; overlap reported |
| organism transfer multistep | 22,588 rows / 10,023 records | same organism-transfer policy |

For the strict core split, source, record, sample, and pathway-family overlap
with training are all zero. The organism-transfer split intentionally retains
all held-out organisms and has 139 pathway families overlapping the current
training subset; it is reported as transfer analysis, not the leakage-free core
score.

Stable identities are:

- `record_id = hash(organism, source_json, pathway_id, pathway_block)`;
- `sample_id = record_id:prefix=<prefix_step_count>`;
- `entry_id` remains only the local block number and is not globally unique.

Inference preserves these identity/source fields as strings, including leading
zeros in `pathway_family_id`.

## Phenotype policy for the current experiment

Every row in the prepared train and evaluation tables is currently
`phenotype_status=not_annotated`. Phenotype is therefore outside the current
SFT/AE/HNN experiment and is not imputed, treated as negative, or used as a
training objective. Task 4 later uses a separate, experimentally grounded
intervention/phenotype dataset.

## Biological sequence and model coordinates

- Different graph layers provide an upstream-to-downstream ordinal sequence.
- The ordinal sequence does not provide time units or unequal biological time
  intervals.
- Same-layer atomic events are treated as a parallel, permutation-invariant
  set unless independent ordering evidence exists.
- Text `max_length=8192` is a prompt-plus-answer token budget, not a dynamics
  step count.
- Dynamics uses at most 128 graph-layer advances with normalized
  `dynamics_dt=1/128`.

At the 8192-token budget, the current training set retains 99.17% of graph-layer
targets and 91.07% of parsed atomic events. Thirty-one of 17,416 rows retain no
complete graph layer and contribute SFT cross-entropy only, not a dynamics
loss. A short atomic sentence remains a valid target as long as one complete
event span is present; short token length increases variance but does not make
the HNN loss undefined.

## Why formal Stage-1 does not use all 32,258,032 rows directly

The full table contains every prefix and therefore weights long biological
records many times more heavily than short ones. Training directly on every row
would confound biological coverage with prefix count and would take months per
epoch at the measured 8192-token throughput. The first formal Stage-1 run uses
all 17,416 rows of the record-balanced 0.1% set. Its validation-selected result
is the gate for a later record-balanced scale-up; the 32-million-row prefix
table is not called the formal training target.
