# ChatPathway2 workspace guidance

## Repository and assets

- Run repository commands from the `ChatPathway2` Git root.
- Keep models, datasets, checkpoints, runs, and generated artifacts outside Git.
- Never commit credentials, `.env` files, model weights, generated CSV datasets, or checkpoints.

## CFFF runtime

- Repository: `/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui/ChatPathway2`.
- Asset root: `/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui`.
- The host provides Python 3.10.12, PyTorch 2.3.0+cu121, and four NVIDIA A100-SXM4-80GB GPUs.
- Activate `/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui/ChatPathway2/.venv` for project commands.
- Source `/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui/codex-env.sh` when the CFFF network proxy or Codex runtime is required. This file is server-local and must not be committed.
- Set `CHATPATHWAY_PROFILE=cfff` or `CHATPATHWAY_ASSET_ROOT` explicitly for runtime commands.

## Current data state

- The 16 `KEGG_all_new_processed` archives passed SHA256 verification on 2026-07-11.
- All 16 archive sentinels are already present under `KEGG_all_new/processed`; do not re-extract them without a concrete reason.
- `KEGG_all_new/processed_graph` is not present. Do not claim phenotype supervision from this dataset until the phenotype source and schema are resolved.

## Verification

Run lightweight checks before committing changes:

```bash
python -m downstream.tests.smoke_test
python -m experiments.validate_matrix
python -m experiments.run_experiment audit
python -m experiments.run_experiment consistency --phase both --quiet
```
