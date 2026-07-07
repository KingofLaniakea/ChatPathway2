# ChatPathway method

| Directory | Contents |
| --- | --- |
| `training/` | Current SFT, latent-AE, FrameworkA, PHNN prototype, and LeJEPA probe training entry points |
| `training/legacy/` | Historical distributed HNN/SFT variants, kept for provenance only |
| `inference/` | Pathway text generation and method-probe inference entry points |
| `dynamics/` | Shared latent-dynamics teacher models and rollout utilities |

The current baseline command is:

```bash
python -m method.inference.pathway
```

It is direct LoRA generation. It does not load or execute the separately saved
HNN or AE checkpoint at runtime. See `docs/METHOD_PROVENANCE.md` for the
training and inference boundary.

For the full run order and source boundary, see `docs/WORKFLOW.md` and
`docs/CODE_PROVENANCE.md`. The PHNN prototype is documented in
`docs/PHNN_TRAINING_DESIGN.md`.
