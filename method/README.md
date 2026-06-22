# ChatPathway method

| Directory | Contents |
| --- | --- |
| `training/` | Current SFT, latent-AE, and FrameworkA training entry points |
| `training/legacy/` | Historical distributed HNN/SFT variants, kept for provenance only |
| `inference/` | Pathway text generation entry points |

The current baseline command is:

```bash
python -m method.inference.pathway
```

It is direct LoRA generation. It does not load or execute the separately saved
HNN or AE checkpoint at runtime. See `docs/METHOD_PROVENANCE.md` for the
training and inference boundary.
