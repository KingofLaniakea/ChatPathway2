# Environment setup

ChatPathway2 is a Python project. Run commands from the repository root.

## Python environment

The CFFF server provides Python 3.10 and PyTorch 2.3.0+cu121 system-wide. Standard
`python -m venv` fails there because Debian `ensurepip` is unavailable, so use
`virtualenv` and reuse the CUDA-enabled system PyTorch build:

```bash
python -m pip install --user virtualenv
python -m virtualenv --system-site-packages --clear .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install optional baseline and Cell2Sentence preparation dependencies only when
those workflows are needed:

```bash
python -m pip install -r requirements-optional.txt
```

## Runtime assets

Large runtime assets live outside Git. The CFFF profile uses:

```text
/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/lihaorui
```

The processed KEGG data is already present under `KEGG_all_new/processed`.
Models and checkpoints must be placed in the configured asset-root directories
before strict runtime checks or model execution.

Select the CFFF profile before training, inference, or asset checks:

```bash
export CHATPATHWAY_PROFILE=cfff
```

`CHATPATHWAY_ASSET_ROOT` can override the configured profile root when needed.

## Verification

After activating the environment, run:

```bash
python -m downstream.tests.smoke_test
python -m experiments.validate_matrix
python -m experiments.run_experiment audit
python -m experiments.run_experiment consistency --phase both --quiet
```
