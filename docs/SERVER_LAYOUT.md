# Server storage layout and legacy links

Canonical server root: `/root/autodl-tmp`.

| Location | Contents | Git status |
| --- | --- | --- |
| `ChatPathway2/` | source code and documentation only | Git repository |
| `models/` | immutable base models | not tracked |
| `data/` | datasets and reference material | not tracked |
| `checkpoints/` | LoRA, AE, HNN checkpoints | not tracked |
| `runs/` | generated outputs, logs, evaluation reports | not tracked |

`ChatPathway2` contains no model, dataset, checkpoint, run-output, or
compatibility symlink. Earlier root-level compatibility links were removed
after all tracked scripts were converted to the canonical layout above.

New output must use `runs/<experiment>/...`; new code should receive paths
through CLI arguments or a configuration file.

Every fresh SSH session must run `source /etc/network_turbo` before GitHub or
Hugging Face access. It is only an academic GitHub/Hugging Face accelerator and
can make unrelated access such as package indexes slower.
