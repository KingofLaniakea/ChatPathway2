# Supporting scripts

The repository root intentionally contains no loose Python scripts. Supporting
workflows are grouped here:

| Directory | Scope |
| --- | --- |
| `data/` | source-data download and preparation helpers |
| `model/` | model acquisition helpers |
| `c2s/prep/` | Cell2Sentence data preparation and split scripts |
| `c2s/train/` | C2S LoRA training variants |
| `c2s/eval/` | C2S and Gemma evaluation variants |
| `inference/` | one-off generation and zero-shot scripts |
| `analysis/` | plotting and exploratory analysis |

The numbered and `_1` variants are preserved verbatim as historical variants;
their names do not imply an endorsed baseline. The current baseline inference
entry point is `method/inference/pathway.py`, and downstream evaluation entry points
are under `downstream/`.

All path references use the server's canonical layout:
`/root/autodl-tmp/{models,data,checkpoints,runs}`. The merged Stage-3 model
path `models/qwen3_8b_stage3_full_merged` is a build target of
`c2s/prep/06_merge_stage3.py`, not an asset currently present on the server.
