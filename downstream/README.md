# ChatPathway downstream task suite

The suite follows the task definitions in `chatpathway_downstream_tasks` and
keeps model generation separate from metric calculation.

| Task | Entry point | Input contract | Result |
| --- | --- | --- | --- |
| I PCER + II entity consistency | `task1_2.py` | inference CSV plus external pathway-to-entity mapping | Hit@1/3/5, MRR, precision/recall/F1, hallucination/omission counts |
| III PCTE | `task3_pcte.py` | precomputed paired latents, or prediction CSV + base model + AE | DTW-aligned cosine/Euclidean PCTE |
| IV CSP | `task4_csp.py` | inference CSV | reactant, reaction, exact-step, parse-validity scores |
| V CKI | `task5_cki.py` | intervention records with calibrated WT/KO survival scores | CSR, GateAcc, JSD, SLM |
| VI perturbed cell | `task6_perturbed_cell.py` | NPZ: `control`, `observed`, `predicted` `[cells, genes]` | expression/delta Pearson and Spearman, Top-K DE delta correlation |

Run the no-network synthetic verification from the repository root:

```bash
python -m downstream.smoke_test
```

After the current server inference completes, evaluate Tasks I, II, and IV
directly from its result CSV. Use an external KEGG/GO/Enrichr-derived reference
for a reportable PCER number; omitting `--reference` is supported only as a
closed-corpus parser/debug check.

```bash
python -m downstream.task1_2 \
  --input /root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv \
  --reference /root/autodl-tmp/data/pathway_reference.json \
  --output-dir /root/autodl-tmp/runs/downstream/task1_2/frameworka_1_epoch4

python -m downstream.task4_csp \
  --input /root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv \
  --output-dir /root/autodl-tmp/runs/downstream/task4/frameworka_1_epoch4
```

PCTE does **not** run the HNN. It compares the generated and gold answer text
trajectories under a shared backbone and trained AE. HNN vector-field
self-consistency is a separate diagnostic and must not be substituted for
PCTE. CKI also needs a curated counterfactual/gate dataset plus a calibrated
phenotype scorer, while Task VI needs a model-to-expression decoder in a shared
gene space. The repository now contains their evaluators and explicit data
contracts, but does not manufacture unavailable biological ground truth.
