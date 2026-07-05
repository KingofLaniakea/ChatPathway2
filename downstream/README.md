# ChatPathway downstream task suite

The suite follows the task definitions in `chatpathway_downstream_tasks` and
keeps model generation separate from metric calculation.

See [PROVENANCE.md](PROVENANCE.md) for a per-task declaration of whether code
is newly authored or migrated/reimplemented from older scripts.

| Task | Entry point | Input contract | Result |
| --- | --- | --- | --- |
| I PCER + II entity consistency | `tasks/task1_2/` | inference CSV plus external pathway-to-entity mapping | Hit@1/3/5, MRR, precision/recall/F1, hallucination/omission counts |
| III PCTE | `tasks/task3_pcte/` | precomputed paired latents, or prediction CSV + base model + AE | DTW-aligned cosine/Euclidean PCTE |
| IV CSP | `tasks/task4_csp/` | inference CSV | reactant, reaction, exact-step, parse-validity scores |
| V CKI | `tasks/task5_cki/` | intervention records with calibrated WT/KO survival scores | CSR, GateAcc, JSD, SLM |
| VI perturbed cell | `tasks/task6_perturbed_cell/` | NPZ matrices, or C2S prediction JSONL plus its training JSONL | expression/delta Pearson and Spearman, Top-K DE delta correlation |
| VII step shuffling | `tasks/task7_step_shuffling/` | explicit gold step continuations, optional base model + adapter | original-order rank, MRR, shuffle rejection, score margin |
| VIII directional reranking | `tasks/task8_directional_reranking/` | expert-validated positive/direction-negative candidate sets | directionality accuracy, wrong-direction rejection, score gap |
| IX counterfactual perturbation | `tasks/task9_counterfactual/` | NPZ paired control/predicted/target trajectories | counterfactual PCTE, endpoint error, intervention-effect cosine |

Run the no-network synthetic verification from the repository root:

```bash
python -m downstream.tests.smoke_test
```

After the current server inference completes, evaluate Tasks I, II, and IV
directly from its result CSV. Use an external KEGG/GO/Enrichr-derived reference
for a reportable PCER number; omitting `--reference` is supported only as a
closed-corpus parser/debug check.

```bash
python -m downstream.tasks.task1_2 \
  --input /root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv \
  --reference /root/autodl-tmp/data/pathway_reference.json \
  --output-dir /root/autodl-tmp/runs/downstream/task1_2/frameworka_1_epoch4

python -m downstream.tasks.task4_csp \
  --input /root/autodl-tmp/runs/inference/frameworka_1/test_7_species_frameworka_1_epoch4.csv \
  --output-dir /root/autodl-tmp/runs/downstream/task4/frameworka_1_epoch4
```

PCTE does **not** run the HNN. It compares the generated and gold answer text
trajectories under a shared backbone and trained AE. This is close to a
"feed both answers back through the model and compare trajectories" description,
but it is distinct from HNN vector-field self-consistency: no `hnn_func.pt` is
loaded, and no HNN rollout is scored. CKI also needs a curated
counterfactual/gate dataset plus a calibrated phenotype scorer. The repository
contains evaluators and explicit data contracts, but does not manufacture
unavailable biological ground truth.

Tasks VII–IX have runnable code but must not be reported before the required
corpora and labels in [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md) are supplied.
In particular, Task VIII requires expert-negative construction; Task IX needs
an intervention-conditioned model.

The existing C2S artifacts can be evaluated without regenerating them. The
vocabulary is built only from training text; it is never inferred from test
predictions. The legacy scripts used different default row counts: Qwen-C2S
used 100 test rows, while Gemma used 500. Keep row count fixed before comparing
them as a benchmark table.

```bash
python -m downstream.tasks.task6_perturbed_cell \
  --c2s-predictions /root/autodl-tmp/runs/c2s/jurkat_ours_results_epoch5.jsonl \
  --c2s-train /root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl \
  --output-dir /root/autodl-tmp/runs/downstream/task6/c2s_epoch5
```

Regenerate the Task VI prediction JSONL artifacts from the server defaults:

```bash
python -m downstream.tasks.task6_perturbed_cell.generation \
  --model qwen_c2s \
  --overwrite

python -m downstream.tasks.task6_perturbed_cell.generation \
  --model gemma \
  --overwrite
```

The default Task VI generation paths are:

| Label | Model input | Adapter | Test input | Prediction output | Legacy limit |
| --- | --- | --- | --- | --- | --- |
| Qwen-C2S | `/root/autodl-tmp/models/qwen3_8B` | `/root/autodl-tmp/checkpoints/qwen3_8b_pathway_c2s_sft_small_5percent/checkpoint_epoch_5` | `/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl` | `/root/autodl-tmp/runs/c2s/jurkat_ours_results_epoch5.jsonl` | 100 |
| Gemma C2S baseline | `/root/autodl-tmp/models/C2S-Scale-Gemma-2-2B` | none | `/root/autodl-tmp/data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl` | `/root/autodl-tmp/runs/c2s/jurkat_test_gemma_predictions_result_5percent_500.jsonl` | 500 |
