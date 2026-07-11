# Design and code provenance

The revised suite was reconciled against four project sources:

- the project core document `ChatPathway.md`;
- the 10-page `Downstream Tasks.pdf` task deck;
- the supplied Task 0-6 redesign discussion;
- the maintained evaluators under `downstream/tasks/` and shared parsers under
  `downstream/common/`.

All files in `downstream/new_tasks/` are new. Historical task code was not
modified. Reuse is explicit:

| Revised task | Reused maintained implementation | New behavior |
| --- | --- | --- |
| Task 0 | maintained `LatentHamiltonianDynamics` checkpoint schema | separate AE and rollout curves, strict held-out/checkpoint manifest, optional direct RK4 rollout |
| Task 1 | `pathway_json` multi-step adapter and conservative entity extraction | explicit atomic-substep schema, ambiguity rejection, parser coverage, layer-set metrics, provenance-gated causal order |
| Task 2 | Task III DTW distance implementation | explicit lengths and fixed-representation provenance manifest |
| Task 3 | conditional LLM scoring utility | three negative types with construction provenance, annotation provenance, HNN diagnostic/calibration boundary |
| Task 4 | metric concepts from CKI only | intervention-evidence/scorer contract, KO effect and rescue ranking, missing-label exclusion, `F(t) != u` validation |
| Task 5 | Task VI expression and delta metrics | gene/cell/perturbation alignment manifest and controlled-ablation comparison |
| Task 6 | none | strict frozen BioMaze option scorer and contamination provenance |

## Deliberate corrections to the design discussion

The implementation does not encode the following statements as facts:

- Damping alone does not guarantee convergence to a correct phenotype,
  especially with forcing.
- A reversed or shuffled text need not produce increasing total Hamiltonian,
  and increasing energy is not a validated causal-direction oracle.
- A time-only force `F(t)` does not represent a gene-specific intervention
  `u`; Task 4 uses prompt/initial-condition intervention unless a separately
  conditioned dynamics model is trained.
- Text/source-list order within one graph layer is not automatically causal;
  within-layer events are permutation-invariant unless an independent DAG or
  expert ordering artifact is supplied.
- Low latent error does not establish biological equivalence.
- Synthetic smoke cases establish software behavior only, never model quality.

These boundaries are enforced in schemas where possible and repeated in every
summary artifact so they survive outside this README.
