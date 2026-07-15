# Training matrix

```mermaid
flowchart TD
  data["audited full-index pathway v4 release"] --> sft["shared stage-1 SFT"]
  sft --> ae["shared pure-MSE reconstruction AE"]
  sft --> base["stage-1 direct baseline"]
  sft --> s2["compute-matched stage-2 SFT only"]
  ae --> s2
  sft --> hnnpre["frozen-latent HNN pretrain"]
  ae --> hnnpre
  hnnpre --> hnngate["validation stability gate"]
  hnngate --> hnn["low-LR stage-2 SFT + HNN"]
  sft --> fdhpre["frozen-latent FDHNN pretrain"]
  ae --> fdhpre
  fdhpre --> fdhgate["validation stability gate"]
  fdhgate --> fdh["low-LR stage-2 SFT + FDHNN"]
  ae --> hnnabl["random HNN direct-joint ablation"]
  ae --> fdhabl["random FDHNN direct-joint ablation"]
  base --> direct["direct greedy JSON inference"]
  s2 --> direct
  hnn --> direct
  fdh --> direct
  hnnabl --> direct
  fdhabl --> direct
```

Each seed has its own `checkpoints/seeds/<seed>` and `runs/seeds/<seed>` tree.
Within that tree, all stage-2 arms share the same SFT/AE artifacts, data,
seed, validation grouping, and registered low-LoRA-LR settings.
SFT, AE, and all stage-2 arms also share the explicit 8192-token budget
and per-process batch size 1; direct inference uses the same prompt budget.
The formal v4 release exposes one globally balanced, seed-fixed prefix per
biological record. Training and validation reuse that registered view on every
epoch, so record weight cannot change with the number of eligible prefixes.
Epoch-wise prefix rotation remains a separately registered rematerialization
study rather than a hidden source of variation in this matrix.
Validation is an explicit CSV containing entire held-out `pathway_family_id`
groups and is reused unchanged by SFT, AE, and all stage-2 arms.

The current matrix uses only direct greedy inference. Generation studies
retained for the next phase are event-boundary rollout with layer-dependent
step size, graph-layer-only rollout, token-by-token generation using a separately
trained token-resolution objective, and a multiscale hybrid. Reusing the
event/layer checkpoint as a per-token vector field is forbidden because it
changes the unit under study. PHNN and Neural ODE remain deferred axes.
