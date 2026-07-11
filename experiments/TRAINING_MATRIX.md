# Training matrix

```mermaid
flowchart TD
  data["record-balanced pathway pilot"] --> sft["shared stage-1 SFT"]
  sft --> ae["shared reconstruction AE"]
  sft --> base["stage-1 direct baseline"]
  sft --> s2["compute-matched stage-2 SFT only"]
  ae --> s2
  sft --> hnn["stage-2 SFT + J grad H"]
  ae --> hnn
  sft --> fdh["stage-2 SFT + (J-rI) grad H + F(t)"]
  ae --> fdh
  base --> direct["direct greedy JSON inference"]
  s2 --> direct
  hnn --> direct
  fdh --> direct
```

Each seed has its own `checkpoints/seeds/<seed>` and `runs/seeds/<seed>` tree.
Within that tree, the three stage-2 arms share the same SFT/AE artifacts, data,
seed, epoch schedule, validation grouping, and LoRA optimizer settings.

The matrix does not contain token-level rollout/mixed inference. Dynamics is
trained at graph-layer resolution, so advancing it per generated token would
change the unit under study. PHNN and Neural ODE are deferred axes.
