# Checkpoint selection

Every seed uses:

```text
checkpoints/seeds/<seed>/shared/pathway_sft/checkpoint_best
checkpoints/seeds/<seed>/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt
checkpoints/seeds/<seed>/experiments/<row>/final_lora/checkpoint_best
```

`best_checkpoint.json` records the monitored validation metric, epoch, value,
and path. SFT monitors grouped validation CE; AE monitors reconstruction loss;
stage-2 monitors the complete validation objective. Direct inference loads the
selected LoRA only. Task 0 additionally loads the selected AE and
`hamiltonian_dynamics.pt`.

A fixed `checkpoint_epoch_N` is not “best” without this selection record. The
current checkpoints do not include exact optimizer/scheduler/RNG resume state;
warm-start and exact resume are different claims.
