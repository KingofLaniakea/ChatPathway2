"""Run the full training pipeline for this experiment."""

from experiments._launch import asset_path, run_steps


if __name__ == "__main__":
    run_steps([
        ('method.training.sft', ['--base-model', asset_path('models/qwen3_8B'), '--train', asset_path('data/train_11_species_dataset.csv'), '--save-dir', asset_path('checkpoints/experiments/exp004_neuralode_reconae_joint_direct/sft')]),
        ('method.training.latent_ae', ['--base-model', asset_path('models/qwen3_8B'), '--sft-lora', asset_path('checkpoints/experiments/exp004_neuralode_reconae_joint_direct/sft/checkpoint_epoch_5'), '--train', asset_path('data/train_11_species_dataset.csv'), '--save-dir', asset_path('checkpoints/experiments/exp004_neuralode_reconae_joint_direct/ae')]),
        ('method.training.joint_lora_dynamics', ['--variant', 'neural_ode', '--base-model', asset_path('models/qwen3_8B'), '--sft-adapter', asset_path('checkpoints/experiments/exp004_neuralode_reconae_joint_direct/sft/checkpoint_epoch_5'), '--ae-ckpt', asset_path('checkpoints/experiments/exp004_neuralode_reconae_joint_direct/ae/ae_epoch_5/ae_proj.pt'), '--train', asset_path('data/train_11_species_dataset.csv'), '--save-dir', asset_path('checkpoints/experiments/exp004_neuralode_reconae_joint_direct/final_lora')]),
    ])
