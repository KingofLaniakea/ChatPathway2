"""Run the full training pipeline for this experiment."""

from experiments._launch import asset_path, run_steps


if __name__ == "__main__":
    run_steps([
        ('method.training.sft', ['--base-model', asset_path('models/qwen3_8B'), '--train', asset_path('data/train_11_species_dataset.csv'), '--save-dir', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/sft')]),
        ('method.training.latent_ae', ['--base-model', asset_path('models/qwen3_8B'), '--sft-lora', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/sft/checkpoint_epoch_5'), '--train', asset_path('data/train_11_species_dataset.csv'), '--save-dir', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/ae')]),
        ('method.training.latent_dynamics_teacher', ['--variant', 'neural_ode', '--base-model', asset_path('models/qwen3_8B'), '--adapter', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/sft/checkpoint_epoch_5'), '--ae-ckpt', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/ae/ae_epoch_5/ae_proj.pt'), '--train', asset_path('data/train_11_species_dataset.csv'), '--save-dir', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/teacher')]),
        ('method.training.dynamics_distilled_lora', ['--base-model', asset_path('models/qwen3_8B'), '--sft-adapter', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/sft/checkpoint_epoch_5'), '--teacher-checkpoint', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/teacher/neural_ode/neural_ode_epoch_3.pt'), '--ae-ckpt', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/ae/ae_epoch_5/ae_proj.pt'), '--train', asset_path('data/train_11_species_dataset.csv'), '--save-dir', asset_path('checkpoints/experiments/exp003_neuralode_reconae_teacher_direct/final_lora')]),
    ])
