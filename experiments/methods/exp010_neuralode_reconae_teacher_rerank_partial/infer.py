"""Run inference/evaluation for this experiment."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        'method.inference.rollout_rerank',
        ['--checkpoint', asset_path('checkpoints/experiments/exp010_neuralode_reconae_teacher_rerank_partial/teacher/neural_ode/neural_ode_epoch_3.pt'), '--base-model', asset_path('models/qwen3_8B'), '--adapter', asset_path('checkpoints/experiments/exp010_neuralode_reconae_teacher_rerank_partial/sft/checkpoint_epoch_5'), '--ae-ckpt', asset_path('checkpoints/experiments/exp010_neuralode_reconae_teacher_rerank_partial/ae/ae_epoch_5/ae_proj.pt'), '--input', asset_path('runs/experiments/exp010_neuralode_reconae_teacher_rerank_partial/candidates.csv'), '--output', asset_path('runs/experiments/exp010_neuralode_reconae_teacher_rerank_partial/reranked.csv')],
    )
