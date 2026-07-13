"""Distributed stage-2 LoRA plus Hamiltonian-dynamics training.

This module preserves :mod:`method.training.framework_a`'s loss, checkpoint,
and validation contract while distributing training examples across one or
more GPUs.  LoRA gradients are synchronized by ``DistributedDataParallel``;
the separate dynamics module is synchronized explicitly at every optimizer
step because its public ``regularization_loss`` API is used outside
``forward``.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator, Sized

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.dynamics.hamiltonian import LatentHamiltonianDynamics
from method.training.common import (
    EarlyStopping,
    accumulation_divisor,
    append_jsonl,
    artifact_sha256,
    base_model_identity,
    configure_logger,
    ensure_new_output_dir,
    file_sha256,
    git_commit,
    seed_everything,
    write_json,
)
from method.training.framework_a import (
    CSVPathwayDataset,
    CascadeProjection,
    TrainConfig,
    batch_losses,
    fixed_time_grid,
    load_frames,
    make_collate_fn,
    parse_args,
    save_checkpoint,
    unwrap_state_dict,
)


LOSS_KEYS = ("total", "sft", "align", "state", "latent_state", "regularization")
COUNTER_KEYS = (
    "dynamics_truncated_samples",
    "dynamics_truncated_semantic_steps",
    "text_truncated_substeps",
    "text_truncated_semantic_steps",
    "prompt_tokens_dropped",
    "answer_tokens_dropped",
)


class StridedEvaluationSampler(Sampler[int]):
    """Partition evaluation rows exactly, without padding or duplication."""

    def __init__(self, data_source: Sized, *, rank: int, world_size: int) -> None:
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.size = len(data_source)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return math.ceil((self.size - self.rank) / self.world_size)


class MatchedGlobalBatchTrainSampler(Sampler[int]):
    """Give different world sizes the same shuffled global optimizer batches.

    The global permutation is padded only to the effective global batch size.
    With the maintained settings, both 4x3 and 2x6 therefore see the same 12
    examples in every optimizer update, including the same deterministic two
    padding examples for the 16,474-row formal split.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        rank: int,
        world_size: int,
        global_batch_size: int,
        seed: int,
    ) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank/world_size")
        if global_batch_size < 1 or global_batch_size % world_size:
            raise ValueError("global_batch_size must be positive and divisible by world_size")
        if len(data_source) < 1:
            raise ValueError("training dataset must not be empty")
        self.size = len(data_source)
        self.rank = rank
        self.world_size = world_size
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.epoch = 0
        self.total_size = math.ceil(self.size / global_batch_size) * global_batch_size

    @property
    def padding_rows(self) -> int:
        return self.total_size - self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def global_indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        if self.padding_rows:
            repeats = math.ceil(self.padding_rows / len(indices))
            indices.extend((indices * repeats)[: self.padding_rows])
        return indices

    def __iter__(self) -> Iterator[int]:
        indices = self.global_indices()
        return iter(indices[self.rank : self.total_size : self.world_size])

    def __len__(self) -> int:
        return self.total_size // self.world_size


def distributed_environment(cfg: TrainConfig) -> tuple[bool, int, int, int, str]:
    """Initialize torchrun state and return distributed/rank/device metadata."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process stage-2 training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = cfg.device
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(int(device.split(":", 1)[1]) if ":" in device else 0)
    return distributed, rank, local_rank, world_size, device


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def effective_global_batch_size(cfg: TrainConfig, world_size: int) -> int:
    return cfg.batch_size * cfg.gradient_accumulation_steps * world_size


def _dataset(
    frame: Any,
    tokenizer: Any,
    cfg: TrainConfig,
    *,
    training: bool,
) -> CSVPathwayDataset:
    """Construct either the original or record-prefix-aware dataset API.

    The prefix-aware pipeline is being developed independently.  Supporting
    its additive constructor arguments here prevents the DDP implementation
    from coupling its commit to that separate worktree change.
    """

    parameters = inspect.signature(CSVPathwayDataset).parameters
    positional: list[Any] = [frame, tokenizer, cfg.max_length, cfg.answer_budget_fraction]
    if "prefix_sampling" in parameters:
        sampling = getattr(cfg, "prefix_sampling", "all_rows") if training else (
            "one_per_record" if "record_id" in frame.columns else "all_rows"
        )
        policy = getattr(cfg, "prefix_policy", "dynamics_cycle") if training else "balanced_cycle"
        positional.extend((sampling, policy, cfg.seed))
    return CSVPathwayDataset(*positional)


def _set_dataset_epoch(dataset: Any, epoch: int) -> None:
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(epoch)


def _broadcast_module(module: nn.Module, *, distributed: bool) -> None:
    if not distributed:
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=0)


def _average_dynamics_gradients(
    dynamics: nn.Module,
    *,
    distributed: bool,
    world_size: int,
) -> None:
    """Average dynamics gradients, treating a missing rank gradient as zero."""

    if not distributed:
        return
    for parameter in dynamics.parameters():
        present = torch.tensor(
            1 if parameter.grad is not None else 0,
            dtype=torch.int64,
            device=parameter.device,
        )
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        if int(present.item()) == 0:
            continue
        gradient = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(world_size)
        if parameter.grad is None:
            parameter.grad = gradient


def _reduce_metrics(
    loss_sums: dict[str, float],
    counters: dict[str, int],
    examples: int,
    *,
    device: str,
    distributed: bool,
) -> dict[str, float | int]:
    loss_values = torch.tensor(
        [*(loss_sums[key] for key in LOSS_KEYS), float(examples)],
        dtype=torch.float64,
        device=device,
    )
    counter_values = torch.tensor(
        [counters[key] for key in COUNTER_KEYS],
        dtype=torch.int64,
        device=device,
    )
    if distributed:
        dist.all_reduce(loss_values, op=dist.ReduceOp.SUM)
        dist.all_reduce(counter_values, op=dist.ReduceOp.SUM)
    denominator = max(float(loss_values[-1].item()), 1.0)
    return {
        **{key: float(loss_values[index].item() / denominator) for index, key in enumerate(LOSS_KEYS)},
        "examples": int(loss_values[-1].item()),
        **{key: int(counter_values[index].item()) for index, key in enumerate(COUNTER_KEYS)},
    }


def evaluate_distributed(
    *,
    model: nn.Module,
    projection: CascadeProjection,
    dynamics: LatentHamiltonianDynamics,
    loader: DataLoader,
    cfg: TrainConfig,
    device: str,
    time_grid: torch.Tensor,
    distributed: bool,
) -> dict[str, float | int]:
    model.eval()
    dynamics.eval()
    sums = {key: 0.0 for key in LOSS_KEYS}
    counters = {key: 0 for key in COUNTER_KEYS}
    examples = 0
    for batch in loader:
        losses = batch_losses(
            model=model,
            projection=projection,
            dynamics=dynamics,
            batch=batch,
            cfg=cfg,
            device=device,
            training=False,
            time_grid=time_grid,
        )
        batch_examples = int(batch["input_ids"].shape[0])
        for key in LOSS_KEYS:
            sums[key] += float(losses[key].detach().item()) * batch_examples  # type: ignore[union-attr]
        for key in COUNTER_KEYS:
            counters[key] += int(losses[key])
        examples += batch_examples
    return _reduce_metrics(sums, counters, examples, device=device, distributed=distributed)


def _logger(save_root: Path, variant: str, *, is_main: bool) -> logging.Logger:
    if is_main:
        return configure_logger(save_root / "train.log", f"framework_a_ddp.{variant}")
    logger = logging.getLogger(f"framework_a_ddp.{variant}.worker")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def train(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or parse_args()
    distributed, rank, local_rank, world_size, device = distributed_environment(cfg)
    cfg = replace(cfg, device=device)
    is_main = rank == 0
    seed_everything(cfg.seed + rank, deterministic=cfg.deterministic)
    save_root = Path(cfg.save_dir)

    try:
        if is_main:
            ensure_new_output_dir(save_root)
        if distributed:
            dist.barrier()
        logger = _logger(save_root, cfg.variant, is_main=is_main)
        if is_main:
            write_json(save_root / "run_config.json", asdict(cfg))

        train_frame, validation_frame = load_frames(cfg)
        if is_main:
            manifest: dict[str, Any] = {
                "git_commit": git_commit(Path(__file__).resolve().parents[2]),
                "train_path": str(Path(cfg.train_path).resolve()),
                "validation_path": (
                    str(Path(cfg.validation_path).resolve())
                    if cfg.validation_path
                    else "deterministic_group_split"
                ),
                "train_rows": len(train_frame),
                "validation_rows": len(validation_frame),
                "seed": cfg.seed,
                "validation_group_column": cfg.validation_group_column,
                "base_model_identity": base_model_identity(cfg.base_model),
                "sft_adapter_sha256": artifact_sha256(cfg.sft_lora),
                "ae_checkpoint_sha256": artifact_sha256(cfg.ae_checkpoint),
                "distributed": {
                    "backend": "nccl" if distributed else "none",
                    "world_size": world_size,
                    "batch_size_per_process": cfg.batch_size,
                    "gradient_accumulation_steps_per_process": cfg.gradient_accumulation_steps,
                    "effective_global_batch_size": effective_global_batch_size(cfg, world_size),
                    "train_sampler": "matched_global_optimizer_batches" if distributed else "random",
                    "train_sampler_padding_rows_per_epoch": (
                        math.ceil(
                            len(train_frame) / effective_global_batch_size(cfg, world_size)
                        )
                        * effective_global_batch_size(cfg, world_size)
                        - len(train_frame)
                        if distributed
                        else 0
                    ),
                    "validation_sampler": "strided_exact_no_padding",
                    "lora_gradient_reducer": "torch_ddp_mean",
                    "dynamics_gradient_reducer": "explicit_all_reduce_mean",
                },
            }
            if cfg.hash_inputs:
                manifest["train_sha256"] = file_sha256(cfg.train_path)
                if cfg.validation_path:
                    manifest["validation_sha256"] = file_sha256(cfg.validation_path)
            write_json(save_root / "run_manifest.json", manifest)
            logger.info(
                "variant=%s structure=%s train_rows=%d validation_rows=%d world_size=%d "
                "effective_global_batch_size=%d",
                cfg.variant,
                cfg.structure_mode,
                len(train_frame),
                len(validation_frame),
                world_size,
                effective_global_batch_size(cfg, world_size),
            )

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        base_model.config.use_cache = False
        if cfg.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        model: nn.Module = PeftModel.from_pretrained(
            base_model,
            cfg.sft_lora,
            is_trainable=True,
        )
        model.enable_input_require_grads()  # type: ignore[attr-defined]
        model.to(device)

        projection = CascadeProjection(
            high_dim=base_model.config.hidden_size,
            latent_dim=cfg.latent_dim,
        ).to(device).float()
        projection.load_state_dict(
            unwrap_state_dict(torch.load(cfg.ae_checkpoint, map_location=device))
        )
        projection.requires_grad_(False)
        projection.eval()

        dynamics = LatentHamiltonianDynamics(
            cfg.latent_dim,
            variant=cfg.variant,
            hidden_dim=cfg.hidden_dim,
            structure_mode=cfg.structure_mode,
            damping_mode=cfg.damping_mode,
        ).to(device).float()
        _broadcast_module(dynamics, distributed=distributed)
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        optimizer = optim.AdamW(
            [
                {
                    "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                    "lr": cfg.lr,
                },
                {"params": dynamics.parameters(), "lr": cfg.dynamics_lr},
            ]
        )

        train_dataset = _dataset(train_frame, tokenizer, cfg, training=True)
        validation_dataset = _dataset(validation_frame, tokenizer, cfg, training=False)
        train_sampler = (
            MatchedGlobalBatchTrainSampler(
                train_dataset,
                rank=rank,
                world_size=world_size,
                global_batch_size=effective_global_batch_size(cfg, world_size),
                seed=cfg.seed,
            )
            if distributed
            else None
        )
        validation_sampler = (
            StridedEvaluationSampler(validation_dataset, rank=rank, world_size=world_size)
            if distributed
            else None
        )
        collate = make_collate_fn(tokenizer.pad_token_id)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=torch.Generator().manual_seed(cfg.seed),
            collate_fn=collate,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=validation_sampler,
            collate_fn=collate,
        )
        time_grid = fixed_time_grid(cfg, device)
        early_stopping = EarlyStopping(
            cfg.early_stopping_patience,
            cfg.early_stopping_min_delta,
        )
        history: list[dict[str, Any]] = []
        stopped_early = False
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, cfg.epochs + 1):
            _set_dataset_epoch(train_dataset, epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            dynamics.train()
            sums = {key: 0.0 for key in LOSS_KEYS}
            counters = {key: 0 for key in COUNTER_KEYS}
            examples = 0
            progress = tqdm(
                train_loader,
                desc=f"{cfg.variant} DDP epoch {epoch}",
                disable=not is_main,
            )
            for step, batch in enumerate(progress):
                group_end = (
                    (step + 1) % cfg.gradient_accumulation_steps == 0
                    or step + 1 == len(train_loader)
                )
                sync_context = (
                    model.no_sync()  # type: ignore[attr-defined]
                    if distributed and not group_end
                    else nullcontext()
                )
                with sync_context:
                    losses = batch_losses(
                        model=model,
                        projection=projection,
                        dynamics=dynamics,
                        batch=batch,
                        cfg=cfg,
                        device=device,
                        training=True,
                        time_grid=time_grid,
                    )
                    divisor = accumulation_divisor(
                        step,
                        len(train_loader),
                        cfg.gradient_accumulation_steps,
                    )
                    (losses["total"] / divisor).backward()  # type: ignore[operator]

                batch_examples = int(batch["input_ids"].shape[0])
                for key in LOSS_KEYS:
                    sums[key] += float(losses[key].detach().item()) * batch_examples  # type: ignore[union-attr]
                for key in COUNTER_KEYS:
                    counters[key] += int(losses[key])
                examples += batch_examples

                if group_end:
                    _average_dynamics_gradients(
                        dynamics,
                        distributed=distributed,
                        world_size=world_size,
                    )
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if is_main:
                    progress.set_postfix(
                        sft=f"{float(losses['sft'].detach().item()):.3f}",  # type: ignore[union-attr]
                        align=f"{float(losses['align'].detach().item()):.3f}",  # type: ignore[union-attr]
                    )

            train_metrics = _reduce_metrics(
                sums,
                counters,
                examples,
                device=device,
                distributed=distributed,
            )
            validation_metrics = evaluate_distributed(
                model=unwrap_ddp(model),
                projection=projection,
                dynamics=dynamics,
                loader=validation_loader,
                cfg=cfg,
                device=device,
                time_grid=time_grid,
                distributed=distributed,
            )
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "dynamics_learning_rate": optimizer.param_groups[1]["lr"],
                "distributed_world_size": world_size,
                "effective_global_batch_size": effective_global_batch_size(cfg, world_size),
            }
            improved, should_stop = early_stopping.update(
                float(validation_metrics["total"]),
                epoch,
            )
            if is_main:
                history.append(record)
                append_jsonl(save_root / "metrics.jsonl", record)
                write_json(save_root / "history.json", history)
                logger.info(
                    "epoch=%d train_total=%.6f val_total=%.6f val_sft=%.6f "
                    "val_velocity=%.6f val_state=%.6f world_size=%d",
                    epoch,
                    train_metrics["total"],
                    validation_metrics["total"],
                    validation_metrics["sft"],
                    validation_metrics["align"],
                    validation_metrics["state"],
                    world_size,
                )
                raw_model = unwrap_ddp(model)
                save_checkpoint(
                    destination=save_root / f"checkpoint_epoch_{epoch}",
                    model=raw_model,
                    dynamics=dynamics,
                    cfg=cfg,
                    epoch=epoch,
                    metrics=record,
                )
                if improved:
                    save_checkpoint(
                        destination=save_root / "checkpoint_best",
                        model=raw_model,
                        dynamics=dynamics,
                        cfg=cfg,
                        epoch=epoch,
                        metrics=record,
                    )
                    write_json(
                        save_root / "best_checkpoint.json",
                        {
                            "epoch": epoch,
                            "monitor": "validation.total",
                            "value": validation_metrics["total"],
                            "path": str(save_root / "checkpoint_best"),
                        },
                    )
            if distributed:
                dist.barrier()
            if should_stop:
                stopped_early = True
                if is_main:
                    logger.info(
                        "early_stop epoch=%d best_epoch=%d best_validation_total=%.6f",
                        epoch,
                        early_stopping.best_epoch,
                        early_stopping.best,
                    )
                break

        if is_main:
            write_json(
                save_root / "run_complete.json",
                {
                    "status": "completed",
                    "completed_epochs": len(history),
                    "best_epoch": early_stopping.best_epoch,
                    "best_validation_total": early_stopping.best,
                    "early_stopped": stopped_early,
                    "variant": cfg.variant,
                    "structure_mode": cfg.structure_mode,
                    "distributed_world_size": world_size,
                    "effective_global_batch_size": effective_global_batch_size(cfg, world_size),
                },
            )
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train()
