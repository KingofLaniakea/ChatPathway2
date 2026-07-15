"""Train the shared pathway SFT LoRA with deterministic validation and selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from method.training.common import (
    EarlyStopping,
    base_model_identity,
    accumulation_divisor,
    append_jsonl,
    configure_logger,
    ensure_disjoint_groups,
    ensure_new_output_dir,
    file_sha256,
    git_commit,
    seed_everything,
    stable_group_split,
    write_json,
)
from method.training.sequence import encode_supervised
from method.training.csv_io import read_training_csv
from method.training.prefix_sampling import (
    EpochPrefixView,
    PREFIX_POLICIES,
    PREFIX_SAMPLING_MODES,
)


DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/qwen3_8B"
DEFAULT_TRAIN = "/root/autodl-tmp/data/pathway_v4_full/train_pathway_continuation_v4.csv"
DEFAULT_SAVE = "/root/autodl-tmp/checkpoints/shared/pathway_sft"


@dataclass
class SFTConfig:
    base_model: str
    train_path: str
    validation_path: str | None
    save_dir: str
    resume_adapter: str | None
    batch_size: int
    dataloader_workers: int
    gradient_accumulation_steps: int
    lr: float
    epochs: int
    max_length: int
    answer_budget_fraction: float
    prefix_sampling: str
    prefix_policy: str
    text_column: str
    limit: int | None
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    attn_implementation: str
    gradient_checkpointing: bool
    validation_fraction: float
    validation_group_column: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    seed: int
    deterministic: bool
    hash_inputs: bool
    device: str


class SFTDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        text_column: str,
        max_length: int,
        answer_budget_fraction: float,
        prefix_sampling: str,
        prefix_policy: str,
        seed: int,
    ):
        self.prefix_view = EpochPrefixView(
            frame.to_dict(orient="records"),
            sampling_mode=prefix_sampling,
            policy=prefix_policy,
            seed=seed,
        )
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length
        self.answer_budget_fraction = answer_budget_fraction

    def __len__(self) -> int:
        return len(self.prefix_view)

    def set_epoch(self, epoch: int) -> None:
        self.prefix_view.set_epoch(epoch)

    def selection_summary(self) -> dict[str, int]:
        return self.prefix_view.selection_summary()

    def estimated_text_length(self, index: int) -> int:
        """Cheap stable proxy used only to group similar DDP sequence lengths."""

        row = self.prefix_view.row(index)
        answer = str(
            row.get(
                self.text_column,
                row.get("answer", row.get("formatted_answer_no_phenotype", "")),
            )
        )
        return len(str(row.get("question", ""))) + len(answer)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.prefix_view.row(index)
        question = str(row.get("question", ""))
        answer = str(row.get(self.text_column, row.get("answer", row.get("formatted_answer_no_phenotype", ""))))
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        encoded = encode_supervised(
            self.tokenizer,
            prompt,
            answer,
            max_length=self.max_length,
            answer_budget_fraction=self.answer_budget_fraction,
        )
        return {
            "input_ids": torch.tensor(encoded.input_ids, dtype=torch.long),
            "labels": torch.tensor(encoded.labels, dtype=torch.long),
            "prompt_tokens_dropped": encoded.prompt_tokens_dropped,
            "answer_tokens_dropped": encoded.answer_tokens_dropped,
        }


class LengthGroupedDistributedSampler(Sampler[int]):
    """Shard length-adjacent global batches across DDP ranks without loss.

    All ranks still receive the same number of samples and collectively cover
    the complete dataset.  Only the order changes: samples processed in one
    synchronized optimizer step have similar text lengths, reducing idle time
    while a rank waits for another rank's much longer sequence.
    """

    def __init__(
        self,
        dataset: SFTDataset,
        *,
        num_replicas: int,
        rank: int,
        batch_size: int,
        seed: int,
    ) -> None:
        if len(dataset) < 1:
            raise ValueError("length-grouped sampler requires a non-empty dataset")
        if num_replicas < 1 or not 0 <= rank < num_replicas or batch_size < 1:
            raise ValueError("invalid distributed sampler geometry")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.global_batch_size = num_replicas * batch_size
        self.global_batches = math.ceil(len(dataset) / self.global_batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.global_batches * self.batch_size

    def __iter__(self):
        ordered = sorted(
            range(len(self.dataset)),
            key=self.dataset.estimated_text_length,
        )
        groups: list[list[int]] = []
        for offset in range(0, len(ordered), self.global_batch_size):
            group = ordered[offset : offset + self.global_batch_size]
            if len(group) < self.global_batch_size:
                original = tuple(group)
                group.extend(
                    original[index % len(original)]
                    for index in range(self.global_batch_size - len(group))
                )
            groups.append(group)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        group_order = torch.randperm(len(groups), generator=generator).tolist()
        start = self.rank * self.batch_size
        end = start + self.batch_size
        selected = [index for group_index in group_order for index in groups[group_index][start:end]]
        return iter(selected)


def make_collate_fn(pad_id: int):
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(item["input_ids"]) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            # Do not infer padding from token values: some chat tokenizers use
            # EOS as PAD, and EOS legitimately occurs inside the prompt.
            "attention_mask": attention_mask,
            "prompt_tokens_dropped": sum(int(item["prompt_tokens_dropped"]) for item in batch),
            "answer_tokens_dropped": sum(int(item["answer_tokens_dropped"]) for item in batch),
        }

    return collate


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train", dest="train_path", default=DEFAULT_TRAIN)
    parser.add_argument("--validation", dest="validation_path")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE)
    parser.add_argument("--resume-adapter")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--answer-budget-fraction", type=float, default=0.5)
    parser.add_argument(
        "--prefix-sampling",
        choices=PREFIX_SAMPLING_MODES,
        default="one_per_record",
        help="Train every materialized prefix or one deterministic prefix per record per epoch.",
    )
    parser.add_argument(
        "--prefix-policy",
        choices=tuple(PREFIX_POLICIES),
        default="sft_cycle",
    )
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-group-column", default="pathway_family_id")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hash-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    values = vars(args)
    values["gradient_checkpointing"] = not values.pop("no_gradient_checkpointing")
    if not 0 < values["answer_budget_fraction"] < 1:
        parser.error("--answer-budget-fraction must be between 0 and 1")
    if values["dataloader_workers"] < 0:
        parser.error("--dataloader-workers must be non-negative")
    return SFTConfig(**values)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[bool, int, str]:
    distributed = is_distributed()
    if not distributed:
        return False, 0, ""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return True, local_rank, f"cuda:{local_rank}"


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def load_frames(cfg: SFTConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_training_csv(cfg.train_path)
    if cfg.limit is not None:
        frame = frame.head(cfg.limit)
    if "question" not in frame.columns:
        raise ValueError("training CSV must contain question")
    if cfg.prefix_sampling == "one_per_record" and "record_id" not in frame.columns:
        raise ValueError("one_per_record prefix sampling requires record_id")
    if cfg.validation_path:
        validation = read_training_csv(cfg.validation_path)
        if frame.empty or validation.empty:
            raise ValueError("training and explicit validation CSVs must both contain rows")
        ensure_disjoint_groups(
            frame,
            validation,
            group_column=cfg.validation_group_column,
        )
        return frame.reset_index(drop=True), validation.reset_index(drop=True)
    return stable_group_split(
        frame,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
        group_column=cfg.validation_group_column,
    )


def distributed_average(total: float, count: int, device: str, distributed: bool) -> float:
    values = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float((values[0] / values[1].clamp(min=1)).item())


def validation_metrics(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float | int]:
    model.eval()
    loss_numerator = 0.0
    supervised_tokens = input_tokens = 0
    prompt_dropped = answer_dropped = 0
    with torch.no_grad():
        for batch in loader:
            batch_supervised_tokens = int((batch["labels"] != -100).sum().item())
            outputs = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                labels=batch["labels"].to(device, non_blocking=True),
                use_cache=False,
            )
            loss_numerator += float(outputs.loss.item()) * batch_supervised_tokens
            supervised_tokens += batch_supervised_tokens
            input_tokens += int(batch["attention_mask"].sum().item())
            prompt_dropped += int(batch["prompt_tokens_dropped"])
            answer_dropped += int(batch["answer_tokens_dropped"])
    return {
        "loss_numerator": loss_numerator,
        "supervised_tokens": supervised_tokens,
        "input_tokens": input_tokens,
        "prompt_tokens_dropped": prompt_dropped,
        "answer_tokens_dropped": answer_dropped,
    }


def save_adapter(model: nn.Module, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(destination)


def train() -> None:
    cfg = parse_args()
    distributed, local_rank, ddp_device = setup_distributed()
    device = ddp_device or cfg.device
    is_main = local_rank == 0
    seed_everything(cfg.seed + local_rank, deterministic=cfg.deterministic)
    save_root = Path(cfg.save_dir)
    release_root = Path(cfg.train_path).expanduser().resolve().parent
    release_manifest_path = release_root / "dataset_manifest.json"
    release_audit_path = release_root / "data_audit.json"
    release_provenance: dict[str, Any] = {}
    if release_manifest_path.is_file():
        if not release_audit_path.is_file():
            raise FileNotFoundError(
                f"dataset manifest exists without immutable audit: {release_audit_path}"
            )
        release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        release_audit = json.loads(release_audit_path.read_text(encoding="utf-8"))
        build_id = release_manifest.get("dataset_build_id")
        if not isinstance(build_id, str) or not re.fullmatch(r"dataset:[0-9a-f]{24}", build_id):
            raise ValueError(f"invalid dataset_build_id in {release_manifest_path}")
        if release_audit.get("status") != "passed" or release_audit.get("failures"):
            raise ValueError(f"dataset audit did not pass: {release_audit_path}")
        if stat.S_IMODE(release_audit_path.stat().st_mode) != 0o444:
            raise PermissionError(f"dataset audit must be read-only 0444: {release_audit_path}")
        expected_namespace = f"{release_root.name}_{build_id.split(':', 1)[1]}"
        configured_namespace = os.environ.get("CHATPATHWAY_DATASET_NAMESPACE")
        if configured_namespace and configured_namespace != expected_namespace:
            raise ValueError(
                "checkpoint dataset namespace disagrees with dataset_manifest.json: "
                f"{configured_namespace!r} != {expected_namespace!r}"
            )
        release_provenance = {
            "dataset_build_id": build_id,
            "dataset_namespace": expected_namespace,
            "dataset_manifest_path": str(release_manifest_path),
            "dataset_manifest_sha256": file_sha256(release_manifest_path),
            "data_audit_path": str(release_audit_path),
            "data_audit_sha256": file_sha256(release_audit_path),
        }
    logger = None
    if is_main:
        ensure_new_output_dir(save_root)
        logger = configure_logger(save_root / "train.log", "pathway_sft")
        write_json(save_root / "run_config.json", asdict(cfg))

    train_frame, validation_frame = load_frames(cfg)
    if is_main:
        manifest: dict[str, Any] = {
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "train_path": str(Path(cfg.train_path).resolve()),
            "validation_path": str(Path(cfg.validation_path).resolve()) if cfg.validation_path else "deterministic_group_split",
            "train_eligible_prefix_rows": len(train_frame),
            "train_records": (
                int(train_frame["record_id"].nunique())
                if "record_id" in train_frame.columns
                else len(train_frame)
            ),
            "train_samples_per_epoch": (
                int(train_frame["record_id"].nunique())
                if cfg.prefix_sampling == "one_per_record"
                else len(train_frame)
            ),
            "validation_prefix_rows": len(validation_frame),
            "validation_records": (
                int(validation_frame["record_id"].nunique())
                if "record_id" in validation_frame.columns
                else len(validation_frame)
            ),
            "validation_samples": (
                int(validation_frame["record_id"].nunique())
                if "record_id" in validation_frame.columns
                else len(validation_frame)
            ),
            "validation_prefix_sampling": (
                "one_per_record"
                if "record_id" in validation_frame.columns
                else "all_rows"
            ),
            "validation_prefix_policy": "balanced_cycle",
            "prefix_sampling": cfg.prefix_sampling,
            "prefix_policy": cfg.prefix_policy,
            "seed": cfg.seed,
            "base_model_identity": base_model_identity(cfg.base_model),
            **release_provenance,
        }
        if cfg.hash_inputs:
            manifest["train_sha256"] = file_sha256(cfg.train_path)
            if cfg.validation_path:
                manifest["validation_sha256"] = file_sha256(cfg.validation_path)
        write_json(save_root / "run_manifest.json", manifest)
        logger.info(
            "train_eligible_prefix_rows=%d train_records=%d train_samples_per_epoch=%d "
            "validation_prefix_rows=%d validation_records=%d validation_samples=%d seed=%d",
            manifest["train_eligible_prefix_rows"],
            manifest["train_records"],
            manifest["train_samples_per_epoch"],
            manifest["validation_prefix_rows"],
            manifest["validation_records"],
            manifest["validation_samples"],
            cfg.seed,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=cfg.attn_implementation,
    )
    base_model.config.use_cache = False
    if cfg.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    if cfg.resume_adapter:
        model = PeftModel.from_pretrained(base_model, cfg.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=[item.strip() for item in cfg.target_modules.split(",") if item.strip()],
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    model.enable_input_require_grads()
    model.to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    collate = make_collate_fn(tokenizer.pad_token_id)
    train_dataset = SFTDataset(
        train_frame,
        tokenizer,
        cfg.text_column,
        cfg.max_length,
        cfg.answer_budget_fraction,
        cfg.prefix_sampling,
        cfg.prefix_policy,
        cfg.seed,
    )
    validation_dataset = SFTDataset(
        validation_frame,
        tokenizer,
        cfg.text_column,
        cfg.max_length,
        cfg.answer_budget_fraction,
        "one_per_record" if "record_id" in validation_frame.columns else "all_rows",
        "balanced_cycle",
        cfg.seed,
    )
    if is_main:
        manifest["validation_prefix_selection"] = validation_dataset.selection_summary()
        write_json(save_root / "run_manifest.json", manifest)
        logger.info(
            "validation_prefix_selection=%s",
            json.dumps(manifest["validation_prefix_selection"], sort_keys=True),
        )
    train_sampler = (
        LengthGroupedDistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=local_rank,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )
        if distributed
        else None
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    loader_options: dict[str, Any] = {
        "num_workers": cfg.dataloader_workers,
        "pin_memory": device.startswith("cuda"),
    }
    if cfg.dataloader_workers:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(cfg.seed),
        **loader_options,
    )
    validation_view: Dataset = validation_dataset
    if distributed:
        validation_view = Subset(
            validation_dataset,
            range(local_rank, len(validation_dataset), dist.get_world_size()),
        )
    validation_loader = DataLoader(
        validation_view,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
        **loader_options,
    )
    optimizer = optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=cfg.lr)
    early_stopping = EarlyStopping(cfg.early_stopping_patience, cfg.early_stopping_min_delta)
    history: list[dict[str, Any]] = []
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        train_dataset.set_epoch(epoch)
        prefix_selection = train_dataset.selection_summary()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            logger.info(
                "epoch=%d train_samples=%d prefix_selection=%s",
                epoch,
                len(train_dataset),
                json.dumps(prefix_selection, sort_keys=True),
            )
        model.train()
        train_started = time.perf_counter()
        train_total = 0.0
        train_steps = 0
        train_prompt_dropped = train_answer_dropped = 0
        train_input_tokens = train_supervised_tokens = 0
        progress = tqdm(train_loader, desc=f"SFT epoch {epoch}", disable=not is_main)
        for step, batch in enumerate(progress):
            train_input_tokens += int(batch["attention_mask"].sum().item())
            train_supervised_tokens += int((batch["labels"] != -100).sum().item())
            outputs = model(
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                labels=batch["labels"].to(device, non_blocking=True),
                use_cache=False,
            )
            divisor = accumulation_divisor(step, len(train_loader), cfg.gradient_accumulation_steps)
            (outputs.loss / divisor).backward()
            train_total += float(outputs.loss.item())
            train_steps += 1
            train_prompt_dropped += int(batch["prompt_tokens_dropped"])
            train_answer_dropped += int(batch["answer_tokens_dropped"])
            if (step + 1) % cfg.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if is_main:
                progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

        train_seconds = time.perf_counter() - train_started
        train_avg = distributed_average(train_total, train_steps, device, distributed)
        train_counts = torch.tensor(
            [
                train_prompt_dropped,
                train_answer_dropped,
                train_input_tokens,
                train_supervised_tokens,
            ],
            dtype=torch.int64,
            device=device,
        )
        if distributed:
            dist.all_reduce(train_counts, op=dist.ReduceOp.SUM)
            elapsed = torch.tensor(train_seconds, dtype=torch.float64, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            train_seconds = float(elapsed.item())
        validation_started = time.perf_counter()
        validation = validation_metrics(unwrap_model(model), validation_loader, device)
        value = torch.tensor(
            [
                validation["loss_numerator"],
                validation["supervised_tokens"],
                validation["input_tokens"],
                validation["prompt_tokens_dropped"],
                validation["answer_tokens_dropped"],
            ],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        val_avg = float((value[0] / value[1].clamp(min=1)).item())
        val_supervised_tokens = int(value[1].item())
        val_input_tokens = int(value[2].item())
        val_prompt_dropped = int(value[3].item())
        val_answer_dropped = int(value[4].item())
        validation_seconds = time.perf_counter() - validation_started
        if distributed:
            elapsed = torch.tensor(validation_seconds, dtype=torch.float64, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            validation_seconds = float(elapsed.item())
        record = {
            "epoch": epoch,
            "prefix_selection": prefix_selection,
            "train_loss": train_avg,
            "validation_loss": val_avg,
            "train_prompt_tokens_dropped": int(train_counts[0].item()),
            "train_answer_tokens_dropped": int(train_counts[1].item()),
            "train_input_tokens": int(train_counts[2].item()),
            "train_supervised_tokens": int(train_counts[3].item()),
            "train_seconds": train_seconds,
            "train_input_tokens_per_second": float(train_counts[2].item())
            / max(train_seconds, 1e-9),
            "train_supervised_tokens_per_second": float(train_counts[3].item())
            / max(train_seconds, 1e-9),
            "validation_seconds": validation_seconds,
            "validation_input_tokens": val_input_tokens,
            "validation_supervised_tokens": val_supervised_tokens,
            "validation_prompt_tokens_dropped": val_prompt_dropped,
            "validation_answer_tokens_dropped": val_answer_dropped,
            "lr": cfg.lr,
        }
        improved, should_stop = early_stopping.update(val_avg, epoch)
        if is_main:
            history.append(record)
            append_jsonl(save_root / "metrics.jsonl", record)
            write_json(save_root / "history.json", history)
            logger.info(
                "epoch=%d train_loss=%.6f validation_loss=%.6f train_seconds=%.1f validation_seconds=%.1f supervised_tokens_per_second=%.1f",
                epoch,
                train_avg,
                val_avg,
                train_seconds,
                validation_seconds,
                record["train_supervised_tokens_per_second"],
            )
            save_adapter(model, save_root / f"checkpoint_epoch_{epoch}")
            if improved:
                save_adapter(model, save_root / "checkpoint_best")
                write_json(
                    save_root / "best_checkpoint.json",
                    {"epoch": epoch, "monitor": "validation_loss", "value": val_avg, "path": str(save_root / "checkpoint_best")},
                )
        if distributed:
            dist.barrier()
        if should_stop:
            stopped_early = True
            if is_main:
                logger.info("early_stop epoch=%d best_epoch=%d best_validation_loss=%.6f", epoch, early_stopping.best_epoch, early_stopping.best)
            break

    if distributed and dist.is_initialized():
        dist.destroy_process_group()
    if is_main:
        write_json(
            save_root / "run_complete.json",
            {
                "status": "completed",
                "completed_epochs": len(history),
                "best_epoch": early_stopping.best_epoch,
                "best_validation_loss": early_stopping.best,
                "early_stopped": stopped_early,
            },
        )


if __name__ == "__main__":
    train()
