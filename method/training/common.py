"""Shared reproducibility, validation, logging, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def stable_group_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if frame.empty:
        raise ValueError("cannot split an empty training table")
    if group_column not in frame.columns:
        raise ValueError(
            f"grouped validation requires column '{group_column}'; "
            "provide a leakage-safe column or an explicit --validation CSV"
        )
    groups = frame[group_column].fillna("").astype(str).str.strip()
    if (groups == "").any():
        raise ValueError(f"grouped validation column '{group_column}' contains empty identities")
    unique_groups = sorted(groups.unique())
    if len(unique_groups) < 2:
        raise ValueError(
            f"grouped validation needs at least two distinct '{group_column}' values; "
            "provide more groups or an explicit --validation CSV"
        )

    def is_validation(value: str) -> bool:
        digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
        fraction = int(digest[:16], 16) / float(16**16)
        return fraction < validation_fraction

    mask = groups.map(is_validation)
    if not mask.any() or mask.all():
        # Keep whole groups together even for small smoke datasets.
        order = sorted(
            unique_groups,
            key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
        )
        count = min(max(1, round(len(order) * validation_fraction)), len(order) - 1)
        validation_groups = set(order[:count])
        mask = groups.isin(validation_groups)
    return frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(drop=True)


def ensure_disjoint_groups(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    group_column: str,
) -> None:
    """Reject explicit validation files that share a source identity with train."""

    for name, frame in (("training", train), ("validation", validation)):
        if group_column not in frame.columns:
            raise ValueError(f"{name} CSV is missing leakage-safe group column {group_column!r}")
        values = frame[group_column].fillna("").astype(str).str.strip()
        if (values == "").any():
            raise ValueError(f"{name} CSV has empty {group_column!r} identities")
    train_groups = set(train[group_column].astype(str).str.strip())
    validation_groups = set(validation[group_column].astype(str).str.strip())
    overlap = sorted(train_groups & validation_groups)
    if overlap:
        examples = ", ".join(overlap[:5])
        raise ValueError(
            f"explicit validation leaks {len(overlap)} {group_column!r} groups from training; "
            f"examples: {examples}"
        )


def accumulation_divisor(step: int, total_steps: int, accumulation_steps: int) -> int:
    """Return the real group size, including a shorter final accumulation group."""

    group_start = (step // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_steps - group_start)


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    """Hash one file or a small checkpoint directory with relative names."""

    source = Path(path)
    if source.is_file():
        return file_sha256(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {source}")
    for item in files:
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def base_model_identity(path: str | Path) -> dict[str, str]:
    source = Path(path)
    manifest = source / "chatpathway_download_manifest.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        return {
            "repo_id": str(value.get("repo_id", "")),
            "resolved_revision": str(value.get("resolved_revision", "")),
            "download_manifest_sha256": file_sha256(manifest),
        }
    config = source / "config.json"
    return {
        "repo_id": "",
        "resolved_revision": "unrecorded",
        "config_sha256": file_sha256(config) if config.is_file() else "",
    }


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def configure_logger(log_path: str | Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def ensure_new_output_dir(path: str | Path) -> Path:
    """Create a run directory, refusing to mix with an earlier run."""

    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse non-empty training directory {destination}; "
            "choose a new --save-dir for each seed/run"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, value: float, epoch: int) -> tuple[bool, bool]:
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, should_stop
