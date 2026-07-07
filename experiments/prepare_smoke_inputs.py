"""Create tiny CSV/JSONL inputs for AutoDL smoke runs.

The script copies the first N records from the pathway CSV and C2S JSONL files
declared by the current runtime layout. It has no torch/pandas dependency.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiments.runtime_config import asset_path


DEFAULT_PATHWAY_TRAIN = "data/train_11_species_dataset.csv"
DEFAULT_PATHWAY_TEST = "data/test_7_species_dataset.csv"
DEFAULT_PATHWAY_TRAIN_SMOKE = "data/train_11_species_dataset_smoke.csv"
DEFAULT_PATHWAY_TEST_SMOKE = "data/test_7_species_dataset_smoke.csv"

DEFAULT_C2S_TRAIN = "data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent.jsonl"
DEFAULT_C2S_TEST = "data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent.jsonl"
DEFAULT_C2S_TRAIN_SMOKE = "data/CRISPR_GSE264667_Data/jurkat_c2s_train_seen_small_5percent_smoke.jsonl"
DEFAULT_C2S_TEST_SMOKE = "data/CRISPR_GSE264667_Data/jurkat_c2s_test_unseen_small_5percent_smoke.jsonl"


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_csv_head(source: Path, target: Path, rows: int, overwrite: bool) -> int:
    ensure_writable(target, overwrite)
    with source.open(newline="", encoding="utf-8-sig") as src, target.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"{source} has no CSV header.")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            writer.writerow(row)
            count += 1
            if count >= rows:
                break
    return count


def copy_jsonl_head(source: Path, target: Path, rows: int, overwrite: bool) -> int:
    ensure_writable(target, overwrite)
    count = 0
    with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
            count += 1
            if count >= rows:
                break
    return count


def maybe_copy(kind: str, source: str, target: str, rows: int, overwrite: bool, skip_missing: bool) -> dict:
    source_path = Path(asset_path(source))
    target_path = Path(asset_path(target))
    if not source_path.exists():
        if skip_missing:
            return {"kind": kind, "source": str(source_path), "target": str(target_path), "rows": 0, "status": "missing_source_skipped"}
        raise FileNotFoundError(f"Missing source for {kind}: {source_path}")
    if kind.endswith("csv"):
        count = copy_csv_head(source_path, target_path, rows, overwrite)
    else:
        count = copy_jsonl_head(source_path, target_path, rows, overwrite)
    return {"kind": kind, "source": str(source_path), "target": str(target_path), "rows": count, "status": "ok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rows", type=int, default=2, help="Number of records to copy per source.")
    parser.add_argument("--pathway-train", default=DEFAULT_PATHWAY_TRAIN)
    parser.add_argument("--pathway-test", default=DEFAULT_PATHWAY_TEST)
    parser.add_argument("--pathway-train-output", default=DEFAULT_PATHWAY_TRAIN_SMOKE)
    parser.add_argument("--pathway-test-output", default=DEFAULT_PATHWAY_TEST_SMOKE)
    parser.add_argument("--c2s-train", default=DEFAULT_C2S_TRAIN)
    parser.add_argument("--c2s-test", default=DEFAULT_C2S_TEST)
    parser.add_argument("--c2s-train-output", default=DEFAULT_C2S_TRAIN_SMOKE)
    parser.add_argument("--c2s-test-output", default=DEFAULT_C2S_TEST_SMOKE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", help="Skip sources not present in this environment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1:
        raise SystemExit("--rows must be >= 1")
    results = [
        maybe_copy("pathway_train_csv", args.pathway_train, args.pathway_train_output, args.rows, args.overwrite, args.skip_missing),
        maybe_copy("pathway_test_csv", args.pathway_test, args.pathway_test_output, args.rows, args.overwrite, args.skip_missing),
        maybe_copy("c2s_train_jsonl", args.c2s_train, args.c2s_train_output, args.rows, args.overwrite, args.skip_missing),
        maybe_copy("c2s_test_jsonl", args.c2s_test, args.c2s_test_output, args.rows, args.overwrite, args.skip_missing),
    ]
    for result in results:
        print(
            f"{result['status']}\t{result['kind']}\trows={result['rows']}\t"
            f"{result['source']} -> {result['target']}"
        )


if __name__ == "__main__":
    main()
