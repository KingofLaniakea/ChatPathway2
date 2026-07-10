#!/usr/bin/env python3
"""Build ChatPathway2 supervised CSVs from processed KEGG pathway JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

try:
    from dataprocess.schemas import (
        CSV_FIELDNAMES,
        PathwayExample,
        PathwayRecord,
        PathwayStep,
        PhenotypeTarget,
    )
except ImportError:  # Allows: python dataprocess/build_pathway_csv.py
    from schemas import (  # type: ignore
        CSV_FIELDNAMES,
        PathwayExample,
        PathwayRecord,
        PathwayStep,
        PhenotypeTarget,
    )


DEFAULT_PROCESSED_ROOT = "../KEGG_all_new/processed"
DEFAULT_PROCESSED_GRAPH_ROOT = "../KEGG_all_new/processed_graph"
DEFAULT_OUTPUT = ""
DEFAULT_TRAIN_OUTPUT = "../data/train_kegg_pathway_dataset.csv"
DEFAULT_TEST_OUTPUT = "../data/test_kegg_pathway_dataset.csv"

LAYER_RE = re.compile(r"^layer\s+(-?\d+)$", re.IGNORECASE)
PATHWAY_BLOCK_RE = re.compile(r"^pathway\s+(\d+)$", re.IGNORECASE)

PHENOTYPE_KEYS = {
    "phenotype",
    "phenotypes",
    "phenotype_text",
    "phenotype_name",
    "phenotype_names",
    "phenotype_description",
    "phenotype_descriptions",
    "phenotype_nodes",
    "phenotype_edges",
}

TEXT_KEYS = (
    "text",
    "name",
    "display_name",
    "label",
    "title",
    "description",
    "value",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ChatPathway2 CSV examples from KEGG processed JSON."
    )
    parser.add_argument(
        "--processed-root",
        default=DEFAULT_PROCESSED_ROOT,
        help="Path to processed/<organism> JSON root, relative to ChatPathway2.",
    )
    parser.add_argument(
        "--processed-graph-root",
        default=DEFAULT_PROCESSED_GRAPH_ROOT,
        help="Path to processed_graph/<organism> JSON root, relative to ChatPathway2.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="All-row CSV output path. Empty by default to avoid duplicating train/test.",
    )
    parser.add_argument(
        "--train-output",
        default=DEFAULT_TRAIN_OUTPUT,
        help="Train CSV output path. Use an empty string to disable.",
    )
    parser.add_argument(
        "--test-output",
        default=DEFAULT_TEST_OUTPUT,
        help="Test CSV output path. Use an empty string to disable.",
    )
    parser.add_argument(
        "--test-organisms",
        default="",
        help="Comma-separated organism codes assigned to test split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Hash-based grouped test fraction when --test-organisms is empty.",
    )
    parser.add_argument(
        "--split-key",
        choices=("source_json", "pathway_id", "organism"),
        default="source_json",
        help="Grouping key for hash split to avoid prefix leakage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed used in deterministic hash split.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=2,
        help="Skip pathway blocks with fewer ordered text layers.",
    )
    parser.add_argument(
        "--include-empty-prefix",
        action="store_true",
        help="Also create the i=0 example with no observed steps.",
    )
    parser.add_argument(
        "--require-phenotype",
        action="store_true",
        help="Keep only rows whose processed_graph source has phenotype supervision.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Debug limit for number of processed JSON files to scan. 0 means no limit.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Debug limit for generated examples. 0 means no limit.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many processed files. 0 disables progress.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV outputs.",
    )
    args = parser.parse_args(argv)
    if args.test_fraction < 0 or args.test_fraction > 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.min_steps < 1:
        parser.error("--min-steps must be at least 1")
    return args


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_json_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def parse_layer_number(key: str) -> Optional[int]:
    match = LAYER_RE.match(key.strip())
    if not match:
        return None
    return int(match.group(1))


def parse_pathway_block_number(key: str) -> Optional[int]:
    match = PATHWAY_BLOCK_RE.match(key.strip())
    if not match:
        return None
    return int(match.group(1))


def is_layer_mapping(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(parse_layer_number(str(key)) is not None for key in value)


def iter_pathway_blocks(data: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    if is_layer_mapping(data):
        yield "pathway 0", data
        return
    if not isinstance(data, dict):
        return

    block_items: list[tuple[int, str, dict[str, Any]]] = []
    for key, value in data.items():
        if not is_layer_mapping(value):
            continue
        block_number = parse_pathway_block_number(str(key))
        sort_index = block_number if block_number is not None else len(block_items)
        block_items.append((sort_index, str(key), value))

    for _, key, value in sorted(block_items, key=lambda item: (item[0], item[1])):
        yield key, value


def stringify_layer_items(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            else:
                text = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if text:
                items.append(text)
        return items
    if value is None:
        return []
    return [json.dumps(value, ensure_ascii=False, sort_keys=True)]


def extract_steps(layer_map: dict[str, Any]) -> list[PathwayStep]:
    indexed_layers: list[tuple[int, str, Any]] = []
    for key, value in layer_map.items():
        layer_number = parse_layer_number(str(key))
        if layer_number is None:
            continue
        indexed_layers.append((layer_number, str(key), value))

    steps = []
    for layer_number, layer_id, value in sorted(indexed_layers):
        items = stringify_layer_items(value)
        if not items:
            continue
        steps.append(
            PathwayStep(
                step_index=layer_number,
                layer_id=layer_id,
                text=" ".join(items),
                source_items=tuple(items),
            )
        )
    return steps


def find_processed_graph_file(
    processed_file: Path,
    processed_root: Path,
    processed_graph_root: Path,
) -> Optional[Path]:
    if not processed_graph_root.exists():
        return None
    try:
        relative = processed_file.relative_to(processed_root)
    except ValueError:
        relative = Path(processed_file.name)

    direct = processed_graph_root / relative
    if direct.exists():
        return direct

    candidates = [
        processed_graph_root / relative.parent / f"{processed_file.stem}_graph.json",
        processed_graph_root / relative.parent / f"{processed_file.stem}.graph.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def collect_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if value is None:
        return texts
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        for item in value:
            texts.extend(collect_texts(item))
        return texts
    if isinstance(value, dict):
        for key in TEXT_KEYS:
            if key in value:
                texts.extend(collect_texts(value[key]))
        for key in PHENOTYPE_KEYS:
            if key in value:
                texts.extend(collect_texts(value[key]))
        return texts
    return texts


def unique_nonempty(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        text = " ".join(value.split())
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def item_mentions_phenotype(item: dict[str, Any]) -> bool:
    fields = (
        "type",
        "entity_type",
        "node_type",
        "relation_type",
        "subtype",
        "category",
        "kind",
    )
    return any("phenotype" in str(item.get(field, "")).lower() for field in fields)


def extract_phenotype(graph_data: Any) -> PhenotypeTarget:
    if not isinstance(graph_data, dict):
        return PhenotypeTarget()

    texts: list[str] = []
    metadata = graph_data.get("metadata")
    if isinstance(metadata, dict):
        for key in PHENOTYPE_KEYS:
            if key in metadata:
                texts.extend(collect_texts(metadata[key]))

    for key in PHENOTYPE_KEYS:
        if key in graph_data:
            texts.extend(collect_texts(graph_data[key]))

    for container_key in ("nodes", "entries", "relations", "edges", "reactions"):
        container = graph_data.get(container_key)
        if not isinstance(container, list):
            continue
        for item in container:
            if isinstance(item, dict) and item_mentions_phenotype(item):
                texts.extend(collect_texts(item))

    unique = unique_nonempty(texts)
    if not unique:
        return PhenotypeTarget()
    return PhenotypeTarget(
        text="; ".join(unique),
        status="available",
        source="processed_graph",
    )


def extract_graph_fields(graph_data: Any) -> tuple[str, str, str, PhenotypeTarget]:
    if not isinstance(graph_data, dict):
        return "", "", "", PhenotypeTarget()

    metadata = graph_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    organism = first_string(
        metadata.get("organism"),
        metadata.get("organism_code"),
        graph_data.get("organism"),
        graph_data.get("organism_code"),
    )
    pathway_id = first_string(
        metadata.get("pathway_id"),
        metadata.get("pathway"),
        metadata.get("name"),
        graph_data.get("pathway_id"),
        graph_data.get("pathway"),
        graph_data.get("name"),
    )
    pathway_title = first_string(
        metadata.get("title"),
        metadata.get("pathway_title"),
        metadata.get("description"),
        graph_data.get("title"),
        graph_data.get("pathway_title"),
        graph_data.get("description"),
    )
    return organism, pathway_id, pathway_title, extract_phenotype(graph_data)


def default_organism(processed_file: Path, processed_root: Path) -> str:
    try:
        relative = processed_file.relative_to(processed_root)
    except ValueError:
        return ""
    if len(relative.parts) > 1:
        return relative.parts[0]
    return ""


def make_record(
    processed_file: Path,
    processed_root: Path,
    processed_graph_root: Path,
    block_name: str,
    layer_map: dict[str, Any],
) -> PathwayRecord:
    graph_file = find_processed_graph_file(
        processed_file,
        processed_root,
        processed_graph_root,
    )
    graph_organism = ""
    graph_pathway_id = ""
    graph_title = ""
    phenotype = PhenotypeTarget()
    if graph_file is not None:
        try:
            graph_organism, graph_pathway_id, graph_title, phenotype = (
                extract_graph_fields(load_json(graph_file))
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"warning: failed to read processed_graph {graph_file}: {exc}",
                file=sys.stderr,
            )

    block_number = parse_pathway_block_number(block_name)
    return PathwayRecord(
        source_json=processed_file.relative_to(processed_root).as_posix(),
        source_graph_json=(
            graph_file.relative_to(processed_graph_root).as_posix()
            if graph_file is not None
            else ""
        ),
        organism=graph_organism or default_organism(processed_file, processed_root),
        pathway_id=graph_pathway_id or processed_file.stem,
        entry_id=str(block_number) if block_number is not None else block_name,
        pathway_block=block_name,
        pathway_title=graph_title,
        steps=tuple(extract_steps(layer_map)),
        phenotype=phenotype,
    )


def prefix_lengths(step_count: int, include_empty_prefix: bool) -> range:
    start = 0 if include_empty_prefix else 1
    return range(start, step_count)


def iter_examples(
    processed_root: Path,
    processed_graph_root: Path,
    include_empty_prefix: bool,
    min_steps: int,
    require_phenotype: bool,
    max_files: int,
    max_rows: int,
    progress_every: int,
    stats: Counter,
) -> Iterator[PathwayExample]:
    row_count = 0
    for file_index, processed_file in enumerate(iter_json_files(processed_root), start=1):
        if max_files and file_index > max_files:
            break
        if progress_every and file_index % progress_every == 0:
            print(
                f"processed {file_index} files, generated {row_count} rows",
                file=sys.stderr,
            )

        stats["processed_files"] += 1
        try:
            data = load_json(processed_file)
        except (OSError, json.JSONDecodeError) as exc:
            stats["invalid_json_files"] += 1
            print(f"warning: failed to read {processed_file}: {exc}", file=sys.stderr)
            continue

        block_count = 0
        for block_name, layer_map in iter_pathway_blocks(data):
            block_count += 1
            stats["pathway_blocks"] += 1
            record = make_record(
                processed_file,
                processed_root,
                processed_graph_root,
                block_name,
                layer_map,
            )
            if len(record.steps) < min_steps:
                stats["skipped_short_blocks"] += 1
                continue
            if require_phenotype and not record.phenotype.has_target:
                stats["skipped_missing_phenotype"] += 1
                continue
            if record.phenotype.has_target:
                stats["blocks_with_phenotype"] += 1
            else:
                stats["blocks_without_phenotype"] += 1

            for prefix_len in prefix_lengths(len(record.steps), include_empty_prefix):
                if max_rows and row_count >= max_rows:
                    return
                row_count += 1
                stats["examples"] += 1
                yield PathwayExample(record=record, prefix_len=prefix_len)

        if block_count == 0:
            stats["files_without_pathway_blocks"] += 1


def parse_organism_set(value: str) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def split_for_example(
    example: PathwayExample,
    test_organisms: set[str],
    test_fraction: float,
    split_key: str,
    seed: int,
) -> str:
    if test_organisms:
        return "test" if example.record.organism in test_organisms else "train"
    if test_fraction <= 0:
        return "train"
    if split_key == "organism":
        key = example.record.organism
    elif split_key == "pathway_id":
        key = example.record.pathway_id
    else:
        key = example.record.source_json
    return "test" if stable_fraction(key, seed) < test_fraction else "train"


def requested_outputs(args: argparse.Namespace) -> dict[str, Path]:
    outputs = {}
    if args.output:
        outputs["all"] = resolve_repo_path(args.output)
    if args.train_output:
        outputs["train"] = resolve_repo_path(args.train_output)
    if args.test_output:
        outputs["test"] = resolve_repo_path(args.test_output)
    return outputs


def validate_outputs(outputs: dict[str, Path], overwrite: bool) -> None:
    for label, path in outputs.items():
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{label} output already exists: {path}. Use --overwrite to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    processed_root = resolve_repo_path(args.processed_root)
    processed_graph_root = resolve_repo_path(args.processed_graph_root)
    if not processed_root.exists():
        raise FileNotFoundError(
            "processed root does not exist: "
            f"{processed_root}. Expected the unzipped dataset next to ChatPathway2."
        )

    outputs = requested_outputs(args)
    if not outputs:
        raise ValueError("no output path requested")
    validate_outputs(outputs, args.overwrite)

    test_organisms = parse_organism_set(args.test_organisms)
    stats: Counter = Counter()

    with ExitStack() as stack:
        writers: dict[str, csv.DictWriter] = {}
        for label, path in outputs.items():
            handle = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writers[label] = writer

        for example in iter_examples(
            processed_root=processed_root,
            processed_graph_root=processed_graph_root,
            include_empty_prefix=args.include_empty_prefix,
            min_steps=args.min_steps,
            require_phenotype=args.require_phenotype,
            max_files=args.max_files,
            max_rows=args.max_rows,
            progress_every=args.progress_every,
            stats=stats,
        ):
            row = example.csv_row()
            if "all" in writers:
                writers["all"].writerow(row)
                stats["all_rows"] += 1
            split = split_for_example(
                example,
                test_organisms=test_organisms,
                test_fraction=args.test_fraction,
                split_key=args.split_key,
                seed=args.seed,
            )
            if split in writers:
                writers[split].writerow(row)
                stats[f"{split}_rows"] += 1

    summary = {
        "processed_root": str(processed_root),
        "processed_graph_root": str(processed_graph_root),
        "outputs": {label: str(path) for label, path in outputs.items()},
        "test_organisms": sorted(test_organisms),
        "test_fraction": args.test_fraction,
        "split_key": args.split_key,
        "include_empty_prefix": args.include_empty_prefix,
        "stats": dict(sorted(stats.items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
