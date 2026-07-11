#!/usr/bin/env python3
"""Download and verify the pinned Qwen3-8B base snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.runtime_config import asset_path


DEFAULT_REPO = "Qwen/Qwen3-8B"
# Latest verified upstream commit visible when this workflow was frozen.
DEFAULT_REVISION = "b968826"


def required_weight_files(target: Path) -> list[Path]:
    index_path = target / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")
        return [target / name for name in sorted(set(weight_map.values()))]
    single = target / "model.safetensors"
    return [single]


def verify_snapshot(target: Path) -> dict[str, Any]:
    required = [
        target / "config.json",
        target / "tokenizer_config.json",
    ]
    tokenizer_candidates = [target / "tokenizer.json", target / "tokenizer.model"]
    if not any(path.is_file() and path.stat().st_size > 0 for path in tokenizer_candidates):
        raise FileNotFoundError("Snapshot has neither a non-empty tokenizer.json nor tokenizer.model")
    weights = required_weight_files(target)
    required.extend(weights)
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError("Missing/empty Qwen snapshot files:\n" + "\n".join(missing))
    return {
        "target": str(target.resolve()),
        "weight_files": [path.name for path in weights],
        "weight_bytes": sum(path.stat().st_size for path in weights),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--target", default=asset_path("models/qwen3_8B"))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--endpoint", help="Optional Hugging Face endpoint/mirror for this process.")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(args.target).expanduser().resolve()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    manifest_path = target / "chatpathway_download_manifest.json"
    resolved_revision = args.revision
    if args.verify_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"verify-only requires the original download manifest: {manifest_path}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("repo_id") != args.repo_id:
            raise ValueError(
                f"Downloaded repo_id={existing.get('repo_id')!r} does not match {args.repo_id!r}"
            )
        existing_revision = str(existing.get("resolved_revision", ""))
        if not existing_revision or not existing_revision.startswith(args.revision):
            raise ValueError(
                f"Downloaded revision {existing_revision!r} does not match requested {args.revision!r}"
            )
        resolved_revision = existing_revision
    else:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi(endpoint=args.endpoint).model_info(args.repo_id, revision=args.revision)
        resolved_revision = info.sha
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo_id,
            revision=resolved_revision,
            local_dir=target,
            repo_type="model",
            max_workers=args.max_workers,
        )

    verification = verify_snapshot(target)
    manifest = {
        "format_version": 1,
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        **verification,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
