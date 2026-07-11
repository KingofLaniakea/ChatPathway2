"""Runtime path configuration loaded from the repository root config file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CONFIG_ENV = "CHATPATHWAY_CONFIG"
PROFILE_ENV = "CHATPATHWAY_PROFILE"
ASSET_ROOT_ENV = "CHATPATHWAY_ASSET_ROOT"
DEFAULT_PROFILE = "autodl"
DEFAULT_ASSET_ROOT = "/root/autodl-tmp"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path() -> Path:
    configured = os.environ.get(CONFIG_ENV)
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else repo_root() / path
    return repo_root() / "chatpathway.config.json"


def load_runtime_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {
            "active_profile": DEFAULT_PROFILE,
            "profiles": {
                DEFAULT_PROFILE: {
                    "asset_root": DEFAULT_ASSET_ROOT,
                    "models_dir": "models",
                    "data_dir": "data",
                    "checkpoints_dir": "checkpoints",
                    "runs_dir": "runs",
                    "artifacts_dir": "artifacts",
                }
            },
        }
    return json.loads(path.read_text(encoding="utf-8"))


def active_profile(config: dict[str, Any] | None = None) -> str:
    config = config or load_runtime_config()
    return os.environ.get(PROFILE_ENV) or config.get("active_profile") or DEFAULT_PROFILE


def profile_config(profile: str | None = None) -> dict[str, Any]:
    config = load_runtime_config()
    profile = profile or active_profile(config)
    profiles = config.get("profiles", {})
    if profile not in profiles:
        available = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(f"Unknown ChatPathway profile '{profile}'. Available profiles: {available}")
    return dict(profiles[profile])


def resolve_configured_path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else repo_root() / path


def asset_root(profile: str | None = None) -> Path:
    overridden = os.environ.get(ASSET_ROOT_ENV)
    if overridden:
        return resolve_configured_path(overridden)
    profile_data = profile_config(profile)
    return resolve_configured_path(profile_data.get("asset_root", DEFAULT_ASSET_ROOT))


def configured_dir(key: str, profile: str | None = None) -> Path:
    profile_data = profile_config(profile)
    relative = profile_data.get(key)
    if relative is None:
        raise KeyError(f"Runtime profile does not define '{key}'.")
    path = Path(os.path.expandvars(relative)).expanduser()
    return path if path.is_absolute() else asset_root(profile) / path


def asset_path(relative_path: str, profile: str | None = None) -> str:
    path = Path(os.path.expandvars(relative_path)).expanduser()
    if path.is_absolute():
        return str(path)
    return str(asset_root(profile) / path)
