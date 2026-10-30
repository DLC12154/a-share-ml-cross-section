"""Configuration loading with paths resolved against the project root."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = project_path(path or DEFAULT_CONFIG_PATH)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")
    return config


def ensure_output_dir(path: str | Path) -> Path:
    output = project_path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
