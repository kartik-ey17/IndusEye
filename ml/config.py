"""Load training config and shared class names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

CLASS_NAMES = ["person", "knife", "gun"]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(root_relative: str | Path) -> Path:
    path = Path(root_relative)
    if path.is_absolute():
        return path
    return ROOT / path


def default_device(configured: Any = None) -> str:
    """Use the configured accelerator when available, otherwise safely use CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(configured if configured is not None else "0")
    except ImportError:
        pass
    return "cpu"
