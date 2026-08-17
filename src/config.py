"""Project paths and parameter loading.

All scripts read their settings from `params.yaml` through `load_params()` so the
DVC pipeline, the training runs and the API stay in sync.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "docs"

CLASS_NAMES = ["cats", "dogs"]

# ImageNet statistics -- used by both the custom CNN and the transfer-learning model
# so that a single preprocessing path serves every checkpoint.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


@functools.lru_cache(maxsize=1)
def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load `params.yaml` once and cache it."""
    params_path = Path(path) if path else PARAMS_PATH
    with open(params_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
