from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIRS = tuple(
    Path(directory)
    for directory in (
        "dataset/raw/images",
        "dataset/raw/labels",
        "dataset/processed/images/train",
        "dataset/processed/images/val",
        "dataset/processed/images/test",
        "dataset/processed/labels/train",
        "dataset/processed/labels/val",
        "dataset/processed/labels/test",
        "models/pretrained",
        "models/trained",
        "models/exported",
        "output/screenshots",
        "output/videos",
        "output/logs",
        "runs/train",
        "runs/detect",
        "runs/val",
    )
)


def ensure_project_directories() -> None:
    for directory in PROJECT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> Any:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@lru_cache(maxsize=16)
def _load_yaml_cached(resolved_path: str) -> Any:
    return load_yaml(resolved_path)


def load_yaml_cached(path: str | Path) -> Any:
    return _load_yaml_cached(str(Path(path)))


load_yaml_cached.cache_clear = _load_yaml_cached.cache_clear


def save_yaml(path: str | Path, data: Any) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    _load_yaml_cached.cache_clear()
