from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIRS = tuple(
    Path(directory)
    for directory in (
        "dataset/object_detection/raw/images",
        "dataset/object_detection/raw/labels",
        "dataset/object_detection/processed/images/train",
        "dataset/object_detection/processed/images/val",
        "dataset/object_detection/processed/images/test",
        "dataset/object_detection/processed/labels/train",
        "dataset/object_detection/processed/labels/val",
        "dataset/object_detection/processed/labels/test",
        "dataset/medical/skin_lesion/raw/images",
        "dataset/medical/skin_lesion/raw/labels",
        "dataset/medical/skin_lesion/processed/images/train",
        "dataset/medical/skin_lesion/processed/images/val",
        "dataset/medical/skin_lesion/processed/images/test",
        "dataset/medical/skin_lesion/processed/labels/train",
        "dataset/medical/skin_lesion/processed/labels/val",
        "dataset/medical/skin_lesion/processed/labels/test",
        "dataset/medical/skin_lesion/metadata",
        "dataset/medical/skin_lesion/reports",
        "models/pretrained",
        "models/trained",
        "models/exported",
        "output/captures",
        "output/medical/reports",
        "output/medical/normalized_images",
        "output/medical/processed_images",
        "output/recordings",
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


def yaml_mapping_issues(data: Any, *, required_keys: tuple[str, ...] = (), label: str = "yaml") -> list[str]:
    issues: list[str] = []
    if not isinstance(data, dict):
        return [f"{label} không phải dạng mapping hợp lệ."]
    for key in required_keys:
        if key not in data:
            issues.append(f"{label} thiếu trường `{key}`.")
    return issues


def yaml_file_issues(path: str | Path, *, required_keys: tuple[str, ...] = (), label: str | None = None) -> list[str]:
    resolved_path = Path(path)
    try:
        data = load_yaml(resolved_path)
    except Exception as exc:
        return [f"Không đọc được {label or resolved_path}: {exc}"]
    return yaml_mapping_issues(data, required_keys=required_keys, label=str(label or resolved_path))


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
