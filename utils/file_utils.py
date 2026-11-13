from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIRS = tuple(
    Path(directory)
    for directory in (
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
        "runs/detect",
    )
)


def ensure_project_directories() -> None:
    for directory in PROJECT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> Any:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_environment_config() -> dict[str, Any]:
    """Load config với environment-specific override.

    Thứ tự ưu tiên:
    1. config/settings.yaml (base)
    2. config/{environment}.yaml (override theo môi trường)

    Environment được xác định bởi biến môi trường ONCOVISION_ENV
    (mặc định: 'development').

    Returns:
        Merged config dict.
    """
    base = load_yaml("config/settings.yaml") or {}
    env = os.environ.get("ONCOVISION_ENV", "development").lower()

    env_file = Path(f"config/{env}.yaml")
    if env_file.exists():
        override = load_yaml(env_file) or {}
        # Deep merge override vào base
        base = _deep_merge(base, override)

    return base


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge hai dict, override ưu tiên hơn base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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


load_yaml_cached.cache_clear = _load_yaml_cached.cache_clear  # type: ignore[attr-defined]


def save_yaml(path: str | Path, data: Any) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    if resolved_path.exists():
        try:
            if resolved_path.read_text(encoding="utf-8") == serialized:
                return
        except Exception:
            pass
    with resolved_path.open("w", encoding="utf-8") as file:
        file.write(serialized)
    _load_yaml_cached.cache_clear()
