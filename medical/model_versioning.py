from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelManifest:
    model_name: str
    version: str
    model_path: Path
    training_date: datetime
    dataset_hash: str
    training_config: dict[str, Any]
    metrics: dict[str, float]
    backbone: str
    num_classes: int
    image_size: int
    file_size_bytes: int
    git_commit: str


def read_model_manifest(model_path: Path) -> ModelManifest | None:
    manifest_path = model_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return ModelManifest(
        model_name=data["model_name"],
        version=data["version"],
        model_path=Path(data["model_path"]),
        training_date=datetime.fromisoformat(data["training_date"]),
        dataset_hash=data["dataset_hash"],
        training_config=data["training_config"],
        metrics=data["metrics"],
        backbone=data["backbone"],
        num_classes=data["num_classes"],
        image_size=data["image_size"],
        file_size_bytes=data["file_size_bytes"],
        git_commit=data["git_commit"],
    )
