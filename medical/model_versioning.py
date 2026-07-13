from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dcm", ".nii", ".nii.gz"})


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


def write_model_manifest(model_path: Path, manifest: ModelManifest) -> Path:
    manifest_path = model_path.with_suffix(".manifest.json")
    data = {
        "model_name": manifest.model_name,
        "version": manifest.version,
        "model_path": str(manifest.model_path),
        "training_date": manifest.training_date.isoformat(),
        "dataset_hash": manifest.dataset_hash,
        "training_config": manifest.training_config,
        "metrics": manifest.metrics,
        "backbone": manifest.backbone,
        "num_classes": manifest.num_classes,
        "image_size": manifest.image_size,
        "file_size_bytes": manifest.file_size_bytes,
        "git_commit": manifest.git_commit,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return manifest_path


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


def compute_dataset_hash(dataset_root: Path) -> str:
    dataset_root = dataset_root.resolve()
    entries = []
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                rel = path.relative_to(dataset_root)
                entries.append(f"{rel}:{path.stat().st_size}")
            except OSError:
                continue
    content = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def get_current_git_commit() -> str | None:
    git_dir = Path(".git")
    if not git_dir.is_dir():
        return None
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return None
    try:
        head_content = head_file.read_text(encoding="utf-8").strip()
        if head_content.startswith("ref: "):
            ref_path = head_content[5:].strip()
            ref_file = git_dir / ref_path
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip()
        elif len(head_content) >= 7:
            return head_content[:40] if len(head_content) >= 40 else head_content
    except OSError:
        pass
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit
    return None
