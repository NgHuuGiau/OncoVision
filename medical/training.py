from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.common import require_yolo
from training.dataset_ops import (
    DEFAULT_SPLITS,
    IMAGE_EXTENSIONS,
    copy_split,
    is_valid_yolo_label_line,
    iter_matching_files,
    reset_processed_dirs,
    split_items,
)
from training.model_paths import resolve_model_source
from utils.file_utils import load_yaml, save_yaml


YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
SPLITS = DEFAULT_SPLITS


@dataclass(frozen=True)
class MedicalTrainingPaths:
    dataset_root: Path
    raw_images_dir: Path
    raw_labels_dir: Path
    processed_images_dir: Path
    processed_labels_dir: Path
    data_yaml_path: Path
    train_runs_dir: Path
    val_runs_dir: Path
    trained_model_path: Path


@dataclass(frozen=True)
class MedicalTrainingSummary:
    train_count: int
    val_count: int
    test_count: int
    trained_model_path: Path | None = None


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    YOLO, ULTRALYTICS_IMPORT_ERROR = require_yolo(YOLO, ULTRALYTICS_IMPORT_ERROR)
    return YOLO


def load_medical_settings() -> dict[str, Any]:
    return load_yaml("config/medical_settings.yaml").get("medical", {})


def medical_training_paths() -> MedicalTrainingPaths:
    settings = load_medical_settings()
    dataset_root = Path(settings.get("dataset_root", "dataset/medical_skin_lesion"))
    disease_name = str(settings.get("disease_name", "skin_cancer_screening"))
    trained_model = Path(f"models/trained/{disease_name}_best.pt")
    return MedicalTrainingPaths(
        dataset_root=dataset_root,
        raw_images_dir=dataset_root / "raw" / "images",
        raw_labels_dir=dataset_root / "raw" / "labels",
        processed_images_dir=dataset_root / "processed" / "images",
        processed_labels_dir=dataset_root / "processed" / "labels",
        data_yaml_path=dataset_root / "data.yaml",
        train_runs_dir=Path("runs/train"),
        val_runs_dir=Path("runs/val"),
        trained_model_path=trained_model,
    )


def _matching_files(root: Path, suffixes: set[str]) -> list[Path]:
    return iter_matching_files(root, suffixes=suffixes)


def _valid_yolo_line(line: str) -> bool:
    return is_valid_yolo_label_line(line)


def audit_medical_raw_dataset(paths: MedicalTrainingPaths | None = None) -> dict[str, Any]:
    paths = paths or medical_training_paths()
    images = _matching_files(paths.raw_images_dir, IMAGE_EXTENSIONS)
    labels = _matching_files(paths.raw_labels_dir, {".txt"})
    label_map = {path.stem: path for path in labels}
    eligible: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []
    invalid_labels: list[Path] = []
    for image_path in images:
        label_path = label_map.get(image_path.stem)
        if label_path is None:
            missing_labels.append(image_path)
            continue
        label_lines = label_path.read_text(encoding="utf-8").splitlines()
        if not all(_valid_yolo_line(line) for line in label_lines if line.strip()):
            invalid_labels.append(label_path)
            continue
        eligible.append((image_path, label_path))
    return {
        "raw_images": images,
        "raw_labels": labels,
        "eligible": eligible,
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
    }


def _reset_processed_dirs(paths: MedicalTrainingPaths) -> None:
    reset_processed_dirs(
        processed_images_dir=paths.processed_images_dir,
        processed_labels_dir=paths.processed_labels_dir,
        splits=SPLITS,
    )


def _split_items(items: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    return split_items(items, splits=SPLITS, seed=42)


def prepare_medical_training_dataset(paths: MedicalTrainingPaths | None = None) -> MedicalTrainingSummary:
    paths = paths or medical_training_paths()
    audit = audit_medical_raw_dataset(paths)
    eligible = audit["eligible"]
    if not eligible:
        raise FileNotFoundError("Medical dataset raw chua co cap image/label hop le de train.")
    _reset_processed_dirs(paths)
    split_map = _split_items(eligible)
    for split_name, items in split_map.items():
        copy_split(
            split_name=split_name,
            items=items,
            processed_images_dir=paths.processed_images_dir,
            processed_labels_dir=paths.processed_labels_dir,
        )
    return MedicalTrainingSummary(
        train_count=len(split_map["train"]),
        val_count=len(split_map["val"]),
        test_count=len(split_map["test"]),
    )


def _medical_training_kwargs(paths: MedicalTrainingPaths, settings: dict[str, Any]) -> dict[str, Any]:
    project = str(paths.train_runs_dir.resolve())
    return {
        "data": str(paths.data_yaml_path.resolve()),
        "epochs": int(settings.get("epochs", 80)),
        "imgsz": int(settings.get("image_size", 640)),
        "batch": int(settings.get("batch", 4)),
        "device": settings.get("device", 0),
        "workers": int(settings.get("workers", 2)),
        "cache": bool(settings.get("cache", False)),
        "amp": bool(settings.get("amp", True)),
        "patience": int(settings.get("patience", 20)),
        "project": project,
        "name": str(settings.get("run_name", "medical_skin_lesion")),
    }


def sync_medical_model_config(model_path: str | Path) -> None:
    payload = load_yaml("config/medical_settings.yaml")
    payload.setdefault("medical", {})
    payload["medical"]["model"] = str(Path(model_path))
    save_yaml("config/medical_settings.yaml", payload)


def train_medical_model(paths: MedicalTrainingPaths | None = None, *, yolo_cls=None) -> Path:
    paths = paths or medical_training_paths()
    settings = load_medical_settings()
    kwargs = _medical_training_kwargs(paths, settings)
    yolo_cls = yolo_cls or _require_yolo()
    base_model = str(settings.get("base_model", settings.get("fallback_model", "yolo11n.pt")))
    results = yolo_cls(str(resolve_model_source(base_model))).train(model=base_model, **kwargs)
    save_dir = Path(getattr(results, "save_dir", kwargs["project"]))
    best_weight = save_dir / "weights" / "best.pt"
    if not best_weight.exists():
        raise FileNotFoundError(f"Khong tim thay best.pt sau khi train tai {best_weight}")
    paths.trained_model_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(best_weight, paths.trained_model_path)
    sync_medical_model_config(paths.trained_model_path)
    return paths.trained_model_path


def validate_medical_model(paths: MedicalTrainingPaths | None = None, *, yolo_cls=None):
    paths = paths or medical_training_paths()
    settings = load_medical_settings()
    yolo_cls = yolo_cls or _require_yolo()
    model_path = Path(settings.get("model", paths.trained_model_path))
    if not model_path.exists():
        model_path = paths.trained_model_path
    if not model_path.exists():
        raise FileNotFoundError("Chua co medical model da train de validate.")
    return yolo_cls(str(resolve_model_source(model_path))).val(
        data=str(paths.data_yaml_path.resolve()),
        project=str(paths.val_runs_dir.resolve()),
        name=str(settings.get("validation_name", "medical_validation")),
    )


def run_full_medical_training_pipeline(*, yolo_cls=None) -> dict[str, Any]:
    paths = medical_training_paths()
    split_summary = prepare_medical_training_dataset(paths)
    trained_model_path = train_medical_model(paths, yolo_cls=yolo_cls)
    validation_metrics = validate_medical_model(paths, yolo_cls=yolo_cls)
    return {
        "train_count": split_summary.train_count,
        "val_count": split_summary.val_count,
        "test_count": split_summary.test_count,
        "trained_model_path": trained_model_path,
        "validation_metrics": validation_metrics,
    }
