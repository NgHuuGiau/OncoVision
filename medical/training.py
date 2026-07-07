from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from medical.cancer_catalog import COMMON_CANCER_TARGETS
from medical.classifier import iter_medical_image_paths, load_medical_classifier, save_medical_classifier, train_medical_classifier
from medical.dataset import create_default_medical_dataset_config, count_medical_class_split_images, ensure_medical_dataset_structure
from utils.file_utils import load_yaml


DEFAULT_MEDICAL_SETTINGS_PATH = Path("config/medical_settings.yaml")
DEFAULT_TRAINED_MODEL_PATH = Path("medical_7_cancers.pt")
DEFAULT_SPLITS = ("train", "val", "test")


@dataclass(frozen=True, init=False)
class MedicalTrainingPaths:
    dataset_root: Path
    data_yaml_path: Path
    trained_model_path: Path
    class_names: tuple[str, ...]
    feature_size: int

    def __init__(
        self,
        dataset_root: str | Path,
        data_yaml_path: str | Path | None = None,
        trained_model_path: str | Path | None = None,
        class_names: tuple[str, ...] | None = None,
        feature_size: int = 32,
        **legacy: Any,
    ) -> None:
        root = Path(dataset_root)
        object.__setattr__(self, "dataset_root", root)
        object.__setattr__(self, "data_yaml_path", Path(data_yaml_path or legacy.get("data_yaml_path") or (root / "data.yaml")))
        object.__setattr__(
            self,
            "trained_model_path",
            Path(trained_model_path or legacy.get("trained_model_path") or DEFAULT_TRAINED_MODEL_PATH),
        )
        object.__setattr__(self, "class_names", tuple(class_names or legacy.get("class_names") or tuple(target.label for target in COMMON_CANCER_TARGETS)))
        object.__setattr__(self, "feature_size", int(feature_size))


@dataclass(frozen=True)
class MedicalTrainingSummary:
    train_count: int
    val_count: int
    test_count: int
    total_count: int
    class_count: int
    trained_model_path: Path | None = None


def _load_medical_settings() -> dict[str, Any]:
    return load_yaml(DEFAULT_MEDICAL_SETTINGS_PATH).get("medical", {})


def medical_training_paths() -> MedicalTrainingPaths:
    settings = _load_medical_settings()
    dataset_root = Path(settings.get("dataset_root", "dataset/medical"))
    feature_size = int(settings.get("feature_size", 32))
    return MedicalTrainingPaths(
        dataset_root=dataset_root,
        data_yaml_path=dataset_root / "data.yaml",
        trained_model_path=DEFAULT_TRAINED_MODEL_PATH,
        class_names=tuple(target.label for target in COMMON_CANCER_TARGETS),
        feature_size=feature_size,
    )


def audit_medical_raw_dataset(paths: MedicalTrainingPaths | None = None) -> dict[str, Any]:
    paths = paths or medical_training_paths()
    split_counts = {split: count_medical_class_split_images(paths.dataset_root, split) for split in DEFAULT_SPLITS}
    class_counts = {
        class_name: sum(split_counts[split][class_name] for split in DEFAULT_SPLITS)
        for class_name in paths.class_names
    }
    missing_classes = [class_name for class_name, count in class_counts.items() if count == 0]
    return {
        "split_counts": split_counts,
        "class_counts": class_counts,
        "missing_classes": missing_classes,
        "train_images": {
            class_name: list(iter_medical_image_paths(paths.dataset_root / class_name / "processed" / "images" / "train"))
            for class_name in paths.class_names
        },
        "val_images": {
            class_name: list(iter_medical_image_paths(paths.dataset_root / class_name / "processed" / "images" / "val"))
            for class_name in paths.class_names
        },
        "test_images": {
            class_name: list(iter_medical_image_paths(paths.dataset_root / class_name / "processed" / "images" / "test"))
            for class_name in paths.class_names
        },
    }


def prepare_medical_training_dataset(paths: MedicalTrainingPaths | None = None) -> MedicalTrainingSummary:
    paths = paths or medical_training_paths()
    ensure_medical_dataset_structure(
        create_default_medical_dataset_config(paths.dataset_root),
    )
    audit = audit_medical_raw_dataset(paths)
    train_count = sum(len(items) for items in audit["train_images"].values())
    val_count = sum(len(items) for items in audit["val_images"].values())
    test_count = sum(len(items) for items in audit["test_images"].values())
    total_count = train_count + val_count + test_count
    if total_count <= 0:
        raise FileNotFoundError("Khong tim thay anh medical hop le trong 7 thu muc ung thu.")
    if audit["missing_classes"]:
        raise FileNotFoundError("Thieu du lieu cho cac lop: " + ", ".join(audit["missing_classes"]))
    return MedicalTrainingSummary(
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        total_count=total_count,
        class_count=len(paths.class_names),
    )


def _samples_for_split(paths: MedicalTrainingPaths, split: str) -> list[tuple[Path, int]]:
    return [
        (image_path, class_index)
        for class_index, class_name in enumerate(paths.class_names)
        for image_path in iter_medical_image_paths(paths.dataset_root / class_name / "processed" / "images" / split)
    ]


def train_medical_model(paths: MedicalTrainingPaths | None = None, *, prepare_dataset: bool = True) -> Path:
    paths = paths or medical_training_paths()
    if prepare_dataset:
        prepare_medical_training_dataset(paths)
    train_samples = _samples_for_split(paths, "train")
    if not train_samples:
        raise FileNotFoundError("Khong co du lieu train medical.")
    classifier_model = train_medical_classifier(
        train_samples,
        class_labels=paths.class_names,
        feature_size=(paths.feature_size, paths.feature_size),
    )
    save_medical_classifier(classifier_model, paths.trained_model_path)
    return paths.trained_model_path


def validate_medical_model(paths: MedicalTrainingPaths | None = None):
    paths = paths or medical_training_paths()
    settings = _load_medical_settings()
    configured_model_path = Path(settings.get("model", paths.trained_model_path))
    candidate_paths = [paths.trained_model_path]
    if configured_model_path != paths.trained_model_path:
        candidate_paths.append(configured_model_path)
    resolved_model_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
    classifier_model = load_medical_classifier(resolved_model_path)
    validation_samples = _samples_for_split(paths, "val")
    if not validation_samples:
        raise FileNotFoundError("Khong co du lieu val medical.")

    correct = 0
    confidences: list[float] = []
    for image_path, class_index in validation_samples:
        prediction = classifier_model.predict(image_path, top_k=1)[0]
        confidences.append(prediction.confidence)
        if prediction.label == paths.class_names[class_index]:
            correct += 1

    total = len(validation_samples)
    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "average_confidence": sum(confidences) / total,
        "model_path": resolved_model_path,
        "class_count": len(paths.class_names),
    }


def run_full_medical_training_pipeline() -> dict[str, Any]:
    paths = medical_training_paths()
    total_start = time.perf_counter()
    prepare_start = time.perf_counter()
    split_summary = prepare_medical_training_dataset(paths)
    prepare_seconds = time.perf_counter() - prepare_start
    # ponytail: reuse the already prepared dataset here; don't scan it twice.
    train_start = time.perf_counter()
    trained_model_path = train_medical_model(paths, prepare_dataset=False)
    train_seconds = time.perf_counter() - train_start
    validate_start = time.perf_counter()
    validation_metrics = validate_medical_model(paths)
    validate_seconds = time.perf_counter() - validate_start
    return {
        "train_count": split_summary.train_count,
        "val_count": split_summary.val_count,
        "test_count": split_summary.test_count,
        "trained_model_path": trained_model_path,
        "validation_metrics": validation_metrics,
        "prepare_seconds": prepare_seconds,
        "train_seconds": train_seconds,
        "validate_seconds": validate_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }
