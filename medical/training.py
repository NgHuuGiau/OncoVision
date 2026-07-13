from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from medical.cancer_catalog import COMMON_CANCER_TARGETS
from medical.classifier import (
    iter_medical_image_paths,
    load_medical_classifier,
    save_medical_classifier,
    train_medical_classifier,
)
from medical.cnn_classifier import train_cnn_classifier
from medical.dataset import (
    create_default_medical_dataset_config,
    count_medical_class_split_images,
    ensure_medical_dataset_structure,
    infer_medical_upload_context,
)
from medical.dashboard import write_training_dashboard
from medical.metrics import compute_multiclass_metrics
from medical.router import (
    IMAGE_TYPE_FAMILIES,
    family_members,
    is_underrepresented,
    route_input,
)
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


def _populate_processed_splits_from_raw_images(paths: MedicalTrainingPaths) -> None:
    for class_name in paths.class_names:
        class_root = paths.dataset_root / class_name
        candidate_roots = [class_root / "raw" / "images", class_root / "raw", class_root]
        raw_images: list[Path] = []
        for candidate_root in candidate_roots:
            if not candidate_root.exists():
                continue
            raw_images.extend(iter_medical_image_paths(candidate_root))
        if not raw_images:
            continue

        split_dirs = {
            split: paths.dataset_root / class_name / "processed" / "images" / split
            for split in DEFAULT_SPLITS
        }
        existing_counts = {split: len(list(iter_medical_image_paths(split_dir))) for split, split_dir in split_dirs.items()}
        if any(existing_counts.values()):
            continue

        raw_images = sorted({path for path in raw_images if path.is_file()}, key=lambda p: p.name.lower())
        if not raw_images:
            continue

        total_images = len(raw_images)
        train_count = max(1, int(round(total_images * 0.7))) if total_images > 1 else 1
        val_count = max(1, int(round(total_images * 0.15))) if total_images > 2 else 0
        test_count = total_images - train_count - val_count
        if test_count < 0:
            test_count = 0
        if total_images > 2 and test_count == 0:
            test_count = 1
            if val_count == 0:
                val_count = 1
            if train_count > total_images - val_count - test_count:
                train_count = total_images - val_count - test_count
        counts = {"train": train_count, "val": val_count, "test": test_count}

        index = 0
        for split in DEFAULT_SPLITS:
            target_dir = split_dirs[split]
            target_dir.mkdir(parents=True, exist_ok=True)
            count = counts[split]
            for image_path in raw_images[index:index + count]:
                destination = target_dir / image_path.name
                if not destination.exists():
                    shutil.copy2(image_path, destination)
            index += count


def prepare_medical_training_dataset(paths: MedicalTrainingPaths | None = None) -> MedicalTrainingSummary:
    paths = paths or medical_training_paths()
    ensure_medical_dataset_structure(
        create_default_medical_dataset_config(paths.dataset_root),
    )
    _populate_processed_splits_from_raw_images(paths)
    audit = audit_medical_raw_dataset(paths)
    train_count = sum(len(items) for items in audit["train_images"].values())
    val_count = sum(len(items) for items in audit["val_images"].values())
    test_count = sum(len(items) for items in audit["test_images"].values())
    total_count = train_count + val_count + test_count
    if total_count <= 0:
        raise FileNotFoundError("Khong tim thay anh medical hop le trong 7 thu muc ung thu.")
    if audit["missing_classes"]:
        missing_classes = ", ".join(audit["missing_classes"])
        if len(paths.class_names) > 1 and len(set(audit["class_counts"].values())) == 1 and list(audit["class_counts"].values())[0] == 0:
            raise FileNotFoundError("Thieu du lieu cho cac lop: " + missing_classes)
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


def _should_use_cnn_backend(paths: MedicalTrainingPaths, train_samples: list[tuple[Path, int]], settings: dict[str, Any]) -> bool:
    backend = str(settings.get("classifier_backend", "centroid")).lower()
    if backend != "cnn":
        return False
    if len(train_samples) < 8:
        return False
    per_class_count = max(1, len(train_samples) // max(1, len(paths.class_names)))
    return per_class_count >= 2


def _compute_class_weights(train_samples: list[tuple[Path, int]], class_names: tuple[str, ...]) -> list[float] | None:
    counts = np.zeros(len(class_names), dtype=np.float32)
    for _, class_index in train_samples:
        if 0 <= class_index < len(counts):
            counts[class_index] += 1.0
    if np.all(counts > 0):
        return [float(value) for value in (counts.sum() / (len(class_names) * counts))]
    return None


def train_medical_model(paths: MedicalTrainingPaths | None = None, *, prepare_dataset: bool = True) -> Path:
    paths = paths or medical_training_paths()
    settings = _load_medical_settings()
    if prepare_dataset:
        prepare_medical_training_dataset(paths)
    train_samples = _samples_for_split(paths, "train")
    if not train_samples:
        raise FileNotFoundError("Khong co du lieu train medical.")
    backend = str(settings.get("classifier_backend", "centroid")).lower()
    if backend == "cnn" and _should_use_cnn_backend(paths, train_samples, settings):
        try:
            return train_cnn_medical_model(paths, prepare_dataset=False)
        except Exception:
            pass
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


def train_cnn_medical_model(paths: MedicalTrainingPaths | None = None, *, prepare_dataset: bool = True) -> Path:
    paths = paths or medical_training_paths()
    settings = _load_medical_settings()
    if prepare_dataset:
        prepare_medical_training_dataset(paths)
    train_samples = _samples_for_split(paths, "train")
    val_samples = _samples_for_split(paths, "val")
    if not train_samples:
        raise FileNotFoundError("Khong co du lieu train medical.")
    class_weights = None
    if bool(settings.get("cnn_class_weighting", True)):
        class_weights = _compute_class_weights(train_samples, paths.class_names)

    wrapper, _history = train_cnn_classifier(
        train_samples,
        class_labels=paths.class_names,
        image_size=int(settings.get("cnn_image_size", 320)),
        backbone=settings.get("cnn_backbone", "resnet50"),
        pretrained=True,
        dropout=float(settings.get("cnn_dropout", 0.25)),
        batch_size=int(settings.get("cnn_batch_size", 16)),
        num_epochs=int(settings.get("cnn_num_epochs", 30)),
        learning_rate=float(settings.get("cnn_learning_rate", 0.00005)),
        val_samples=val_samples or None,
        early_stopping_patience=int(settings.get("cnn_early_stopping_patience", 7)),
        label_smoothing=float(settings.get("cnn_label_smoothing", 0.08)),
        mixed_precision=bool(settings.get("cnn_mixed_precision", True)),
        warmup_epochs=int(settings.get("cnn_warmup_epochs", 4)),
        class_weights=class_weights,
    )
    wrapper.save(paths.trained_model_path)
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
    class_metrics = compute_multiclass_metrics if _is_cnn_wrapper(classifier_model) else None

    if class_metrics is not None:
        truths: list[int] = []
        preds: list[int] = []
        class_to_index = {name: idx for idx, name in enumerate(paths.class_names)}
        for image_path, class_index in validation_samples:
            prediction = classifier_model.predict(image_path, top_k=1)[0]
            pred_label = prediction["label"] if isinstance(prediction, dict) else prediction.label
            confidences.append(prediction["confidence"] if isinstance(prediction, dict) else prediction.confidence)
            if pred_label == paths.class_names[class_index]:
                correct += 1
            truths.append(class_index)
            preds.append(class_to_index.get(pred_label, -1))
        multiclass = compute_multiclass_metrics(truths, preds, list(paths.class_names))
    else:
        for image_path, class_index in validation_samples:
            prediction = classifier_model.predict(image_path, top_k=1)[0]
            confidences.append(prediction.confidence)
            if prediction.label == paths.class_names[class_index]:
                correct += 1
        multiclass = None

    total = len(validation_samples)
    accuracy = correct / total
    result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "average_confidence": sum(confidences) / total,
        "model_path": resolved_model_path,
        "class_count": len(paths.class_names),
    }
    if multiclass is not None:
        result["multiclass_metrics"] = multiclass
    return result


def _is_cnn_wrapper(model: Any) -> bool:
    return hasattr(model, "predict") and hasattr(model, "model") and hasattr(model, "class_labels") and not hasattr(model, "centroids")


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
    report = {
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
    try:
        write_training_dashboard(Path("output/medical/reports"), report)
    except (PermissionError, OSError):
        pass
    return report


def _body_region_to_class_label(body_region: str) -> str | None:
    if body_region == "cervix":
        body_region = "cervical"
    for target in COMMON_CANCER_TARGETS:
        if target.key == body_region:
            return target.label
    return None


def _samples_for_family(paths: MedicalTrainingPaths, family_key: str) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], tuple[str, ...]]:
    members = family_members(family_key)
    if not members:
        return [], [], ()
    per_member_train: dict[str, list[Path]] = {member: [] for member in members}
    per_member_val: dict[str, list[Path]] = {member: [] for member in members}
    for split, bucket in (("train", per_member_train), ("val", per_member_val)):
        for member in members:
            class_label = _body_region_to_class_label(member)
            if class_label is None or class_label not in paths.class_names:
                continue
            split_dir = paths.dataset_root / class_label / "processed" / "images" / split
            for image_path in iter_medical_image_paths(split_dir):
                # Phổi chia theo modality: X-quang -> xray_mammo, CT/MRI/PET -> ct_volume.
                if member == "lung":
                    _, modality = infer_medical_upload_context(image_path)
                    if route_input(modality, "lung").family != family_key:
                        continue
                bucket[member].append(image_path)
    # Chỉ giữ thành viên thực sự có ảnh trong family này (tránh lớp rỗng sau khi chia modality).
    present = [member for member in members if per_member_train[member] or per_member_val[member]]
    index_of = {member: index for index, member in enumerate(present)}
    train_samples = [(path, index_of[member]) for member in present for path in per_member_train[member]]
    val_samples = [(path, index_of[member]) for member in present for path in per_member_val[member]]
    class_labels = tuple(_body_region_to_class_label(member) or member for member in present)
    return train_samples, val_samples, class_labels


def _oversample_underrepresented(samples: list[tuple[Path, int]], members: tuple[str, ...]) -> list[tuple[Path, int]]:
    if not samples or len(members) < 2:
        return samples
    per_class: dict[int, list[tuple[Path, int]]] = {index: [] for index in range(len(members))}
    for sample in samples:
        per_class.setdefault(sample[1], []).append(sample)
    counts = [len(per_class.get(index, [])) for index in range(len(members))]
    if not counts or max(counts) == 0:
        return samples
    median_count = int(np.median(counts))
    if median_count <= 0:
        return samples
    balanced: list[tuple[Path, int]] = []
    for index, items in per_class.items():
        if not items:
            continue
        member = members[index] if 0 <= index < len(members) else None
        if member is not None and is_underrepresented(member) and len(items) < median_count:
            repeats = max(1, median_count // len(items))
            balanced.extend(items * repeats)
        else:
            balanced.extend(items)
    return balanced


def train_medical_submodels(paths: MedicalTrainingPaths | None = None, *, prepare_dataset: bool = True) -> dict[str, Path]:
    paths = paths or medical_training_paths()
    settings = _load_medical_settings()
    if prepare_dataset:
        prepare_medical_training_dataset(paths)
    backend = str(settings.get("classifier_backend", "cnn")).lower()
    submodel_dir = Path(settings.get("submodel_dir", "output/medical/submodels"))
    submodel_dir.mkdir(parents=True, exist_ok=True)
    reweight = bool(settings.get("submodel_reweight_underrepresented", True))

    trained: dict[str, Path] = {}
    for family_key in IMAGE_TYPE_FAMILIES:
        train_samples, val_samples, class_labels = _samples_for_family(paths, family_key)
        # Cần ít nhất 2 lớp để huấn luyện submodel có ý nghĩa.
        if len(class_labels) < 2 or not train_samples:
            continue
        members = tuple(label for label in class_labels)
        if reweight:
            train_samples = _oversample_underrepresented(train_samples, members)
        model_path = submodel_dir / f"{family_key}.pt"
        if backend == "cnn" and _should_use_cnn_backend(paths, train_samples, settings):
            class_weights = None
            if reweight and bool(settings.get("cnn_class_weighting", True)):
                class_weights = _compute_class_weights(train_samples, class_labels)
            wrapper, _history = train_cnn_classifier(
                train_samples,
                class_labels=class_labels,
                image_size=int(settings.get("cnn_image_size", 320)),
                backbone=settings.get("cnn_backbone", "resnet50"),
                pretrained=True,
                dropout=float(settings.get("cnn_dropout", 0.25)),
                batch_size=int(settings.get("cnn_batch_size", 16)),
                num_epochs=int(settings.get("cnn_num_epochs", 30)),
                learning_rate=float(settings.get("cnn_learning_rate", 0.00005)),
                val_samples=val_samples or None,
                early_stopping_patience=int(settings.get("cnn_early_stopping_patience", 7)),
                label_smoothing=float(settings.get("cnn_label_smoothing", 0.08)),
                mixed_precision=bool(settings.get("cnn_mixed_precision", True)),
                warmup_epochs=int(settings.get("cnn_warmup_epochs", 4)),
                class_weights=class_weights,
            )
            wrapper.save(model_path)
        else:
            classifier_model = train_medical_classifier(
                train_samples,
                class_labels=class_labels,
                feature_size=(paths.feature_size, paths.feature_size),
            )
            save_medical_classifier(classifier_model, model_path)
        trained[family_key] = model_path
    return trained


def run_full_medical_submodel_pipeline() -> dict[str, Any]:
    paths = medical_training_paths()
    total_start = time.perf_counter()
    prepare_start = time.perf_counter()
    split_summary = prepare_medical_training_dataset(paths)
    prepare_seconds = time.perf_counter() - prepare_start
    train_start = time.perf_counter()
    trained_submodels = train_medical_submodels(paths, prepare_dataset=False)
    train_seconds = time.perf_counter() - train_start
    report = {
        "train_count": split_summary.train_count,
        "val_count": split_summary.val_count,
        "test_count": split_summary.test_count,
        "trained_submodels": {key: str(value) for key, value in trained_submodels.items()},
        "families": list(trained_submodels.keys()),
        "prepare_seconds": prepare_seconds,
        "train_seconds": train_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }
    try:
        write_training_dashboard(Path("output/medical/reports"), report)
    except (PermissionError, OSError):
        pass
    return report
