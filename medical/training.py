from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
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
from medical.model_versioning import (
    ModelManifest,
    compute_dataset_hash,
    get_current_git_commit,
    write_model_manifest,
)
from medical.router import (
    IMAGE_TYPE_FAMILIES,
    family_members,
    is_underrepresented,
    route_input,
)
from utils.file_utils import load_yaml


DEFAULT_MEDICAL_SETTINGS_PATH = Path("config/medical_settings.yaml")
DEFAULT_TRAINED_MODEL_PATH = Path("medical_7_cancers.pt")
DEFAULT_CNN_MODEL_PATH = Path("medical_7_cancers_cnn.pt")
DEFAULT_SPLITS = ("train", "val", "test")


@dataclass(frozen=True, init=False)
class MedicalTrainingPaths:
    dataset_root: Path
    data_yaml_path: Path
    trained_model_path: Path
    cnn_model_path: Path
    class_names: tuple[str, ...]
    feature_size: int

    def __init__(
        self,
        dataset_root: str | Path,
        data_yaml_path: str | Path | None = None,
        trained_model_path: str | Path | None = None,
        cnn_model_path: str | Path | None = None,
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
        object.__setattr__(
            self,
            "cnn_model_path",
            Path(cnn_model_path or legacy.get("cnn_model_path") or DEFAULT_CNN_MODEL_PATH),
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
        cnn_model_path=DEFAULT_CNN_MODEL_PATH,
        class_names=tuple(target.label for target in COMMON_CANCER_TARGETS),
        feature_size=feature_size,
    )


def _write_training_manifest(
    model_path: Path,
    paths: MedicalTrainingPaths,
    settings: dict[str, Any],
    history: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> None:
    try:
        file_size = model_path.stat().st_size if model_path.exists() else 0
        metrics: dict[str, float] = {}
        if history:
            val_acc = history.get("val_acc", [])
            train_loss = history.get("train_loss", [])
            if val_acc:
                metrics["best_val_acc"] = float(max(val_acc))
                metrics["final_val_acc"] = float(val_acc[-1])
            if train_loss:
                metrics["final_train_loss"] = float(train_loss[-1])
                metrics["num_epochs"] = float(len(train_loss))
            lr = history.get("lr", [])
            if lr:
                metrics["final_lr"] = float(lr[-1])
        manifest = ModelManifest(
            model_name=model_name or model_path.name,
            version=datetime.now().strftime("%Y%m%d-%H%M%S"),
            model_path=model_path.resolve(),
            training_date=datetime.now(),
            dataset_hash=compute_dataset_hash(paths.dataset_root),
            training_config=dict(settings),
            metrics=metrics,
            backbone=str(settings.get("cnn_backbone", "centroid")),
            num_classes=len(paths.class_names),
            image_size=int(settings.get("cnn_image_size", 320)),
            file_size_bytes=file_size,
            git_commit=get_current_git_commit() or "unknown",
        )
        write_model_manifest(model_path, manifest)
    except (OSError, ValueError):
        pass


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


def _patient_id_from_path(path: Path) -> str:
    """Trich id benh nhan tu duong dan de tranh leak giua cac split.

    Thu tu uu tien:
    1. Ten thu muc cha (vi du: .../BN001/scan_01.dcm -> 'bn001').
    2. Tiền tố có chứa chữ số trong tên file (vi du: BN001_slice02.jpg -> 'bn001',
       sub-01_T1.nii -> 'sub-01', P012_img3.dcm -> 'p012').

    Nếu không tìm thấy mã bệnh nhân hợp lệ (tên chung như 'sample_0', 'img_3'
    mà không có định danh bệnh nhân), mỗi ảnh được coi là một bệnh nhân riêng
    để giữ nguyên hành vi phân split theo ảnh như cũ.
    """
    parent = path.parent.name
    if parent and parent.lower() not in {"images", "raw", "processed", ""} and re.search(r"\d", parent):
        return parent.lower()
    stem = path.stem
    # Tien to den dau '_' hoac '-' dau tien, phai co chu so moi la patient id.
    prefix = re.split(r"[_-]", stem, maxsplit=1)[0]
    if re.search(r"\d", prefix):
        return prefix.lower()
    return stem.lower()


def _group_images_by_patient(raw_images: list[Path]) -> list[list[Path]]:
    groups: dict[str, list[Path]] = {}
    for image_path in raw_images:
        groups.setdefault(_patient_id_from_path(image_path), []).append(image_path)
    return [sorted(items, key=lambda p: p.name.lower()) for items in groups.values()]


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

        # Nhóm theo bệnh nhân để cùng một bệnh nhân không bị chia sang cả 2 train/val/test.
        patient_groups = _group_images_by_patient(raw_images)
        total_groups = len(patient_groups)
        if total_groups < 2:
            # Qua it benh nhan de phan tang: gom tat ca vao train nhu cu.
            train_groups = patient_groups
            val_groups: list[list[Path]] = []
            test_groups: list[list[Path]] = []
        else:
            train_count = max(1, int(round(total_groups * 0.7)))
            val_count = max(1, int(round(total_groups * 0.15)))
            if val_count == 0 and total_groups >= 3:
                val_count = 1
            test_count = total_groups - train_count - val_count
            if test_count < 0:
                test_count = 0
            if total_groups > 2 and test_count == 0 and val_count > 0:
                test_count = 1
                if train_count > total_groups - val_count - test_count:
                    train_count = total_groups - val_count - test_count
            train_groups = patient_groups[:train_count]
            val_groups = patient_groups[train_count:train_count + val_count]
            test_groups = patient_groups[train_count + val_count:]

        grouped_splits = {"train": train_groups, "val": val_groups, "test": test_groups}
        for split, groups in grouped_splits.items():
            target_dir = split_dirs[split]
            target_dir.mkdir(parents=True, exist_ok=True)
            for group in groups:
                for image_path in group:
                    destination = target_dir / image_path.name
                    if not destination.exists():
                        shutil.copy2(image_path, destination)


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


def _compute_class_weights(train_samples: list[tuple[Path, int]], class_names: tuple[str, ...]) -> list[float]:
    counts = np.zeros(len(class_names), dtype=np.float32)
    for _, class_index in train_samples:
        if 0 <= class_index < len(counts):
            counts[class_index] += 1.0
    smoothed_counts = counts + 1.0
    return [float(smoothed_counts.sum() / (len(class_names) * value)) for value in smoothed_counts]


def train_medical_model(paths: MedicalTrainingPaths | None = None, *, prepare_dataset: bool = True, verbose: bool = False, resume_path: str | Path | None = None) -> Path:
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
            return train_cnn_medical_model(paths, prepare_dataset=False, verbose=verbose, resume_path=resume_path)
        except Exception:
            pass
    train_samples = _samples_for_split(paths, "train")
    if not train_samples:
        raise FileNotFoundError("Khong co du lieu train medical.")
    classifier_model = train_medical_classifier(
        train_samples,
        class_labels=paths.class_names,
        feature_size=(paths.feature_size, paths.feature_size),
        progress_tag="train",
    )
    save_medical_classifier(classifier_model, paths.trained_model_path)
    _write_training_manifest(paths.trained_model_path, paths, settings)
    return paths.trained_model_path


def train_cnn_medical_model(paths: MedicalTrainingPaths | None = None, *, prepare_dataset: bool = True, verbose: bool = False, resume_path: str | Path | None = None, checkpoint_path: str | Path | None = None, settings_override: dict[str, Any] | None = None, max_train_samples: int | None = None, output_model_path: str | Path | None = None) -> Path:
    paths = paths or medical_training_paths()
    settings = _load_medical_settings()
    if settings_override:
        settings = {**settings, **settings_override}
    if prepare_dataset:
        prepare_medical_training_dataset(paths)
    train_samples = _samples_for_split(paths, "train")
    if max_train_samples is not None and len(train_samples) > max_train_samples:
        train_samples = train_samples[:max_train_samples]
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
        pretrained=bool(settings.get("cnn_pretrained", True)),
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
        progress_tag="train",
        verbose=verbose,
        resume_path=resume_path,
        checkpoint_path=checkpoint_path,
    )
    # Luu CNN vao path rieng de KHONG ghi de model centroid cu (medical_7_cancers.pt).
    target_path = Path(output_model_path) if output_model_path else paths.cnn_model_path
    wrapper.save(target_path)
    _write_training_manifest(target_path, paths, settings, _history)
    return target_path


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


DEFAULT_KFOLD_MODEL_DIR = Path("output/medical/kfold_models")

# Reverse map cua COMMON_CANCER_TARGETS: nhan lop (label) -> body region key.
_CLASS_LABEL_TO_BODY_REGION = {target.label: target.key for target in COMMON_CANCER_TARGETS}


def _router_body_region(body_region_key: str | None) -> str | None:
    # dataset dung "cervical" nhung router dung "cervix".
    if body_region_key == "cervical":
        return "cervix"
    return body_region_key


def _family_for_sample(paths: MedicalTrainingPaths, image_path: Path, class_index: int) -> str:
    """Tra ve image family (ct_volume, xray_mammo, endoscopy, ...) cho mot sample."""
    if class_index < 0 or class_index >= len(paths.class_names):
        return "unknown"
    label = paths.class_names[class_index]
    body_region_key = _CLASS_LABEL_TO_BODY_REGION.get(label)
    if body_region_key is None:
        return "unknown"
    body_region = _router_body_region(body_region_key)
    modality = None
    # Chỉ phổi (lung) cần suy luận modality vì nó tách làm xray_mammo và ct_volume.
    if body_region == "lung":
        try:
            _, modality = infer_medical_upload_context(image_path)
        except Exception:
            modality = None
    route = route_input(modality, body_region)
    return route.family or "unknown"


def _all_labeled_samples(paths: MedicalTrainingPaths) -> list[tuple[Path, int]]:
    """Gom toan bo sample co nhan tu tat ca cac split (train/val/test)."""
    samples: list[tuple[Path, int]] = []
    for split in DEFAULT_SPLITS:
        samples.extend(_samples_for_split(paths, split))
    return samples


def _stratify_samples_by_family(
    samples: list[tuple[Path, int]],
    class_labels: tuple[str, ...],
    num_folds: int = 5,
    *,
    paths: MedicalTrainingPaths | None = None,
    seed: int = 42,
) -> list[tuple[list[tuple[Path, int]], list[tuple[Path, int]]]]:
    """Chia sample thành các fold, phân tăng theo (lớp, image family).

    Mỗi (lớp, family) được rải đều (round-robin) qua các fold nên mỗi fold có
    đại diện từng family. Trả về danh sách (train_samples, val_samples) cho
    từng fold.
    """
    if num_folds < 2:
        raise ValueError("num_folds phai >= 2 de thuc hien cross-validation.")
    if not samples:
        return [([], []) for _ in range(num_folds)]

    paths = paths or medical_training_paths()

    groups: dict[tuple[int, str], list[tuple[Path, int]]] = {}
    for sample in samples:
        image_path, class_index = sample
        family = _family_for_sample(paths, image_path, class_index)
        groups.setdefault((class_index, family), []).append(sample)

    rng = np.random.default_rng(seed)
    fold_buckets: list[list[tuple[Path, int]]] = [[] for _ in range(num_folds)]
    # Duyet nhom theo thu tu on dinh de ket qua co the tai lap.
    for key in sorted(groups.keys(), key=lambda item: (item[0], str(item[1]))):
        items = list(groups[key])
        rng.shuffle(items)
        for offset, sample in enumerate(items):
            fold_buckets[offset % num_folds].append(sample)

    folds: list[tuple[list[tuple[Path, int]], list[tuple[Path, int]]]] = []
    for fold_index in range(num_folds):
        val_samples = list(fold_buckets[fold_index])
        train_samples = [
            sample
            for other_index in range(num_folds)
            if other_index != fold_index
            for sample in fold_buckets[other_index]
        ]
        folds.append((train_samples, val_samples))
    return folds


def _cnn_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_size": int(config.get("cnn_image_size", 320)),
        "backbone": config.get("cnn_backbone", "resnet50"),
        "pretrained": True,
        "dropout": float(config.get("cnn_dropout", 0.25)),
        "batch_size": int(config.get("cnn_batch_size", 16)),
        "num_epochs": int(config.get("cnn_num_epochs", 30)),
        "learning_rate": float(config.get("cnn_learning_rate", 0.00005)),
        "early_stopping_patience": int(config.get("cnn_early_stopping_patience", 7)),
        "label_smoothing": float(config.get("cnn_label_smoothing", 0.08)),
        "mixed_precision": bool(config.get("cnn_mixed_precision", True)),
        "warmup_epochs": int(config.get("cnn_warmup_epochs", 4)),
    }


def _train_fold_model(
    train_samples: list[tuple[Path, int]],
    class_labels: tuple[str, ...],
    config: dict[str, Any],
    paths: MedicalTrainingPaths,
    *,
    val_samples: list[tuple[Path, int]] | None = None,
    fold_index: int | None = None,
    verbose: bool = False,
    resume_path: str | Path | None = None,
) -> Any:
    """Huan luyen mot model cho mot fold dung cung logic voi train_cnn_classifier."""
    fold_tag = f"fold:{fold_index}" if fold_index is not None else "fold"
    backend = str(config.get("classifier_backend", "cnn")).lower()
    if backend == "cnn" and _should_use_cnn_backend(paths, train_samples, config):
        class_weights = None
        if bool(config.get("cnn_class_weighting", True)):
            class_weights = _compute_class_weights(train_samples, class_labels)
        wrapper, _history = train_cnn_classifier(
            train_samples,
            class_labels=class_labels,
            val_samples=val_samples or None,
            class_weights=class_weights,
            progress_tag=fold_tag,
            verbose=verbose,
            resume_path=resume_path,
            **_cnn_kwargs_from_config(config),
        )
        return wrapper
    return train_medical_classifier(
        train_samples,
        class_labels=class_labels,
        feature_size=(paths.feature_size, paths.feature_size),
        progress_tag=fold_tag,
    )


def _save_fold_model(model: Any, model_path: Path) -> None:
    if hasattr(model, "save"):
        model.save(model_path)
    else:
        save_medical_classifier(model, model_path)


def _extract_prediction(prediction: Any) -> tuple[str, float]:
    if isinstance(prediction, dict):
        return str(prediction.get("label", "") or ""), float(prediction.get("confidence", 0.0))
    return str(getattr(prediction, "label", "") or ""), float(getattr(prediction, "confidence", 0.0))


def _evaluate_model_on_samples(
    model: Any,
    samples: list[tuple[Path, int]],
    class_labels: tuple[str, ...],
) -> dict[str, Any]:
    class_to_index = {name: index for index, name in enumerate(class_labels)}
    truths: list[int] = []
    preds: list[int] = []
    confidences: list[float] = []
    for image_path, class_index in samples:
        prediction = model.predict(image_path, top_k=1)[0]
        label, confidence = _extract_prediction(prediction)
        confidences.append(confidence)
        truths.append(class_index)
        predicted_index = class_to_index.get(label, len(class_labels))
        if predicted_index < 0 or predicted_index >= len(class_labels):
            predicted_index = len(class_labels)
        preds.append(predicted_index)
    metrics = compute_multiclass_metrics(truths, preds, list(class_labels) + ["__unknown__"])
    metrics["average_confidence"] = float(np.mean(confidences)) if confidences else 0.0
    return metrics


def _per_class_value(per_class_entry: Any, attribute: str) -> float:
    if isinstance(per_class_entry, dict):
        return float(per_class_entry.get(attribute, 0.0))
    return float(getattr(per_class_entry, attribute, 0.0))


def _per_class_label(per_class_entry: Any) -> str | None:
    if isinstance(per_class_entry, dict):
        return per_class_entry.get("label")
    return getattr(per_class_entry, "label", None)


def _aggregate_fold_metrics(
    fold_metrics: list[dict[str, Any]],
    class_labels: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not fold_metrics:
        return {}, {}

    scalar_keys = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "average_confidence",
    )
    mean_metrics: dict[str, Any] = {}
    std_metrics: dict[str, Any] = {}
    for key in scalar_keys:
        values = [float(metrics.get(key, 0.0)) for metrics in fold_metrics]
        mean_metrics[key] = float(np.mean(values)) if values else 0.0
        std_metrics[key] = float(np.std(values)) if values else 0.0

    per_class_mean: dict[str, dict[str, float]] = {}
    per_class_std: dict[str, dict[str, float]] = {}
    for label in class_labels:
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        supports: list[float] = []
        for metrics in fold_metrics:
            for entry in metrics.get("per_class", []) or []:
                if _per_class_label(entry) == label:
                    precisions.append(_per_class_value(entry, "precision"))
                    recalls.append(_per_class_value(entry, "recall"))
                    f1s.append(_per_class_value(entry, "f1_score"))
                    supports.append(_per_class_value(entry, "support"))
        if not f1s:
            continue
        per_class_mean[label] = {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1_score": float(np.mean(f1s)),
            "support": float(np.mean(supports)),
        }
        per_class_std[label] = {
            "precision": float(np.std(precisions)),
            "recall": float(np.std(recalls)),
            "f1_score": float(np.std(f1s)),
            "support": float(np.std(supports)),
        }
    mean_metrics["per_class"] = per_class_mean
    std_metrics["per_class"] = per_class_std
    return mean_metrics, std_metrics


def train_with_stratified_kfold(
    samples: list[tuple[Path, int]],
    class_labels: tuple[str, ...],
    num_folds: int = 5,
    *,
    config: dict[str, Any] | None = None,
    paths: MedicalTrainingPaths | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = False,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Huan luyen voi stratified K-Fold cross-validation.

    - Phan tang theo nhan lop VA image family (ct_volume, xray_mammo, ...).
    - Moi fold: train tren k-1 fold, validate tren fold giu lai.
    - Luu model tung fold vao output/medical/kfold_models/fold_{i}.pt.
    - Tra ve dict gom fold_metrics, mean_metrics, std_metrics.
    """
    if not samples:
        raise FileNotFoundError("Khong co du lieu de chay stratified K-Fold.")

    paths = paths or medical_training_paths()
    config = dict(config or _load_medical_settings())
    class_labels = tuple(class_labels)
    output_dir = Path(output_dir or DEFAULT_KFOLD_MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("cv_seed", 42))
    folds = _stratify_samples_by_family(samples, class_labels, num_folds, paths=paths, seed=seed)

    fold_metrics: list[dict[str, Any]] = []
    for fold_index, (train_samples, val_samples) in enumerate(folds):
        if not train_samples or not val_samples:
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "skipped": True,
                    "reason": "fold rong (thieu du lieu train hoac val)",
                    "train_count": len(train_samples),
                    "val_count": len(val_samples),
                }
            )
            continue
        model_path = output_dir / f"fold_{fold_index}.pt"
        try:
            model = _train_fold_model(
                train_samples,
                class_labels,
                config,
                paths,
                val_samples=val_samples,
                fold_index=fold_index,
                verbose=verbose,
                resume_path=resume_path,
            )
            _save_fold_model(model, model_path)
            metrics = _evaluate_model_on_samples(model, val_samples, class_labels)
        except Exception as error:  # pragma: no cover - phong ve tinh mach lac
            fold_metrics.append(
                {
                    "fold": fold_index,
                    "skipped": True,
                    "reason": f"loi khi huan luyen/danh gia fold: {error}",
                    "train_count": len(train_samples),
                    "val_count": len(val_samples),
                }
            )
            continue
        metrics["fold"] = fold_index
        metrics["skipped"] = False
        metrics["model_path"] = model_path
        metrics["train_count"] = len(train_samples)
        metrics["val_count"] = len(val_samples)
        fold_metrics.append(metrics)

    evaluated = [metrics for metrics in fold_metrics if not metrics.get("skipped")]
    mean_metrics, std_metrics = _aggregate_fold_metrics(evaluated, class_labels)
    return {
        "num_folds": num_folds,
        "evaluated_folds": len(evaluated),
        "class_labels": list(class_labels),
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }


def _select_best_hyperparams(
    config: dict[str, Any],
    cv_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Chọn bộ hyperparam tốt nhất sau CV.

    Hiện tại CV chạy với một bộ cấu hình cố định (không grid-search) nên bộ
    hyperparam tốt nhất chính là config đã được CV kiểm chứng. Hàm này giữ điểm
    mở rộng để sau này có thể chọn theo fold có macro_f1 cao nhất.
    """
    best = dict(config)
    if cv_results:
        evaluated = [m for m in cv_results.get("fold_metrics", []) if not m.get("skipped")]
        if evaluated:
            best_fold = max(evaluated, key=lambda metrics: float(metrics.get("macro_f1", 0.0)))
            best["_cv_best_fold"] = best_fold.get("fold")
            best["_cv_best_macro_f1"] = float(best_fold.get("macro_f1", 0.0))
    return best


def _train_final_model_on_all_data(
    paths: MedicalTrainingPaths,
    samples: list[tuple[Path, int]],
    config: dict[str, Any],
    cv_results: dict[str, Any] | None = None,
    verbose: bool = False,
    resume_path: str | Path | None = None,
) -> Path:
    """Huan luyen model cuoi cung tren toan bo du lieu voi hyperparam tot nhat tu CV."""
    best_config = _select_best_hyperparams(config, cv_results)
    model = _train_fold_model(samples, paths.class_names, best_config, paths, val_samples=None, verbose=verbose, resume_path=resume_path)
    _save_fold_model(model, paths.trained_model_path)
    return paths.trained_model_path


def run_full_medical_training_pipeline(
    *,
    run_kfold: bool | None = None,
    num_folds: int = 5,
    verbose: bool = False,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = medical_training_paths()
    settings = _load_medical_settings()
    if run_kfold is None:
        run_kfold = bool(settings.get("nested_cross_validation", False))
    total_start = time.perf_counter()
    print("=" * 60, flush=True)
    print("BAT DAU TRAINING MEDICAL 7 UNG THU", flush=True)
    print("=" * 60, flush=True)
    print("[1/4] Dang chuan bi du lieu (quet dataset, tao split)...", flush=True)
    prepare_start = time.perf_counter()
    split_summary = prepare_medical_training_dataset(paths)
    prepare_seconds = time.perf_counter() - prepare_start
    print(
        f"[1/4] Xong chuan bi: train={split_summary.train_count} "
        f"val={split_summary.val_count} test={split_summary.test_count} ({prepare_seconds:.1f}s)",
        flush=True,
    )
    print("[2/4] Dang huan luyen model...", flush=True)
    train_start = time.perf_counter()
    trained_model_path = train_medical_model(paths, prepare_dataset=False, verbose=verbose, resume_path=resume_path)
    train_seconds = time.perf_counter() - train_start
    print(f"[2/4] Xong huan luyen: {trained_model_path} ({train_seconds:.1f}s)", flush=True)
    print("[3/4] Dang danh gia model...", flush=True)
    validate_start = time.perf_counter()
    validation_metrics = validate_medical_model(paths)
    validate_seconds = time.perf_counter() - validate_start
    print(
        f"[3/4] Xong danh gia: accuracy={validation_metrics.get('accuracy', 0):.4f} ({validate_seconds:.1f}s)",
        flush=True,
    )
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

    if run_kfold:
        print("[4/4] Dang chay K-Fold cross-validation...", flush=True)
        cv_start = time.perf_counter()
        all_samples = _all_labeled_samples(paths)
        cv_results = train_with_stratified_kfold(
            all_samples,
            paths.class_names,
            num_folds=num_folds,
            config=settings,
            paths=paths,
            verbose=verbose,
            resume_path=resume_path,
        )
        report["cv_results"] = cv_results
        report["cv_seconds"] = time.perf_counter() - cv_start
        print(f"[4/4] Xong K-Fold ({report['cv_seconds']:.1f}s)", flush=True)
        print("[bonus] Dang huan luyen model cuoi cung tren toan bo du lieu...", flush=True)
        final_start = time.perf_counter()
        final_model_path = _train_final_model_on_all_data(
            paths, all_samples, settings, cv_results, verbose=verbose, resume_path=resume_path
        )
        report["final_model_path"] = final_model_path
        report["final_train_seconds"] = time.perf_counter() - final_start
        report["trained_model_path"] = final_model_path
        report["best_hyperparams"] = _select_best_hyperparams(settings, cv_results)
        print(f"[bonus] Xong model cuoi cung ({report['final_train_seconds']:.1f}s)", flush=True)
        print("[bonus] Dang danh gia lai model cuoi cung...", flush=True)
        try:
            report["validation_metrics"] = validate_medical_model(paths)
        except FileNotFoundError:
            pass
        report["total_seconds"] = time.perf_counter() - total_start

    print("-" * 60, flush=True)
    print(f"HOAN TAT: {report['total_seconds']:.1f}s", flush=True)
    print(f"Model: {report['trained_model_path']}", flush=True)
    if "validation_metrics" in report and report["validation_metrics"]:
        print(f"Accuracy: {report['validation_metrics'].get('accuracy', 0):.4f}", flush=True)
    print("=" * 60, flush=True)

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
            cnn_kwargs = _cnn_kwargs_from_config(dict(settings))
            wrapper, _history = train_cnn_classifier(
                train_samples,
                class_labels=class_labels,
                val_samples=val_samples or None,
                class_weights=class_weights,
                progress_tag=f"submodel:{family_key}",
                **cnn_kwargs,
            )
            wrapper.save(model_path)
            _write_training_manifest(model_path, paths, settings, _history, model_name=family_key)
        else:
            classifier_model = train_medical_classifier(
                train_samples,
                class_labels=class_labels,
                feature_size=(paths.feature_size, paths.feature_size),
                progress_tag=f"submodel:{family_key}",
            )
            save_medical_classifier(classifier_model, model_path)
            _write_training_manifest(model_path, paths, settings, model_name=family_key)
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
