from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from medical.cnn_classifier import is_cnn_classifier_path, load_cnn_classifier

MEDICAL_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
DEFAULT_FEATURE_SIZE = (32, 32)


@dataclass(frozen=True)
class MedicalClassifierPrediction:
    label: str
    confidence: float


@dataclass
class MedicalClassifierModel:
    class_labels: tuple[str, ...]
    centroids: np.ndarray
    feature_size: tuple[int, int] = DEFAULT_FEATURE_SIZE

    def predict(self, source, *, top_k: int = 3) -> list[MedicalClassifierPrediction]:
        features = _extract_medical_features(source, feature_size=self.feature_size)
        distances = np.linalg.norm(self.centroids - features, axis=1)
        scores = -distances
        probabilities = _softmax(scores)
        ranked_indexes = np.argsort(-probabilities)
        return [
            MedicalClassifierPrediction(label=self.class_labels[index], confidence=float(probabilities[index]))
            for index in ranked_indexes[: max(1, top_k)]
        ]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if total <= 0.0:
        return np.full_like(exp_values, 1.0 / len(exp_values))
    return exp_values / total


def _resample_filter() -> int:
    return getattr(Image, "Resampling", Image).BILINEAR


def is_supported_medical_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MEDICAL_IMAGE_EXTENSIONS


def _extract_medical_features(source, *, feature_size=DEFAULT_FEATURE_SIZE) -> np.ndarray:
    if isinstance(source, np.ndarray):
        array = source
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3 and array.shape[-1] == 3:
            array = array[:, :, ::-1]
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    else:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
    resized = image.resize(feature_size, _resample_filter())
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array.reshape(-1)


def iter_medical_image_paths(directory: str | Path) -> Iterable[Path]:
    root = Path(directory)
    if not root.exists():
        return ()
    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and is_supported_medical_image_path(path)
    )


def train_medical_classifier(
    samples: Iterable[tuple[str | Path, int]],
    *,
    class_labels: tuple[str, ...],
    feature_size: tuple[int, int] = DEFAULT_FEATURE_SIZE,
) -> MedicalClassifierModel:
    centroid_sums: np.ndarray | None = None
    counts = np.zeros(len(class_labels), dtype=np.int64)

    for source, class_index in samples:
        features = _extract_medical_features(source, feature_size=feature_size)
        if centroid_sums is None:
            centroid_sums = np.zeros((len(class_labels), features.size), dtype=np.float32)
        centroid_sums[class_index] += features
        counts[class_index] += 1

    if centroid_sums is None:
        raise FileNotFoundError("Khong co anh hop le de huan luyen medical classifier.")
    if np.any(counts == 0):
        missing = ", ".join(class_labels[index] for index, count in enumerate(counts) if count == 0)
        raise FileNotFoundError(f"Thiếu dữ liệu cho các lớp: {missing}")

    centroids = centroid_sums / counts[:, None]
    return MedicalClassifierModel(class_labels=class_labels, centroids=centroids, feature_size=feature_size)


def save_medical_classifier(model: MedicalClassifierModel, path: str | Path) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    return target_path


def load_medical_classifier(path: str | Path):
    cnn_model = _try_load_cnn_classifier(path)
    if cnn_model is not None:
        return cnn_model
    with Path(path).open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, MedicalClassifierModel):
        raise TypeError("Model medical khong hop le.")
    return model


def _try_load_cnn_classifier(path: str | Path):
    try:
        if is_cnn_classifier_path(path):
            return load_cnn_classifier(path)
    except Exception:
        pass
    return None
