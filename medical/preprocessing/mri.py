from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import PreprocessingResult, _resize_and_pad, _to_uint8_rgb


def preprocess_mri(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corrected = _n4_bias_field_correction(gray)
    normalized = _slice_normalization(corrected)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    result = _resize_and_pad(result, target_size)
    metadata.update({"bias_correction": "n4_approx", "normalization": "slice_zscore"})
    return PreprocessingResult(image=result, metadata=metadata, modality="mri")


def _n4_bias_field_correction(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    mask = (image_f > 0).astype(np.uint8)
    small = cv2.resize(image_f, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=min(small.shape) // 4)
    blur_large = cv2.resize(blurred, (image_f.shape[1], image_f.shape[0]), interpolation=cv2.INTER_LINEAR)
    blur_large = np.maximum(blur_large, 1e-6)
    corrected = image_f / blur_large * np.mean(blur_large)
    corrected = np.where(mask == 1, corrected, image_f)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _slice_normalization(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    mean = float(np.mean(image_f))
    std = float(np.std(image_f))
    if std < 1e-6:
        return image
    return np.clip((image_f - mean) / std * 128 + 128, 0, 255).astype(np.uint8)
