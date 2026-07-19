from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import PreprocessingResult, _resize_and_pad, _to_uint8_rgb


def preprocess_pet(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    suv_normalized = _suv_normalization(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(suv_normalized)
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    result = _resize_and_pad(result, target_size)
    metadata.update({"normalization": "suv_approx", "contrast_enhancement": True})
    return PreprocessingResult(image=result, metadata=metadata, modality="pet_ct")


def preprocess_pet_ct(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    suv_normalized = _suv_normalization(gray)
    min_val, max_val = float(np.percentile(suv_normalized, 1)), float(np.percentile(suv_normalized, 99))
    if max_val > min_val:
        stretched = np.clip((suv_normalized.astype(np.float32) - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
    else:
        stretched = suv_normalized
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(stretched)
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    result = _resize_and_pad(result, target_size)
    metadata.update({"normalization": "suv_contrast", "suv_normalized": True})
    return PreprocessingResult(image=result, metadata=metadata, modality="pet_ct")


def _suv_normalization(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    p99 = float(np.percentile(image_f, 99))
    if p99 > 0:
        normalized = image_f / p99 * 255
    else:
        normalized = image_f
    return np.clip(normalized, 0, 255).astype(np.uint8)
