from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import PreprocessingResult, _resize_and_pad, _to_uint8_rgb


def preprocess_ultrasound(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = _speckle_noise_reduction(gray)
    adaptive = _adaptive_contrast(denoised)
    result = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    result = _resize_and_pad(result, target_size)
    metadata.update({"speckle_reduced": True, "adaptive_contrast": True})
    return PreprocessingResult(image=result, metadata=metadata, modality="ultrasound")


def _speckle_noise_reduction(image: np.ndarray) -> np.ndarray:
    median = cv2.medianBlur(image, 3)
    bilateral = cv2.bilateralFilter(median, 5, 50, 50)
    result = cv2.addWeighted(image, 0.6, bilateral, 0.4, 0)
    return result


def _adaptive_contrast(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    min_val, max_val = float(np.percentile(enhanced, 2)), float(np.percentile(enhanced, 98))
    if max_val > min_val:
        stretched = np.clip((enhanced.astype(np.float32) - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
    else:
        stretched = enhanced
    return stretched
