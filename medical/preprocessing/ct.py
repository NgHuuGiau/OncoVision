from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import PreprocessingResult, _resize_and_pad, _to_uint8_rgb


def preprocess_ct(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lung_win = apply_hu_window(gray, center=-600, width=1500)
    soft_win = apply_hu_window(gray, center=40, width=400)
    bone_win = apply_hu_window(gray, center=400, width=1800)
    liver_win = apply_hu_window(gray, center=50, width=350)
    combined = cv2.addWeighted(lung_win, 0.35, soft_win, 0.25, 0)
    combined = cv2.addWeighted(combined, 0.8, bone_win, 0.15, 0)
    combined = cv2.addWeighted(combined, 0.9, liver_win, 0.1, 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    z_score = _z_score_normalize(enhanced.astype(np.float32))
    z_score = np.clip((z_score + 3) / 6 * 255, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(z_score, cv2.COLOR_GRAY2BGR)
    result = _resize_and_pad(result, target_size)
    metadata.update({"window": "multi-window", "normalization": "z-score"})
    return PreprocessingResult(image=result, metadata=metadata, modality="ct")


def apply_hu_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    min_val = center - width / 2
    max_val = center + width / 2
    windowed = np.clip(image, min_val, max_val)
    return ((windowed - min_val) / max(width, 1e-6) * 255).astype(np.uint8)


def _z_score_normalize(image: np.ndarray) -> np.ndarray:
    mean = float(np.mean(image))
    std = float(np.std(image))
    if std < 1e-6:
        return image - mean
    return (image - mean) / std
