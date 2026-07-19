from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import PreprocessingResult, _resize_and_pad, _to_uint8_rgb


def preprocess_xray(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    min_val, max_val = float(np.percentile(enhanced, 1)), float(np.percentile(enhanced, 99))
    if max_val > min_val:
        stretched = np.clip((enhanced.astype(np.float32) - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
    else:
        stretched = enhanced
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    bone_enhanced = cv2.filter2D(stretched, cv2.CV_8U, kernel)
    final = cv2.addWeighted(stretched, 0.6, bone_enhanced, 0.4, 0)
    result = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    result = _resize_and_pad(result, target_size)
    metadata.update({"clahe": True, "contrast_stretching": True, "bone_enhancement": True})
    return PreprocessingResult(image=result, metadata=metadata, modality="xray")


def preprocess_xray_chest(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    min_val, max_val = float(np.percentile(enhanced, 2)), float(np.percentile(enhanced, 98))
    if max_val > min_val:
        stretched = np.clip((enhanced.astype(np.float32) - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
    else:
        stretched = enhanced
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    bone_enhanced = cv2.filter2D(stretched, cv2.CV_8U, kernel)
    final = cv2.addWeighted(stretched, 0.6, bone_enhanced, 0.4, 0)
    result = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    result = _resize_and_pad(result, target_size)
    metadata.update({"clahe": True, "contrast_stretching": True, "bone_enhancement": True})
    return PreprocessingResult(image=result, metadata=metadata, modality="xray")
