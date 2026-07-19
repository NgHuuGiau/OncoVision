from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import PreprocessingResult, _resize_and_pad, _to_uint8_rgb


def preprocess_mammogram(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = _detect_and_invert(gray)
    denoised = cv2.fastNlMeansDenoising(inverted, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, breast_mask = _segment_breast(enhanced)
    masked = cv2.bitwise_and(enhanced, enhanced, mask=breast_mask)
    result = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    result = _resize_and_pad(result, target_size)
    metadata.update({"inversion_corrected": True, "breast_segmented": True, "clahe": True})
    return PreprocessingResult(image=result, metadata=metadata, modality="mammogram")


def _detect_and_invert(gray: np.ndarray) -> np.ndarray:
    mean_val = float(np.mean(gray))
    if mean_val > 127:
        return cv2.bitwise_not(gray)
    return gray


def _segment_breast(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return gray, mask
