from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import PreprocessingResult, _resize_and_pad, _to_uint8_rgb


def preprocess_endoscopy(image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None) -> PreprocessingResult:
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    result = _remove_specular_highlights(image)
    result = _correct_color_distribution(result)
    result = _correct_lens_distortion(result)
    deblurred = _deblur_image(result)
    deblurred = _resize_and_pad(deblurred, target_size)
    metadata.update({"specular_removed": True, "color_normalized": True, "lens_corrected": True, "deblurred": True})
    return PreprocessingResult(image=deblurred, metadata=metadata, modality="endoscopy")


def _remove_specular_highlights(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    result = image.copy()
    result[mask > 0] = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)[mask > 0]
    return result


def _correct_color_distribution(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _correct_lens_distortion(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    camera_matrix = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    dist_coeffs[0, 0] = -0.3
    dist_coeffs[1, 0] = 0.1
    return cv2.undistort(image, camera_matrix, dist_coeffs)


def _deblur_image(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, cv2.CV_8U, kernel)
