from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import (
    PreprocessingResult,
    _resize_and_pad,
    _to_uint8_rgb,
)


def preprocess_colonoscopy(
    image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None
) -> PreprocessingResult:
    """Tiền xử lý ảnh nội soi đại tràng (colonoscopy).

    Tương tự endoscopy nhưng tối ưu cho phát hiện polyp:
    - Tăng cường độ tương phản vùng polyp
    - Giảm noise từ bong bóng khí
    - Chuẩn hóa màu niêm mạc
    """
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    result = _remove_specular_highlights(image)
    result = _remove_bubbles(result)
    result = _enhance_polyp_contrast(result)
    result = _correct_color_distribution(result)
    result = _deblur_image(result)
    result = _resize_and_pad(result, target_size)
    metadata.update({
        "specular_removed": True,
        "bubbles_removed": True,
        "polyp_contrast_enhanced": True,
        "color_normalized": True,
        "deblurred": True,
    })
    return PreprocessingResult(image=result, metadata=metadata, modality="colonoscopy")


def _remove_specular_highlights(image: np.ndarray) -> np.ndarray:
    """Loại bỏ điểm phản quang (specular highlights)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    kernel: np.ndarray = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    result = image.copy()
    result[mask > 0] = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)[mask > 0]
    return result


def _remove_bubbles(image: np.ndarray) -> np.ndarray:
    """Giảm noise từ bong bóng khí trong lòng đại tràng."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bong bóng thường là vùng rất sáng, tròn
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = image.copy()
    if np.count_nonzero(mask) > 0:
        result[mask > 0] = cv2.inpaint(result, mask, 5, cv2.INPAINT_TELEA)[mask > 0]
    return result


def _enhance_polyp_contrast(image: np.ndarray) -> np.ndarray:
    """Tăng cường độ tương phản để làm nổi bật polyp."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # CLAHE mạnh hơn cho polyp detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _correct_color_distribution(image: np.ndarray) -> np.ndarray:
    """Chuẩn hóa màu niêm mạc đại tràng."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _deblur_image(image: np.ndarray) -> np.ndarray:
    """Làm sắc nét ảnh."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, cv2.CV_8U, kernel)
