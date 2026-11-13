from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from medical.preprocessing.base import (
    PreprocessingResult,
    _resize_and_pad,
    _to_uint8_rgb,
)


def preprocess_colposcopy(
    image: np.ndarray, target_size: int = 320, metadata: dict[str, Any] | None = None
) -> PreprocessingResult:
    """Tiền xử lý ảnh soi cổ tử cung (colposcopy).

    Tối ưu cho phát hiện tổn thương cổ tử cung:
    - Tăng cường độ tương phản vùng tổn thương
    - Chuẩn hóa màu sau khi bôi dung dịch acetic acid/Lugol
    - Giảm noise từ ánh sáng phản xạ
    """
    metadata = metadata or {}
    image = _to_uint8_rgb(image)
    result = _remove_specular_highlights(image)
    result = _enhance_lesion_contrast(result)
    result = _normalize_acetowhite_color(result)
    result = _deblur_image(result)
    result = _resize_and_pad(result, target_size)
    metadata.update({
        "specular_removed": True,
        "lesion_contrast_enhanced": True,
        "acetowhite_normalized": True,
        "deblurred": True,
    })
    return PreprocessingResult(image=result, metadata=metadata, modality="colposcopy")


def _remove_specular_highlights(image: np.ndarray) -> np.ndarray:
    """Loại bỏ điểm phản quang từ ánh sáng colposcope."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    kernel: np.ndarray = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    result = image.copy()
    if np.count_nonzero(mask) > 0:
        result[mask > 0] = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)[mask > 0]
    return result


def _enhance_lesion_contrast(image: np.ndarray) -> np.ndarray:
    """Tăng cường độ tương phản để làm nổi bật tổn thương."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # CLAHE mạnh cho lesion detection
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _normalize_acetowhite_color(image: np.ndarray) -> np.ndarray:
    """Chuẩn hóa màu acetowhite (vùng trắng sau khi bôi acetic acid)."""
    # Chuyển sang HSV để phát hiện vùng acetowhite
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Tăng brightness nhẹ cho vùng tối
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _deblur_image(image: np.ndarray) -> np.ndarray:
    """Làm sắc nét ảnh."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, cv2.CV_8U, kernel)
