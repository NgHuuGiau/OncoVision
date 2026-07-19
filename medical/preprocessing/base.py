from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingResult:
    image: np.ndarray
    metadata: dict[str, Any]
    modality: str


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def _resize_and_pad(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_size and w == target_size:
        return image
    scale = min(target_size / max(h, w), target_size / max(h, w))
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    background = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y = (target_size - new_h) // 2
    x = (target_size - new_w) // 2
    background[y : y + new_h, x : x + new_w] = resized
    return background
