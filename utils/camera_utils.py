from __future__ import annotations

import platform
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraOpenResult:
    capture: cv2.VideoCapture | None
    index_used: int | None
    attempted_indexes: tuple[int, ...]


def open_camera_capture(index: int) -> cv2.VideoCapture | None:
    system_name = platform.system().lower()
    if system_name == "windows" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if system_name == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def open_camera_capture_with_fallback(index: int, *, fallback_indices: tuple[int, ...] = (0, 1, 2, 3)) -> CameraOpenResult:
    attempted: list[int] = []
    for candidate in (index, *fallback_indices):
        if candidate in attempted:
            continue
        attempted.append(candidate)
        capture = open_camera_capture(candidate)
        if capture is not None and capture.isOpened():
            return CameraOpenResult(capture=capture, index_used=candidate, attempted_indexes=tuple(attempted))
        if capture is not None:
            capture.release()
    return CameraOpenResult(capture=None, index_used=None, attempted_indexes=tuple(attempted))
