from __future__ import annotations

import cv2


def open_camera_capture(index: int) -> cv2.VideoCapture | None:
    if hasattr(cv2, "CAP_DSHOW"):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)
