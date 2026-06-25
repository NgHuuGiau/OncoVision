# Frame processing operations

import cv2
import numpy as np

CAMERA_ONLY_WIDTH = 800
CAMERA_ONLY_HEIGHT = 600

def _compute_motion_score(current_frame: np.ndarray, previous_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (7, 7), 0)
    current_gray = cv2.resize(current_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if previous_gray is None:
        return 0.0, current_gray
    if previous_gray.shape != current_gray.shape:
        previous_gray = cv2.resize(previous_gray, (current_gray.shape[1], current_gray.shape[0]), interpolation=cv2.INTER_AREA)
    return float(cv2.absdiff(current_gray, previous_gray).mean()), current_gray


def _mean_luminance(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 255.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _enhance_low_light_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _compose_camera_only_layout(frame: np.ndarray, profile_name: str | None = None) -> np.ndarray:
    if profile_name == "high":
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[1] > CAMERA_ONLY_WIDTH or frame.shape[0] > CAMERA_ONLY_HEIGHT else cv2.INTER_LINEAR
    return cv2.resize(frame, (CAMERA_ONLY_WIDTH, CAMERA_ONLY_HEIGHT), interpolation=interpolation)
