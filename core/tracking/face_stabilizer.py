# Face stabilization module

FACE_LABEL = "face"
FACE_TRACKING_STICKY_ALPHA = 0.90
FACE_TRACKING_STABLE_MOTION_RATIO = 0.14
FACE_JITTER_FREEZE_RATIO = 0.035
FACE_JITTER_MAX_SIZE_CHANGE_RATIO = 0.08

from core.tracking.bbox_math import (
    _bbox_movement_ratio,
    _bbox_size_change_ratio,
    _smooth_bbox,
)

def _is_refined_face_label(label: str) -> bool:
    return str(label).lower() == FACE_LABEL


def _stabilize_face_bbox(
    previous_display_bbox: tuple[int, int, int, int],
    previous_observed_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    adaptive_alpha: float,
) -> tuple[int, int, int, int]:
    movement_ratio = _bbox_movement_ratio(previous_observed_bbox, current_bbox)
    size_change_ratio = _bbox_size_change_ratio(previous_observed_bbox, current_bbox)
    if movement_ratio <= FACE_JITTER_FREEZE_RATIO and size_change_ratio <= FACE_JITTER_MAX_SIZE_CHANGE_RATIO:
        return previous_display_bbox
    face_alpha = adaptive_alpha
    if movement_ratio <= FACE_TRACKING_STABLE_MOTION_RATIO:
        face_alpha = max(face_alpha, FACE_TRACKING_STICKY_ALPHA)
    return _smooth_bbox(previous_display_bbox, current_bbox, alpha=face_alpha)
