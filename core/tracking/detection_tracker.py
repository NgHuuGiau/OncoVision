# Detection tracker module

TRACKING_MATCH_IOU = 0.35
TRACKING_SMOOTHING_ALPHA = 0.65
TRACKING_FAST_SMOOTHING_ALPHA = 0.30
TRACKING_STABLE_SMOOTHING_ALPHA = 0.80
TRACKING_MATCH_CENTER_RATIO = 0.42
TRACKING_PREDICTION_MOTION_RATIO = 0.22
TRACKING_STABLE_MOTION_RATIO = 0.06

from core.tracking.bbox_math import (
    _bbox_iou,
    _bbox_center_distance,
    _bbox_reference_size,
    _bbox_movement_ratio,
    _smooth_bbox,
    _estimate_motion_bbox,
)
from core.tracking.face_stabilizer import (
    _is_refined_face_label,
    _stabilize_face_bbox,
)

def _adaptive_tracking_alpha(
    previous_observed_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> float:
    movement_ratio = _bbox_movement_ratio(previous_observed_bbox, current_bbox)
    if movement_ratio <= TRACKING_STABLE_MOTION_RATIO:
        return TRACKING_STABLE_SMOOTHING_ALPHA
    if movement_ratio >= 0.35:
        return TRACKING_FAST_SMOOTHING_ALPHA
    blend = max(0.0, min(1.0, (movement_ratio - TRACKING_STABLE_MOTION_RATIO) / (0.35 - TRACKING_STABLE_MOTION_RATIO)))
    return TRACKING_STABLE_SMOOTHING_ALPHA + ((TRACKING_FAST_SMOOTHING_ALPHA - TRACKING_STABLE_SMOOTHING_ALPHA) * blend)


def _can_match_detection(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    iou_threshold: float = TRACKING_MATCH_IOU,
    center_ratio_threshold: float = TRACKING_MATCH_CENTER_RATIO,
) -> bool:
    overlap = _bbox_iou(previous_bbox, current_bbox)
    if overlap >= iou_threshold:
        return True
    center_distance = _bbox_center_distance(previous_bbox, current_bbox)
    reference_size = max(_bbox_reference_size(previous_bbox), _bbox_reference_size(current_bbox), 1.0)
    return center_distance <= (reference_size * center_ratio_threshold)


def _tracking_match_score(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> float:
    overlap = _bbox_iou(previous_bbox, current_bbox)
    center_distance = _bbox_center_distance(previous_bbox, current_bbox)
    reference_size = max(_bbox_reference_size(previous_bbox), _bbox_reference_size(current_bbox), 1.0)
    normalized_distance = min(1.0, center_distance / reference_size)
    return overlap + (1.0 - normalized_distance)


def _match_and_smooth_detections(
    current_detections: list,
    previous_detections: list,
    previous_observed_detections: list | None = None,
    iou_threshold: float = TRACKING_MATCH_IOU,
) -> list:
    # Avoid circular import at runtime by importing DetectionRecord inside the function if needed,
    # or just treat items as objects dynamically.
    if not current_detections:
        return []
    if not previous_detections:
        return list(current_detections)

    # We import DetectionRecord dynamically to avoid circular import issues
    from core.camera_runner import DetectionRecord

    smoothed: list = []
    matched_previous_indices: set[int] = set()
    for current in current_detections:
        best_match_index = -1
        best_score = -1.0
        for index, previous in enumerate(previous_detections):
            if index in matched_previous_indices:
                continue
            if previous.class_id != current.class_id or previous.label != current.label:
                continue
            if not _can_match_detection(previous.bbox, current.bbox, iou_threshold=iou_threshold):
                continue
            score = _tracking_match_score(previous.bbox, current.bbox)
            if score > best_score:
                best_score = score
                best_match_index = index
        if best_match_index >= 0:
            matched_previous_indices.add(best_match_index)
            previous = previous_detections[best_match_index]
            previous_observed = (
                previous_observed_detections[best_match_index]
                if previous_observed_detections is not None and best_match_index < len(previous_observed_detections)
                else previous
            )
            adaptive_alpha = _adaptive_tracking_alpha(previous_observed.bbox, current.bbox)
            movement_ratio = _bbox_movement_ratio(previous_observed.bbox, current.bbox)
            if movement_ratio >= TRACKING_PREDICTION_MOTION_RATIO:
                smoothing_source_bbox = _estimate_motion_bbox(previous.bbox, previous_observed.bbox, current.bbox)
            else:
                smoothing_source_bbox = previous.bbox
            smoothed_bbox = (
                _stabilize_face_bbox(
                    previous_display_bbox=previous.bbox,
                    previous_observed_bbox=previous_observed.bbox,
                    current_bbox=current.bbox,
                    adaptive_alpha=adaptive_alpha,
                )
                if _is_refined_face_label(current.label)
                else _smooth_bbox(smoothing_source_bbox, current.bbox, alpha=adaptive_alpha)
            )
            smoothed.append(
                DetectionRecord(
                    class_id=current.class_id,
                    label=current.label,
                    confidence=current.confidence,
                    bbox=smoothed_bbox,
                    track_id=previous.track_id,
                )
            )
            continue
        smoothed.append(current)
    return smoothed
