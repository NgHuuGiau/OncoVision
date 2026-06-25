# Detection filter module

DISPLAY_NMS_IOU = 0.45
PERSON_MIN_CONFIDENCE = 0.60
PHONE_MIN_CONFIDENCE = 0.55
DISPLAY_MIN_CONFIDENCE = 0.50
PERSON_MAX_AREA_RATIO = 0.6
PERSON_MAX_WIDTH_HEIGHT_RATIO = 1.35
PERSON_EDGE_TOUCH_RATIO = 0.02
FACE_LABEL = "face"
PERSON_LABEL = "person"
PHONE_LABEL = "phone"

from core.tracking.bbox_math import _bbox_iou

def _dedupe_display_detections(detections: list, iou_threshold: float = DISPLAY_NMS_IOU) -> list:
    selected: list = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(
            detection.label == existing.label and _bbox_iou(detection.bbox, existing.bbox) >= iou_threshold
            for existing in selected
        ):
            continue
        selected.append(detection)
    return selected


def _box_area_ratio(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> float:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return (max(0, x2 - x1) * max(0, y2 - y1)) / max(1, frame_w * frame_h)


def _touches_frame_edge(box: tuple[int, int, int, int], frame_shape: tuple[int, ...], margin_ratio: float = PERSON_EDGE_TOUCH_RATIO) -> bool:
    frame_h, frame_w = frame_shape[:2]
    margin_x = max(1, int(frame_w * margin_ratio))
    margin_y = max(1, int(frame_h * margin_ratio))
    x1, y1, x2, y2 = box
    return x1 <= margin_x or y1 <= margin_y or x2 >= frame_w - margin_x or y2 >= frame_h - margin_y


def _person_shape_is_plausible(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return (width / height) <= PERSON_MAX_WIDTH_HEIGHT_RATIO


def _filter_person_detections(
    detections: list,
    frame_shape: tuple[int, ...],
    *,
    person_confidence: float = PERSON_MIN_CONFIDENCE,
    phone_confidence: float = PHONE_MIN_CONFIDENCE,
    display_confidence: float = DISPLAY_MIN_CONFIDENCE,
) -> list:
    filtered: list = []

    for item in detections:
        label = str(item.label).lower()

        if label == PERSON_LABEL:
            if (
                item.confidence >= person_confidence
                and _person_shape_is_plausible(item.bbox)
                and not (_touches_frame_edge(item.bbox, frame_shape) and _box_area_ratio(item.bbox, frame_shape) > PERSON_MAX_AREA_RATIO)
            ):
                filtered.append(item)

        elif label == FACE_LABEL:
            if (
                item.confidence >= person_confidence
                and _person_shape_is_plausible(item.bbox)
                and not (_touches_frame_edge(item.bbox, frame_shape) and _box_area_ratio(item.bbox, frame_shape) > PERSON_MAX_AREA_RATIO)
            ):
                filtered.append(item)

        elif label == PHONE_LABEL:
            if item.confidence >= phone_confidence:
                filtered.append(item)

        else:
            if item.confidence >= display_confidence:
                filtered.append(item)

    return filtered
