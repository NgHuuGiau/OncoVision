# Bounding box mathematical operations

def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _smooth_bbox(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    alpha: float,
) -> tuple[int, int, int, int]:
    alpha = max(0.0, min(1.0, float(alpha)))
    return tuple(
        int(round((previous_value * alpha) + (current_value * (1.0 - alpha))))
        for previous_value, current_value in zip(previous_bbox, current_bbox)
    )  # type: ignore[return-value]


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_center_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    center_ax, center_ay = _bbox_center(box_a)
    center_bx, center_by = _bbox_center(box_b)
    return float(((center_ax - center_bx) ** 2 + (center_ay - center_by) ** 2) ** 0.5)


def _bbox_reference_size(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return float(max(width, height))


def _bbox_movement_ratio(
    previous_observed_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> float:
    movement_distance = _bbox_center_distance(previous_observed_bbox, current_bbox)
    reference_size = max(_bbox_reference_size(previous_observed_bbox), _bbox_reference_size(current_bbox), 1.0)
    return movement_distance / reference_size


def _bbox_size_change_ratio(
    previous_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> float:
    previous_width = max(1, previous_bbox[2] - previous_bbox[0])
    previous_height = max(1, previous_bbox[3] - previous_bbox[1])
    current_width = max(1, current_bbox[2] - current_bbox[0])
    current_height = max(1, current_bbox[3] - current_bbox[1])
    width_ratio = abs(current_width - previous_width) / max(previous_width, current_width, 1)
    height_ratio = abs(current_height - previous_height) / max(previous_height, current_height, 1)
    return max(width_ratio, height_ratio)


def _estimate_motion_bbox(
    previous_display_bbox: tuple[int, int, int, int],
    previous_observed_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    deltas = [current_value - previous_value for previous_value, current_value in zip(previous_observed_bbox, current_bbox)]
    predicted = tuple(previous_display_value + delta for previous_display_value, delta in zip(previous_display_bbox, deltas))
    return predicted  # type: ignore[return-value]
