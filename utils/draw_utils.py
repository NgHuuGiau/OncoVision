from __future__ import annotations

import hashlib
from collections.abc import Iterable

import cv2
import numpy as np

_COLOR_PALETTE: tuple[tuple[int, int, int], ...] = (
    (46, 125, 255),
    (255, 159, 67),
    (46, 204, 113),
    (235, 87, 87),
    (155, 89, 182),
    (241, 196, 15),
    (38, 198, 218),
    (230, 126, 34),
)
_TAG_PADDING = 6
_TAG_MARGIN = 4


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    index = digest[0] % len(_COLOR_PALETTE)
    return _COLOR_PALETTE[index]


def _text_box_metrics(text: str, font_scale: float, thickness: int, padding: int = _TAG_PADDING) -> tuple[int, int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_width = text_width + (padding * 2)
    box_height = text_height + baseline + (padding * 2)
    return text_width, text_height, baseline, box_width, box_height


def _fits_rect(
    image_shape: tuple[int, ...],
    top_left: tuple[int, int],
    box_width: int,
    box_height: int,
    margin: int = _TAG_MARGIN,
) -> bool:
    image_height, image_width = image_shape[:2]
    x, y = top_left
    return (
        x >= margin
        and y >= margin
        and (x + box_width) <= max(margin, image_width - margin)
        and (y + box_height) <= max(margin, image_height - margin)
    )


def _clamp_top_left(
    image_shape: tuple[int, ...],
    top_left: tuple[int, int],
    box_width: int,
    box_height: int,
    margin: int = _TAG_MARGIN,
) -> tuple[int, int]:
    image_height, image_width = image_shape[:2]
    max_x = max(margin, image_width - box_width - margin)
    max_y = max(margin, image_height - box_height - margin)
    x = max(margin, min(int(top_left[0]), max_x))
    y = max(margin, min(int(top_left[1]), max_y))
    return x, y


def _draw_text_tag(
    image: np.ndarray,
    text: str,
    top_left: tuple[int, int],
    font_scale: float,
    text_color: tuple[int, int, int],
    background_color: tuple[int, int, int],
    thickness: int = 1,
    padding: int = _TAG_PADDING,
    border_color: tuple[int, int, int] | None = None,
    border_thickness: int = 1,
) -> tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    _text_width, text_height, _baseline, box_width, box_height = _text_box_metrics(
        text=text,
        font_scale=font_scale,
        thickness=thickness,
        padding=padding,
    )
    x, y = _clamp_top_left(image.shape, top_left, box_width, box_height)
    cv2.rectangle(image, (x, y), (x + box_width, y + box_height), background_color, -1)
    if border_color is not None and border_thickness > 0:
        cv2.rectangle(image, (x, y), (x + box_width, y + box_height), border_color, border_thickness)
    text_origin = (x + padding, y + padding + text_height)
    cv2.putText(image, text, text_origin, font, font_scale, text_color, thickness, cv2.LINE_AA)
    return (x, y, box_width, box_height)


def _clamp_bbox_to_image(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    image_height, image_width = image_shape[:2]
    if image_height <= 0 or image_width <= 0:
        return None
    x1, y1, x2, y2 = [int(value) for value in bbox]
    left = max(0, min(x1, x2, image_width - 1))
    top = max(0, min(y1, y2, image_height - 1))
    right = max(0, min(max(x1, x2), image_width - 1))
    bottom = max(0, min(max(y1, y2), image_height - 1))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _draw_fps_text(
    image: np.ndarray,
    fps: float,
    font_scale: float,
    thickness: int,
    margin: int = 12,
) -> None:
    text = f"{fps:.1f}" if fps > 0 else "--"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(margin, image.shape[1] - text_width - margin)
    y = max(text_height + margin, image.shape[0] - baseline - margin)
    cv2.putText(image, text, (x, y), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)


def _boxes_overlap(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
    iou_threshold: float = 0.1,
) -> bool:
    """Check if two bounding boxes overlap significantly (IoU > threshold)."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return False
    inter = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return False
    return (inter / union) > iou_threshold


def _apply_nms(
    detection_list: list,
    image_shape: tuple[int, ...],
    iou_threshold: float = 0.1,
) -> list:
    """Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Strategy:
    1. Sort detections by confidence (descending)
    2. For each detection, keep it if IoU with kept detections is below threshold
    3. Return only the kept detections
    """
    if not detection_list:
        return []
    
    # Filter out invalid bboxes first
    valid_detections = []
    for det in detection_list:
        if det.bbox is None:
            continue
        clamped = _clamp_bbox_to_image(det.bbox, image_shape)
        if clamped is not None:
            valid_detections.append((det, clamped))
    
    if not valid_detections:
        return []
    
    # Sort by confidence descending
    valid_detections.sort(key=lambda x: x[0].confidence, reverse=True)
    
    kept_detections = []
    kept_bboxes = []
    
    for det, clamped_bbox in valid_detections:
        # Check if this detection overlaps with any kept detection
        overlaps = False
        for kept_bbox in kept_bboxes:
            if _boxes_overlap(clamped_bbox, kept_bbox, iou_threshold):
                overlaps = True
                break
        
        if not overlaps:
            kept_detections.append(det)
            kept_bboxes.append(clamped_bbox)
    
    return kept_detections


def _select_labels_to_show(
    detection_list: list,
    image_shape: tuple[int, ...],
    iou_threshold: float = 0.5,
) -> set[int]:
    """Select which detections should show labels.
    
    Strategy:
    1. Group detections by label
    2. Within each label group, pick the highest confidence detection
    3. Among selected detections, use NMS-like IoU filtering to avoid overlapping labels
    """
    if not detection_list:
        return set()
    
    # Step 1: Group by label and pick highest confidence per label
    label_best = {}  # label -> (idx, detection)
    for idx, det in enumerate(detection_list):
        if det.bbox is None:
            continue
        clamped = _clamp_bbox_to_image(det.bbox, image_shape)
        if clamped is None:
            continue
        label = det.label
        if label not in label_best or det.confidence > label_best[label][1].confidence:
            label_best[label] = (idx, det)
    
    # Step 2: Among label-best detections, use IoU filtering to avoid overlapping labels
    selected = set()
    labeled_bboxes = []
    # Sort by confidence descending
    sorted_label_best = sorted(label_best.values(), key=lambda x: x[1].confidence, reverse=True)
    
    for idx, det in sorted_label_best:
        clamped = _clamp_bbox_to_image(det.bbox, image_shape)
        if clamped is None:
            continue
        # Check if this detection overlaps with any already-labeled detection
        overlaps = False
        for labeled_bbox in labeled_bboxes:
            if _boxes_overlap(clamped, labeled_bbox, iou_threshold):
                overlaps = True
                break
        if not overlaps:
            selected.add(idx)
            labeled_bboxes.append(clamped)
    
    return selected


def _find_non_overlapping_label_position(
    image_shape: tuple[int, ...],
    bbox_x1: int,
    bbox_y1: int,
    label_width: int,
    label_height: int,
    drawn_rects: list[tuple[int, int, int, int]],
    margin: int = 8,
) -> tuple[int, int]:
    """Find a position for the label that doesn't overlap with already drawn labels.
    
    Tries positions in order: above box, right of box, below box, left of box, inside box.
    """
    image_h, image_w = image_shape[:2]
    
    def _overlaps_any(x: int, y: int) -> bool:
        for rx, ry, rw, rh in drawn_rects:
            if not (x >= rx + rw + margin or x + label_width <= rx - margin or
                    y >= ry + rh + margin or y + label_height <= ry - margin):
                return True
        return False
    
    def _fits(x: int, y: int) -> bool:
        return (x >= 0 and y >= 0 and 
                x + label_width <= image_w and 
                y + label_height <= image_h)
    
    # Try 1: Above the box
    x, y = max(0, bbox_x1), max(0, bbox_y1 - label_height - margin)
    if _fits(x, y) and not _overlaps_any(x, y):
        return x, y
    
    # Try 2: Right of the box
    x, y = max(0, bbox_x1 + label_width + margin), max(0, bbox_y1)
    if _fits(x, y) and not _overlaps_any(x, y):
        return x, y
    
    # Try 3: Below the box
    x, y = max(0, bbox_x1), max(0, bbox_y1 + label_height + margin)
    if _fits(x, y) and not _overlaps_any(x, y):
        return x, y
    
    # Try 4: Left of the box
    x, y = max(0, bbox_x1 - label_width - margin), max(0, bbox_y1)
    if _fits(x, y) and not _overlaps_any(x, y):
        return x, y
    
    # Try 5: Inside box, top-left corner
    x, y = bbox_x1 + 2, bbox_y1 + 2
    if _fits(x, y) and not _overlaps_any(x, y):
        return x, y
    
    # Fallback: scan for any non-overlapping position
    for scan_y in range(0, image_h - label_height, 10):
        for scan_x in range(0, image_w - label_width, 10):
            if _fits(scan_x, scan_y) and not _overlaps_any(scan_x, scan_y):
                return scan_x, scan_y
    
    # Last resort: return above box
    return max(0, bbox_x1), max(0, bbox_y1 - label_height - margin)


def _draw_semi_transparent_tag(
    image: np.ndarray,
    text: str,
    top_left: tuple[int, int],
    font_scale: float,
    text_color: tuple[int, int, int],
    background_color: tuple[int, int, int],
    thickness: int = 1,
    padding: int = _TAG_PADDING,
    alpha: float = 0.7,
) -> tuple[int, int, int, int]:
    """Draw a semi-transparent text tag to avoid completely obscuring the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    _text_width, text_height, _baseline, box_width, box_height = _text_box_metrics(
        text=text,
        font_scale=font_scale,
        thickness=thickness,
        padding=padding,
    )
    x, y = _clamp_top_left(image.shape, top_left, box_width, box_height)
    # Create semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), background_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    text_origin = (x + padding, y + padding + text_height)
    cv2.putText(image, text, text_origin, font, font_scale, text_color, thickness, cv2.LINE_AA)
    return (x, y, box_width, box_height)


def draw_detection_results(
    image: np.ndarray,
    detections: Iterable,
    box_thickness: int = 1,
    label_font_scale: float = 0.5,
    motion_trails: dict[int, list[tuple[int, int]]] | None = None,
    fps: float | None = None,
    show_fps: bool = False,
    show_heatmap: bool = True,
    show_all_labels: bool = False,
    label_alpha: float = 0.85,
) -> np.ndarray:
    detection_list = list(detections)
    trail_overlay = image.copy() if (motion_trails and len(motion_trails) > 0) else None
    drew_trail = False
    if trail_overlay is not None:
        safe_trails = motion_trails or {}
        for detection in detection_list:
            trail_points = safe_trails.get(getattr(detection, "track_id", -1), [])
            if len(trail_points) < 2:
                continue
            drew_trail = True
            box_color = _color_for_label(detection.label)
            for index in range(1, len(trail_points)):
                start = trail_points[index - 1]
                end = trail_points[index]
                segment_ratio = index / max(1, len(trail_points) - 1)
                thickness = max(1, round(max(1, box_thickness) * (0.6 + (segment_ratio * 0.8))))
                cv2.line(trail_overlay, start, end, box_color, thickness, cv2.LINE_AA)
                cv2.circle(trail_overlay, end, max(1, thickness // 2), box_color, -1, cv2.LINE_AA)
        if drew_trail:
            image = cv2.addWeighted(trail_overlay, 0.30, image, 0.70, 0.0)
    if detection_list and show_heatmap:
        heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        for detection in detection_list:
            if detection.bbox is None:
                continue
            clamped_bbox = _clamp_bbox_to_image(detection.bbox, image.shape)
            if clamped_bbox is None:
                continue
            x1, y1, x2, y2 = clamped_bbox
            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)
            radius_x = max(8, (x2 - x1) // 2)
            radius_y = max(8, (y2 - y1) // 2)
            cv2.ellipse(heatmap, (int(center_x), int(center_y)), (radius_x, radius_y), 0, 0, 360, 1.0, -1)
        if np.max(heatmap) > 0:
            heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0).astype(np.float32)
            max_heat = float(np.max(heatmap))
            heatmap = heatmap / max(max_heat, 1e-6)
            heatmap_color = np.zeros(image.shape, dtype=np.uint8)
            heatmap_color[..., 0] = np.clip(heatmap * 255 * 0.35, 0, 255).astype(np.uint8)
            heatmap_color[..., 1] = np.clip(heatmap * 255 * 0.20, 0, 255).astype(np.uint8)
            heatmap_color[..., 2] = np.clip(heatmap * 255 * 0.90, 0, 255).astype(np.uint8)
            image = cv2.addWeighted(image, 0.75, heatmap_color, 0.25, 0.0)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    nms_detections = _apply_nms(detection_list, image.shape, iou_threshold=0.1)
    
    # Track drawn label positions to avoid overlapping labels
    drawn_label_rects = []  # List of (x, y, w, h) for drawn labels
    box_counter = 0  # Unique counter for each box

    for detection in nms_detections:
        if detection.bbox is None:
            continue
        clamped_bbox = _clamp_bbox_to_image(detection.bbox, image.shape)
        if clamped_bbox is None:
            continue
        x1, y1, x2, y2 = clamped_bbox
        box_color = _color_for_label(detection.label)
        # Draw thin bounding box (thickness=1)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 1)
        
        # Increment box counter for unique indexing
        box_counter += 1
        
        # Draw label with unique box number
        label_text = f"Box {box_counter}: {detection.label} {detection.confidence:.2f}"
        label_scale = label_font_scale
        label_thickness = 1
        _text_width, _text_height, _baseline, _box_width, box_height = _text_box_metrics(
            label_text,
            label_scale,
            label_thickness,
        )
        # Find non-overlapping position for label
        label_x, label_y = _find_non_overlapping_label_position(
            image.shape, x1, y1, _box_width, box_height, drawn_label_rects, margin=8
        )
        drawn_label_rects.append((label_x, label_y, _box_width, box_height))
        _draw_semi_transparent_tag(
            image=image,
            text=label_text,
            top_left=(label_x, label_y),
            font_scale=label_scale,
            text_color=(255, 255, 255),
            background_color=box_color,
            thickness=label_thickness,
            alpha=label_alpha,
        )
    if show_fps and fps is not None:
        _draw_fps_text(
            image=image,
            fps=fps,
            font_scale=max(0.72, label_font_scale * 0.95),
            thickness=max(1, box_thickness - 1),
        )
    return image
