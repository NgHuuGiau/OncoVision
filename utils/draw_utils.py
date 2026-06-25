from __future__ import annotations

import hashlib
from typing import Iterable

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
    text_width, text_height, baseline, box_width, box_height = _text_box_metrics(
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


def draw_detection_results(
    image: np.ndarray,
    detections: Iterable,
    box_thickness: int = 2,
    label_font_scale: float = 0.8,
    motion_trails: dict[int, list[tuple[int, int]]] | None = None,
    fps: float | None = None,
    show_fps: bool = False,
) -> np.ndarray:
    detection_list = list(detections)
    trail_overlay = image.copy() if (motion_trails and len(motion_trails) > 0) else None
    drew_trail = False
    if trail_overlay is not None:
        for detection in detection_list:
            trail_points = motion_trails.get(getattr(detection, "track_id", -1), [])
            if len(trail_points) < 2:
                continue
            drew_trail = True
            box_color = _color_for_label(detection.label)
            for index in range(1, len(trail_points)):
                start = trail_points[index - 1]
                end = trail_points[index]
                segment_ratio = index / max(1, len(trail_points) - 1)
                thickness = max(1, int(round(max(1, box_thickness) * (0.6 + (segment_ratio * 0.8)))))
                cv2.line(trail_overlay, start, end, box_color, thickness, cv2.LINE_AA)
                cv2.circle(trail_overlay, end, max(1, thickness // 2), box_color, -1, cv2.LINE_AA)
        if drew_trail:
            image = cv2.addWeighted(trail_overlay, 0.30, image, 0.70, 0.0)
    for detection in detection_list:
        clamped_bbox = _clamp_bbox_to_image(detection.bbox, image.shape)
        if clamped_bbox is None:
            continue
        x1, y1, x2, y2 = clamped_bbox
        box_color = _color_for_label(detection.label)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, max(2, box_thickness))
        label_text = f"{detection.label} {detection.confidence:.2f}"
        label_scale = max(0.62, label_font_scale * 0.92)
        label_thickness = max(1, box_thickness - 1)
        _text_width, _text_height, _baseline, _box_width, box_height = _text_box_metrics(
            label_text,
            label_scale,
            label_thickness,
        )
        _draw_text_tag(
            image=image,
            text=label_text,
            top_left=(max(0, x1), y1 - box_height - 6),
            font_scale=label_scale,
            text_color=(255, 255, 255),
            background_color=box_color,
            thickness=label_thickness,
        )
    if show_fps and fps is not None:
        _draw_fps_text(
            image=image,
            fps=fps,
            font_scale=max(0.72, label_font_scale * 0.95),
            thickness=max(1, box_thickness - 1),
        )
    return image
