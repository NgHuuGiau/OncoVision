"""Section 3: Detection helpers.

Các hàm xử lý detection findings: coerce, decode raw image, map to canvas.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from medical.pipeline.dataclasses import DetectionFinding


def _coerce_detections(items: list[Any]) -> list[DetectionFinding]:
    detections: list[DetectionFinding] = []
    for item in items:
        if isinstance(item, DetectionFinding):
            detections.append(item)
            continue
        if isinstance(item, dict) and "label" in item and "confidence" in item:
            raw_bbox = item.get("bbox")
            bbox = (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])) if raw_bbox and len(raw_bbox) == 4 else None
            detections.append(
                DetectionFinding(
                    label=str(item["label"]),
                    confidence=float(item["confidence"]),
                    bbox=bbox,
                    segmentation=cast(
                        tuple[tuple[int, int], ...],
                        tuple(tuple(map(int, point)) for point in (item.get("segmentation") or ())),
                    ) or None,
                    frame_index=int(item["frame_index"]) if item.get("frame_index") is not None else None,
                    timestamp_seconds=(
                        float(item["timestamp_seconds"])
                        if item.get("timestamp_seconds") is not None
                        else None
                    ),
                    bbox_reliable=item.get("bbox_reliable", True),
                )
            )
    return detections


def _decode_raw_image(source: str | Path | None) -> np.ndarray | None:
    """Đọc pixel ảnh gốc (chưa thumbnail/canvas) cho detector; file hỏng/video trả None."""
    if source is None:
        return None
    path = Path(source)
    if not path.is_file():
        return None
    try:
        buffer: np.ndarray = np.frombuffer(path.read_bytes(), np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _map_detections_to_canvas(
    detections: list[DetectionFinding],
    raw_hw: tuple[int, int],
    canvas_hw: tuple[int, int],
) -> list[DetectionFinding]:
    """Đổi bbox/polygon từ hệ tọa độ ảnh gốc sang canvas.

    Phép biến đổi khớp normalize_uploaded_image: thumbnail (không phóng to) rồi
    dán giữa canvas vuông nền đen.
    """
    raw_h, raw_w = raw_hw
    canvas_h, canvas_w = canvas_hw
    if raw_h <= 0 or raw_w <= 0 or (raw_h, raw_w) == (canvas_h, canvas_w):
        return detections
    scale = min(1.0, canvas_w / raw_w, canvas_h / raw_h)
    off_x = (canvas_w - round(raw_w * scale)) // 2
    off_y = (canvas_h - round(raw_h * scale)) // 2
    mapped: list[DetectionFinding] = []
    for item in detections:
        bbox = item.bbox
        new_bbox = bbox
        if bbox is not None:
            new_bbox = (
                round(bbox[0] * scale) + off_x,
                round(bbox[1] * scale) + off_y,
                round(bbox[2] * scale) + off_x,
                round(bbox[3] * scale) + off_y,
            )
        new_seg = item.segmentation
        if new_seg:
            new_seg = tuple(
                (round(x * scale) + off_x, round(y * scale) + off_y)
                for x, y in new_seg
            )
        mapped.append(replace(item, bbox=new_bbox, segmentation=new_seg))
    return mapped
