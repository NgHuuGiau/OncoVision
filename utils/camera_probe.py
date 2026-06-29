from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from utils.camera_utils import open_camera_capture
from utils.console_ui import CYAN, GREEN, RED, YELLOW


class CameraCapture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self): ...

    def release(self) -> None: ...


@dataclass
class CameraProbeResult:
    level: str
    summary: str
    detail: str

    @property
    def color(self) -> str:
        return {"PASS": GREEN, "WARN": YELLOW, "ERROR": RED}.get(self.level, CYAN)

    @property
    def style(self) -> str:
        return self.color

    @property
    def ok(self) -> bool:
        return self.level == "PASS"


def probe_camera(
    *,
    index: int = 0,
    attempts: int = 3,
    unavailable_detail: str,
    open_camera_capture_fn: Callable[[int], CameraCapture | None] = open_camera_capture,
) -> CameraProbeResult:
    try:
        capture = open_camera_capture_fn(index)
    except Exception as exc:
        return CameraProbeResult(
            level="ERROR",
            summary=f"Camera thật       ERROR | Không tạo được camera index {index}",
            detail=f"Lý do không chạy  {exc}",
        )

    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        return CameraProbeResult(
            level="WARN",
            summary=f"Camera thật       WARN  | Không mở được camera index {index}",
            detail=unavailable_detail,
        )

    frame_width = 0
    frame_height = 0
    got_frame = False
    try:
        for _ in range(max(1, attempts)):
            success, frame = capture.read()
            if success and frame is not None:
                frame_height, frame_width = frame.shape[:2]
                got_frame = True
                break
    finally:
        capture.release()

    if not got_frame:
        return CameraProbeResult(
            level="WARN",
            summary=f"Camera thật       WARN  | Mở được camera index {index} nhưng không đọc được frame",
            detail="Lý do không chạy  Webcam mở được nhưng không trả về khung hình hợp lệ.",
        )

    return CameraProbeResult(
        level="PASS",
        summary=f"Camera thật       PASS  | Đọc frame thành công tại index {index}",
        detail=f"Chi tiết          {frame_width}x{frame_height}",
    )
