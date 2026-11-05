from __future__ import annotations

from dataclasses import dataclass

from utils.camera_utils import open_camera_capture


@dataclass(frozen=True)
class CameraProbeResult:
    level: str
    summary: str
    detail: str


def probe_camera(
    *,
    index: int = 0,
    attempts: int = 3,
    unavailable_detail: str = "Lý do không chạy  Webcam không sẵn sàng, đang bị app khác chiếm hoặc chưa cắm.",
    open_camera_capture_fn=open_camera_capture,
) -> CameraProbeResult:
    capture = open_camera_capture_fn(index=index)
    if capture is None:
        return CameraProbeResult(
            level="WARN",
            summary=f"Camera thật       WARN  | Không mở được camera index {index}",
            detail=f"Lý do không chạy  Không mở được camera index {index}. {unavailable_detail}",
        )

    try:
        for _ in range(max(1, attempts)):
            result = capture.read()
            if not isinstance(result, tuple) or len(result) != 2:
                continue
            ok, frame = result
            if ok and frame is not None:
                height, width = frame.shape[:2]
                return CameraProbeResult(
                    level="PASS",
                    summary=f"Camera thật       PASS  | Đọc frame thành công tại index {index}",
                    detail=f"Chi tiết          {width}x{height}",
                )
    finally:
        capture.release()

    return CameraProbeResult(
        level="WARN",
        summary=f"Camera thật       WARN  | Mở được camera index {index} nhưng không đọc được frame",
        detail="Lý do không chạy  Webcam mở được nhưng không trả về khung hình hợp lệ.",
    )
