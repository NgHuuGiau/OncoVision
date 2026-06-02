from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.fallback_manager import iter_fallback_configs
from core.model_selector import RuntimeConfig
from core.yolo_loader import LoadedModel, load_yolo_model
from utils.file_utils import ensure_project_directories
from utils.logger import get_logger
from utils.visualization import draw_detection_results


logger = get_logger(__name__)
WINDOW_NAME = "YOLO Realtime Camera"
ASSISTANT_WINDOW_NAME = "YOLO Capture Assistant"
SAMPLE_IMAGE_DIR = Path("dataset/sample/images")
SAMPLE_LABEL_DIR = Path("dataset/sample/labels")
CAPTURE_STABILITY_SECONDS = 5.0
MOTION_RESET_THRESHOLD = 3.5
ALLOWED_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
ASSISTANT_WINDOW_SIZE = (860, 320)


@dataclass
class DetectionRecord:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class CapturePreparationState:
    stable_since: float
    previous_gray: np.ndarray | None = None
    motion_score: float = 0.0
    status: str = "Giu camera on dinh trong 5 giay."


def _to_yolo_bbox_line(class_id: int, bbox: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> str:
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    x_center = x1 + (box_width / 2.0)
    y_center = y1 + (box_height / 2.0)
    return (
        f"{class_id} "
        f"{x_center / image_width:.6f} "
        f"{y_center / image_height:.6f} "
        f"{box_width / image_width:.6f} "
        f"{box_height / image_height:.6f}"
    )


def _sanitize_sample_name(value: str) -> str:
    cleaned = "".join(char if char in ALLOWED_NAME_CHARS else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_-")


def _draw_message_panel(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    panel = frame.copy()
    panel_width = min(frame.shape[1] - 40, 700)
    panel_height = 80 + (len(lines) * 34)
    x1 = 20
    y1 = 20
    x2 = x1 + panel_width
    y2 = min(frame.shape[0] - 20, y1 + panel_height)
    cv2.rectangle(panel, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.putText(panel, title, (x1 + 16, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        y = y1 + 66 + (index * 30)
        cv2.putText(panel, line, (x1 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return panel


@lru_cache(maxsize=8)
def _load_unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in font_candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _render_assistant_window(title: str, lines: list[str]) -> np.ndarray:
    width, height = ASSISTANT_WINDOW_SIZE
    image = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(image)
    title_font = _load_unicode_font(34)
    body_font = _load_unicode_font(22)
    draw.rectangle([(8, 8), (width - 8, height - 8)], outline=(0, 220, 255), width=3)
    draw.text((24, 22), title, fill=(0, 220, 255), font=title_font)
    y = 78
    for line in lines:
        draw.text((24, y), line, fill=(240, 240, 240), font=body_font)
        y += 36
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


class CameraDetector:
    def __init__(self, runtime: RuntimeConfig, camera_index: int = 0) -> None:
        self.runtime = runtime
        self.camera_index = camera_index
        self.capture: cv2.VideoCapture | None = None
        self.loaded_model: LoadedModel | None = None
        self.last_frame_ts = time.perf_counter()
        self.smoothed_fps = 0.0
        self.consecutive_read_failures = 0
        self.max_consecutive_read_failures = 5
        self.recovery_count = 0
        self.last_status_message = "San sang khoi tao camera."
        self.last_error_message = ""
        self.active_runtime_summary = ""
        self.last_raw_frame: np.ndarray | None = None
        self.last_detections: list[DetectionRecord] = []

    def initialize(self) -> None:
        last_error: Exception | None = None
        runtime_candidates = [self.runtime, *list(iter_fallback_configs(self.runtime))]
        for runtime in runtime_candidates:
            try:
                self.runtime = runtime
                self.loaded_model, self.runtime.resolved_device = load_yolo_model(runtime)
                self.release()
                self.capture = self._open_capture()
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.runtime.camera_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.runtime.camera_height)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.capture.isOpened():
                    raise RuntimeError("Khong mo duoc camera.")
                self.consecutive_read_failures = 0
                self.last_frame_ts = time.perf_counter()
                self.last_error_message = ""
                self.active_runtime_summary = (
                    f"{self.runtime.active_model_name or self.runtime.primary_model_name} | "
                    f"{self.runtime.resolved_device} | imgsz {self.runtime.imgsz}"
                )
                self.last_status_message = (
                    "Da khoi tao camera thanh cong. "
                    f"Dang chay voi {self.active_runtime_summary}."
                )
                logger.info("Detector initialized with %s", self.runtime.summary())
                return
            except Exception as exc:
                last_error = exc
                self.last_error_message = str(exc)
                self.last_status_message = "Khoi tao runtime that bai, dang thu fallback."
                logger.warning("Runtime failed, trying fallback: %s", exc)
                self.release()
        raise RuntimeError(f"Khong khoi tao duoc detector. Loi cuoi: {last_error}")

    def read_and_detect(self) -> tuple[bool, Any, list[DetectionRecord], float]:
        if self.capture is None or self.loaded_model is None:
            raise RuntimeError("Detector chua duoc khoi tao.")
        ok, frame = self.capture.read()
        if not ok:
            self.consecutive_read_failures += 1
            self.last_error_message = "Khong doc duoc frame tu camera."
            self.last_status_message = (
                f"Mat frame camera ({self.consecutive_read_failures}/{self.max_consecutive_read_failures})."
            )
            if self.consecutive_read_failures >= self.max_consecutive_read_failures:
                raise RuntimeError("Camera lien tuc khong tra ve frame.")
            return False, None, [], 0.0
        self.consecutive_read_failures = 0

        try:
            results = self.loaded_model.model.predict(
                source=frame,
                imgsz=self.runtime.imgsz,
                conf=self.runtime.conf,
                device=self.runtime.resolved_device,
                half=self.runtime.use_half,
                max_det=self.runtime.max_det,
                verbose=False,
                stream=False,
            )
        except Exception as exc:
            logger.warning("Inference failed on %s: %s", self.runtime.primary_model_name, exc)
            self.recovery_count += 1
            self.last_error_message = str(exc)
            self.last_status_message = (
                "Suy luan bi loi, he thong dang tu phuc hoi va thu cau hinh an toan hon."
            )
            self.initialize()
            return False, None, [], 0.0

        detections = self._parse_results(results)
        raw_frame = frame.copy()
        processed_frame = draw_detection_results(
            image=frame,
            detections=detections,
            box_thickness=self.runtime.box_thickness,
            label_font_scale=self.runtime.label_font_scale,
        )
        self.last_raw_frame = raw_frame
        self.last_detections = detections
        now = time.perf_counter()
        current_fps = 1.0 / max(now - self.last_frame_ts, 1e-6)
        self.last_frame_ts = now
        if self.smoothed_fps == 0.0:
            self.smoothed_fps = current_fps
        else:
            self.smoothed_fps = (self.smoothed_fps * 0.85) + (current_fps * 0.15)
        self.last_status_message = f"Dang nhan dien on dinh voi {len(detections)} doi tuong."
        return True, processed_frame, detections, self.smoothed_fps

    def _parse_results(self, results: list) -> list[DetectionRecord]:
        parsed: list[DetectionRecord] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                parsed.append(
                    DetectionRecord(
                        class_id=cls_id,
                        label=names.get(cls_id, str(cls_id)),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return parsed

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.last_status_message = "Camera da dung."

    def save_training_sample(
        self,
        *,
        frame: np.ndarray,
        detections: list[DetectionRecord],
        sample_name: str | None = None,
    ) -> tuple[Path, Path]:
        ensure_project_directories()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name_prefix = _sanitize_sample_name(sample_name or "")
        if name_prefix:
            base_name = f"{name_prefix}_{timestamp}"
        else:
            base_name = f"capture_{timestamp}_{int(time.time() * 1000) % 1000:03d}"
        image_path = SAMPLE_IMAGE_DIR / f"{base_name}.jpg"
        label_path = SAMPLE_LABEL_DIR / f"{base_name}.txt"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Khong luu duoc anh: {image_path}")
        label_lines = [_to_yolo_bbox_line(item.class_id, item.bbox, frame.shape) for item in detections]
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        self.last_status_message = (
            f"Da luu mau train: {image_path.name} va {label_path.name} ({len(label_lines)} nhan)."
        )
        logger.info("Saved training sample: %s | %s", image_path, label_path)
        return image_path, label_path

    def save_current_training_sample(self, sample_name: str | None = None) -> tuple[Path, Path]:
        if self.last_raw_frame is None:
            raise RuntimeError("Chua co frame nao de luu.")
        return self.save_training_sample(
            frame=self.last_raw_frame,
            detections=self.last_detections,
            sample_name=sample_name,
        )

    def runtime_health(self) -> dict:
        return {
            "status": self.last_status_message,
            "last_error": self.last_error_message,
            "recovery_count": self.recovery_count,
            "runtime": self.active_runtime_summary,
        }

    def _open_capture(self) -> cv2.VideoCapture:
        if hasattr(cv2, "CAP_DSHOW"):
            return cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        return cv2.VideoCapture(self.camera_index)


def _compute_motion_score(current_frame: np.ndarray, previous_gray: np.ndarray | None) -> tuple[float, np.ndarray]:
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    if previous_gray is None:
        return 0.0, current_gray
    diff = cv2.absdiff(current_gray, previous_gray)
    return float(diff.mean()), current_gray


def _update_capture_preparation(
    state: CapturePreparationState,
    frame: np.ndarray,
    now: float,
) -> tuple[CapturePreparationState, bool, float]:
    motion_score, current_gray = _compute_motion_score(frame, state.previous_gray)
    stable_since = state.stable_since
    status = state.status
    if motion_score > MOTION_RESET_THRESHOLD:
        stable_since = now
        status = "Phat hien rung/lac. Giu yen lai de dem nguoc tu dau."
    else:
        status = "Khung hinh dang on dinh. Tiep tuc giu yen."
    remaining = max(0.0, CAPTURE_STABILITY_SECONDS - (now - stable_since))
    updated = CapturePreparationState(
        stable_since=stable_since,
        previous_gray=current_gray,
        motion_score=motion_score,
        status=status,
    )
    return updated, remaining <= 0.0, remaining


def _render_capture_preparation_overlay(
    frame: np.ndarray,
    *,
    remaining_seconds: float,
    motion_score: float,
    status: str,
) -> np.ndarray:
    seconds = int(np.ceil(remaining_seconds))
    return _render_assistant_window(
        title="Chuẩn bị chụp mẫu train",
        lines=[
            "Bạn đã bấm T. Hệ thống đang kiểm tra độ ổn định khung hình.",
            f"Đếm ngược: {seconds}s",
            f"Mức rung/lắc: {motion_score:.2f} | Ngưỡng reset: {MOTION_RESET_THRESHOLD:.2f}",
            status,
            "Giữ camera và vật thể yên. Nếu rung, bộ đếm sẽ bắt đầu lại.",
        ],
    )


def _render_name_prompt(sample_name: str, detection_count: int) -> np.ndarray:
    typed_value = sample_name or "(để trống sẽ dùng tên mặc định)"
    return _render_assistant_window(
        title="Đặt tên mẫu train",
        lines=[
            f"Số nhãn sẽ lưu: {detection_count}",
            f"Tên hiện tại: {typed_value}",
            "Gõ tên bằng bàn phím rồi nhấn Enter để lưu.",
            "Backspace để xóa ký tự, Esc để hủy.",
        ],
    )


@lru_cache(maxsize=1)
def _render_idle_assistant() -> np.ndarray:
    return _render_assistant_window(
        title="Trợ lý chụp mẫu train",
        lines=[
            "Bấm T để bắt đầu chụp dữ liệu huấn luyện.",
            "Hệ thống sẽ đếm 5 giây ổn định trước khi lưu.",
            "Cửa sổ này độc lập với camera nên bạn có thể phóng to/thu nhỏ camera.",
        ],
    )


def _handle_name_input(current_name: str, key: int) -> tuple[str, bool, bool]:
    if key in (13, 10):
        return current_name, True, False
    if key == 27:
        return current_name, False, True
    if key in (8, 127):
        return current_name[:-1], False, False
    if 32 <= key <= 126:
        char = chr(key)
        if char in ALLOWED_NAME_CHARS and len(current_name) < 48:
            return current_name + char, False, False
    return current_name, False, False


def run_camera_session(runtime: RuntimeConfig, camera_index: int = 0) -> None:
    detector = CameraDetector(runtime=runtime, camera_index=camera_index)
    detector.initialize()
    capture_prep: CapturePreparationState | None = None
    naming_mode = False
    typed_name = ""
    frozen_frame: np.ndarray | None = None
    frozen_detections: list[DetectionRecord] = []
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(ASSISTANT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, runtime.camera_width, runtime.camera_height)
        cv2.resizeWindow(ASSISTANT_WINDOW_NAME, *ASSISTANT_WINDOW_SIZE)
        cv2.imshow(ASSISTANT_WINDOW_NAME, _render_idle_assistant())
        while True:
            ok, frame, detections, _fps = detector.read_and_detect()
            if not ok:
                continue

            display_frame = frame
            if capture_prep is not None:
                now = time.perf_counter()
                capture_prep, ready, remaining = _update_capture_preparation(capture_prep, detector.last_raw_frame, now)
                assistant_panel = _render_capture_preparation_overlay(
                    frame=display_frame,
                    remaining_seconds=remaining,
                    motion_score=capture_prep.motion_score,
                    status=capture_prep.status,
                )
                cv2.imshow(ASSISTANT_WINDOW_NAME, assistant_panel)
                if ready and detector.last_raw_frame is not None:
                    capture_prep = None
                    naming_mode = True
                    typed_name = ""
                    frozen_frame = detector.last_raw_frame.copy()
                    frozen_detections = list(detector.last_detections)
                    detector.last_status_message = "Khung hinh da on dinh. Hay dat ten de luu."
                    cv2.imshow(ASSISTANT_WINDOW_NAME, _render_name_prompt(typed_name, len(frozen_detections)))
            elif naming_mode and frozen_frame is not None:
                display_frame = draw_detection_results(
                    image=frozen_frame.copy(),
                    detections=frozen_detections,
                    box_thickness=runtime.box_thickness,
                    label_font_scale=runtime.label_font_scale,
                )
                cv2.imshow(ASSISTANT_WINDOW_NAME, _render_name_prompt(typed_name, len(frozen_detections)))
            else:
                cv2.imshow(ASSISTANT_WINDOW_NAME, _render_idle_assistant())

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if naming_mode:
                typed_name, should_save, should_cancel = _handle_name_input(typed_name, key)
                if should_cancel:
                    naming_mode = False
                    frozen_frame = None
                    frozen_detections = []
                    detector.last_status_message = "Da huy luu mau train."
                    cv2.imshow(ASSISTANT_WINDOW_NAME, _render_idle_assistant())
                    continue
                if should_save and frozen_frame is not None:
                    image_path, label_path = detector.save_training_sample(
                        frame=frozen_frame,
                        detections=frozen_detections,
                        sample_name=typed_name,
                    )
                    logger.info("Da luu %s va %s", image_path.name, label_path.name)
                    naming_mode = False
                    frozen_frame = None
                    frozen_detections = []
                    typed_name = ""
                    cv2.imshow(ASSISTANT_WINDOW_NAME, _render_idle_assistant())
                continue

            if capture_prep is not None:
                if key == 27:
                    capture_prep = None
                    detector.last_status_message = "Da huy che do chup mau train."
                    cv2.imshow(ASSISTANT_WINDOW_NAME, _render_idle_assistant())
                continue

            if key in (ord("t"), ord("T")):
                capture_prep = CapturePreparationState(stable_since=time.perf_counter())
                detector.last_status_message = "Bat dau dem nguoc 5 giay de chup mau train."
                continue
            if key == 27:
                break
    finally:
        detector.release()
        cv2.destroyAllWindows()
