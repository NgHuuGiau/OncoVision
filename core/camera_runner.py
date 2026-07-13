from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.frame_capture import FrameCapture
from core.fallback_manager import iter_fallback_configs
from core.model_loader import LoadedModel, load_yolo_model
from core.model_selector import RuntimeConfig
from core.recorder import VideoRecorder
from utils.camera_utils import open_camera_capture
from utils.draw_utils import draw_detection_results
from utils.file_utils import load_yaml_cached
from utils.logger import get_logger

# Import helpers from submodules
from core.tracking.bbox_math import _bbox_center
from core.tracking.detection_filter import (
    _filter_person_detections,
    _dedupe_display_detections,
    PERSON_MIN_CONFIDENCE,
    PHONE_MIN_CONFIDENCE,
    DISPLAY_MIN_CONFIDENCE,
)
from core.tracking.detection_tracker import _match_and_smooth_detections
from core.frame_processing import (
    _compute_motion_score,
    _mean_luminance,
    _enhance_low_light_frame,
    _compose_camera_only_layout,
)

logger = get_logger(__name__)
WINDOW_NAME = "OncoVision Camera Realtime"
MOTION_STABLE_THRESHOLD = 2.8
WINDOW_MARGIN = 16
TRACK_TRAIL_MAX_POINTS = 10
TRACK_TRAIL_MIN_MOVEMENT_PX = 4.0
FRAME_READY_MAX_AGE_SECONDS = 0.25
SETTINGS_PATH = Path("config/settings.yaml")

PHONE_LABEL = "phone"
PHONE_LABEL_ALIASES = {PHONE_LABEL, "cell phone", "mobile phone", "smartphone"}


@dataclass(frozen=True)
class CameraSessionControls:
    show_overlays: bool = True
    show_fps: bool = True
    show_trails: bool = True


@dataclass(frozen=True)
class CameraSessionOutputConfig:
    capture_dir: Path
    recording_dir: Path
    recording_codec: str
    recording_fps: float


def _load_camera_session_output_config() -> CameraSessionOutputConfig:
    settings = load_yaml_cached(SETTINGS_PATH) or {}
    output = settings.get("output", {})
    recording = settings.get("recording", {})
    return CameraSessionOutputConfig(
        capture_dir=Path(output.get("captures_dir", "output/captures")),
        recording_dir=Path(output.get("recordings_dir", "output/recordings")),
        recording_codec=str(recording.get("codec", "mp4v")),
        recording_fps=float(recording.get("fps", 20.0)),
    )


@dataclass
class DetectionRecord:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: int = -1


def _normalize_detection_label_and_bbox(
    label: str,
    bbox: tuple[int, int, int, int],
) -> tuple[str, tuple[int, int, int, int]]:
    normalized_label = str(label).strip().lower()
    if normalized_label in PHONE_LABEL_ALIASES:
        return PHONE_LABEL, bbox
    return normalized_label, bbox


class CameraStream:
    def __init__(self, camera_index: int, max_consecutive_read_failures: int = 5) -> None:
        self.camera_index = camera_index
        self.max_consecutive_read_failures = max_consecutive_read_failures
        self.consecutive_read_failures = 0
        self.last_status_message = "Sẵn sàng khởi tạo camera."
        self.last_error_message = ""
        self.capture: cv2.VideoCapture | None = None
        self.capture_thread: threading.Thread | None = None
        self.capture_stop_event = threading.Event()
        self.capture_ready_event = threading.Event()
        self.capture_lock = threading.Lock()
        self.latest_captured_frame: np.ndarray | None = None
        self.latest_frame_timestamp = 0.0

    def open(self, width: int, height: int) -> None:
        self.release()
        self.capture = self._open_capture()
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError("Không mở được camera.")
        self.consecutive_read_failures = 0
        self.last_error_message = ""
        self.last_status_message = "Đã mở camera thành công."
        self.latest_captured_frame = None
        self.latest_frame_timestamp = 0.0
        self.capture_ready_event.clear()
        self._start_capture_worker()

    def read_latest_frame(self, wait_seconds: float = 0.05) -> np.ndarray | None:
        frame = self._take_latest_frame()
        if frame is not None:
            return frame
        self.capture_ready_event.wait(wait_seconds)
        return self._take_latest_frame()

    def release(self) -> None:
        self._stop_capture_worker()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.last_status_message = "Camera đã dừng."

    def _start_capture_worker(self) -> None:
        self.capture_stop_event.clear()
        self.capture_ready_event.clear()
        # Thread creation is intentionally simple: keep the capture loop isolated
        # and avoid passing unsupported constructor arguments across Python versions.
        self.capture_thread = threading.Thread(
            target=self._capture_worker_loop,
            name="camera-capture-worker",
            daemon=True,
        )
        self.capture_thread.start()

    def _stop_capture_worker(self) -> None:
        if self.capture_thread is None:
            return
        self.capture_stop_event.set()
        self.capture_thread.join(timeout=1.0)
        self.capture_thread = None
        self.capture_stop_event.clear()
        self.capture_ready_event.clear()
        with self.capture_lock:
            self.latest_captured_frame = None
            self.latest_frame_timestamp = 0.0

    def _capture_worker_loop(self) -> None:
        while not self.capture_stop_event.is_set():
            ok, frame = self._read_capture_frame(track_failures=True)
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self.capture_lock:
                self.latest_captured_frame = frame
                self.latest_frame_timestamp = time.perf_counter()
            self.capture_ready_event.set()

    def _read_capture_frame(self, track_failures: bool) -> tuple[bool, np.ndarray | None]:
        if self.capture is None:
            return False, None
        try:
            result = self.capture.read()
        except Exception:
            result = (False, None)
        if not isinstance(result, tuple) or len(result) != 2:
            result = (False, None)
        ok, frame = result
        if ok and frame is not None:
            if track_failures:
                self.consecutive_read_failures = 0
            return True, frame
        if track_failures:
            self.consecutive_read_failures += 1
            self.last_error_message = "Không đọc được frame từ camera."
            self.last_status_message = f"Mat frame camera ({self.consecutive_read_failures}/{self.max_consecutive_read_failures})."
        return False, None  # type: ignore[return-value]

    def _take_latest_frame(self) -> np.ndarray | None:
        with self.capture_lock:
            if self.latest_captured_frame is None:
                return None
            if (time.perf_counter() - self.latest_frame_timestamp) > FRAME_READY_MAX_AGE_SECONDS:
                return None
            return self.latest_captured_frame.copy()

    def _open_capture(self) -> cv2.VideoCapture:
        capture = open_camera_capture(self.camera_index)
        if capture is None:
            raise RuntimeError("Không tạo được camera capture.")
        return capture


class CameraDetector:
    def __init__(self, runtime: RuntimeConfig, camera_index: int = 0) -> None:
        self.runtime = runtime
        self.camera_index = camera_index
        self.camera_stream: CameraStream | None = None
        self.loaded_model: LoadedModel | None = None
        self.last_frame_ts = time.perf_counter()
        self.smoothed_fps = 0.0
        self.recovery_count = 0
        self.last_status_message = "Sẵn sàng khởi tạo camera."
        self.last_error_message = ""
        self.last_detections: list[DetectionRecord] = []
        self.previous_display_detections: list[DetectionRecord] = []
        self.previous_observed_detections: list[DetectionRecord] = []
        self.display_trails: dict[int, list[tuple[int, int]]] = {}
        self.next_track_id = 1
        self.frame_index = 0
        self.previous_gray: np.ndarray | None = None
        self.last_motion_score: float = 0.0
        self.last_raw_frame: np.ndarray | None = None
        self.no_frame_poll_failures = 0
        self.fallback_chain_tried: list[dict[str, Any]] = []
        self.runtime_step_errors: list[dict[str, Any]] = []
        self.active_runtime_summary = ""

    @property
    def capture(self) -> cv2.VideoCapture | None:
        return self.camera_stream.capture if self.camera_stream is not None else None

    @property
    def max_consecutive_read_failures(self) -> int:
        return self.camera_stream.max_consecutive_read_failures if self.camera_stream is not None else 5

    @property
    def consecutive_read_failures(self) -> int:
        return self.camera_stream.consecutive_read_failures if self.camera_stream is not None else 0

    def initialize(self) -> None:
        last_error: Exception | None = None
        self.fallback_chain_tried = []
        self.runtime_step_errors = []
        for runtime in [self.runtime, *list(iter_fallback_configs(self.runtime))]:
            attempt = {
                "profile_name": runtime.profile_name,
                "model_name": runtime.primary_model_name,
                "resolved_device": runtime.resolved_device,
                "imgsz": int(runtime.imgsz),
                "use_half": bool(runtime.use_half),
            }
            self.fallback_chain_tried.append(attempt)
            try:
                self.runtime = runtime
                self.loaded_model = None
                self.release()
                self.camera_stream = CameraStream(camera_index=self.camera_index)
                self.camera_stream.open(self.runtime.camera_width, self.runtime.camera_height)
                self.loaded_model, resolved_device = load_yolo_model(self.runtime)
                self.runtime.resolved_device = resolved_device
                self.runtime.active_model_name = self.loaded_model.model_name
                dummy_frame = np.zeros((self.runtime.imgsz, self.runtime.imgsz, 3), dtype=np.uint8)
                try:
                    self.loaded_model.model.predict(source=dummy_frame, verbose=False)
                except Exception:
                    pass
                self.last_frame_ts = time.perf_counter()
                self._reset_runtime_state()
                self.last_error_message = ""
                self.active_runtime_summary = (
                    f"camera {self.runtime.camera_width}x{self.runtime.camera_height} | "
                    f"profile {self.runtime.profile_name}"
                )
                self.last_status_message = f"Đã khởi tạo camera thành công. Đang chạy với {self.active_runtime_summary}."
                logger.info("Detector initialized with %s", self.runtime.summary())
                return
            except Exception as exc:
                last_error = exc
                self.last_error_message = str(exc)
                self.last_status_message = "Khởi tạo runtime thất bại, đang thử fallback."
                self.runtime_step_errors.append({**attempt, "error": str(exc)})
                logger.warning("Runtime failed, trying fallback: %s", exc)
                self.release()
        raise RuntimeError(f"Không khởi tạo được detector. Lỗi cuối: {last_error}")

    def read_and_detect(
        self,
        controls: CameraSessionControls | None = None,
    ) -> tuple[bool, Any, list[DetectionRecord], float]:
        controls = controls or CameraSessionControls(show_fps=bool(getattr(self.runtime, "show_fps", True)))
        if self.camera_stream is None or self.loaded_model is None:
            raise RuntimeError("Detector chưa được khởi tạo.")

        frame = self.camera_stream.read_latest_frame(wait_seconds=0.05)
        if frame is None:
            self.no_frame_poll_failures += 1
            self.last_error_message = self.camera_stream.last_error_message
            self.last_status_message = self.camera_stream.last_status_message
            if self.no_frame_poll_failures >= self.camera_stream.max_consecutive_read_failures:
                raise RuntimeError("Camera liên tục không trả về frame.")
            return False, None, [], 0.0
        self.no_frame_poll_failures = 0
        self.last_raw_frame = frame
        self.frame_index += 1
        try:
            current_detections = self._predict_frame(frame)
            self.last_detections = list(current_detections)
        except Exception as exc:
            logger.warning("Inference failed on %s: %s", self.runtime.primary_model_name, exc)
            self.recovery_count += 1
            self.last_error_message = str(exc)
            self.last_status_message = "Suy luận bị lỗi, hệ thống đang tự phục hồi và thử cấu hình an toàn hơn."
            self.initialize()
            return False, None, [], 0.0

        motion_score, current_gray = _compute_motion_score(frame, self.previous_gray)
        self.last_motion_score = motion_score
        self.previous_gray = current_gray
        detections = self._smooth_display_detections(self._effective_display_detections(current_detections))
        if motion_score > MOTION_STABLE_THRESHOLD and detections:
            self.last_status_message = f"Đang theo dõi vật thể chuyển động với {len(detections)} đối tượng."
        else:
            if len(detections) > 0:
                self.last_status_message = f"Đang nhận diện ổn định với {len(detections)} đối tượng."
            else:
                self.last_status_message = "Chưa có vật thể nào được nhận diện."
        fps = self._update_fps()
        if controls.show_overlays:
            processed_frame = draw_detection_results(
                image=frame,
                detections=detections,
                box_thickness=self._effective_box_thickness(),
                label_font_scale=self._effective_label_font_scale(),
                motion_trails=self.display_trails if controls.show_trails else None,
                fps=fps,
                show_fps=controls.show_fps,
            )
        else:
            processed_frame = frame.copy()
        return True, processed_frame, detections, fps

    def _predict_frame(self, frame: np.ndarray) -> list[DetectionRecord]:
        if self.loaded_model is None:
            return []
        inference_frame = self._prepare_frame_for_inference(frame)
        results = self.loaded_model.model.predict(
            source=inference_frame,
            imgsz=self._effective_inference_imgsz(),
            conf=self._effective_confidence(),
            iou=float(getattr(self.runtime, "iou", 0.50)),
            device=self.runtime.resolved_device,
            half=self.runtime.use_half,
            max_det=self._effective_max_det(),
            verbose=False,
            stream=False,
        )
        parsed = self._parse_results(results)
        return _filter_person_detections(
            parsed,
            frame.shape,
            person_confidence=float(getattr(self.runtime, "person_confidence", PERSON_MIN_CONFIDENCE)),
            phone_confidence=float(getattr(self.runtime, "phone_confidence", PHONE_MIN_CONFIDENCE)),
            display_confidence=float(getattr(self.runtime, "display_confidence", DISPLAY_MIN_CONFIDENCE)),
        )

    def _prepare_frame_for_inference(self, frame: np.ndarray) -> np.ndarray:
        if not bool(getattr(self.runtime, "enhance_low_light", True)):
            return frame
        brightness = _mean_luminance(frame)
        if brightness >= float(getattr(self.runtime, "low_light_mean_threshold", 96.0)):
            return frame
        if brightness >= max(24.0, float(getattr(self.runtime, "low_light_mean_threshold", 96.0)) * 0.72):
            return frame
        return _enhance_low_light_frame(frame)

    def _effective_inference_imgsz(self) -> int:
        if self.runtime.profile_name == "low":
            return self.runtime.imgsz
        if self.runtime.profile_name == "medium":
            return min(self.runtime.imgsz, 640)
        return self.runtime.imgsz

    def _effective_max_det(self) -> int:
        if self.runtime.profile_name == "low":
            return min(self.runtime.max_det, 6)
        if self.runtime.profile_name == "medium":
            return min(self.runtime.max_det, 16)
        if self.runtime.profile_name == "high":
            return min(self.runtime.max_det, 24)
        return self.runtime.max_det

    def _effective_confidence(self) -> float:
        if self.runtime.profile_name == "low":
            return max(self.runtime.conf, 0.42)
        if self.runtime.profile_name == "medium":
            return max(self.runtime.conf, 0.38)
        if self.runtime.profile_name == "high":
            return max(self.runtime.conf, 0.32)
        return self.runtime.conf

    def _effective_display_detections(
        self,
        detections: list[DetectionRecord],
    ) -> list[DetectionRecord]:
        cleaned = _dedupe_display_detections(detections)

        if self.runtime.profile_name == "low":
            return cleaned[:5]
        if self.runtime.profile_name == "medium":
            return cleaned[:10]
        return cleaned[:20]

    def _smooth_display_detections(self, detections: list[DetectionRecord]) -> list[DetectionRecord]:
        smoothed = _match_and_smooth_detections(
            current_detections=detections,
            previous_detections=self.previous_display_detections,
            previous_observed_detections=self.previous_observed_detections,
        )
        self._assign_track_ids(smoothed)
        self._update_display_trails(smoothed)
        self.previous_display_detections = list(smoothed)
        self.previous_observed_detections = list(detections)
        return smoothed

    def _reset_runtime_state(self) -> None:
        self.frame_index = 0
        self.last_detections = []
        self.previous_display_detections = []
        self.previous_observed_detections = []
        self.display_trails = {}
        self.next_track_id = 1
        self.previous_gray = None
        self.last_motion_score = 0.0
        self.last_raw_frame = None
        self.no_frame_poll_failures = 0

    def _assign_track_ids(self, detections: list[DetectionRecord]) -> None:
        for detection in detections:
            if detection.track_id >= 0:
                continue
            detection.track_id = self.next_track_id
            self.next_track_id += 1

    def _update_display_trails(self, detections: list[DetectionRecord]) -> None:
        next_trails: dict[int, list[tuple[int, int]]] = {}
        for detection in detections:
            center_x, center_y = _bbox_center(detection.bbox)
            center = (int(round(center_x)), int(round(center_y)))
            trail = list(self.display_trails.get(detection.track_id, []))
            if not trail:
                trail.append(center)
            else:
                last_x, last_y = trail[-1]
                movement = float((((center[0] - last_x) ** 2) + ((center[1] - last_y) ** 2)) ** 0.5)
                if movement >= TRACK_TRAIL_MIN_MOVEMENT_PX:
                    trail.append(center)
                else:
                    trail[-1] = center
            next_trails[detection.track_id] = trail[-TRACK_TRAIL_MAX_POINTS:]
        self.display_trails = next_trails

    def _effective_box_thickness(self) -> int:
        if self.runtime.profile_name == "low":
            return max(1, self.runtime.box_thickness - 2)
        if self.runtime.profile_name == "medium":
            return max(1, self.runtime.box_thickness - 1)
        return self.runtime.box_thickness

    def _effective_label_font_scale(self) -> float:
        if self.runtime.profile_name == "low":
            return max(0.62, self.runtime.label_font_scale * 0.72)
        if self.runtime.profile_name == "medium":
            return max(0.72, self.runtime.label_font_scale * 0.86)
        return self.runtime.label_font_scale

    def _update_fps(self) -> float:
        self.smoothed_fps, self.last_frame_ts = _next_smoothed_fps(self.smoothed_fps, self.last_frame_ts)
        return self.smoothed_fps

    def _parse_results(self, results: list) -> list[DetectionRecord]:
        parsed: list[DetectionRecord] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                label, bbox = _normalize_detection_label_and_bbox(
                    names.get(cls_id, str(cls_id)),
                    (x1, y1, x2, y2),
                )
                parsed.append(
                    DetectionRecord(
                        class_id=cls_id,
                        label=label,
                        confidence=float(box.conf[0].item()),
                        bbox=bbox,
                    )
                )
        return parsed

    def release(self) -> None:
        self.previous_display_detections = []
        self.previous_observed_detections = []
        self.display_trails = {}
        if self.camera_stream is not None:
            self.camera_stream.release()
        self.camera_stream = None
        self.last_status_message = "Camera đã dừng."


def _center_window(window_name: str, width: int, height: int) -> None:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return

        monitor = user32.MonitorFromWindow(hwnd, 2)

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", RECT), ("rcWork", RECT), ("dwFlags", ctypes.c_ulong)]

        window_rect = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(window_rect))
        window_width = int(window_rect.right - window_rect.left)
        window_height = int(window_rect.bottom - window_rect.top)

        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(monitor, ctypes.byref(info))

        work_width = int(info.rcWork.right - info.rcWork.left)
        work_height = int(info.rcWork.bottom - info.rcWork.top)
        x = int(info.rcWork.left + max(WINDOW_MARGIN, (work_width - window_width) // 2))
        y = int(info.rcWork.top + max(WINDOW_MARGIN, (work_height - window_height) // 2))
        cv2.moveWindow(window_name, x, y)
    except Exception:
        return


def _next_smoothed_fps(previous_fps: float, last_frame_ts: float) -> tuple[float, float]:
    now = time.perf_counter()
    current_fps = 1.0 / max(now - last_frame_ts, 1e-6)
    if previous_fps == 0.0:
        return current_fps, now
    return (previous_fps * 0.85) + (current_fps * 0.15), now


def _poll_window_key(wait_ms: int, poll_slice_ms: int = 8) -> int:
    remaining = max(1, wait_ms)
    while remaining > 0:
        step = max(1, min(poll_slice_ms, remaining))
        key = cv2.waitKey(step) & 0xFF
        if key != 255:
            return key
        remaining -= step
    return 255


def _window_escape_requested(key: int) -> bool:
    return key == 27


def _ensure_window_positioned(*, composed: np.ndarray, window_positioned: bool) -> bool:
    if window_positioned:
        return True
    cv2.resizeWindow(WINDOW_NAME, composed.shape[1], composed.shape[0])
    _center_window(WINDOW_NAME, composed.shape[1], composed.shape[0])
    return True


def _toggle_session_controls(controls: CameraSessionControls, key: int) -> CameraSessionControls:
    if key in (ord("o"), ord("O")):
        return CameraSessionControls(
            show_overlays=not controls.show_overlays,
            show_fps=controls.show_fps,
            show_trails=controls.show_trails,
        )
    if key in (ord("f"), ord("F")):
        return CameraSessionControls(
            show_overlays=controls.show_overlays,
            show_fps=not controls.show_fps,
            show_trails=controls.show_trails,
        )
    if key in (ord("t"), ord("T")):
        return CameraSessionControls(
            show_overlays=controls.show_overlays,
            show_fps=controls.show_fps,
            show_trails=not controls.show_trails,
        )
    return controls


def _handle_recording_and_capture_keys(
    *,
    key: int,
    latest_frame: np.ndarray | None,
    frame_capture: FrameCapture,
    recorder: VideoRecorder,
) -> None:
    if latest_frame is None:
        return
    if key in (ord("s"), ord("S")):
        result = frame_capture.save_frame(latest_frame)
        if result.success:
            logger.info("Saved snapshot to %s", result.path)
    elif key in (ord("r"), ord("R")):
        if recorder.is_recording:
            stopped = recorder.stop()
            logger.info("Stopped recording: %s", stopped.path)
        else:
            started = recorder.start((latest_frame.shape[1], latest_frame.shape[0]))
            logger.info("Started recording: %s", started.path)


def run_camera_session(runtime: RuntimeConfig, camera_index: int = 0) -> None:
    detector = CameraDetector(runtime=runtime, camera_index=camera_index)
    detector.initialize()
    window_positioned = False
    output_config = _load_camera_session_output_config()
    frame_capture = FrameCapture(output_config.capture_dir)
    recorder = VideoRecorder(
        output_config.recording_dir,
        codec=output_config.recording_codec,
        fps=output_config.recording_fps,
    )
    controls = CameraSessionControls(show_fps=bool(getattr(runtime, "show_fps", True)))

    def _handle_runtime_key(key: int, latest_frame: np.ndarray | None) -> bool:
        nonlocal controls
        if key == 255:
            return False
        if _window_escape_requested(key):
            return True
        controls = _toggle_session_controls(controls, key)
        _handle_recording_and_capture_keys(
            key=key,
            latest_frame=latest_frame,
            frame_capture=frame_capture,
            recorder=recorder,
        )
        return False

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            ok, frame, _detections, _fps = detector.read_and_detect(controls=controls)
            if not ok:
                key = _poll_window_key(1)
                if _handle_runtime_key(key, detector.last_raw_frame):
                    break
                continue

            display_frame = frame
            if recorder.is_recording:
                recorder.write(display_frame)

            active_runtime = detector.runtime
            composed = _compose_camera_only_layout(display_frame, active_runtime.profile_name)

            cv2.imshow(WINDOW_NAME, composed)
            window_positioned = _ensure_window_positioned(composed=composed, window_positioned=window_positioned)
            key = _poll_window_key(1)
            if _handle_runtime_key(key, detector.last_raw_frame):
                break
    finally:
        recorder.stop()
        detector.release()
        cv2.destroyAllWindows()


def run_camera_preview_session(runtime: RuntimeConfig, camera_index: int = 0) -> None:
    camera_stream = CameraStream(camera_index=camera_index)
    camera_stream.open(runtime.camera_width, runtime.camera_height)
    window_positioned = False
    smoothed_fps = 0.0
    last_frame_ts = time.perf_counter()

    def _handle_runtime_key(key: int) -> bool:
        if key == 255:
            return False
        if _window_escape_requested(key):
            return True
        return False

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame = camera_stream.read_latest_frame(wait_seconds=0.05)
            if frame is None:
                key = _poll_window_key(1)
                if _handle_runtime_key(key):
                    break
                continue

            smoothed_fps, last_frame_ts = _next_smoothed_fps(smoothed_fps, last_frame_ts)
            preview_frame = draw_detection_results(
                image=frame,
                detections=[],
                box_thickness=max(2, runtime.box_thickness),
                label_font_scale=max(0.62, runtime.label_font_scale),
                fps=smoothed_fps,
                show_fps=bool(getattr(runtime, "show_fps", True)),
            )
            composed = _compose_camera_only_layout(preview_frame, runtime.profile_name)

            cv2.imshow(WINDOW_NAME, composed)
            window_positioned = _ensure_window_positioned(composed=composed, window_positioned=window_positioned)
            key = _poll_window_key(1)
            if _handle_runtime_key(key):
                break
    finally:
        camera_stream.release()
        cv2.destroyAllWindows()
