from __future__ import annotations

from app.chat_ui.paths import build_chat_capture_path, get_chat_capture_dir
from app.chat_ui.content import translate as tr
from utils.camera_utils import open_camera_capture_with_fallback

try:
    import cv2
except ImportError:
    cv2 = None

from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
class CameraCaptureDialog(QDialog):
    captured = Signal(str)

    def __init__(self, *, language: str, camera_index_value: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.language = language
        self.camera_index_value = camera_index_value
        self.capture = None
        self.latest_frame = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = 0.0
        self.setWindowTitle(tr(language, "camera_window"))
        self.setModal(True)
        self.resize(760, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(720, 420)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setObjectName("GreetingCard")
        self.preview_label.setStyleSheet("border-radius: 18px;")
        layout.addWidget(self.preview_label)

        self.status_label = QLabel()
        self.status_label.setObjectName("Subtle")
        layout.addWidget(self.status_label)

        self.save_path_label = QLabel()
        self.save_path_label.setObjectName("Subtle")
        self.save_path_label.setWordWrap(True)
        self.save_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.save_path_label)

        row = QHBoxLayout()
        row.addStretch(1)
        self.capture_button = QPushButton(tr(language, "capture"))
        self.capture_button.clicked.connect(self.capture_frame)
        row.addWidget(self.capture_button)
        layout.addLayout(row)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self._refresh_save_path_label()
        self.start_camera()

    def _refresh_save_path_label(self) -> None:
        self.save_path_label.setText(tr(self.language, "camera_save_dir").format(path=str(get_chat_capture_dir())))

    def start_camera(self) -> None:
        if cv2 is None:
            self.status_label.setText(tr(self.language, "camera_missing"))
            self.capture_button.setEnabled(False)
            return
        open_result = open_camera_capture_with_fallback(self.camera_index_value)
        self.capture = open_result.capture
        if self.capture is None or not self.capture.isOpened():
            self.capture = None
            tried = ", ".join(str(index) for index in open_result.attempted_indexes)
            self.status_label.setText(
                tr(self.language, "camera_unavailable")
                + " "
                + tr(self.language, "camera_fallback_hint")
                + " "
                + tr(self.language, "camera_fallback_tried").format(indexes=tried)
            )
            self.capture_button.setEnabled(False)
            return
        if open_result.index_used is not None and open_result.index_used != self.camera_index_value:
            self.status_label.setText(
                tr(self.language, "camera_ready")
                + " "
                + tr(self.language, "camera_fallback_used").format(index=open_result.index_used)
            )
        else:
            self.status_label.setText(tr(self.language, "camera_ready"))
        self.timer.start(30)

    def update_preview(self) -> None:
        import time
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.status_label.setText(
                tr(self.language, "camera_unavailable") + " " + tr(self.language, "camera_fallback_hint")
            )
            return
        self.latest_frame = frame
        self.frame_count += 1
        now = time.monotonic()
        if self.last_fps_time == 0.0:
            self.last_fps_time = now
        elapsed = now - self.last_fps_time
        if elapsed >= 0.5:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(rgb, (8, 8), (110, 36), (0, 0, 0), -1)
        cv2.putText(rgb, f"FPS: {self.fps:.0f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 100), 2, cv2.LINE_AA)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(image).scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def capture_frame(self) -> None:
        if self.latest_frame is None or cv2 is None:
            return
        output_path = build_chat_capture_path()
        if not cv2.imwrite(str(output_path), self.latest_frame):
            self.status_label.setText(tr(self.language, "camera_save_failed"))
            return
        self.captured.emit(str(output_path))
        self.accept()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        super().closeEvent(event)
