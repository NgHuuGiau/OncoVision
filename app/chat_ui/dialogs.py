from __future__ import annotations
import os
import time

from app.chat_ui.content import translate
from app.chat_ui.paths import build_chat_capture_path, get_chat_capture_dir

try:
    import cv2
except ImportError:
    cv2 = None

from PySide6.QtCore import QParallelAnimationGroup, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsBlurEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

def tr(language: str, key: str) -> str:
    return translate(language, key)

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
            save_dir = str(get_chat_capture_dir())
            self.save_path_label.setText(
                tr(self.language, "camera_save_dir").format(path=save_dir)
            )

        def start_camera(self) -> None:
            if cv2 is None:
                self.status_label.setText(tr(self.language, "camera_missing"))
                self.capture_button.setEnabled(False)
                return
            self.capture = cv2.VideoCapture(self.camera_index_value, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = None
                self.status_label.setText(tr(self.language, "camera_unavailable"))
                self.capture_button.setEnabled(False)
                return
            self.status_label.setText(tr(self.language, "camera_ready"))
            self.timer.start(30)

        def update_preview(self) -> None:
            import time
            if self.capture is None:
                return
            ok, frame = self.capture.read()
            if not ok:
                self.status_label.setText(tr(self.language, "camera_unavailable"))
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
            fps_text = f"FPS: {self.fps:.0f}"
            cv2.rectangle(rgb, (8, 8), (110, 36), (0, 0, 0), -1)
            cv2.putText(rgb, fps_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 100), 2, cv2.LINE_AA)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(pixmap)

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

class SettingsDialog(QDialog):
        def __init__(self, *, parent_window: "ChatWindow") -> None:
            super().__init__(parent_window)
            self.window = parent_window
            self.setWindowTitle(tr(self.window.language, "settings_title"))
            self.setModal(True)
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.resize(1060, 700)
            self.theme_buttons: dict[str, QPushButton] = {}
            self.build_ui()

        def build_ui(self) -> None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(20, 20, 20, 20)
            outer.setSpacing(0)

            shell = QFrame()
            shell.setObjectName("SettingsShell")
            outer.addWidget(shell)

            shell_layout = QVBoxLayout(shell)
            shell_layout.setContentsMargins(0, 0, 0, 0)
            shell_layout.setSpacing(0)

            header_row = QHBoxLayout()
            header_row.setContentsMargins(28, 24, 28, 20)
            header_row.setSpacing(12)
            self.dialog_title = QLabel(tr(self.window.language, "settings_title"))
            self.dialog_title.setStyleSheet("font-size: 28px; font-weight: 700;")
            header_row.addWidget(self.dialog_title)
            header_row.addStretch(1)
            close_button = QPushButton("✕")
            close_button.setObjectName("SettingsCloseButton")
            close_button.setFixedSize(32, 32)
            close_button.clicked.connect(self.accept)
            header_row.addWidget(close_button)
            shell_layout.addLayout(header_row)

            header_divider = QFrame()
            header_divider.setFixedHeight(1)
            header_divider.setStyleSheet(
                "background: rgba(255, 255, 255, 0.08); border: none;"
                if self.window.effective_theme == "dark"
                else "background: rgba(15, 23, 42, 0.08); border: none;"
            )
            shell_layout.addWidget(header_divider)

            body_row = QHBoxLayout()
            body_row.setSpacing(0)

            sidebar = QFrame()
            sidebar.setObjectName("SettingsNav")
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.setContentsMargins(16, 16, 16, 16)
            sidebar_layout.setSpacing(14)

            self.general_button = QPushButton()
            self.general_button.setObjectName("SettingsNavButton")
            self.general_button.setMinimumHeight(52)
            sidebar_layout.addWidget(self.general_button)
            sidebar_layout.addStretch(1)
            body_row.addWidget(sidebar, 2)

            divider = QFrame()
            divider.setFrameShape(QFrame.VLine)
            divider.setStyleSheet(
                "background: transparent; border: none; border-left: 1px solid rgba(255, 255, 255, 0.08);"
                if self.window.effective_theme == "dark"
                else "background: transparent; border: none; border-left: 1px solid rgba(17, 24, 39, 0.08);"
            )
            body_row.addWidget(divider)

            content = QFrame()
            content.setObjectName("SettingsContent")
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(28, 22, 28, 28)
            content_layout.setSpacing(22)

            self.section_title = QLabel()
            self.section_title.setObjectName("SectionTitle")
            self.section_title.setStyleSheet("font-size: 22px; font-weight: 700;")
            content_layout.addWidget(self.section_title)

            self.appearance_card = QFrame()
            self.appearance_card.setObjectName("SettingsOptionCard")
            appearance_card_layout = QVBoxLayout(self.appearance_card)
            appearance_card_layout.setContentsMargins(22, 20, 22, 20)
            appearance_card_layout.setSpacing(18)
            appearance_row = QHBoxLayout()
            appearance_row.setSpacing(10)
            self.appearance_icon = QLabel("◔")
            self.appearance_icon.setStyleSheet("font-size: 18px; font-weight: 700;")
            appearance_row.addWidget(self.appearance_icon, 0, Qt.AlignTop)
            self.appearance_label = QLabel()
            self.appearance_label.setStyleSheet("font-size: 18px; font-weight: 700;")
            appearance_row.addWidget(self.appearance_label, 1)
            appearance_card_layout.addLayout(appearance_row)

            theme_options = QHBoxLayout()
            theme_options.setSpacing(12)
            for key in ("light", "dark", "system"):
                button = QPushButton()
                button.setObjectName("ThemeChoiceButton")
                button.setMinimumHeight(44)
                button.clicked.connect(lambda _checked=False, mode=key: self.set_theme_mode(mode))
                self.theme_buttons[key] = button
                theme_options.addWidget(button)
            appearance_card_layout.addLayout(theme_options)
            content_layout.addWidget(self.appearance_card)

            self.language_card = QFrame()
            self.language_card.setObjectName("SettingsOptionCard")
            language_card_layout = QVBoxLayout(self.language_card)
            language_card_layout.setContentsMargins(22, 20, 22, 20)
            language_card_layout.setSpacing(18)
            language_row = QHBoxLayout()
            language_row.setSpacing(10)
            self.language_icon = QLabel("🌐")
            self.language_icon.setStyleSheet("font-size: 18px;")
            language_row.addWidget(self.language_icon, 0, Qt.AlignTop)
            self.language_label = QLabel()
            self.language_label.setStyleSheet("font-size: 18px; font-weight: 700;")
            language_row.addWidget(self.language_label, 1)
            language_card_layout.addLayout(language_row)

            self.language_combo = QComboBox()
            self.language_combo.setObjectName("SettingsCombo")
            self.language_combo.setMinimumHeight(46)
            self.language_combo.currentIndexChanged.connect(self.on_language_changed)
            language_card_layout.addWidget(self.language_combo)
            content_layout.addWidget(self.language_card)

            content_layout.addStretch(1)
            body_row.addWidget(content, 5)
            shell_layout.addLayout(body_row)
            self.retranslate_dialog()

        def set_theme_mode(self, mode: str) -> None:
            self.window.theme_mode = mode
            self.window.apply_theme()
            self.refresh_theme_buttons()

        def on_language_changed(self, index: int) -> None:
            self.window.language = "en" if index == 0 else "vi"
            self.window.retranslate_ui()
            self.retranslate_dialog()

        def retranslate_dialog(self) -> None:
            language = self.window.language
            self.setWindowTitle(tr(language, "settings_title"))
            self.dialog_title.setText(tr(language, "settings_title"))
            self.general_button.setText(f"⌂  {tr(language, 'general')}")
            self.section_title.setText(tr(language, "general"))
            self.appearance_label.setText(tr(language, "appearance"))
            self.language_label.setText(tr(language, "language"))
            self.theme_buttons["light"].setText(f"☼  {tr(language, 'light')}")
            self.theme_buttons["dark"].setText(f"☾  {tr(language, 'dark')}")
            self.theme_buttons["system"].setText(f"▣  {tr(language, 'system')}")
            self.refresh_theme_buttons()

            language_index = 0 if self.window.language == "en" else 1
            self.language_combo.blockSignals(True)
            self.language_combo.clear()
            self.language_combo.addItems([tr(language, "english"), tr(language, "vietnamese")])
            self.language_combo.setCurrentIndex(language_index)
            self.language_combo.blockSignals(False)

        def refresh_theme_buttons(self) -> None:
            active_mode = self.window.theme_mode
            preview_theme = active_mode
            if preview_theme == "system":
                preview_theme = self.window.detect_system_theme()
            dark_mode = preview_theme == "dark"
            for mode, button in self.theme_buttons.items():
                is_active = mode == active_mode
                if dark_mode:
                    button.setStyleSheet(
                        "background: rgba(37, 99, 255, 0.22); color: #f8fbff; border: 1px solid #2563ff; border-radius: 14px; font-weight: 600; font-size: 15px; text-align: center; padding: 0 14px;"
                        if is_active
                        else "background: rgba(15, 23, 42, 0.88); color: #e2e8f0; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; font-weight: 600; font-size: 15px; text-align: center; padding: 0 14px;"
                    )
                else:
                    button.setStyleSheet(
                        "background: #eef4ff; color: #0f172a; border: 1px solid #2563ff; border-radius: 14px; font-weight: 700; font-size: 15px; text-align: center; padding: 0 14px;"
                        if is_active
                        else "background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 14px; font-weight: 600; font-size: 15px; text-align: center; padding: 0 14px;"
                    )

class ImagePreviewDialog(QDialog):
        def __init__(self, image_path: str, parent: QWidget | None = None, effective_theme: str = "dark") -> None:
            super().__init__(parent)
            self.setWindowTitle("Image Preview")
            self.setModal(True)
            self.resize(1000, 800)
            self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setWindowOpacity(0.0)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(10, 10, 10, 10)
            shell = QFrame()
            shell.setObjectName("ImagePreviewShell")
            shell_layout = QVBoxLayout(shell)
            shell_layout.setContentsMargins(20, 20, 20, 20)
            layout.addWidget(shell)

            self.blur = QGraphicsBlurEffect()
            self.blur.setBlurRadius(20)
            shell.setGraphicsEffect(self.blur)

            if effective_theme == "light":
                shell.setStyleSheet("background: rgba(255,255,255,0.85); border-radius: 28px; border: 1px solid rgba(0,0,0,0.08);")
            else:
                shell.setStyleSheet("background: rgba(18,19,24,0.92); border-radius: 28px; border: 1px solid rgba(255,255,255,0.06);")
            self.scroll = QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setAlignment(Qt.AlignCenter)
            self.scroll.setStyleSheet("background: transparent; border: none; border-radius: 20px;")
            self.img_label = QLabel()
            self.img_label.setPixmap(QPixmap(image_path).scaled(960, 740, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.scroll.setWidget(self.img_label)
            shell_layout.addWidget(self.scroll)
            close_btn = QPushButton("✕", shell)
            close_btn.setObjectName("SettingsCloseButton")
            close_btn.setFixedSize(40, 40)
            close_btn.move(940, 10)
            close_btn.clicked.connect(self.accept)

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.accept()
            else:
                super().keyPressEvent(event)

        def showEvent(self, event):
            self.group = QParallelAnimationGroup(self)
            self.fade = QPropertyAnimation(self, b"windowOpacity")
            self.fade.setDuration(300)
            self.fade.setStartValue(0.0)
            self.fade.setEndValue(1.0)
            
            self.blur_anim = QPropertyAnimation(self.blur, b"blurRadius")
            self.blur_anim.setDuration(500)
            self.blur_anim.setStartValue(20)
            self.blur_anim.setEndValue(0)
            
            self.group.addAnimation(self.fade)
            self.group.addAnimation(self.blur_anim)
            self.group.start()
            super().showEvent(event)

