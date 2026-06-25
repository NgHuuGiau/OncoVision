from __future__ import annotations

import sys
import time
import random
import platform
from pathlib import Path

from app.chat_ui.medical_controller import MedicalChatController
from app.chat_ui.models import ChatMessage, Conversation
from app.chat_ui.medical_worker import build_patient_code, create_medical_worker_base
from app.chat_ui.paths import CHAT_HISTORY_DB_PATH
from app.chat_ui.storage import ChatDatabase
from utils.logger import get_logger


logger = get_logger(__name__)

try:
    from pygments import highlight  # noqa: F401
    from pygments.lexers import get_lexer_by_name, guess_lexer  # noqa: F401
    from pygments.formatters import HtmlFormatter  # noqa: F401
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None


def launch_chat_app(*, window_title: str, camera_index: int = 0, app_mode: str = "medium", selected_model: str | None = None) -> int:
    try:
        from PySide6.QtCore import (  # noqa: F401
            QParallelAnimationGroup,
            QPropertyAnimation,
            Qt,
            QTimer,
            Signal,
            QThread,
            QVariantAnimation,
            QEasingCurve,
            QPoint,
        )
        from PySide6.QtGui import (  # noqa: F401
            QAction,
            QPixmap,
            QColor,
            QShortcut,
            QKeySequence,
        )
        from PySide6.QtSvg import QSvgRenderer  # noqa: F401
        from PySide6.QtWidgets import (  # noqa: F401
            QApplication,
            QDialog,
            QFileDialog,
            QFrame,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMenu,
            QMessageBox,
            QSystemTrayIcon,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSpacerItem,
            QGraphicsDropShadowEffect,
            QGraphicsOpacityEffect,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("Missing PySide6. Install with: pip install PySide6 opencv-python") from exc

    from medical.chat_service import MedicalChatService

    MedicalAnalysisWorker = create_medical_worker_base(QThread, Signal)

    try:
        import numpy as np  # noqa: F401
        import pyaudio  # noqa: F401
        import wave  # noqa: F401
        import torch  # noqa: F401
        from faster_whisper import WhisperModel
        VOICE_LOCAL_AVAILABLE = True
    except ImportError:
        VOICE_LOCAL_AVAILABLE = False

    voice_model_cache: dict[tuple[str, str], object] = {}

    def get_cached_whisper_model(*, language: str, device: str):
        cache_key = (language, device)
        model = voice_model_cache.get(cache_key)
        if model is None:
            compute_type = "int8" if device == "cpu" else "float16"
            model = WhisperModel("base", device=device, compute_type=compute_type)
            voice_model_cache[cache_key] = model
        return model

    try:
        import cv2  # noqa: F401
    except ImportError:
        pass

    from app.chat_ui.content import translate
    from app.chat_ui.theme_styles import DARK_STYLESHEET, LIGHT_STYLESHEET
    from app.chat_ui.icons import themed_icon, themed_pixmap
    from app.chat_ui.voice_worker import build_voice_worker_class

    def tr(language: str, key: str) -> str:
        return translate(language, key)

    from app.chat_ui.dialogs import CameraCaptureDialog, SettingsDialog

    from app.chat_ui.widgets import (
        ChatBubble,
        ComposerPreviewThumb,
        HistoryItemWidget,
        MessageInput,
        RecordingPanel,
    )

    VoiceWorker = None
    if VOICE_LOCAL_AVAILABLE:
        VoiceWorker = build_voice_worker_class(
            qthread_cls=QThread,
            signal_cls=Signal,
            np_module=np,
            pyaudio_module=pyaudio,
            torch_module=torch,
            get_cached_whisper_model=get_cached_whisper_model,
        )

    class ChatWindow(QMainWindow):
        def __init__(self, *, title: str, initial_camera_index: int, mode_label: str, model_label: str | None) -> None:
            super().__init__()
            self.language = "vi"
            self.theme_mode = "system"
            self.effective_theme = "system"
            self.theme_overlay: QLabel | None = None
            self.theme_overlay_effect = None
            self.theme_transition_anim = None
            self.sidebar_expanded = True
            self.is_refreshing_history = False
            self.is_recording = False
            self.typing_timer = QTimer()
            self.initial_camera_index = initial_camera_index
            self.mode_label = mode_label
            self.model_label = model_label or ""
            self.pending_image_attachments: list[tuple[str, str]] = []
            self.conversations: list[Conversation] = []
            self.db = ChatDatabase(str(CHAT_HISTORY_DB_PATH))
            self.medical_controller = MedicalChatController(MedicalChatService())
            self.medical_worker: MedicalAnalysisWorker | None = None
            self.medical_status_message = ""

            self.language = self.db.get_setting("language", "vi")
            self.theme_mode = self.db.get_setting("theme", "system")
            
            self.active_conversation_index = 0
            self.setup_tray_icon()
            self.setup_voice_animation()
            self.setWindowTitle(title)
            self.resize(1480, 920)
            self.setAcceptDrops(True)
            self.build_ui()
            self._initialize_medical_status()
            self.conversations = self.db.get_all_conversations()
            if not self.conversations:
                self.conversations = [self._build_empty_conversation()]
            self.retranslate_ui()
            QTimer.singleShot(0, self.message_input.setFocus)

        def _build_empty_conversation(self) -> Conversation:
            return Conversation(
                title=tr(self.language, "new_chat"),
                subtitle=tr(self.language, "today"),
                messages=[],
                id=None,
            )

        def setup_tray_icon(self) -> None:
            self.tray_icon = QSystemTrayIcon(self)
            self.tray_icon.setIcon(themed_icon("sidebar_app.svg", self.icon_color(), 24))
            self.tray_icon.show()

        def show_tray_notification(self, message: str) -> None:
            if QSystemTrayIcon.supportsMessages() and hasattr(self, "tray_icon"):
                snippet = (message[:60] + "...") if len(message) > 60 else message
                self.tray_icon.showMessage(
                    tr(self.language, "new_message"),
                    snippet,
                    QSystemTrayIcon.Information,
                    3000
                )

        def _initialize_medical_status(self) -> None:
            try:
                model_path = self.medical_controller.ensure_ready()
                self.medical_status_message = (
                    f"Medical model \u0111\u00e3 s\u1eb5n s\u00e0ng: {Path(model_path).name}"
                )
            except Exception as exc:
                self.medical_status_message = (
                    f"Medical model ch\u01b0a s\u1eb5n s\u00e0ng: {exc}"
                )

        def scroll_to_bottom(self) -> None:
            if hasattr(self, "scroll_area"):
                bar = self.scroll_area.verticalScrollBar()
                QTimer.singleShot(50, lambda: bar.setValue(bar.maximum()))

        def setup_voice_animation(self) -> None:
            self.pulse_anim = QVariantAnimation(self)
            self.pulse_anim.setDuration(700)
            self.pulse_anim.setStartValue(QColor("#FF5252"))  # Bright red
            self.pulse_anim.setEndValue(QColor("#7F2929"))  # Dark red
            self.pulse_anim.setEasingCurve(QEasingCurve.InOutSine)
            self.pulse_anim.setLoopCount(-1)
            self.pulse_anim.valueChanged.connect(self.update_mic_style)

        def update_mic_style(self, color: QColor) -> None:
            self.micro_button.setStyleSheet(f"""
                QPushButton#RoundButton {{
                    background-color: {color.name()};
                    border: 2px solid {color.name()};
                }}
            """)

        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()

        def dropEvent(self, event):
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.handle_dropped_image(path)

        def handle_dropped_image(self, path: str):
            self.queue_image_attachment(path, "image")

        def build_ui(self) -> None:
            root_widget = QWidget()
            root_widget.setObjectName("Root")
            
            self.sidebar_shadow = QGraphicsDropShadowEffect(root_widget)
            self.sidebar_shadow.setBlurRadius(30)
            self.sidebar_shadow.setXOffset(10)
            self.sidebar_shadow.setYOffset(0)
            self.sidebar_shadow.setColor(QColor(0, 0, 0, 100))
            
            root = QHBoxLayout(root_widget)
            root.setContentsMargins(14, 14, 14, 14)
            root.setSpacing(16)
            self.setCentralWidget(root_widget)

            self.sidebar = QFrame()
            self.sidebar.setObjectName("Sidebar")
            self.sidebar.setMinimumWidth(240)
            self.sidebar.setMaximumWidth(240)
            self.sidebar.setGraphicsEffect(self.sidebar_shadow)
            sidebar_layout = QVBoxLayout(self.sidebar)
            sidebar_layout.setContentsMargins(14, 14, 14, 14)
            sidebar_layout.setSpacing(14)

            header_frame = QFrame()
            header_frame.setObjectName("SidebarHeader")
            header = QHBoxLayout(header_frame)
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(12)
            self.sidebar_header_layout = header
            self.sidebar_header_left_spacer = QWidget()
            self.sidebar_header_left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.sidebar_app_button = QPushButton()
            self.sidebar_app_button.setObjectName("SidebarAppButton")
            self.sidebar_app_button.setIconSize(QSize(28, 28))
            self.sidebar_app_button.setCursor(Qt.PointingHandCursor)
            self.sidebar_app_button.clicked.connect(self.toggle_sidebar)
            self.brand_text = QLabel("YOLO Chat AI")
            self.brand_text.setObjectName("BrandText")
            self.sidebar_header_right_spacer = QWidget()
            self.sidebar_header_right_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.sidebar_toggle_button = QPushButton()
            self.sidebar_toggle_button.setObjectName("SidebarToggleButton")
            self.sidebar_toggle_button.setFixedSize(28, 28)
            self.sidebar_toggle_button.hide()
            header.addWidget(self.brand_text, 0, Qt.AlignVCenter | Qt.AlignLeft)
            header.addWidget(self.sidebar_header_right_spacer, 1)
            header.addWidget(self.sidebar_app_button, 0, Qt.AlignCenter)
            header.addWidget(self.sidebar_header_left_spacer, 1)
            header.addWidget(self.sidebar_toggle_button, 0, Qt.AlignRight)
            sidebar_layout.addWidget(header_frame)

            self.new_chat_button = QPushButton()
            self.new_chat_button.setObjectName("SidebarPrimaryButton")
            self.new_chat_button.setIconSize(QSize(20, 20))
            self.new_chat_button.clicked.connect(self.start_new_chat)
            sidebar_layout.addWidget(self.new_chat_button)

            self.search_box = QFrame()
            self.search_box.setObjectName("SearchBox")
            search_layout = QHBoxLayout(self.search_box)
            search_layout.setContentsMargins(14, 10, 14, 10)
            search_layout.setSpacing(10)
            self.search_icon = QLabel()
            self.search_icon.setFixedSize(18, 18)
            search_layout.addWidget(self.search_icon)
            self.search_input = QLineEdit()
            self.search_input.setFrame(False)
            self.search_input.setStyleSheet("border: none; background: transparent; padding: 0;")
            self.search_input.textChanged.connect(self.refresh_history)
            search_layout.addWidget(self.search_input, 1)
            self.search_filter_label = QLabel("☰")
            self.search_filter_label.setObjectName("Subtle")
            self.search_filter_label.setAlignment(Qt.AlignCenter)
            self.search_filter_label.setFixedSize(18, 18)
            search_layout.addWidget(self.search_filter_label)
            sidebar_layout.addWidget(self.search_box)

            self.search_compact_button = QPushButton()
            self.search_compact_button.setObjectName("SidebarCompactSearchButton")
            self.search_compact_button.setIconSize(QSize(22, 22))
            self.search_compact_button.setToolTip(tr(self.language, "search"))
            self.search_compact_button.clicked.connect(self.focus_search)
            sidebar_layout.addWidget(self.search_compact_button, 0, Qt.AlignHCenter)

            self.history_title = QLabel()
            self.history_title.setObjectName("SectionTitle")
            sidebar_layout.addWidget(self.history_title)

            self.history_panel = QFrame()
            self.history_panel.setObjectName("HistoryPanel")
            history_panel_layout = QVBoxLayout(self.history_panel)
            history_panel_layout.setContentsMargins(0, 0, 0, 0)
            history_panel_layout.setSpacing(0)

            self.history_list = QListWidget()
            self.history_list.setFrameShape(QFrame.NoFrame)
            self.history_list.setSpacing(10)
            self.history_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Ẩn thanh cuộn dọc
            self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # Ẩn thanh cuộn ngang
            self.history_list.setContextMenuPolicy(Qt.CustomContextMenu) # Kích hoạt menu ngữ cảnh
            self.history_list.customContextMenuRequested.connect(self.show_history_context_menu) # Kết nối sự kiện chuột phải
            self.history_list.currentRowChanged.connect(self.select_conversation)
            history_panel_layout.addWidget(self.history_list)
            sidebar_layout.addWidget(self.history_panel)

            self.sidebar_spacer = QWidget()
            self.sidebar_spacer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            sidebar_layout.addWidget(self.sidebar_spacer)

            self.settings_button = QPushButton()
            self.settings_button.setObjectName("SidebarFooterButton")
            self.settings_button.setIconSize(QSize(20, 20))
            self.settings_button.clicked.connect(self.open_settings)
            sidebar_layout.addWidget(self.settings_button)
            root.addWidget(self.sidebar, 0)

            self.chat_panel = QWidget()
            self.chat_panel.setObjectName("ChatPanel")
            chat_layout = QVBoxLayout(self.chat_panel)
            chat_layout.setContentsMargins(18, 12, 18, 16)
            chat_layout.setSpacing(18)

            top_row = QHBoxLayout()
            top_row.setSpacing(12)
            top_row.addStretch(1)
            self.light_mode_button = QPushButton()
            self.light_mode_button.setObjectName("ModeButton")
            self.light_mode_button.setMinimumSize(86, 40)
            self.light_mode_button.clicked.connect(lambda: self.set_theme_mode("light"))
            top_row.addWidget(self.light_mode_button)
            self.dark_mode_button = QPushButton()
            self.dark_mode_button.setObjectName("ModeButton")
            self.dark_mode_button.setMinimumSize(72, 40)
            self.dark_mode_button.clicked.connect(lambda: self.set_theme_mode("dark"))
            top_row.addWidget(self.dark_mode_button)
            self.desktop_button = QPushButton()
            self.desktop_button.setObjectName("ModeButton")
            self.desktop_button.setMinimumSize(220, 40)
            top_row.addWidget(self.desktop_button)
            chat_layout.addLayout(top_row)

            self.greeting_card = QFrame()
            self.greeting_card.setObjectName("GreetingCard")
            greeting_layout = QVBoxLayout(self.greeting_card)
            greeting_layout.setContentsMargins(12, 6, 12, 0)
            greeting_layout.setSpacing(8)
            self.greeting_title = QLabel()
            self.greeting_title.setObjectName("GreetingTitle")
            self.greeting_text = QLabel()
            self.greeting_text.setObjectName("Subtle")
            self.greeting_text.setStyleSheet("font-size: 16px;")
            greeting_layout.addWidget(self.greeting_title)
            self.greeting_title.setAlignment(Qt.AlignLeft)
            greeting_layout.addWidget(self.greeting_text)
            self.greeting_text.setAlignment(Qt.AlignLeft)
            chat_layout.addWidget(self.greeting_card)

            self.empty_state = QFrame()
            self.empty_state.setObjectName("ChatBoard")
            empty_layout = QVBoxLayout(self.empty_state)
            empty_layout.setContentsMargins(24, 32, 24, 24)
            empty_layout.setSpacing(12)
            empty_layout.addStretch(1)
            self.robot_mark = QLabel("🤖")
            self.robot_mark.setAlignment(Qt.AlignCenter)
            self.robot_mark.setStyleSheet("font-size: 86px;")
            empty_layout.addWidget(self.robot_mark)
            self.empty_title = QLabel("Bắt đầu cuộc trò chuyện")
            self.empty_title.setAlignment(Qt.AlignCenter)
            self.empty_title.setStyleSheet("font-size: 20px; font-weight: 700;")
            empty_layout.addWidget(self.empty_title)
            self.empty_subtitle = QLabel("Đặt câu hỏi cho YOLO Chat AI để nhận câu trả lời hữu ích!")
            self.empty_subtitle.setObjectName("Subtle")
            self.empty_subtitle.setAlignment(Qt.AlignCenter)
            empty_layout.addWidget(self.empty_subtitle)
            empty_layout.addStretch(1)
            chat_layout.addWidget(self.empty_state, 1)

            self.scroll_area = QScrollArea()
            self.scroll_area.setObjectName("MessageScroll")
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setFrameShape(QFrame.NoFrame)
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.messages_host = QWidget()
            self.messages_host.setObjectName("MessagesHost")
            self.messages_layout = QVBoxLayout(self.messages_host)
            self.messages_layout.setContentsMargins(0, 6, 0, 6)
            self.messages_layout.setSpacing(16)
            self.messages_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
            self.scroll_area.setWidget(self.messages_host)
            self.scroll_area.hide()
            chat_layout.addWidget(self.scroll_area, 1)

            self.recording_panel = RecordingPanel(self.language, window=self)
            self.recording_panel.hide()
            chat_layout.addWidget(self.recording_panel, 0, Qt.AlignCenter)

            self.composer = QFrame()
            self.composer.setObjectName("Composer")
            composer_layout = QVBoxLayout(self.composer)
            composer_layout.setContentsMargins(8, 8, 8, 8)
            composer_layout.setSpacing(6)
            self.composer.setMinimumHeight(82)
            self.composer.setMaximumHeight(180)

            self.image_preview_area = QScrollArea()
            self.image_preview_area.setObjectName("ComposerPreviewScroll")
            self.image_preview_area.setWidgetResizable(True)
            self.image_preview_area.setFrameShape(QFrame.NoFrame)
            self.image_preview_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.image_preview_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.image_preview_area.setFixedHeight(92)
            self.image_preview_area.hide()

            self.image_preview_host = QWidget()
            self.image_preview_host.setObjectName("ComposerPreviewHost")
            self.image_preview_layout = QHBoxLayout(self.image_preview_host)
            self.image_preview_layout.setContentsMargins(2, 2, 2, 2)
            self.image_preview_layout.setSpacing(8)
            self.image_preview_layout.addStretch(1)
            self.image_preview_area.setWidget(self.image_preview_host)
            composer_layout.addWidget(self.image_preview_area)

            self.message_input_row = QFrame()
            self.message_input_row.setObjectName("ComposerInputRow")
            input_row_layout = QHBoxLayout(self.message_input_row)
            input_row_layout.setContentsMargins(6, 4, 6, 4)
            input_row_layout.setSpacing(6)

            self.plus_button = QPushButton("")
            self.plus_button.setObjectName("RoundButton")
            self.plus_button.setFixedSize(44, 44)
            self.plus_button.clicked.connect(self.show_plus_menu)
            input_row_layout.addWidget(self.plus_button, 0, Qt.AlignVCenter)

            self.message_input = MessageInput()
            self.message_input.setObjectName("ComposerInput")
            self.message_input.setMinimumHeight(40)
            self.message_input.setMaximumHeight(88)
            self.message_input.setFrameShape(QFrame.NoFrame)
            self.message_input.setPlaceholderText(tr(self.language, "input_placeholder"))
            self.message_input.viewport().setAutoFillBackground(False)
            self.message_input.viewport().setStyleSheet("background: transparent;")
            self.message_input.enter_pressed.connect(self.send_message)
            input_row_layout.addWidget(self.message_input, 1, Qt.AlignVCenter)

            self.micro_button = QPushButton("")
            self.micro_button.setObjectName("RoundButton")
            self.micro_button.setFixedSize(44, 44)
            self.micro_button.clicked.connect(self.start_voice_input)
            input_row_layout.addWidget(self.micro_button, 0, Qt.AlignVCenter)

            self.send_button = QPushButton("")
            self.send_button.setObjectName("SendButton")
            self.send_button.setFixedSize(44, 44)
            self.send_button.clicked.connect(self.send_message)
            input_row_layout.addWidget(self.send_button, 0, Qt.AlignVCenter)
            composer_layout.addWidget(self.message_input_row)
            chat_layout.addWidget(self.composer)

            self.disclaimer_label = QLabel("Kết quả AI chỉ mang tính hỗ trợ. Hãy kiểm tra lại thông tin quan trọng với bác sĩ chuyên khoa.")
            self.disclaimer_label.setObjectName("Subtle")
            self.disclaimer_label.setAlignment(Qt.AlignCenter)
            chat_layout.addWidget(self.disclaimer_label)
            self.medical_status_label = QLabel(self.medical_status_message)
            self.medical_status_label.setObjectName("Subtle")
            self.medical_status_label.setAlignment(Qt.AlignCenter)
            chat_layout.addWidget(self.medical_status_label)

            self.search_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
            self.search_shortcut.activated.connect(self.focus_search)

            root.addWidget(self.chat_panel, 1)

        def apply_dark_theme(self) -> None:
            self.effective_theme = "dark"
            app = QApplication.instance()
            if app:
                app.setStyleSheet(DARK_STYLESHEET)
            self.apply_theme_assets()
            self.refresh_history()

        def apply_light_theme(self) -> None:
            self.effective_theme = "light"
            app = QApplication.instance()
            if app:
                app.setStyleSheet(LIGHT_STYLESHEET)
            self.apply_theme_assets()
            self.refresh_history()

        def detect_system_theme(self) -> str:
            if platform.system() == "Windows":
                try:
                    import winreg
                    registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
                    key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    return "light" if value == 1 else "dark"
                except Exception:
                    pass
            return "dark"

        def _resolved_theme_target(self) -> str:
            target = self.theme_mode
            if target == "system":
                target = self.detect_system_theme()
            return "light" if target == "light" else "dark"

        def _clear_theme_overlay(self) -> None:
            if self.theme_transition_anim is not None:
                try:
                    self.theme_transition_anim.stop()
                except Exception:
                    pass
            self.theme_transition_anim = None
            self.theme_overlay_effect = None
            if self.theme_overlay is not None:
                try:
                    self.theme_overlay.deleteLater()
                except Exception:
                    pass
            self.theme_overlay = None

        def _apply_theme_target(self, target: str) -> None:
            self.effective_theme = target
            app = QApplication.instance()
            if app:
                app.setStyleSheet(LIGHT_STYLESHEET if target == "light" else DARK_STYLESHEET)
            self.db.set_setting("theme", self.theme_mode)
            self.apply_theme_assets()
            self.refresh_history()

        def _animate_theme_overlay(self, snapshot: QPixmap) -> None:
            if snapshot.isNull():
                return
            self._clear_theme_overlay()
            overlay = QLabel(self)
            overlay.setObjectName("ThemeTransitionOverlay")
            overlay.setPixmap(snapshot)
            overlay.setScaledContents(True)
            overlay.setGeometry(self.rect())
            overlay.show()
            overlay.raise_()
            effect = QGraphicsOpacityEffect(overlay)
            effect.setOpacity(1.0)
            overlay.setGraphicsEffect(effect)

            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(280)
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            animation.setEasingCurve(QEasingCurve.InOutCubic)

            def _finish_overlay() -> None:
                self._clear_theme_overlay()

            animation.finished.connect(_finish_overlay)
            self.theme_overlay = overlay
            self.theme_overlay_effect = effect
            self.theme_transition_anim = animation
            animation.start()

        def apply_theme(self) -> None:
            target = self._resolved_theme_target()
            if target == self.effective_theme:
                self._apply_theme_target(target)
                return
            snapshot = self.grab()
            self._apply_theme_target(target)
            self.repaint()
            QApplication.processEvents()
            self._animate_theme_overlay(snapshot)

        def cycle_theme_mode(self) -> None:
            order = ["system", "dark", "light"]
            current = self.theme_mode if self.theme_mode in order else "system"
            next_index = (order.index(current) + 1) % len(order)
            self.theme_mode = order[next_index]
            self.apply_theme()

        def set_theme_mode(self, mode: str) -> None:
            self.theme_mode = mode
            self.apply_theme()

        def icon_color(self) -> str:
            return "#ECECF1" if self.effective_theme == "dark" else "#111827"

        def subtle_icon_color(self) -> str:
            return "#AAB0BC" if self.effective_theme == "dark" else "#6B7280"

        def apply_theme_assets(self) -> None:
            strong = self.icon_color()
            subtle = self.subtle_icon_color()
            self.sidebar_app_button.setIcon(themed_icon("sidebar_app.svg", strong, 28))
            self.new_chat_button.setIcon(themed_icon("new_chat.svg", strong, 22))
            self.search_compact_button.setIcon(themed_icon("search.svg", subtle, 22))
            self.settings_button.setIcon(themed_icon("settings.svg", strong, 22))
            self.search_icon.setPixmap(themed_pixmap("search.svg", subtle, 18))
            self.plus_button.setIcon(themed_icon("plus.svg", strong, 18))
            self.plus_button.setIconSize(QSize(18, 18))
            self.micro_button.setIcon(themed_icon("mic.svg", strong, 18))
            self.micro_button.setIconSize(QSize(18, 18))
            self.send_button.setIcon(themed_icon("send.svg", strong, 18))
            self.send_button.setIconSize(QSize(18, 18))
            self.message_input.apply_visual_style(dark_mode=self.effective_theme == "dark")
            self.recording_panel.setup_styles()
            self.refresh_topbar_buttons()
            self.refresh_empty_state_theme()
            for i in range(self.messages_layout.count()):
                item = self.messages_layout.itemAt(i)
                widget = item.widget()
                if isinstance(widget, ChatBubble):
                    widget.refresh_theme(self.effective_theme)

        def refresh_topbar_buttons(self) -> None:
            dark_mode = self.effective_theme == "dark"
            active = self.theme_mode
            palette = {
                "base": "#0f172a" if dark_mode else "#ffffff",
                "border": "rgba(255, 255, 255, 0.08)" if dark_mode else "#e2e8f0",
                "text": "#f8fafc" if dark_mode else "#0f172a",
                "selected_bg": "rgba(37, 99, 255, 0.22)" if dark_mode else "#eef4ff",
                "selected_text": "#f8fbff" if dark_mode else "#2563ff",
                "selected_border": "#2563ff",
            }
            buttons = (
                (self.light_mode_button, "light"),
                (self.dark_mode_button, "dark"),
            )
            for button, mode in buttons:
                is_active = active == mode
                button.setStyleSheet(
                    f"background: {palette['selected_bg'] if is_active else palette['base']};"
                    f"color: {palette['selected_text'] if is_active else palette['text']};"
                    f"border: 1px solid {palette['selected_border'] if is_active else palette['border']};"
                    "border-radius: 14px; font-size: 14px; font-weight: 600; padding: 0 16px;"
                )
            self.desktop_button.setStyleSheet(
                f"background: {palette['base']}; color: {palette['text']}; border: 1px solid {palette['border']};"
                "border-radius: 14px; font-size: 14px; font-weight: 600; padding: 0 16px;"
            )

        def runtime_badge_text(self) -> str:
            parts = ["Desktop"]
            mode_label = str(self.mode_label or "").strip()
            if mode_label and mode_label.lower() not in {"desktop", "ui"}:
                parts.append(mode_label)
            model_label = Path(str(self.model_label)).name if self.model_label else ""
            if model_label:
                parts.append(model_label)
            return "▣  " + " | ".join(parts)

        def refresh_empty_state_theme(self) -> None:
            if self.effective_theme == "dark":
                self.robot_mark.setStyleSheet("font-size: 86px; color: #3b82f6;")
                self.empty_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f8fafc;")
            else:
                self.robot_mark.setStyleSheet("font-size: 86px; color: #2563ff;")
                self.empty_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #0f172a;")

        def retranslate_ui(self) -> None:
            self.new_chat_button.setText(tr(self.language, "new_chat"))
            self.search_input.setPlaceholderText(tr(self.language, "search"))
            self.history_title.setText(tr(self.language, "history"))
            self.settings_button.setText(tr(self.language, "settings"))
            self.search_input.setPlaceholderText(tr(self.language, "search"))
            self.greeting_title.setText(tr(self.language, "greeting_title"))
            self.db.set_setting("language", self.language)
            self.greeting_text.setText(tr(self.language, "greeting_text"))
            self.message_input.setPlaceholderText(tr(self.language, "input_placeholder"))
            self.plus_button.setToolTip(tr(self.language, "attach"))
            self.light_mode_button.setText(f"☼  {tr(self.language, 'light')}")
            self.dark_mode_button.setText(f"☾  {tr(self.language, 'dark')}")
            self.desktop_button.setText(self.runtime_badge_text())
            self.desktop_button.setToolTip(self.runtime_badge_text())
            if hasattr(self, "medical_status_label"):
                self.medical_status_label.setText(self.medical_status_message)
            self.update_sidebar_ui()
            self.refresh_history()
            self.apply_theme()
            self.render_messages()

        def toggle_sidebar(self) -> None:
            target_width = 88 if self.sidebar_expanded else 240
            
            self.sidebar_anim = QPropertyAnimation(self.sidebar, b"minimumWidth")
            self.sidebar_anim.setDuration(250)
            self.sidebar_anim.setStartValue(self.sidebar.width())
            self.sidebar_anim.setEndValue(target_width)
            self.sidebar_anim.setEasingCurve(QEasingCurve.InOutQuad)
            
            self.sidebar_max_anim = QPropertyAnimation(self.sidebar, b"maximumWidth")
            self.sidebar_max_anim.setDuration(250)
            self.sidebar_max_anim.setStartValue(self.sidebar.width())
            self.sidebar_max_anim.setEndValue(target_width)
            self.sidebar_max_anim.setEasingCurve(QEasingCurve.InOutQuad)

            self.sidebar_group = QParallelAnimationGroup()
            self.sidebar_group.addAnimation(self.sidebar_anim)
            self.sidebar_group.addAnimation(self.sidebar_max_anim)
            
            def on_finished():
                self.sidebar_expanded = not self.sidebar_expanded
                self.update_sidebar_ui()
            
            self.sidebar_group.finished.connect(on_finished)
            self.sidebar_group.start()

        def focus_search(self) -> None:
            if not self.sidebar_expanded:
                self.sidebar_expanded = True
                self.update_sidebar_ui()
            self.search_input.setFocus()

        def update_sidebar_ui(self) -> None:
            expanded = self.sidebar_expanded
            self.brand_text.setVisible(expanded)
            self.search_box.setVisible(expanded)
            self.search_compact_button.setVisible(not expanded)
            self.history_title.setVisible(expanded)
            self.history_panel.setVisible(expanded)
            self.sidebar_header_left_spacer.setVisible(not expanded)
            self.sidebar_header_right_spacer.setVisible(True)
            self.sidebar_header_layout.setSpacing(12 if expanded else 0)
            self.sidebar_app_button.setToolTip(
                tr(self.language, "sidebar_collapse") if expanded else tr(self.language, "sidebar_expand")
            )
            self.sidebar.layout().setStretchFactor(self.history_panel, 1 if expanded else 0)
            self.sidebar.layout().setStretchFactor(self.sidebar_spacer, 0 if expanded else 1)

            for button in (self.new_chat_button, self.settings_button):
                button.setObjectName(
                    ("SidebarPrimaryButton" if button is self.new_chat_button else "SidebarFooterButton")
                    if expanded
                    else "SidebarCompactButton"
                )
                button.style().unpolish(button)
                button.style().polish(button)

            if expanded:
                self.new_chat_button.setText(tr(self.language, "new_chat"))
                self.settings_button.setText(tr(self.language, "settings"))
                self.search_compact_button.setToolTip("")
                self.new_chat_button.setToolTip("")
                self.settings_button.setToolTip("")
                self.new_chat_button.setIconSize(QSize(20, 20))
                self.settings_button.setIconSize(QSize(20, 20))
            else:
                self.new_chat_button.setText("")
                self.settings_button.setText("")
                self.search_compact_button.setToolTip(tr(self.language, "search"))
                self.new_chat_button.setToolTip(tr(self.language, "new_chat"))
                self.settings_button.setToolTip(tr(self.language, "settings"))
                self.new_chat_button.setIconSize(QSize(22, 22))
                self.settings_button.setIconSize(QSize(22, 22))

        def active_conversation(self) -> Conversation:
            if not self.conversations:
                self.conversations.append(
                    Conversation(
                        title=tr(self.language, "new_chat"),
                        subtitle=tr(self.language, "today"),
                        messages=[],
                        id=None,
                    )
                )
                self.active_conversation_index = 0
            return self.conversations[self.active_conversation_index]

        def refresh_history(self) -> None:
            query = self.search_input.text().strip().lower() if hasattr(self, "search_input") else ""

            matching_conv_ids = set()
            if query:
                matching_conv_ids = set(self.db.search_conversations_by_message(query))

            self.is_refreshing_history = True
            self.history_list.blockSignals(True)
            self.history_list.clear()
            current_item_row = 0
            visible_row = 0
            for index, conversation in enumerate(self.conversations):
                if conversation.id is None and not conversation.messages:
                    continue
                matches_title = query in conversation.title.lower()
                matches_message = conversation.id in matching_conv_ids
                if query and not (matches_title or matches_message):
                    continue

                item = QListWidgetItem()
                item.setData(Qt.UserRole, index)
                item.setSizeHint(QSize(0, 68))
                self.history_list.addItem(item)
                widget = HistoryItemWidget(
                    conversation.title,
                    conversation.subtitle,
                    icon_pixmap=themed_pixmap("chat_history.svg", self.subtle_icon_color(), 20),
                    selected=index == self.active_conversation_index,
                )
                self.history_list.setItemWidget(item, widget)
                if index == self.active_conversation_index:
                    current_item_row = visible_row
                visible_row += 1
            self.history_list.blockSignals(False)
            if self.history_list.count():
                self.history_list.setCurrentRow(min(current_item_row, self.history_list.count() - 1))
            self.is_refreshing_history = False

        def select_conversation(self, row: int) -> None:
            if self.is_refreshing_history:
                return
            item = self.history_list.item(row)
            if item is None:
                return
            source_index = item.data(Qt.UserRole)
            if source_index is None:
                return
            self.active_conversation_index = int(source_index)
            self.render_messages()

        def show_history_context_menu(self, pos: QPoint) -> None:
            item = self.history_list.itemAt(pos)
            if not item:
                return
            
            index = item.data(Qt.UserRole) # Lấy chỉ mục thực của đoạn chat
            menu = QMenu(self)
            delete_action = QAction(tr(self.language, "delete_chat"), self)
            delete_action.triggered.connect(lambda: self.delete_conversation(index))
            menu.addAction(delete_action)
            menu.exec(self.history_list.mapToGlobal(pos))

        def delete_conversation(self, index_to_delete: int) -> None:
            if not (0 <= index_to_delete < len(self.conversations)):
                return
            
            conv_id = self.conversations[index_to_delete].id
            if conv_id is not None:
                self.db.delete_conversation(conv_id)

            self.conversations.pop(index_to_delete)
            if not self.conversations:
                self.conversations.append(self._build_empty_conversation())
                self.active_conversation_index = 0
            elif self.active_conversation_index >= index_to_delete:
                self.active_conversation_index = max(0, self.active_conversation_index - 1)
            
            self.refresh_history()
            self.render_messages()

        def clear_all_history(self) -> None:
            self.db.clear_all_conversations()
            self.conversations = []
            self.start_new_chat()
            self.refresh_history()
            self.render_messages()

        def render_messages(self) -> None:
            while self.messages_layout.count() > 1:
                item = self.messages_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            
            messages = self.active_conversation().messages
            self.greeting_card.setVisible(len(messages) == 0)
            self.empty_state.setVisible(len(messages) == 0)
            self.scroll_area.setVisible(len(messages) > 0)
            
            for message in messages:
                bubble = ChatBubble(message, language=self.language, align_right=message.sender == "user", window=self)
                self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
            QTimer.singleShot(
                0,
                lambda: self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum()
                ),
            )

        def start_new_chat(self) -> None:
            conversation = self._build_empty_conversation()
            self.conversations.insert(0, conversation)
            self.active_conversation_index = 0
            self.refresh_history()
            self.history_list.setCurrentRow(0)
            self.render_messages()

        def _add_popup_menu_button(
            self,
            menu: QMenu,
            *,
            icon_name: str,
            text: str,
            icon_color: str,
            callback,
        ) -> None:
            action = QWidgetAction(menu)
            button = QPushButton(text, menu)
            button.setObjectName("PopupMenuButton")
            button.setIcon(themed_icon(icon_name, icon_color, 18))
            button.setIconSize(QSize(18, 18))
            button.setCursor(Qt.PointingHandCursor)

            def trigger_action() -> None:
                menu.close()
                callback()

            button.clicked.connect(trigger_action)
            action.setDefaultWidget(button)
            menu.addAction(action)

        def show_plus_menu(self) -> None:
            menu = QMenu(self)
            strong = self.icon_color()
            menu.setMinimumWidth(210)
            self._add_popup_menu_button(
                menu,
                icon_name="image.svg",
                text=tr(self.language, "choose_image"),
                icon_color=strong,
                callback=self.pick_image,
            )
            self._add_popup_menu_button(
                menu,
                icon_name="camera.svg",
                text=tr(self.language, "camera"),
                icon_color=strong,
                callback=self.open_camera,
            )

            anchor = self.plus_button if hasattr(self, "plus_button") else self.send_button
            menu.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

        def add_message(self, message: ChatMessage) -> None:
            conversation = self.active_conversation()
            if conversation.id is None:
                conversation.id = self.db.create_conversation(conversation.title, conversation.subtitle)
            message.id = self.db.add_message(conversation.id, message)
            conversation.messages.append(message)
            if conversation.title == tr(self.language, "new_chat") and message.sender == "user":
                first_line = message.text.strip().splitlines()[0] if message.text.strip() else ""
                new_title = (first_line[:28] or tr(self.language, "new_chat")).strip()
                conversation.title = new_title
                self.db.update_conversation_title(conversation.id, new_title)
            conversation.subtitle = tr(self.language, "today")
            self.refresh_history()
            self.render_messages()

        def queue_image_attachment(self, path: str, attachment_kind: str = "image") -> None:
            self.pending_image_attachments.append((path, attachment_kind))
            self.refresh_image_previews()
            self.message_input.setFocus()

        def refresh_image_previews(self) -> None:
            while self.image_preview_layout.count() > 1:
                item = self.image_preview_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            has_attachments = bool(self.pending_image_attachments)
            self.image_preview_area.setVisible(has_attachments)
            self.composer.setMinimumHeight(152 if has_attachments else 82)

            if not has_attachments:
                return

            for index, (path, attachment_kind) in enumerate(self.pending_image_attachments):
                thumb_widget = ComposerPreviewThumb(
                    path=path,
                    attachment_kind=attachment_kind,
                    remove_callback=lambda _checked=False, idx=index: self.remove_pending_image_attachment(idx),
                )
                self.image_preview_layout.insertWidget(self.image_preview_layout.count() - 1, thumb_widget)

        def remove_pending_image_attachment(self, index: int) -> None:
            if not (0 <= index < len(self.pending_image_attachments)):
                return
            self.pending_image_attachments.pop(index)
            self.refresh_image_previews()
            self.message_input.setFocus()

        def clear_pending_image_previews(self) -> None:
            self.pending_image_attachments.clear()
            self.refresh_image_previews()

        def pending_attachment_prompt(self) -> str:
            if not self.pending_image_attachments:
                return ""
            if len(self.pending_image_attachments) == 1:
                _, attachment_kind = self.pending_image_attachments[0]
                return tr(self.language, "attach_camera_label") if attachment_kind == "camera" else tr(self.language, "attach_image_label")
            return f"{len(self.pending_image_attachments)} attachments"

        def send_message(self) -> None:
            if self.medical_controller.active:
                QMessageBox.information(self, tr(self.language, "info_title"), tr(self.language, "medical_pending"))
                return
            text = self.message_input.toPlainText().strip()
            if not text and not self.pending_image_attachments:
                QMessageBox.information(self, tr(self.language, "info_title"), tr(self.language, "empty_send"))
                return

            for index, (path, attachment_kind) in enumerate(self.pending_image_attachments):
                attachment_text = text if index == 0 and text else (
                    tr(self.language, "attach_camera_label") if attachment_kind == "camera" else tr(self.language, "attach_image_label")
                )
                self.add_message(
                    ChatMessage(
                        sender="user",
                        text=attachment_text,
                        attachment_path=path,
                        attachment_kind=attachment_kind,
                    )
                )

            if text and not self.pending_image_attachments:
                self.add_message(ChatMessage(sender="user", text=text))

            self.message_input.clear()
            prompt = text or self.pending_attachment_prompt()
            first_attachment = self.pending_image_attachments[0] if self.pending_image_attachments else None
            self.clear_pending_image_previews()
            self.generate_system_response(
                prompt,
                first_attachment[0] if first_attachment else None,
                first_attachment[1] if first_attachment else None,
            )

        def generate_system_response(self, prompt: str, attach_path: str = None, attach_kind: str = None):
            source = attach_kind or "chat"
            if source in {"image", "camera"} and attach_path:
                self._start_medical_analysis(prompt=prompt, attach_path=attach_path)
            else:
                self.add_message(ChatMessage(sender="assistant", text=self.build_system_reply(text=prompt, source=source)))
            self.scroll_to_bottom()

        def _start_medical_analysis(self, *, prompt: str, attach_path: str) -> None:
            if self.medical_controller.active:
                self.add_message(ChatMessage(sender="assistant", text=tr(self.language, "medical_pending")))
                return
            self.send_button.setEnabled(False)
            self.plus_button.setEnabled(False)
            self.micro_button.setEnabled(False)
            self.message_input.setEnabled(False)
            state, status_message = self.medical_controller.begin_analysis(tr(self.language, "medical_analyzing"))
            self.message_input.setPlaceholderText(state.placeholder)
            self.add_message(status_message)
            self.medical_worker = MedicalAnalysisWorker(
                self.medical_controller.service,
                image_path=attach_path,
                patient_code=self._build_patient_code(),
                user_prompt=prompt,
            )
            self.medical_worker.result_ready.connect(self._on_medical_analysis_success)
            self.medical_worker.error.connect(self._on_medical_analysis_error)
            self.medical_worker.finished.connect(self._on_medical_analysis_finished)
            self.medical_worker.start()

        def _on_medical_analysis_success(self, medical_response) -> None:
            self.add_message(
                ChatMessage(
                    sender="assistant",
                    text=medical_response.reply_text,
                    attachment_path=medical_response.attachment_path,
                    attachment_kind=medical_response.attachment_kind,
                    metadata_json=medical_response.metadata_json,
                )
            )
            self.scroll_to_bottom()

        def _on_medical_analysis_error(self, err: str) -> None:
            self.add_message(ChatMessage(sender="assistant", text=self._format_error(err)))
            self.scroll_to_bottom()

        def _on_medical_analysis_finished(self) -> None:
            state = self.medical_controller.finish_analysis(tr(self.language, "input_placeholder"))
            self.send_button.setEnabled(True)
            self.plus_button.setEnabled(True)
            self.micro_button.setEnabled(True)
            self.message_input.setEnabled(True)
            self.message_input.setPlaceholderText(state.placeholder)
            self.message_input.setFocus()
            self.medical_worker = None

        def _build_patient_code(self) -> str:
            conversation = self.active_conversation()
            return build_patient_code(conversation.id, int(time.time()))

        def start_typewriter(self, full_text: str):
            self.typewriter_content = full_text
            self.typewriter_idx = 0
            self.current_system_text = ""
            
            if self.typing_timer.isActive():
                self.typing_timer.stop()
            
            self.typewriter_tick()

        def typewriter_tick(self):
            if self.typewriter_idx < len(self.typewriter_content):
                char = self.typewriter_content[self.typewriter_idx]
                self.current_system_text += char
                self.update_last_message(self.current_system_text)
                self.typewriter_idx += 1
                
                delay = random.randint(10, 45)
                
                if char in ".?!":
                    delay += 400
                elif char in ",;:":
                    delay += 200
                
                QTimer.singleShot(delay, self.typewriter_tick)
                self.scroll_to_bottom()
            else:
                self.show_tray_notification(self.typewriter_content)

        def update_last_message(self, text: str):
            conv = self.active_conversation()
            if conv.messages:
                conv.messages[-1].text = text
                idx = self.messages_layout.count() - 2
                if idx >= 0:
                    item = self.messages_layout.itemAt(idx)
                    if item and item.widget():
                        last_bubble = item.widget()
                        if isinstance(last_bubble, ChatBubble):
                            last_bubble.update_display_text(text)
                            if text.startswith("Error:") or text.startswith("Phan tich loi:"):
                                self.shake_bubble(last_bubble)

        def shake_bubble(self, bubble: ChatBubble):
            orig_pos = bubble.pos()
            shake = QPropertyAnimation(bubble, b"pos", self)
            shake.setDuration(300)
            shake.setStartValue(orig_pos)
            shake.setKeyValueAt(0.15, orig_pos + QPoint(-8, 0))
            shake.setKeyValueAt(0.3, orig_pos + QPoint(8, 0))
            shake.setKeyValueAt(0.45, orig_pos + QPoint(-6, 0))
            shake.setKeyValueAt(0.6, orig_pos + QPoint(6, 0))
            shake.setKeyValueAt(0.75, orig_pos + QPoint(-3, 0))
            shake.setKeyValueAt(0.9, orig_pos + QPoint(3, 0))
            shake.setEndValue(orig_pos)
            shake.setEasingCurve(QEasingCurve.OutBounce)
            shake.start(QPropertyAnimation.DeleteWhenStopped)

        def start_voice_input(self) -> None:
            if not VOICE_LOCAL_AVAILABLE:
                return

            if self.is_recording:
                self.worker.stop()
                return

            self.is_recording = True
            self.message_input.setPlaceholderText(tr(self.language, "loading_model"))
            self.message_input.setEnabled(False)
            self.recording_panel.show()
            self.pulse_anim.start()

            self.worker = VoiceWorker(self.language)
            self.worker.result_ready.connect(self.on_voice_success)
            self.worker.intensity_changed.connect(self.recording_panel.waveform.set_intensity)
            self.worker.error.connect(self.on_voice_error)
            self.worker.finished.connect(self.on_voice_complete)
            self.worker.start()
            self.message_input.setPlaceholderText(tr(self.language, "recording_status"))

        def on_voice_success(self, text: str):
            if text:
                existing_text = self.message_input.toPlainText().strip()
                combined_text = f"{existing_text} {text}".strip() if existing_text else text
                self.message_input.setPlainText(combined_text)

        def on_voice_error(self):
            QMessageBox.warning(self, tr(self.language, "info_title"), tr(self.language, "voice_error"))

        def on_voice_complete(self):
            self.is_recording = False
            self.pulse_anim.stop()
            self.recording_panel.hide()
            self.micro_button.setStyleSheet("")
            self.micro_button.setIcon(themed_icon("mic.svg", self.icon_color(), 18))
            self.message_input.setPlaceholderText(tr(self.language, "input_placeholder"))
            self.message_input.setEnabled(True)
            self.message_input.setFocus()

        def build_system_reply(self, *, text: str, source: str) -> str:
            if source == "image":
                return tr(self.language, "system_reply_image")
            if source == "camera":
                return tr(self.language, "system_reply_camera")
            if source == "text_file":
                return tr(self.language, "system_reply_text")
            if source == "chat":
                return tr(self.language, "system_unavailable")
            return f"{tr(self.language, 'system_reply_text')} {text[:120]}".strip()

        def _format_error(self, err: str) -> str:
            if "does not support image input" in err or "Cannot read" in err:
                return tr(self.language, "image_model_error")
            return f"Phan tich loi: {err}"

        def pick_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                tr(self.language, "choose_image"),
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
            )
            if not path:
                return
            self.queue_image_attachment(path, "image")

        def open_camera(self) -> None:
            dialog = CameraCaptureDialog(
                language=self.language,
                camera_index_value=self.initial_camera_index,
                parent=self,
            )
            dialog.captured.connect(self.handle_camera_capture)
            dialog.exec()

        def handle_camera_capture(self, path: str) -> None:
            self.queue_image_attachment(path, "camera")

        def open_settings(self) -> None:
            SettingsDialog(parent_window=self).exec()

    app = QApplication.instance() or QApplication(sys.argv)
    window = ChatWindow(title=window_title, initial_camera_index=camera_index, mode_label=app_mode, model_label=selected_model)
    window.show()
    return app.exec()
