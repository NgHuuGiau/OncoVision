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
            QPropertyAnimation,
            Qt,
            QTimer,
            Signal,
            QThread,
            QVariantAnimation,
            QEasingCurve,
            QPoint,
            QSize,
        )
        from PySide6.QtGui import (  # noqa: F401
            QPixmap,
            QColor,
            QShortcut,
            QKeySequence,
        )
        from PySide6.QtSvg import QSvgRenderer  # noqa: F401
        from PySide6.QtWidgets import (  # noqa: F401
            QApplication,
            QDialog,
            QFrame,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
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
    from app.chat_ui.icons import themed_icon
    from app.chat_ui.theme_flow import (
        apply_dark_theme as apply_dark_theme_flow,
        apply_light_theme as apply_light_theme_flow,
        apply_theme_assets as apply_theme_assets_flow,
        icon_color as icon_color_flow,
        refresh_empty_state_theme as refresh_empty_state_theme_flow,
        refresh_topbar_buttons as refresh_topbar_buttons_flow,
        runtime_badge_text as runtime_badge_text_flow,
        subtle_icon_color as subtle_icon_color_flow,
    )
    from app.chat_ui.voice_worker import build_voice_worker_class

    def tr(language: str, key: str) -> str:
        return translate(language, key)

    from app.chat_ui.dialogs import CameraCaptureDialog, SettingsDialog

    from app.chat_ui.widgets import (
        ChatBubble,
    )
    from app.chat_ui.message_flow import (
        clear_pending_image_previews,
        handle_camera_capture as handle_camera_capture_flow,
        handle_dropped_image as handle_dropped_image_flow,
        pick_image as pick_image_flow,
        refresh_image_previews as refresh_image_previews_flow,
        remove_pending_image_attachment as remove_pending_image_attachment_flow,
        send_message as send_message_flow,
        submit_user_message as submit_user_message_flow,
    )
    from app.chat_ui.menu_flow import show_plus_menu as show_plus_menu_flow
    from app.chat_ui.composer_flow import build_composer_section
    from app.chat_ui.sidebar_flow import (
        clear_all_history as clear_all_history_flow,
        delete_conversation as delete_conversation_flow,
        refresh_history as refresh_history_flow,
        select_conversation as select_conversation_flow,
        show_history_context_menu as show_history_context_menu_flow,
        toggle_sidebar as toggle_sidebar_flow,
        update_sidebar_ui as update_sidebar_ui_flow,
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
            for conv in self.conversations:
                normalized = conv.title.strip().lower()
                if normalized in {"vi", "en", ""} or (len(normalized) <= 2 and normalized.isalpha()):
                    conv.title = tr(self.language, "new_chat")
                    if conv.id is not None:
                        self.db.update_conversation_title(conv.id, conv.title)
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
            handled = False
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.handle_dropped_image(path)
                    handled = True
            if handled:
                event.acceptProposedAction()
            else:
                event.ignore()

        def handle_dropped_image(self, path: str):
            handle_dropped_image_flow(self, path)

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
            self.sidebar.setMinimumWidth(280)
            self.sidebar.setMaximumWidth(280)
            self.sidebar.setGraphicsEffect(self.sidebar_shadow)
            sidebar_layout = QVBoxLayout(self.sidebar)
            sidebar_layout.setContentsMargins(12, 12, 12, 12)
            sidebar_layout.setSpacing(12)

            header_frame = QFrame()
            header_frame.setObjectName("SidebarHeader")
            header = QHBoxLayout(header_frame)
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(10)
            self.sidebar_header_layout = header
            self.sidebar_header_left_spacer = QWidget()
            self.sidebar_header_left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.sidebar_app_button = QPushButton()
            self.sidebar_app_button.setObjectName("SidebarAppButton")
            self.sidebar_app_button.setIconSize(QSize(28, 28))
            self.sidebar_app_button.setLayoutDirection(Qt.LeftToRight)
            self.sidebar_app_button.setCursor(Qt.PointingHandCursor)
            self.sidebar_app_button.clicked.connect(self.toggle_sidebar)
            self.brand_text = QLabel("OncoVision Chat AI")
            self.brand_text.setObjectName("BrandText")
            self.brand_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            self.brand_text.setStyleSheet("font-size: 12px; font-weight: 700;")
            self.brand_text.setWordWrap(False)
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
            self.new_chat_button.setLayoutDirection(Qt.LeftToRight)
            self.new_chat_button.clicked.connect(self.start_new_chat)
            self.new_chat_button.setText("Cuộc trò chuyện mới")
            sidebar_layout.addWidget(self.new_chat_button)

            self.search_box = QFrame()
            self.search_box.setObjectName("SearchBox")
            search_layout = QHBoxLayout(self.search_box)
            search_layout.setContentsMargins(12, 10, 12, 10)
            search_layout.setSpacing(10)
            self.search_icon = QLabel()
            self.search_icon.setFixedSize(18, 18)
            search_layout.addWidget(self.search_icon)
            self.search_input = QLineEdit()
            self.search_input.setFrame(False)
            self.search_input.setStyleSheet("border: none; background: transparent; padding: 0;")
            self.search_input.setPlaceholderText("Tìm kiếm cuộc trò chuyện...")
            self.search_input.textChanged.connect(self.refresh_history)
            search_layout.addWidget(self.search_input, 1)
            self.search_filter_label = QLabel("\u2630")
            self.search_filter_label.setObjectName("Subtle")
            self.search_filter_label.setAlignment(Qt.AlignCenter)
            self.search_filter_label.setFixedSize(18, 18)
            search_layout.addWidget(self.search_filter_label)
            self.search_box.setMinimumHeight(50)
            sidebar_layout.addWidget(self.search_box)

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
            self.history_list.setSpacing(8)
            self.history_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Ẩn thanh cuộn dọc
            self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Ẩn thanh cuộn ngang
            self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)  # Kích hoạt menu ngữ cảnh
            self.history_list.customContextMenuRequested.connect(self.show_history_context_menu)  # Kết nối sự kiện chuột phải
            self.history_list.currentRowChanged.connect(self.select_conversation)
            history_panel_layout.addWidget(self.history_list)
            sidebar_layout.addWidget(self.history_panel)

            self.sidebar_spacer = QWidget()
            self.sidebar_spacer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            sidebar_layout.addWidget(self.sidebar_spacer)

            self.settings_button = QPushButton()
            self.settings_button.setObjectName("SidebarFooterButton")
            self.settings_button.setIconSize(QSize(20, 20))
            self.settings_button.setLayoutDirection(Qt.LeftToRight)
            self.settings_button.clicked.connect(self.open_settings)
            self.settings_button.setText("Cài đặt")
            sidebar_layout.addWidget(self.settings_button)
            root.addWidget(self.sidebar, 0)

            self.chat_panel = QWidget()
            self.chat_panel.setObjectName("ChatPanel")
            chat_layout = QVBoxLayout(self.chat_panel)
            chat_layout.setContentsMargins(16, 10, 16, 12)
            chat_layout.setSpacing(14)

            top_row = QHBoxLayout()
            top_row.setSpacing(10)
            top_row.addStretch(1)
            self.light_mode_button = QPushButton()
            self.light_mode_button.setObjectName("ModeButton")
            self.light_mode_button.setMinimumSize(80, 38)
            self.light_mode_button.setLayoutDirection(Qt.LeftToRight)
            self.light_mode_button.clicked.connect(lambda: self.set_theme_mode("light"))
            top_row.addWidget(self.light_mode_button)
            self.dark_mode_button = QPushButton()
            self.dark_mode_button.setObjectName("ModeButton")
            self.dark_mode_button.setMinimumSize(70, 38)
            self.dark_mode_button.setLayoutDirection(Qt.LeftToRight)
            self.dark_mode_button.clicked.connect(lambda: self.set_theme_mode("dark"))
            top_row.addWidget(self.dark_mode_button)
            self.desktop_button = QPushButton()
            self.desktop_button.setObjectName("ModeButton")
            self.desktop_button.setMinimumSize(190, 38)
            self.desktop_button.setLayoutDirection(Qt.LeftToRight)
            top_row.addWidget(self.desktop_button)
            chat_layout.addLayout(top_row)

            self.greeting_card = QFrame()
            self.greeting_card.setObjectName("GreetingCard")
            greeting_layout = QVBoxLayout(self.greeting_card)
            greeting_layout.setContentsMargins(10, 4, 10, 0)
            greeting_layout.setSpacing(6)
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
            empty_layout.setContentsMargins(20, 24, 20, 20)
            empty_layout.setSpacing(10)
            empty_layout.addStretch(1)
            self.robot_mark = QLabel("🤖")
            self.robot_mark.setAlignment(Qt.AlignCenter)
            self.robot_mark.setStyleSheet("font-size: 86px;")
            empty_layout.addWidget(self.robot_mark)
            self.empty_title = QLabel("Bác sĩ AI luôn sẵn sàng hỗ trợ bạn")
            self.empty_title.setAlignment(Qt.AlignCenter)
            self.empty_title.setStyleSheet("font-size: 20px; font-weight: 700;")
            empty_layout.addWidget(self.empty_title)
            self.empty_subtitle = QLabel("Đặt câu hỏi về sức khỏe, triệu chứng, xét nghiệm, điều trị và nhận thông tin đáng tin cậy từ AI.")
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
            self.messages_layout.setSpacing(14)
            self.messages_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
            self.scroll_area.setWidget(self.messages_host)
            self.scroll_area.hide()
            chat_layout.addWidget(self.scroll_area, 1)

            build_composer_section(self, chat_layout)

            root.addWidget(self.chat_panel, 1)

        def apply_dark_theme(self) -> None:
            apply_dark_theme_flow(self, QApplication.instance())

        def apply_light_theme(self) -> None:
            apply_light_theme_flow(self, QApplication.instance())

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
            return icon_color_flow(self)

        def subtle_icon_color(self) -> str:
            return subtle_icon_color_flow(self)

        def apply_theme_assets(self) -> None:
            apply_theme_assets_flow(self)

        def refresh_topbar_buttons(self) -> None:
            refresh_topbar_buttons_flow(self)

        def runtime_badge_text(self) -> str:
            return runtime_badge_text_flow(self)

        def refresh_empty_state_theme(self) -> None:
            refresh_empty_state_theme_flow(self)

        def retranslate_ui(self) -> None:
            self.update_sidebar_texts()
            self.greeting_title.setText(tr(self.language, "greeting_title"))
            self.db.set_setting("language", self.language)
            self.greeting_text.setText(tr(self.language, "greeting_text"))
            self.search_input.setPlaceholderText("Tìm kiếm cuộc trò chuyện...")
            self.empty_title.setText("Bác sĩ AI luôn sẵn sàng hỗ trợ bạn")
            self.empty_subtitle.setText("Đặt câu hỏi về sức khỏe, triệu chứng, xét nghiệm, điều trị và nhận thông tin đáng tin cậy từ AI.")
            self.message_input.setPlaceholderText(tr(self.language, "input_placeholder"))
            self.plus_button.setToolTip(tr(self.language, "attach"))
            self.light_mode_button.setText(f"\u263c  {tr(self.language, 'light')}")
            self.dark_mode_button.setText(f"\u263e  {tr(self.language, 'dark')}")
            self.desktop_button.setText(self.runtime_badge_text())
            self.desktop_button.setToolTip(self.runtime_badge_text())
            if hasattr(self, "medical_status_label"):
                self.medical_status_label.setText(self.medical_status_message)
            self.update_sidebar_ui()
            self.refresh_history()
            self.apply_theme()
            self.render_messages()

        def toggle_sidebar(self) -> None:
            toggle_sidebar_flow(self)

        def update_sidebar_ui(self) -> None:
            update_sidebar_ui_flow(self)

        def update_sidebar_texts(self) -> None:
            from app.chat_ui.sidebar_flow import update_sidebar_texts as update_sidebar_texts_flow
            update_sidebar_texts_flow(self)

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
            refresh_history_flow(self)

        def select_conversation(self, row: int) -> None:
            select_conversation_flow(self, row)

        def show_history_context_menu(self, pos: QPoint) -> None:
            show_history_context_menu_flow(self, pos)

        def delete_conversation(self, index_to_delete: int) -> None:
            delete_conversation_flow(self, index_to_delete)

        def clear_all_history(self) -> None:
            clear_all_history_flow(self)

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

        def show_plus_menu(self) -> None:
            show_plus_menu_flow(self)

        def add_message(self, message: ChatMessage) -> None:
            conversation = self.active_conversation()
            if conversation.id is None:
                conversation.id = self.db.create_conversation(conversation.title, conversation.subtitle)
            message.id = self.db.add_message(conversation.id, message)
            conversation.messages.append(message)
            if conversation.title == tr(self.language, "new_chat") and message.sender == "user":
                first_line = message.text.strip().splitlines()[0] if message.text.strip() else ""
                attachment_labels = {
                    tr(self.language, "attach_image_label"),
                    tr(self.language, "attach_camera_label"),
                }
                if first_line and first_line not in attachment_labels and first_line not in {"vi", "en"} and not (len(first_line) <= 2 and first_line.isalpha()):
                    new_title = first_line[:28].strip()
                    conversation.title = new_title
                    self.db.update_conversation_title(conversation.id, new_title)
            conversation.subtitle = tr(self.language, "today")
            self.refresh_history()
            self.render_messages()

        def refresh_image_previews(self) -> None:
            refresh_image_previews_flow(self)

        def remove_pending_image_attachment(self, index: int) -> None:
            remove_pending_image_attachment_flow(self, index)

        def clear_pending_image_previews(self) -> None:
            clear_pending_image_previews(self)

        def pending_attachment_prompt(self) -> str:
            if not self.pending_image_attachments:
                return ""
            if len(self.pending_image_attachments) == 1:
                _, attachment_kind = self.pending_image_attachments[0]
                return tr(self.language, "attach_camera_label") if attachment_kind == "camera" else tr(self.language, "attach_image_label")
            return f"{len(self.pending_image_attachments)} attachments"

        def _submit_user_message(self, text: str = "", attachments: list[tuple[str, str]] | None = None) -> None:
            submit_user_message_flow(self, text, attachments)

        def send_message(self) -> None:
            send_message_flow(self)

        def generate_system_response(self, prompt: str, attach_path: str = None, attach_kind: str = None):
            source = attach_kind or "chat"
            if source in {"image", "camera"} and attach_path:
                self.add_message(
                    ChatMessage(
                        sender="assistant",
                        text=(
                            f"Đã nhận {source} để phân tích: {Path(attach_path).name}. "
                            "Hệ thống sẽ hiển thị ảnh gốc, ảnh đã xử lý và nhãn dự đoán ngay bên dưới."
                        ),
                        attachment_path=attach_path,
                        attachment_kind=source,
                    )
                )
            elif source == "chat":
                self.add_message(ChatMessage(sender="assistant", text=self.build_system_reply(text=prompt, source=source)))
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
            self._pending_voice_text = ""
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
                self._pending_voice_text = combined_text
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
            pending_voice_text = getattr(self, "_pending_voice_text", "").strip()
            if pending_voice_text:
                self._pending_voice_text = ""
                self._submit_user_message(pending_voice_text, [])

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
            pick_image_flow(self)

        def open_camera(self) -> None:
            dialog = CameraCaptureDialog(
                language=self.language,
                camera_index_value=self.initial_camera_index,
                parent=self,
            )
            dialog.captured.connect(self.handle_camera_capture)
            dialog.exec()

        def handle_camera_capture(self, path: str) -> None:
            handle_camera_capture_flow(self, path)

        def open_settings(self) -> None:
            SettingsDialog(parent_window=self).exec()

    app = QApplication.instance() or QApplication(sys.argv)
    window = ChatWindow(title=window_title, initial_camera_index=camera_index, mode_label=app_mode, model_label=selected_model)
    window.show()
    return app.exec()


