from __future__ import annotations

import math
import time
from pathlib import Path

from app.chat_ui.content import translate
from app.chat_ui.dialogs import ImagePreviewDialog
from app.chat_ui.image_utils import load_preview_pixmap
from app.chat_ui.models import ChatMessage
from utils.logger import get_logger

from PySide6.QtCore import QEasingCurve, QPoint, QRectF, Qt, QPropertyAnimation, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPalette, QPixmap, QTextOption
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = get_logger(__name__)


def tr(language: str, key: str) -> str:
    return translate(language, key)


class TypingIndicator(QWidget):
    def __init__(self, color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(60, 24)
        self.color = color
        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        t = time.time() - self.start_time
        for i in range(3):
            offset = math.sin(t * 7 - i * 0.9) * 4
            alpha = int(160 + 95 * math.sin(t * 7 - i * 0.9))
            color = QColor(self.color)
            color.setAlpha(max(0, min(255, alpha)))
            painter.setBrush(color)
            painter.drawEllipse(10 + i * 14, 12 + offset, 5, 5)


class HistoryItemWidget(QFrame):
    def __init__(
        self,
        title: str,
        subtitle: str,
        *,
        icon_pixmap: QPixmap | None = None,
        selected: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("HistoryItem")
        self.setProperty("selected", selected)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        icon_label = QLabel()
        if icon_pixmap is not None:
            icon_label.setPixmap(icon_pixmap)
        icon_label.setFixedSize(20, 20)
        layout.addWidget(icon_label, 0, Qt.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        title_label.setWordWrap(True)
        text_layout.addWidget(title_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("Subtle")
        text_layout.addWidget(subtitle_label)

        layout.addLayout(text_layout, 1)
        self.setFixedHeight(68)


class WaveformWidget(QWidget):
    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(100, 30)
        self.values = [0] * 12
        self.color = QColor(color)

    def set_intensity(self, value: int) -> None:
        self.values.pop(0)
        self.values.append(value)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        spacing = 4
        bar_width = (width - (len(self.values) - 1) * spacing) / len(self.values)
        for index, value in enumerate(self.values):
            bar_height = max(4, (value / 100.0) * height)
            x = index * (bar_width + spacing)
            y = (height - bar_height) / 2
            painter.setBrush(self.color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(x, y, bar_width, bar_height), 2, 2)


class RecordingPanel(QFrame):
    def __init__(self, language: str, parent: QWidget | None = None, window: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RecordingPanel")
        self._window = window
        self.setGraphicsEffect(
            QGraphicsDropShadowEffect(blurRadius=15, xOffset=0, yOffset=4, color=QColor(0, 0, 0, 60))
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        self.waveform = WaveformWidget("#FF5252")
        self.waveform.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.setAlignment(self.waveform, Qt.AlignVCenter)
        layout.addWidget(self.waveform)

        self.label = QLabel(tr(language, "recording_status"))
        self.label.setStyleSheet("color: #FF5252; font-weight: 700; font-size: 14px;")
        self.label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        layout.addWidget(self.label)
        self.hide()

    def setup_styles(self) -> None:
        if self._window and getattr(self._window, "effective_theme", "dark") == "light":
            self.setStyleSheet("background: transparent; border: none;")
            self.label.setStyleSheet("color: #0f172a; font-weight: 700; font-size: 14px;")
            self.waveform.color = QColor("#ef4444")
            self.waveform.update()
            return
        self.setStyleSheet("background: transparent; border: none;")
        self.label.setStyleSheet("color: white; font-weight: 700; font-size: 14px;")


class MessageInput(QPlainTextEdit):
    enter_pressed = Signal()
    MAX_VISIBLE_LINES = 5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.textChanged.connect(self._adjust_height)
        self._height_animation = QVariantAnimation(self)
        self._height_animation.setDuration(120)
        self._height_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._height_animation.valueChanged.connect(lambda value: self.setFixedHeight(int(value)))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setCenterOnScroll(False)
        self.setCursorWidth(3)
        self.document().setDocumentMargin(4)
        self.setContentsMargins(0, 0, 0, 0)
        self.setTabChangesFocus(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setPlaceholderText("Nhập tin nhắn...")
        self.setFixedHeight(40)
        self._adjust_height()

    def apply_visual_style(self, *, dark_mode: bool) -> None:
        palette = self.palette()
        if dark_mode:
            palette.setColor(QPalette.Text, QColor("#F3F4F6"))
            palette.setColor(QPalette.PlaceholderText, QColor("#C2CAD6"))
            palette.setColor(QPalette.Base, QColor(0, 0, 0, 0))
            palette.setColor(QPalette.Highlight, QColor("#4DB8FF"))
            palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
            self.setStyleSheet(
                "background: transparent;"
                "border: none;"
                "color: #F3F4F6;"
                "selection-background-color: #4DB8FF;"
                "selection-color: #FFFFFF;"
                "font-size: 16px;"
                "padding: 6px 4px;"
            )
        else:
            palette.setColor(QPalette.Text, QColor("#111827"))
            palette.setColor(QPalette.PlaceholderText, QColor("#6B7280"))
            palette.setColor(QPalette.Base, QColor(0, 0, 0, 0))
            palette.setColor(QPalette.Highlight, QColor("#93C5FD"))
            palette.setColor(QPalette.HighlightedText, QColor("#111827"))
            self.setStyleSheet(
                "background: transparent;"
                "border: none;"
                "color: #111827;"
                "selection-background-color: #93C5FD;"
                "selection-color: #111827;"
                "font-size: 16px;"
                "padding: 6px 4px;"
            )
        self.setPalette(palette)
        self.viewport().setPalette(palette)
        self.viewport().setStyleSheet("background: transparent;")

    def _adjust_height(self) -> None:
        self.document().setTextWidth(self.viewport().width())
        lines = max(1.0, self.document().size().height())
        height = (lines * self.fontMetrics().lineSpacing()) + self.contentsMargins().top() + self.contentsMargins().bottom() + 12
        max_height = max(40, int(self.fontMetrics().lineSpacing() * self.MAX_VISIBLE_LINES + 18))
        target_height = max(40, min(int(height), max_height))
        if target_height == self.height():
            return
        self._height_animation.stop()
        self._height_animation.setStartValue(self.height())
        self._height_animation.setEndValue(target_height)
        self._height_animation.start()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._adjust_height()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.enter_pressed.emit()
            return
        super().keyPressEvent(event)


class ComposerPreviewThumb(QFrame):
    def __init__(self, *, path: str, attachment_kind: str, remove_callback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerPreviewThumb")
        self.setFixedSize(72, 72)
        self.remove_callback = remove_callback

        thumb_layout = QVBoxLayout(self)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(0)

        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setFixedSize(72, 72)
        self.thumb_label.setToolTip(path)
        pixmap = load_preview_pixmap(path, max_size=(72, 72))
        if not pixmap.isNull():
            self.thumb_label.setPixmap(pixmap.scaled(72, 72, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            self.thumb_label.setText("Ảnh" if attachment_kind == "image" else "Chụp")
            self.thumb_label.setStyleSheet("font-size: 12px; font-weight: 700;")
        thumb_layout.addWidget(self.thumb_label)

        self.delete_button = QPushButton("x", self)
        self.delete_button.setObjectName("ComposerPreviewDeleteButton")
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.setFixedSize(24, 24)
        self.delete_button.move(self.width() - 28, 4)
        self.delete_button.hide()
        self.delete_button.clicked.connect(self.remove_callback)

    def enterEvent(self, event) -> None:
        self.delete_button.show()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.delete_button.hide()
        super().leaveEvent(event)


class ChatBubble(QWidget):
    def __init__(
        self,
        message: ChatMessage,
        *,
        language: str,
        align_right: bool,
        parent: QWidget | None = None,
        window: QMainWindow | None = None,
    ) -> None:
        super().__init__(parent)
        self.chat_window = window
        self.effective_theme = getattr(window, "effective_theme", "dark")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 8)
        outer.setSpacing(12)

        if align_right:
            outer.addStretch(1)

        self.bubble = QFrame()
        self.bubble.setObjectName("BubbleUser" if align_right else "BubbleSystem")
        self.bubble.setMaximumWidth(680)
        self.bubble.setMinimumWidth(96)
        self.bubble.setMinimumHeight(52)
        self.bubble.setAttribute(Qt.WA_StyledBackground, True)

        shadow = QGraphicsDropShadowEffect(self.bubble)
        shadow.setBlurRadius(15 if align_right else 10)
        shadow.setXOffset(0)
        shadow.setYOffset(8 if align_right else 4)
        shadow.setColor(QColor(0, 0, 0, 90 if align_right else 40))
        self.bubble.setGraphicsEffect(shadow)

        self.bubble_layout = QVBoxLayout(self.bubble)
        self.bubble_layout.setContentsMargins(18, 12, 18, 12)
        self.bubble_layout.setSpacing(8)

        if message.attachment_path:
            attachment_label = QLabel(Path(message.attachment_path).name)
            attachment_label.setObjectName("Attachment")
            attachment_label.setToolTip(message.attachment_path)
            self.bubble_layout.addWidget(attachment_label)
            if message.attachment_kind in {"image", "camera"}:
                pixmap = load_preview_pixmap(message.attachment_path, max_size=(360, 240))
                if not pixmap.isNull():
                    self.preview_img = QLabel()
                    self.preview_img.setPixmap(pixmap.scaled(360, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.preview_img.setCursor(Qt.PointingHandCursor)
                    self.preview_img.setToolTip(message.attachment_path)
                    self.preview_img.mousePressEvent = lambda e: self.show_image_full(message.attachment_path)
                    self.bubble_layout.addWidget(self.preview_img)

        self.text_label = QLabel()
        self.typing_indicator: TypingIndicator | None = None
        self.update_display_text(message.text or "")
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.PlainText)
        self.text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_label.setStyleSheet("font-size: 15px;")
        self.bubble_layout.addWidget(self.text_label)
        outer.addWidget(self.bubble, 0, Qt.AlignTop)

        if not align_right:
            outer.addStretch(1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.anim = QPropertyAnimation(self.bubble, b"pos")
        self.anim.setDuration(500)
        curr = self.bubble.pos()
        self.anim.setStartValue(QPoint(curr.x(), curr.y() + 20))
        self.anim.setEndValue(curr)
        self.anim.setEasingCurve(QEasingCurve.OutBack)
        self.anim.start()

    def show_image_full(self, path: str) -> None:
        try:
            dialog = ImagePreviewDialog(path, self.window(), effective_theme=self.window().effective_theme)
            dialog.exec()
        except Exception:
            logger.exception("Failed to open image preview dialog for %s", path)

    def update_display_text(self, text: str) -> None:
        if text == "[TYPING]":
            self.text_label.hide()
            if not self.typing_indicator:
                self.typing_indicator = TypingIndicator(QColor("#4db8ff"))
                self.bubble_layout.addWidget(self.typing_indicator)
            self.typing_indicator.show()
            return

        if self.typing_indicator:
            self.typing_indicator.hide()
        self.text_label.show()

        is_error = text.startswith("Phân tích lỗi:") or text.startswith("Error:")
        if self.effective_theme == "light" and is_error:
            self.text_label.setStyleSheet(
                "font-size: 15px; color: #b00020; font-weight: 700; "
                "background: rgba(176,0,32,0.08); padding: 6px; border-radius: 8px;"
            )
        elif is_error:
            self.text_label.setStyleSheet(
                "font-size: 15px; color: #ff5252; font-weight: 700; "
                "background: rgba(255,82,82,0.1); padding: 6px; border-radius: 8px;"
            )
        else:
            base_color = "#111827" if self.effective_theme == "light" else "#f8fafc"
            self.text_label.setStyleSheet(f"font-size: 15px; color: {base_color}; background: transparent; padding: 0;")
        self.text_label.setText(text)

    def refresh_theme(self, effective_theme: str) -> None:
        self.effective_theme = effective_theme
        self.update_display_text(self.text_label.text())
