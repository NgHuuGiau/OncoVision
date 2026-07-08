from __future__ import annotations

from PySide6.QtCore import QParallelAnimationGroup, QPropertyAnimation, Qt
from PySide6.QtWidgets import QDialog, QFrame, QGraphicsBlurEffect, QHBoxLayout, QScrollArea, QSlider, QVBoxLayout, QPushButton, QLabel, QWidget, QSizePolicy

from app.chat_ui.image_utils import load_medical_volume_pixmaps, load_preview_pixmap

class ImagePreviewDialog(QDialog):
    def __init__(self, image_path: str, parent: QWidget | None = None, effective_theme: str = "dark") -> None:
        super().__init__(parent)
        self.image_path = image_path
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
        shell_layout.setSpacing(16)
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
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 0px; height: 0px; background: transparent; }"
            "QScrollBar:horizontal { width: 0px; height: 0px; background: transparent; }"
        )
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.volume_pixmaps = load_medical_volume_pixmaps(image_path)
        self.slice_slider: QSlider | None = None
        self.prev_button: QPushButton | None = None
        self.next_button: QPushButton | None = None
        self.slice_caption: QLabel | None = None
        if len(self.volume_pixmaps) > 1:
            controls_row = QHBoxLayout()
            controls_row.setSpacing(10)
            self.prev_button = QPushButton("‹")
            self.prev_button.setFixedSize(42, 42)
            self.prev_button.clicked.connect(self._show_previous_slice)
            controls_row.addWidget(self.prev_button)

            self.slice_caption = QLabel()
            self.slice_caption.setAlignment(Qt.AlignCenter)
            self.slice_caption.setStyleSheet("font-size: 20px; font-weight: 800;")
            controls_row.addWidget(self.slice_caption, 1)

            self.next_button = QPushButton("›")
            self.next_button.setFixedSize(42, 42)
            self.next_button.clicked.connect(self._show_next_slice)
            controls_row.addWidget(self.next_button)

            shell_layout.addLayout(controls_row)

            self.slice_slider = QSlider(Qt.Horizontal)
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(len(self.volume_pixmaps) - 1)
            self.slice_slider.valueChanged.connect(self._update_slice_preview)
            shell_layout.addWidget(self.slice_slider)
            self._update_slice_preview(0)
        else:
            self.img_label.setPixmap(load_preview_pixmap(image_path, max_size=(960, 740)).scaled(960, 740, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("\u2715", shell)
        close_btn.setObjectName("SettingsCloseButton")
        close_btn.setFixedSize(40, 40)
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn, 0, Qt.AlignTop)
        shell_layout.addLayout(close_row)
        self.scroll.setWidget(self.img_label)
        shell_layout.addWidget(self.scroll, 1)

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

    def _update_slice_preview(self, index: int) -> None:
        if not self.volume_pixmaps:
            return
        index = max(0, min(index, len(self.volume_pixmaps) - 1))
        if self.slice_slider is not None and self.slice_slider.value() != index:
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(index)
            self.slice_slider.blockSignals(False)
        pixmap = self.volume_pixmaps[index]
        if not pixmap.isNull():
            self.img_label.setPixmap(pixmap.scaled(960, 740, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.slice_caption is not None:
            self.slice_caption.setText(f"Slice {index + 1}/{len(self.volume_pixmaps)}")

    def _show_previous_slice(self) -> None:
        if self.slice_slider is None:
            return
        self.slice_slider.setValue(max(0, self.slice_slider.value() - 1))

    def _show_next_slice(self) -> None:
        if self.slice_slider is None:
            return
        self.slice_slider.setValue(min(self.slice_slider.maximum(), self.slice_slider.value() + 1))
__all__ = ["ImagePreviewDialog"]
