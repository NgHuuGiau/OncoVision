from __future__ import annotations

from PySide6.QtCore import QParallelAnimationGroup, QPropertyAnimation, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QFrame, QGraphicsBlurEffect, QHBoxLayout, QScrollArea, QVBoxLayout, QPushButton, QLabel, QWidget, QSizePolicy

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
        self.img_label.setPixmap(QPixmap(image_path).scaled(960, 740, Qt.KeepAspectRatio, Qt.SmoothTransformation))
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
__all__ = ["ImagePreviewDialog"]
