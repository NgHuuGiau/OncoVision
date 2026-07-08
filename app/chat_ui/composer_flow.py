from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from app.chat_ui.widgets import MessageInput, RecordingPanel


def build_composer_section(window, chat_layout) -> None:
    window.recording_panel = RecordingPanel(window.language, window=window)
    window.recording_panel.hide()
    chat_layout.addWidget(window.recording_panel, 0, Qt.AlignHCenter)

    window.composer = QFrame()
    window.composer.setObjectName("Composer")
    composer_layout = QVBoxLayout(window.composer)
    composer_layout.setContentsMargins(8, 8, 8, 8)
    composer_layout.setSpacing(5)
    window.composer.setMinimumHeight(76)
    window.composer.setMaximumHeight(260)

    window.image_preview_area = QScrollArea()
    window.image_preview_area.setObjectName("ComposerPreviewScroll")
    window.image_preview_area.setWidgetResizable(True)
    window.image_preview_area.setFrameShape(QFrame.NoFrame)
    window.image_preview_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    window.image_preview_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    window.image_preview_area.setFixedHeight(80)
    window.image_preview_area.hide()

    window.image_preview_host = QWidget()
    window.image_preview_host.setObjectName("ComposerPreviewHost")
    window.image_preview_layout = QHBoxLayout(window.image_preview_host)
    window.image_preview_layout.setContentsMargins(0, 0, 0, 0)
    window.image_preview_layout.setSpacing(6)
    window.image_preview_layout.addStretch(1)
    window.image_preview_area.setWidget(window.image_preview_host)
    composer_layout.addWidget(window.image_preview_area)

    window.message_input_row = QFrame()
    window.message_input_row.setObjectName("ComposerInputRow")
    input_row_layout = QHBoxLayout(window.message_input_row)
    input_row_layout.setContentsMargins(5, 4, 5, 4)
    input_row_layout.setSpacing(5)

    window.plus_button = window.plus_button if hasattr(window, "plus_button") else None
    if window.plus_button is None:
        window.plus_button = QPushButton("")
    window.plus_button.setObjectName("RoundButton")
    window.plus_button.setFixedSize(40, 40)
    window.plus_button.clicked.connect(window.show_plus_menu)
    input_row_layout.addWidget(window.plus_button, 0, Qt.AlignBottom)

    window.message_input = MessageInput()
    window.message_input.setObjectName("ComposerInput")
    window.message_input.setMinimumHeight(38)
    window.message_input.setMaximumHeight(180)
    window.message_input.setFrameShape(QFrame.NoFrame)
    window.message_input.setPlaceholderText(window.tr(window.language, "input_placeholder"))
    window.message_input.viewport().setAutoFillBackground(False)
    window.message_input.viewport().setStyleSheet("background: transparent;")
    window.message_input.enter_pressed.connect(window.send_message)
    input_row_layout.addWidget(window.message_input, 1, Qt.AlignVCenter)

    window.micro_button = QPushButton("")
    window.micro_button.setObjectName("RoundButton")
    window.micro_button.setFixedSize(40, 40)
    window.micro_button.clicked.connect(window.start_voice_input)
    input_row_layout.addWidget(window.micro_button, 0, Qt.AlignBottom)

    window.send_button = QPushButton("")
    window.send_button.setObjectName("SendButton")
    window.send_button.setFixedSize(40, 40)
    window.send_button.clicked.connect(window.send_message)
    input_row_layout.addWidget(window.send_button, 0, Qt.AlignBottom)
    composer_layout.addWidget(window.message_input_row)
    chat_layout.addWidget(window.composer)

    window.disclaimer_label = window.disclaimer_label if hasattr(window, "disclaimer_label") else None
    if window.disclaimer_label is None:
        window.disclaimer_label = QLabel(
            "Kết quả AI chỉ mang tính hỗ trợ. Hãy kiểm tra lại thông tin quan trọng với bác sĩ chuyên khoa."
        )
    window.disclaimer_label.setObjectName("Subtle")
    window.disclaimer_label.setAlignment(Qt.AlignCenter)
    chat_layout.addWidget(window.disclaimer_label)

    window.medical_status_label = window.medical_status_label if hasattr(window, "medical_status_label") else None
    if window.medical_status_label is None:
        window.medical_status_label = QLabel(window.medical_status_message)
    window.medical_status_label.setObjectName("Subtle")
    window.medical_status_label.setAlignment(Qt.AlignCenter)
    chat_layout.addWidget(window.medical_status_label)
