from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from app.chat_ui.content import translate as tr


class SettingsDialog(QDialog):
    def __init__(self, *, parent_window) -> None:
        super().__init__(parent_window)
        self.window = parent_window
        self.setWindowTitle(tr(self.window.language, "settings_title"))
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(1120, 780)
        self.theme_buttons: dict[str, QPushButton] = {}
        self.language_buttons: dict[str, QPushButton] = {}
        self.language_options = ("vi", "en")
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
        self.dialog_title.setStyleSheet("font-size: 30px; font-weight: 800;")
        header_row.addWidget(self.dialog_title)
        header_row.addStretch(1)
        close_button = QPushButton("\u2715")
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
        self.general_button.setMinimumHeight(54)
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
        content_layout.setContentsMargins(28, 26, 28, 28)
        content_layout.setSpacing(20)

        self.section_title = QLabel()
        self.section_title.setObjectName("SectionTitle")
        self.section_title.setStyleSheet("font-size: 24px; font-weight: 800;")
        content_layout.addWidget(self.section_title)

        self.appearance_card = QFrame()
        self.appearance_card.setObjectName("SettingsOptionCard")
        appearance_card_layout = QVBoxLayout(self.appearance_card)
        appearance_card_layout.setContentsMargins(22, 20, 22, 20)
        appearance_card_layout.setSpacing(18)
        appearance_row = QHBoxLayout()
        appearance_row.setSpacing(10)
        self.appearance_icon = QLabel("\u25d4")
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
            button.setMinimumHeight(48)
            button.clicked.connect(lambda _checked=False, mode=key: self.set_theme_mode(mode))
            self.theme_buttons[key] = button
            theme_options.addWidget(button)
        appearance_card_layout.addLayout(theme_options)
        content_layout.addWidget(self.appearance_card)

        self.language_card = QFrame()
        self.language_card.setObjectName("SettingsOptionCard")
        language_card_layout = QVBoxLayout(self.language_card)
        language_card_layout.setContentsMargins(22, 20, 22, 20)
        language_card_layout.setSpacing(14)
        language_row = QHBoxLayout()
        language_row.setSpacing(10)
        self.language_icon = QLabel("\U0001f310")
        self.language_icon.setStyleSheet("font-size: 18px;")
        language_row.addWidget(self.language_icon, 0, Qt.AlignTop)
        self.language_label = QLabel()
        self.language_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        language_row.addWidget(self.language_label, 1)
        language_card_layout.addLayout(language_row)

        language_options = QHBoxLayout()
        language_options.setSpacing(12)
        for key in self.language_options:
            button = QPushButton()
            button.setObjectName("ThemeChoiceButton")
            button.setMinimumHeight(48)
            button.clicked.connect(lambda _checked=False, code=key: self.set_language(code))
            self.language_buttons[key] = button
            language_options.addWidget(button)
        language_card_layout.addLayout(language_options)
        content_layout.addWidget(self.language_card)

        content_layout.addStretch(1)
        body_row.addWidget(content, 5)
        shell_layout.addLayout(body_row)
        self.retranslate_dialog()

    def set_theme_mode(self, mode: str) -> None:
        self.window.theme_mode = mode
        self.window.apply_theme()
        self.refresh_theme_buttons()

    def set_language(self, language_code: str) -> None:
        self.window.language = language_code
        self.window.db.set_setting("language", self.window.language)
        self.window.retranslate_ui()
        self.retranslate_dialog()

    def retranslate_dialog(self) -> None:
        language = self.window.language
        self.setWindowTitle(tr(language, "settings_title"))
        self.dialog_title.setText(tr(language, "settings_title"))
        self.general_button.setText(f"\u2302  {tr(language, 'general')}")
        self.section_title.setText(tr(language, "general"))
        self.appearance_label.setText(tr(language, "appearance"))
        self.language_label.setText(tr(language, "language"))
        self.theme_buttons["light"].setText(f"\u263c  {tr(language, 'light')}")
        self.theme_buttons["dark"].setText(f"\u263e  {tr(language, 'dark')}")
        self.theme_buttons["system"].setText(f"\u25a3  {tr(language, 'system')}")
        self.language_buttons["vi"].setText(tr(language, "vietnamese"))
        self.language_buttons["en"].setText(tr(language, "english"))
        self.refresh_theme_buttons()

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

        active_language = self.window.language
        for code, button in self.language_buttons.items():
            is_active = code == active_language
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
