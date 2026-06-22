from __future__ import annotations

_BASE_SHARED = """
    QWidget#ChatPanel,
    QWidget#ScrollContainer,
    QWidget#MessagesHost {
        background: transparent;
    }
    QLabel {
        background: transparent;
        color: {text_color};
        font-weight: normal;
    }
    QLabel#Subtle {{
        color: {subtle_color};
    }}
    QLabel#GreetingTitle {{
        font-size: 32px;
        font-weight: 700;
        color: {heading_color};
    }}
    QLabel#BrandText,
    QLabel#ChatHeaderTitle,
    QLabel#SectionTitle {{
        color: {heading_color};
        font-weight: 700;
    }}
    QLabel#BrandText {{
        font-size: 18px;
    }}
    QLabel#ChatHeaderTitle {{
        font-size: 24px;
    }}
    QLabel#SectionTitle {{
        font-size: 15px;
        font-weight: 600;
    }}
    QLabel#ModeBadge {{
        color: {subtle_color};
        font-size: 13px;
    }}
    QLabel#GreetingAvatar {{
        min-width: 56px;
        max-width: 56px;
        min-height: 56px;
        max-height: 56px;
        border-radius: 28px;
        border: 1px solid {avatar_border};
        background: {avatar_background};
    }}
    QLabel#Attachment {{
        color: {attachment_color};
        font-weight: 600;
    }}
    QLabel#Avatar {{
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        border-radius: 16px;
        border: none;
        background: #4db8ff;
        color: {avatar_text};
        font-weight: 700;
        qproperty-alignment: AlignCenter;
    }}
    QFrame#Sidebar {{
        background: {sidebar_bg};
        border: 1px solid {sidebar_border};
        border-radius: 20px;
    }}
    QFrame#TopBar,
    QFrame#SidebarHeader,
    QFrame#MessageScroll,
    QFrame#SettingsContent,
    QFrame#SidebarButton,
    QFrame#GreetingCard,
    QFrame#ChatBoard,
    QFrame#SettingsCard,
    QWidget#ComposerPreviewHost {{
        background: transparent;
        border: none;
    }}
    QFrame#TopBar {{
        border-bottom: 1px solid {divider_color};
    }}
    QFrame#Composer {{
        background: {composer_bg};
        border: 1px solid {composer_border};
        border-radius: 20px;
    }}
    QFrame#ComposerInputRow {{
        background: {composer_input_bg};
        border: 1px solid {composer_input_border};
        border-radius: 14px;
        padding: 2px 4px;
    }}
    QScrollArea#ComposerPreviewScroll,
    QScrollArea#MessageScroll {{
        background: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {scrollbar_color};
        border-radius: 5px;
        min-height: 36px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {scrollbar_color};
        border-radius: 5px;
        min-width: 36px;
    }}
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical,
    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {{
        background: transparent;
        border: none;
        height: 0px;
        width: 0px;
    }}
    QFrame#ComposerPreviewThumb {{
        background: {preview_thumb_bg};
        border: 1px solid {preview_thumb_border};
        border-radius: 18px;
    }}
    QPushButton#ComposerPreviewDeleteButton {{
        background: rgba(239, 68, 68, 0.92);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 700;
        padding: 0;
        text-align: center;
    }}
    QPushButton#ComposerPreviewDeleteButton:hover {{
        background: rgba(220, 38, 38, 0.98);
    }}
    QFrame#HistoryPanel {{
        background: {history_panel_bg};
        border: 1px solid {history_panel_border};
        border-radius: 24px;
    }}
    QFrame#SearchBox {{
        background: {search_bg};
        border: 1px solid {search_border};
        border-radius: 14px;
    }}
    QFrame#SettingsShell,
    QFrame#ImagePreviewShell {{
        background: {dialog_shell_bg};
        border: 1px solid {dialog_shell_border};
        border-radius: 20px;
    }}
    QFrame#SettingsNav {{
        background: {settings_nav_bg};
        border: none;
        border-radius: 0px;
    }}
    QFrame#SettingsOptionCard {{
        background: {settings_card_bg};
        border: 1px solid {settings_card_border};
        border-radius: 16px;
    }}
    QFrame#HistoryItem {{
        background: transparent;
        border: none;
        border-radius: 16px;
    }}
    QFrame#HistoryItem:hover {{
        background: {history_hover};
    }}
    QFrame#HistoryItem[selected="true"] {{
        background: {history_selected_bg};
        border: 1px solid {history_selected_border};
    }}
    QFrame#BubbleUser {{
        background: rgba(77, 184, 255, 0.14);
        border: 1px solid rgba(77, 184, 255, 0.12);
        border-top-left-radius: 24px;
        border-top-right-radius: 8px;
        border-bottom-left-radius: 24px;
        border-bottom-right-radius: 24px;
    }}
    QFrame#BubbleSystem {{
        background: {bubble_system_bg};
        border: 1px solid {bubble_system_border};
        border-top-left-radius: 8px;
        border-top-right-radius: 24px;
        border-bottom-left-radius: 24px;
        border-bottom-right-radius: 24px;
    }}
    QLineEdit, QPlainTextEdit, QComboBox, QListWidget {{
        background: {input_bg};
        border: 1px solid {input_border};
        border-radius: 16px;
        color: {text_color};
        padding: 12px 14px;
    }}
    QLineEdit[state="error"] {{
        border: 1.5px solid #ff5252;
    }}
    QLineEdit[state="success"] {{
        border: 1.5px solid #4caf50;
    }}
    QFrame#Composer QPlainTextEdit {{
        background: transparent;
        border: none;
        border-radius: 0;
        color: {text_color};
        font-size: 16px;
        padding: 12px 8px;
        min-height: 36px;
        max-height: 88px;
    }}
    QPlainTextEdit#ComposerInput,
    QPlainTextEdit#ComposerInput QWidget {{
        background: transparent;
        border: none;
    }}
    QPlainTextEdit {{
        padding-top: 0px;
    }}
    QPushButton#ModeButton {{
        background: {mode_button_bg};
        border: 1px solid {mode_button_border};
        border-radius: 14px;
        color: {mode_button_text};
        font-size: 15px;
        font-weight: 600;
        padding: 6px 14px;
        text-align: center;
        min-height: 34px;
    }}
    QPushButton#ModeButton:hover {{
        background: {mode_button_hover};
    }}
    QListWidget {{
        outline: none;
        padding: 0;
        background: transparent;
        border: none;
    }}
    QListWidget::item {{
        margin: 0;
        padding: 0;
        border: none;
    }}
    QPushButton {{
        background: transparent;
        color: {button_text};
        border: none;
        border-radius: 18px;
        padding: 10px 14px;
        text-align: left;
    }}
    QPushButton:hover {{
        background: {button_hover};
    }}
    QPushButton#SidebarPrimaryButton {{
        min-height: 44px;
        padding-left: 16px;
        font-size: 15px;
        font-weight: 600;
        border: none;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563ff, stop:1 #1d4ed8);
        color: #ffffff;
        border-radius: 12px;
    }}
    QPushButton#TopActionButton,
    QPushButton#SidebarAppButton,
    QPushButton#SidebarCompactButton,
    QPushButton#SidebarCompactSearchButton,
    QPushButton#RoundButton {{
        text-align: center;
        padding: 0;
    }}
    QPushButton#TopActionButton {{
        min-width: 40px;
        max-width: 40px;
        min-height: 40px;
        max-height: 40px;
        border-radius: 14px;
        color: {button_text};
        font-size: 18px;
    }}
    QPushButton#TopActionButton:hover {{
        background: {top_action_hover};
    }}
    QPushButton#SidebarPrimaryButton:hover,
    QPushButton#SidebarCompactButton:hover,
    QPushButton#SidebarCompactSearchButton:hover,
    QFrame#SearchBox:hover,
    QPushButton#SidebarAppButton:hover {{
        background: {sidebar_hover};
    }}
    QPushButton#SidebarAppButton,
    QPushButton#SidebarCompactButton,
    QPushButton#SidebarCompactSearchButton {{
        min-width: 52px;
        max-width: 52px;
        min-height: 52px;
        max-height: 52px;
        border-radius: 20px;
    }}
    QPushButton#RoundButton {{
        min-width: 44px;
        max-width: 44px;
        min-height: 44px;
        max-height: 44px;
        border-radius: 20px;
        border: 1px solid {round_button_border};
        background: {round_button_bg};
        color: {round_button_text};
        font-size: 20px;
    }}
    QPushButton#RoundButton:hover {{
        background: {round_button_hover};
    }}
    QPushButton#SendButton {{
        min-width: 44px;
        max-width: 44px;
        min-height: 44px;
        max-height: 44px;
        border-radius: 20px;
        border: none;
        background: #2563ff;
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
        padding: 0;
        text-align: center;
    }}
    QPushButton#SendButton:hover {{
        background: #1d4ed8;
    }}
    QPushButton#SidebarFooterButton {{
        min-height: 44px;
        border-radius: 14px;
        background: {footer_button_bg};
        border: 1px solid {footer_button_border};
        padding: 0 14px;
        font-size: 15px;
        font-weight: 600;
    }}
    QPushButton#SidebarFooterButton:hover {{
        background: {footer_button_hover};
    }}
    QPushButton#SidebarToggleButton,
    QPushButton#SettingsCloseButton {{
        background: transparent;
        border: none;
        text-align: center;
        padding: 0;
    }}
    QPushButton#SettingsNavButton {{
        background: {settings_nav_button_bg};
        border: 1px solid {settings_nav_button_border};
        border-radius: 16px;
        text-align: left;
        padding: 14px 18px;
        font-weight: 600;
        color: {settings_nav_button_text};
    }}
    QPushButton#SettingsCloseButton {{
        color: {button_text};
        font-size: 18px;
    }}
    QPushButton#SettingsCloseButton:hover {{
        color: {button_text};
        background: transparent;
    }}
    QComboBox#SettingsCombo {{
        min-width: 220px;
        min-height: 44px;
        padding: 10px 14px;
        font-size: 15px;
        font-weight: 600;
        color: {combo_text};
        background: {combo_bg};
        border: 1px solid {combo_border};
        border-radius: 18px;
    }}
    QComboBox#SettingsCombo::drop-down {{
        border: none;
        width: 0px;
    }}
    QComboBox#SettingsCombo::down-arrow {{
        image: none;
    }}
    QComboBox#SettingsCombo QAbstractItemView {{
        background: {menu_bg};
        color: {menu_text};
        selection-background-color: {combo_selection_bg};
        selection-color: {combo_selection_text};
        border: 1px solid {menu_border};
        outline: none;
    }}
    QMenu {{
        background: {menu_bg};
        border: 1px solid {menu_border};
        border-radius: 16px;
        padding: 8px;
    }}
    QMenu::item {{
        padding: 10px 18px;
        border-radius: 10px;
        color: {menu_text};
    }}
    QMenu::item:selected {{
        background: {menu_hover};
    }}
    QPushButton#PopupMenuButton {{
        min-height: 52px;
        padding: 0 16px;
        border-radius: 14px;
        background: transparent;
        color: {button_text};
        font-size: 15px;
        font-weight: 600;
        text-align: left;
    }}
    QPushButton#PopupMenuButton:hover {{
        background: {button_hover};
    }}
"""


def _build_stylesheet(**colors: str) -> str:
    return """
    QMainWindow, QDialog, QWidget#Root {{
        background: {root_bg};
        font-family: "Inter", "Segoe UI", "Roboto", "Arial", sans-serif;
    }}
    """.format(**colors) + _BASE_SHARED.format(**colors)


DARK_STYLESHEET = _build_stylesheet(
    root_bg="#07111f",
    text_color="#e3e3e3",
    subtle_color="#94a3b8",
    heading_color="#ffffff",
    avatar_border="rgba(255, 255, 255, 0.08)",
    avatar_background="rgba(255, 255, 255, 0.03)",
    attachment_color="#d7e0ff",
    avatar_text="#ffffff",
    sidebar_bg="#0f172a",
    sidebar_border="rgba(255, 255, 255, 0.08)",
    divider_color="rgba(255, 255, 255, 0.06)",
    composer_bg="rgba(255, 255, 255, 0.06)",
    composer_border="rgba(255, 255, 255, 0.08)",
    composer_input_bg="rgba(255, 255, 255, 0.04)",
    composer_input_border="rgba(255, 255, 255, 0.07)",
    scrollbar_color="rgba(255, 255, 255, 0.18)",
    preview_thumb_bg="rgba(255, 255, 255, 0.04)",
    preview_thumb_border="rgba(255, 255, 255, 0.08)",
    history_panel_bg="rgba(255, 255, 255, 0.02)",
    history_panel_border="rgba(255, 255, 255, 0.05)",
    search_bg="rgba(255, 255, 255, 0.03)",
    search_border="rgba(255, 255, 255, 0.06)",
    dialog_shell_bg="#0f172a",
    dialog_shell_border="rgba(255, 255, 255, 0.08)",
    settings_nav_bg="rgba(255, 255, 255, 0.02)",
    settings_card_bg="rgba(255, 255, 255, 0.03)",
    settings_card_border="rgba(255, 255, 255, 0.06)",
    history_hover="rgba(255, 255, 255, 0.05)",
    history_selected_bg="rgba(255, 255, 255, 0.08)",
    history_selected_border="rgba(255, 255, 255, 0.06)",
    bubble_system_bg="rgba(255, 255, 255, 0.03)",
    bubble_system_border="rgba(255, 255, 255, 0.06)",
    input_bg="rgba(255, 255, 255, 0.03)",
    input_border="rgba(255, 255, 255, 0.08)",
    mode_button_bg="#3a3d45",
    mode_button_border="#50545e",
    mode_button_text="#ececf1",
    mode_button_hover="#454851",
    button_text="#ffffff",
    button_hover="rgba(255, 255, 255, 0.05)",
    top_action_hover="rgba(255, 255, 255, 0.08)",
    sidebar_hover="rgba(255, 255, 255, 0.07)",
    round_button_border="rgba(255, 255, 255, 0.07)",
    round_button_bg="rgba(255, 255, 255, 0.04)",
    round_button_text="#f2f4f8",
    round_button_hover="rgba(255, 255, 255, 0.12)",
    footer_button_bg="rgba(255, 255, 255, 0.04)",
    footer_button_border="rgba(255, 255, 255, 0.08)",
    footer_button_hover="rgba(255, 255, 255, 0.08)",
    settings_nav_button_bg="rgba(255, 255, 255, 0.04)",
    settings_nav_button_border="rgba(255, 255, 255, 0.07)",
    settings_nav_button_text="#f3f4f6",
    combo_text="#f3f4f6",
    combo_bg="rgba(255, 255, 255, 0.06)",
    combo_border="rgba(255, 255, 255, 0.08)",
    menu_bg="#1b1c20",
    menu_text="#ffffff",
    menu_border="#3a3a3a",
    combo_selection_bg="rgba(77, 184, 255, 0.2)",
    combo_selection_text="#ffffff",
    menu_hover="#242424",
)


LIGHT_STYLESHEET = _build_stylesheet(
    root_bg="#f7f9fd",
    text_color="#111827",
    subtle_color="#64748b",
    heading_color="#000000",
    avatar_border="rgba(17, 24, 39, 0.08)",
    avatar_background="rgba(17, 24, 39, 0.03)",
    attachment_color="#000000",
    avatar_text="#111827",
    sidebar_bg="#ffffff",
    sidebar_border="#e8edf5",
    divider_color="rgba(17, 24, 39, 0.08)",
    composer_bg="#ffffff",
    composer_border="#e2e8f0",
    composer_input_bg="#ffffff",
    composer_input_border="#e2e8f0",
    scrollbar_color="rgba(17, 24, 39, 0.18)",
    preview_thumb_bg="rgba(255, 255, 255, 0.82)",
    preview_thumb_border="rgba(17, 24, 39, 0.08)",
    history_panel_bg="rgba(255, 255, 255, 0.65)",
    history_panel_border="rgba(17, 24, 39, 0.06)",
    search_bg="rgba(17, 24, 39, 0.03)",
    search_border="rgba(17, 24, 39, 0.06)",
    dialog_shell_bg="#ffffff",
    dialog_shell_border="#e2e8f0",
    settings_nav_bg="rgba(37, 99, 255, 0.03)",
    settings_card_bg="#ffffff",
    settings_card_border="#e2e8f0",
    history_hover="rgba(17, 24, 39, 0.05)",
    history_selected_bg="rgba(17, 24, 39, 0.08)",
    history_selected_border="rgba(17, 24, 39, 0.05)",
    bubble_system_bg="rgba(255, 255, 255, 0.9)",
    bubble_system_border="rgba(17, 24, 39, 0.06)",
    input_bg="rgba(255, 255, 255, 0.9)",
    input_border="rgba(17, 24, 39, 0.08)",
    mode_button_bg="#e4e7ed",
    mode_button_border="#c8cdd8",
    mode_button_text="#374151",
    mode_button_hover="#d8dbe4",
    button_text="#111827",
    button_hover="rgba(17, 24, 39, 0.05)",
    top_action_hover="rgba(17, 24, 39, 0.08)",
    sidebar_hover="rgba(17, 24, 39, 0.06)",
    round_button_border="rgba(17, 24, 39, 0.08)",
    round_button_bg="rgba(255, 255, 255, 0.92)",
    round_button_text="#111827",
    round_button_hover="rgba(17, 24, 39, 0.08)",
    footer_button_bg="#ffffff",
    footer_button_border="#e2e8f0",
    footer_button_hover="#f8fbff",
    settings_nav_button_bg="#eef4ff",
    settings_nav_button_border="#dbe7ff",
    settings_nav_button_text="#111827",
    combo_text="#000000",
    combo_bg="#ffffff",
    combo_border="#e2e8f0",
    menu_bg="#ffffff",
    menu_text="#111827",
    menu_border="#d9d9df",
    combo_selection_bg="#dbeafe",
    combo_selection_text="#000000",
    menu_hover="#f7f7f8",
)
