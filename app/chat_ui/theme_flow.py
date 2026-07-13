from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize

from app.chat_ui.theme_styles import DARK_STYLESHEET, LIGHT_STYLESHEET
from app.chat_ui.widgets import ChatBubble
from app.chat_ui.icons import themed_icon


def icon_color(window) -> str:
    return "#ECECF1" if window.effective_theme == "dark" else "#111827"


def subtle_icon_color(window) -> str:
    return "#AAB0BC" if window.effective_theme == "dark" else "#6B7280"


def apply_theme_assets(window) -> None:
    strong = icon_color(window)
    window.sidebar_app_button.setIcon(themed_icon("sidebar_app.svg", strong, 28))
    window.sidebar_toggle_button.setIcon(themed_icon("sidebar_app.svg", strong, 20))
    window.sidebar_collapsed_button.setIcon(themed_icon("sidebar_app.svg", strong, 18))
    window.new_chat_button.setIcon(themed_icon("new_chat.svg", strong, 22))
    window.settings_button.setIcon(themed_icon("settings.svg", strong, 22))
    if hasattr(window, "compact_new_chat_button"):
        window.compact_new_chat_button.setIcon(themed_icon("new_chat.svg", strong, 18))
    if hasattr(window, "compact_search_button"):
        window.compact_search_button.setIcon(themed_icon("search.svg", strong, 18))
    if hasattr(window, "compact_settings_button"):
        window.compact_settings_button.setIcon(themed_icon("settings.svg", strong, 18))
    window.plus_button.setIcon(themed_icon("plus.svg", strong, 18))
    window.plus_button.setIconSize(QSize(18, 18))
    window.micro_button.setIcon(themed_icon("mic.svg", strong, 18))
    window.micro_button.setIconSize(QSize(18, 18))
    window.send_button.setIcon(themed_icon("send.svg", strong, 18))
    window.send_button.setIconSize(QSize(18, 18))
    window.message_input.apply_visual_style(dark_mode=window.effective_theme == "dark")
    window.recording_panel.setup_styles()
    refresh_topbar_buttons(window)
    refresh_empty_state_theme(window)
    for i in range(window.messages_layout.count()):
        item = window.messages_layout.itemAt(i)
        widget = item.widget()
        if isinstance(widget, ChatBubble):
            widget.refresh_theme(window.effective_theme)


def refresh_topbar_buttons(window) -> None:
    dark_mode = window.effective_theme == "dark"
    active = window.theme_mode
    palette = {
        "base": "#0f172a" if dark_mode else "#ffffff",
        "border": "rgba(255, 255, 255, 0.08)" if dark_mode else "#e2e8f0",
        "text": "#f8fafc" if dark_mode else "#0f172a",
        "selected_bg": "rgba(37, 99, 255, 0.22)" if dark_mode else "#eef4ff",
        "selected_text": "#f8fbff" if dark_mode else "#2563ff",
        "selected_border": "#2563ff",
    }
    for button, mode in ((window.light_mode_button, "light"), (window.dark_mode_button, "dark")):
        is_active = active == mode
        button.setStyleSheet(
            f"background: {palette['selected_bg'] if is_active else palette['base']};"
            f"color: {palette['selected_text'] if is_active else palette['text']};"
            f"border: 1px solid {palette['selected_border'] if is_active else palette['border']};"
            "border-radius: 14px; font-size: 14px; font-weight: 600; padding: 0 16px;"
        )
    window.desktop_button.setStyleSheet(
        f"background: {palette['base']}; color: {palette['text']}; border: 1px solid {palette['border']};"
        "border-radius: 14px; font-size: 14px; font-weight: 600; padding: 0 16px;"
    )


def refresh_empty_state_theme(window) -> None:
    if window.effective_theme == "dark":
        window.robot_mark.setStyleSheet("font-size: 86px;")
        window.empty_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f8fafc;")
    else:
        window.robot_mark.setStyleSheet("font-size: 86px;")
        window.empty_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #0f172a;")


def apply_light_theme(window, app) -> None:
    window.effective_theme = "light"
    if app:
        app.setStyleSheet(LIGHT_STYLESHEET)
    apply_theme_assets(window)
    window.refresh_history()


def apply_dark_theme(window, app) -> None:
    window.effective_theme = "dark"
    if app:
        app.setStyleSheet(DARK_STYLESHEET)
    apply_theme_assets(window)
    window.refresh_history()


def runtime_badge_text(window) -> str:
    parts = ["Hệ thống"]
    mode_label = str(window.mode_label or "").strip()
    if mode_label and mode_label.lower() not in {"desktop", "ui"}:
        parts.append(mode_label)
    model_label = Path(str(window.model_label)).name if window.model_label else ""
    if model_label:
        parts.append(model_label)
    return "  ".join(parts)



