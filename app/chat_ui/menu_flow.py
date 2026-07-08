from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QMenu, QPushButton, QWidgetAction

from app.chat_ui.content import translate
from app.chat_ui.icons import themed_icon


def add_popup_menu_button(window, menu: QMenu, *, icon_name: str, text: str, icon_color: str, callback) -> None:
    action = QWidgetAction(menu)
    button = QPushButton(text, menu)
    button.setObjectName("PopupMenuButton")
    button.setIcon(themed_icon(icon_name, icon_color, 18))
    button.setIconSize(QSize(18, 18))
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumWidth(220)
    button.setMinimumHeight(48)

    def trigger_action() -> None:
        menu.close()
        callback()

    button.clicked.connect(trigger_action)
    action.setDefaultWidget(button)
    menu.addAction(action)


def show_plus_menu(window) -> None:
    menu = QMenu(window)
    strong = window.icon_color()
    menu.setMinimumWidth(240)
    add_popup_menu_button(window, menu, icon_name="image.svg", text=translate(window.language, "choose_image"), icon_color=strong, callback=window.pick_image)
    add_popup_menu_button(
        window,
        menu,
        icon_name="camera.svg",
        text=translate(window.language, "camera"),
        icon_color=strong,
        callback=window.open_camera,
    )

    anchor = window.plus_button if hasattr(window, "plus_button") else window.send_button
    menu.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))
