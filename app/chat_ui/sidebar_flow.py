from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QListWidgetItem, QMenu, QSizePolicy

from app.chat_ui.icons import themed_icon, themed_pixmap
from app.chat_ui.widgets import HistoryItemWidget


def build_sidebar_ui(window) -> None:
    return None


def apply_theme_assets(window) -> None:
    strong = window.icon_color()
    window.sidebar_app_button.setIcon(themed_icon("sidebar_app.svg", strong, 28))
    window.new_chat_button.setIcon(themed_icon("new_chat.svg", strong, 22))
    window.settings_button.setIcon(themed_icon("settings.svg", strong, 22))
    window.search_icon.setPixmap(themed_pixmap("search.svg", window.subtle_icon_color(), 18))
    window.search_filter_label.setText("\u2630")


def update_sidebar_texts(window) -> None:
    window.new_chat_button.setText("Cuộc trò chuyện mới")
    window.history_title.setText("Lịch sử trò chuyện")
    window.settings_button.setText("Cài đặt")
    window.search_input.setPlaceholderText("Tìm kiếm cuộc trò chuyện...")


def retranslate_sidebar(window) -> None:
    update_sidebar_texts(window)
    window.update_sidebar_ui()


def toggle_sidebar(window) -> None:
    target_width = 88 if window.sidebar_expanded else 280

    window.sidebar_anim = QPropertyAnimation(window.sidebar, b"minimumWidth")
    window.sidebar_anim.setDuration(250)
    window.sidebar_anim.setStartValue(window.sidebar.width())
    window.sidebar_anim.setEndValue(target_width)
    window.sidebar_anim.setEasingCurve(QEasingCurve.InOutQuad)

    window.sidebar_max_anim = QPropertyAnimation(window.sidebar, b"maximumWidth")
    window.sidebar_max_anim.setDuration(250)
    window.sidebar_max_anim.setStartValue(window.sidebar.width())
    window.sidebar_max_anim.setEndValue(target_width)
    window.sidebar_max_anim.setEasingCurve(QEasingCurve.InOutQuad)

    window.sidebar_group = QParallelAnimationGroup()
    window.sidebar_group.addAnimation(window.sidebar_anim)
    window.sidebar_group.addAnimation(window.sidebar_max_anim)

    def on_finished() -> None:
        window.sidebar_expanded = not window.sidebar_expanded
        window.update_sidebar_ui()

    window.sidebar_group.finished.connect(on_finished)
    window.sidebar_group.start()


def update_sidebar_ui(window) -> None:
    expanded = window.sidebar_expanded
    window.sidebar_toggle_button.setVisible(not expanded)
    window.sidebar_header_left_spacer.setVisible(not expanded)
    window.sidebar_header_right_spacer.setVisible(True)
    window.sidebar_header_layout.setSpacing(12 if expanded else 0)
    window.brand_text.setVisible(expanded)
    window.search_box.setVisible(True)
    window.history_title.setVisible(expanded)
    window.history_panel.setVisible(expanded)
    window.sidebar.layout().setStretchFactor(window.history_panel, 1 if expanded else 0)
    window.sidebar.layout().setStretchFactor(window.sidebar_spacer, 0 if expanded else 1)
    window.search_icon.setPixmap(themed_pixmap("search.svg", window.subtle_icon_color(), 18))

    if expanded:
        window.search_box.setObjectName("SearchBox")
        window.search_input.setVisible(True)
        window.search_filter_label.setVisible(True)
        window.new_chat_button.setObjectName("SidebarPrimaryButton")
        window.settings_button.setObjectName("SidebarFooterButton")
        window.new_chat_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        window.settings_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        window.new_chat_button.setMinimumSize(0, 44)
        window.settings_button.setMinimumSize(0, 44)
        update_sidebar_texts(window)
        window.new_chat_button.setIconSize(QSize(20, 20))
        window.settings_button.setIconSize(QSize(20, 20))
    else:
        window.search_box.setObjectName("SidebarCompactSearchButton")
        window.search_input.setVisible(False)
        window.search_filter_label.setVisible(False)
        window.new_chat_button.setObjectName("SidebarCompactButton")
        window.settings_button.setObjectName("SidebarCompactButton")
        window.new_chat_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        window.settings_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        window.new_chat_button.setMinimumSize(52, 52)
        window.settings_button.setMinimumSize(52, 52)
        window.new_chat_button.setText("")
        window.settings_button.setText("")
        window.new_chat_button.setIconSize(QSize(22, 22))
        window.settings_button.setIconSize(QSize(22, 22))

    for button in (window.new_chat_button, window.settings_button):
        button.style().unpolish(button)
        button.style().polish(button)
    window.search_box.style().unpolish(window.search_box)
    window.search_box.style().polish(window.search_box)


def refresh_history(window) -> None:
    query = window.search_input.text().strip().lower() if hasattr(window, "search_input") else ""
    matching_conv_ids = set(window.db.search_conversations_by_message(query)) if query else set()
    window.is_refreshing_history = True
    window.history_list.blockSignals(True)
    window.history_list.clear()
    current_item_row = 0
    visible_row = 0

    for index, conversation in enumerate(window.conversations):
        if conversation.id is None and not conversation.messages:
            continue

        normalized_title = conversation.title.strip().lower()
        if normalized_title in {"vi", "en", ""}:
            conversation.title = window.tr(window.language, "new_chat")
            if conversation.id is not None:
                window.db.update_conversation_title(conversation.id, conversation.title)

        matches_title = query in conversation.title.lower()
        matches_message = conversation.id in matching_conv_ids
        if query and not (matches_title or matches_message):
            continue

        item = QListWidgetItem()
        item.setData(Qt.UserRole, index)
        item.setSizeHint(QSize(0, 68))
        window.history_list.addItem(item)
        widget = HistoryItemWidget(
            conversation.title,
            conversation.subtitle,
            icon_pixmap=themed_pixmap("chat_history.svg", window.subtle_icon_color(), 20),
            selected=index == window.active_conversation_index,
        )
        window.history_list.setItemWidget(item, widget)
        if index == window.active_conversation_index:
            current_item_row = visible_row
        visible_row += 1

    window.history_list.blockSignals(False)
    if window.history_list.count():
        window.history_list.setCurrentRow(min(current_item_row, window.history_list.count() - 1))
    window.is_refreshing_history = False


def select_conversation(window, row: int) -> None:
    if window.is_refreshing_history:
        return
    item = window.history_list.item(row)
    if item is None:
        return
    source_index = item.data(Qt.UserRole)
    if source_index is None:
        return
    window.active_conversation_index = int(source_index)
    window.render_messages()


def show_history_context_menu(window, pos: QPoint) -> None:
    item = window.history_list.itemAt(pos)
    if not item:
        return
    index = item.data(Qt.UserRole)
    menu = QMenu(window)
    delete_action = QAction("Xóa trò chuyện", window)
    delete_action.triggered.connect(lambda: window.delete_conversation(index))
    menu.addAction(delete_action)
    menu.exec(window.history_list.mapToGlobal(pos))


def start_new_chat(window) -> None:
    window.conversations.append(window._build_empty_conversation())
    window.active_conversation_index = len(window.conversations) - 1
    window.refresh_history()
    window.render_messages()


def delete_conversation(window, index_to_delete: int) -> None:
    if not (0 <= index_to_delete < len(window.conversations)):
        return
    conversation = window.conversations.pop(index_to_delete)
    if conversation.id is not None:
        window.db.delete_conversation(conversation.id)
    if not window.conversations:
        window.conversations = [window._build_empty_conversation()]
    window.active_conversation_index = min(window.active_conversation_index, len(window.conversations) - 1)
    window.refresh_history()
    window.render_messages()


def clear_all_history(window) -> None:
    window.db.clear_conversations()
    window.conversations = [window._build_empty_conversation()]
    window.active_conversation_index = 0
    window.refresh_history()
    window.render_messages()
