from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import QListWidgetItem, QMenu, QSizePolicy

from app.chat_ui.icons import themed_icon, themed_pixmap
from app.chat_ui.widgets import HistoryItemWidget


def build_sidebar_ui(window) -> None:
    return None


def apply_theme_assets(window) -> None:
    strong = window.icon_color()
    window.sidebar_app_button.setIcon(themed_icon("sidebar_app.svg", strong, 28))
    window.sidebar_toggle_button.setIcon(themed_icon("sidebar_app.svg", strong, 20))
    window.sidebar_collapsed_button.setIcon(themed_icon("sidebar_app.svg", strong, 20))
    window.new_chat_button.setIcon(themed_icon("new_chat.svg", strong, 22))
    window.settings_button.setIcon(themed_icon("settings.svg", strong, 22))
    if hasattr(window, "compact_new_chat_button"):
        window.compact_new_chat_button.setIcon(themed_icon("new_chat.svg", strong, 20))
    if hasattr(window, "compact_search_button"):
        window.compact_search_button.setIcon(themed_icon("search.svg", strong, 20))
    if hasattr(window, "compact_settings_button"):
        window.compact_settings_button.setIcon(themed_icon("settings.svg", strong, 20))
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
    target_width = 96 if window.sidebar_expanded else 288

    window.sidebar_anim = QPropertyAnimation(window.sidebar, b"minimumWidth")
    window.sidebar_anim.setDuration(220)
    window.sidebar_anim.setStartValue(window.sidebar.width())
    window.sidebar_anim.setEndValue(target_width)
    window.sidebar_anim.setEasingCurve(QEasingCurve.InOutCubic)

    window.sidebar_max_anim = QPropertyAnimation(window.sidebar, b"maximumWidth")
    window.sidebar_max_anim.setDuration(220)
    window.sidebar_max_anim.setStartValue(window.sidebar.width())
    window.sidebar_max_anim.setEndValue(target_width)
    window.sidebar_max_anim.setEasingCurve(QEasingCurve.InOutCubic)

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
    if hasattr(window, "sidebar_shadow"):
        if expanded:
            window.sidebar_shadow.setBlurRadius(30)
            window.sidebar_shadow.setXOffset(10)
            window.sidebar_shadow.setYOffset(0)
            window.sidebar_shadow.setColor(QColor(0, 0, 0, 100))
        else:
            window.sidebar_shadow.setBlurRadius(4)
            window.sidebar_shadow.setXOffset(0)
            window.sidebar_shadow.setYOffset(0)
            window.sidebar_shadow.setColor(QColor(0, 0, 0, 18))
    if hasattr(window, "sidebar_stack"):
        window.sidebar_stack.setCurrentWidget(window.sidebar_open_page if expanded else window.sidebar_compact_page)
    window.search_icon.setPixmap(themed_pixmap("search.svg", window.subtle_icon_color(), 18))

    if expanded:
        window.new_chat_button.setObjectName("SidebarPrimaryButton")
        window.settings_button.setObjectName("SidebarFooterButton")
        window.new_chat_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        window.settings_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        window.new_chat_button.setMinimumSize(0, 44)
        window.settings_button.setMinimumSize(0, 44)
        update_sidebar_texts(window)
        window.new_chat_button.setIconSize(QSize(20, 20))
        window.settings_button.setIconSize(QSize(20, 20))
        window.sidebar_toggle_button.setVisible(True)
        window.sidebar_collapsed_button.setVisible(False)
        window.brand_text.setVisible(True)
        window.sidebar_app_button.setVisible(False)
        window.history_title.setVisible(True)
        window.history_panel.setVisible(True)
        window.search_box.setObjectName("SearchBox")
        window.search_input.setVisible(True)
        window.search_filter_label.setVisible(True)
    else:
        window.sidebar_toggle_button.setVisible(False)
        window.sidebar_collapsed_button.setVisible(True)
        window.brand_text.setVisible(False)
        window.sidebar_app_button.setVisible(False)
        window.history_title.setVisible(False)
        window.history_panel.setVisible(False)
        window.search_box.setObjectName("SidebarCompactSearchButton")
        window.search_input.setVisible(False)
        window.search_filter_label.setVisible(False)
        window.new_chat_button.setObjectName("SidebarPrimaryButton")
        window.settings_button.setObjectName("SidebarFooterButton")
        window.new_chat_button.setText("Cuộc trò chuyện mới")
        window.settings_button.setText("Cài đặt")
        window.new_chat_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        window.settings_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        window.new_chat_button.setMinimumSize(46, 46)
        window.settings_button.setMinimumSize(46, 46)
        window.new_chat_button.setIconSize(QSize(18, 18))
        window.settings_button.setIconSize(QSize(18, 18))
        if hasattr(window, "compact_new_chat_button"):
            window.compact_new_chat_button.setIconSize(QSize(18, 18))
        if hasattr(window, "compact_search_button"):
            window.compact_search_button.setIconSize(QSize(18, 18))
        if hasattr(window, "compact_settings_button"):
            window.compact_settings_button.setIconSize(QSize(18, 18))
        window.sidebar_collapsed_button.style().unpolish(window.sidebar_collapsed_button)
        window.sidebar_collapsed_button.style().polish(window.sidebar_collapsed_button)

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
