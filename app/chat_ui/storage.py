from __future__ import annotations

from contextlib import contextmanager
import os
import sqlite3
from collections import defaultdict

from app.chat_ui.models import ChatMessage, Conversation
from utils.logger import get_logger


logger = get_logger(__name__)


class ChatDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    subtitle TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    sender TEXT,
                    text TEXT,
                    attachment_path TEXT,
                    attachment_kind TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id
                ON messages (conversation_id, id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_text
                ON messages (text)
                """
            )

    def get_setting(self, key: str, default: str) -> str:
        try:
            with self._connect() as conn:
                cursor = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else default
        except Exception:
            return default

    def set_setting(self, key: str, value: str):
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))

    def get_all_conversations(self) -> list[Conversation]:
        try:
            with self._connect() as conn:
                conversation_rows = conn.execute(
                    "SELECT id, title, subtitle FROM conversations ORDER BY id DESC"
                ).fetchall()
                message_rows = conn.execute(
                    """
                    SELECT conversation_id, sender, text, attachment_path, attachment_kind, id
                    FROM messages
                    ORDER BY conversation_id ASC, id ASC
                    """
                ).fetchall()
        except Exception:
            logger.exception("Failed to load conversations from chat history database")
            return []

        messages_by_conversation: dict[int, list[ChatMessage]] = defaultdict(list)
        for conversation_id, sender, text, path, kind, message_id in message_rows:
            messages_by_conversation[conversation_id].append(
                ChatMessage(
                    sender=sender,
                    text=text,
                    attachment_path=path,
                    attachment_kind=kind,
                    id=message_id,
                )
            )

        return [
            Conversation(
                title=title,
                subtitle=subtitle,
                messages=messages_by_conversation.get(conversation_id, []),
                id=conversation_id,
            )
            for conversation_id, title, subtitle in conversation_rows
        ]

    def create_conversation(self, title: str, subtitle: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("INSERT INTO conversations (title, subtitle) VALUES (?, ?)", (title, subtitle))
            return cursor.lastrowid

    def update_conversation_title(self, conv_id: int, title: str):
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))

    def add_message(self, conv_id: int, msg: ChatMessage) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, sender, text, attachment_path, attachment_kind) VALUES (?, ?, ?, ?, ?)",
                (conv_id, msg.sender, msg.text, msg.attachment_path, msg.attachment_kind),
            )
            return cursor.lastrowid

    def delete_conversation(self, conv_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    def clear_all_conversations(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations")

    def clear_all_messages(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM messages")

    def search_conversations_by_message(self, query: str) -> list[int]:
        if not query:
            return []
        try:
            with self._connect() as conn:
                cursor = conn.execute("SELECT DISTINCT conversation_id FROM messages WHERE text LIKE ?", (f"%{query}%",))
                return [row[0] for row in cursor]
        except Exception:
            return []
