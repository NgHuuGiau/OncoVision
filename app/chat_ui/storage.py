from __future__ import annotations

from contextlib import contextmanager
import os
import sqlite3
from collections import defaultdict

from app.chat_ui.models import ChatMessage, Conversation
from utils.logger import get_logger
from utils.sqlite_utils import DEFAULT_SQLITE_TIMEOUT_SECONDS, create_sqlite_connection


logger = get_logger(__name__)


class ChatDatabase:
    CONNECT_TIMEOUT_SECONDS = DEFAULT_SQLITE_TIMEOUT_SECONDS

    def __init__(self, db_path: str):
        self.db_path = db_path
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        return create_sqlite_connection(
            self.db_path,
            timeout_seconds=self.CONNECT_TIMEOUT_SECONDS,
            enable_foreign_keys=True,
        )

    @contextmanager
    def _connect(self):
        conn = self._create_connection()
        try:
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
                    metadata_json TEXT,
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
            self._ensure_column(conn, "messages", "metadata_json", "TEXT")
            self._ensure_indexes(conn)

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
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
        except sqlite3.Error:
            logger.exception("Failed to read setting '%s' from chat history database", key)
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
                    SELECT conversation_id, sender, text, attachment_path, attachment_kind, metadata_json, id
                    FROM messages
                    ORDER BY conversation_id ASC, id ASC
                    """
                ).fetchall()
        except sqlite3.Error:
            logger.exception("Failed to load conversations from chat history database")
            return []

        messages_by_conversation: dict[int, list[ChatMessage]] = defaultdict(list)
        for conversation_id, sender, text, path, kind, metadata_json, message_id in message_rows:
            messages_by_conversation[conversation_id].append(
                ChatMessage(
                    sender=sender,
                    text=text,
                    attachment_path=path,
                    attachment_kind=kind,
                    metadata_json=metadata_json,
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
                "INSERT INTO messages (conversation_id, sender, text, attachment_path, attachment_kind, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, msg.sender, msg.text, msg.attachment_path, msg.attachment_kind, msg.metadata_json),
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
        except sqlite3.Error:
            logger.exception("Failed to search conversations for query '%s'", query)
            return []
