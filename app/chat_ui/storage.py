from __future__ import annotations

from contextlib import contextmanager
import os
import sqlite3

from app.chat_ui.models import ChatMessage, Conversation


class ChatDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
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
        convs = []
        try:
            with self._connect() as conn:
                cursor = conn.execute("SELECT id, title, subtitle FROM conversations ORDER BY id DESC")
                for row in cursor:
                    c_id, title, subtitle = row
                    messages = []
                    m_cursor = conn.execute(
                        "SELECT sender, text, attachment_path, attachment_kind, id FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                        (c_id,),
                    )
                    for m_row in m_cursor:
                        sender, text, path, kind, m_id = m_row
                        messages.append(
                            ChatMessage(
                                sender=sender,
                                text=text,
                                attachment_path=path,
                                attachment_kind=kind,
                                id=m_id,
                            )
                        )
                    convs.append(Conversation(title=title, subtitle=subtitle, messages=messages, id=c_id))
        except Exception as e:
            print(f"Database error: {e}")
        return convs

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
