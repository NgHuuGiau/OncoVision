from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.chat_ui.models import ChatMessage
from app.chat_ui.storage import ChatDatabase


class ChatStorageTests(unittest.TestCase):
    def test_delete_conversation_cascades_to_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "chat.db"
            db = ChatDatabase(str(db_path))
            conv_id = db.create_conversation("Test", "Today")
            db.add_message(conv_id, ChatMessage(sender="user", text="hello"))

            db.delete_conversation(conv_id)

            self.assertEqual(db.get_all_conversations(), [])
            self.assertEqual(db.search_conversations_by_message("hello"), [])

    def test_clear_all_conversations_cascades_to_all_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "chat.db"
            db = ChatDatabase(str(db_path))
            first_id = db.create_conversation("A", "Today")
            second_id = db.create_conversation("B", "Today")
            db.add_message(first_id, ChatMessage(sender="user", text="alpha"))
            db.add_message(second_id, ChatMessage(sender="ai", text="beta"))

            db.clear_all_conversations()

            self.assertEqual(db.get_all_conversations(), [])
            self.assertEqual(db.search_conversations_by_message("alpha"), [])
            self.assertEqual(db.search_conversations_by_message("beta"), [])


if __name__ == "__main__":
    unittest.main()
