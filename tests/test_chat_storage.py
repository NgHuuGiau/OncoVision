from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from app.chat_ui.models import ChatMessage
from app.chat_ui.storage import ChatDatabase


class ChatStorageTests(unittest.TestCase):
    def test_get_all_conversations_returns_messages_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "chat.db"
            db = ChatDatabase(str(db_path))
            older_id = db.create_conversation("Older", "Yesterday")
            newer_id = db.create_conversation("Newer", "Today")
            db.add_message(older_id, ChatMessage(sender="user", text="old-1"))
            db.add_message(older_id, ChatMessage(sender="ai", text="old-2"))
            db.add_message(newer_id, ChatMessage(sender="user", text="new-1"))

            conversations = db.get_all_conversations()

            self.assertEqual([conversation.title for conversation in conversations], ["Newer", "Older"])
            self.assertEqual([message.text for message in conversations[0].messages], ["new-1"])
            self.assertEqual([message.text for message in conversations[1].messages], ["old-1", "old-2"])

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

    def test_init_accepts_db_path_without_directory_component(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                db = ChatDatabase("chat.db")
                conv_id = db.create_conversation("Local", "Now")
                db.add_message(conv_id, ChatMessage(sender="user", text="hello"))
                self.assertEqual(db.get_all_conversations()[0].title, "Local")
            finally:
                os.chdir(original_cwd)

    def test_message_metadata_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "chat.db"
            db = ChatDatabase(str(db_path))
            conv_id = db.create_conversation("Medical", "Today")
            db.add_message(
                conv_id,
                ChatMessage(
                    sender="ai",
                    text="medical reply",
                    attachment_path="overlay.jpg",
                    attachment_kind="image",
                    metadata_json='{"risk_level":"high","medical_case_id":12}',
                ),
            )

            conversations = db.get_all_conversations()

            self.assertEqual(conversations[0].messages[0].metadata_json, '{"risk_level":"high","medical_case_id":12}')


if __name__ == "__main__":
    unittest.main()
