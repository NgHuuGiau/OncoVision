from __future__ import annotations

import unittest
from pathlib import Path

from utils.icons import ICONS


class ChatUiIconAssetsTests(unittest.TestCase):
    def test_required_chat_ui_icons_exist_in_registry(self) -> None:
        required = {
            "sidebar_app.svg",
            "new_chat.svg",
            "search.svg",
            "settings.svg",
            "plus.svg",
            "mic.svg",
            "send.svg",
            "chat_history.svg",
            "image.svg",
            "camera.svg",
        }
        self.assertTrue(required.issubset(ICONS.keys()))

    def test_required_chat_ui_icons_exist_on_disk(self) -> None:
        icons_dir = Path("assets/icons")
        required = (
            "sidebar_app.svg",
            "new_chat.svg",
            "search.svg",
            "settings.svg",
            "plus.svg",
            "mic.svg",
            "send.svg",
            "chat_history.svg",
            "image.svg",
            "camera.svg",
        )
        for name in required:
            with self.subTest(name=name):
                self.assertTrue((icons_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
