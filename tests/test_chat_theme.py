from __future__ import annotations

import os
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from app.chat_ui.icons import themed_icon
from app.chat_ui.theme_styles import DARK_STYLESHEET, LIGHT_STYLESHEET


class ChatThemeStylesTests(unittest.TestCase):
    def test_no_unrendered_template_tags_in_stylesheets(self) -> None:
        """Verify no unrendered template variables like {var} remain in stylesheets."""
        import re
        unrendered_dark = re.findall(r"\{[a-z_]+\}", DARK_STYLESHEET)
        unrendered_light = re.findall(r"\{[a-z_]+\}", LIGHT_STYLESHEET)
        self.assertEqual(unrendered_dark, [], f"Unrendered tags in DARK_STYLESHEET: {unrendered_dark}")
        self.assertEqual(unrendered_light, [], f"Unrendered tags in LIGHT_STYLESHEET: {unrendered_light}")

    def test_avatar_text_is_white_in_both_themes(self) -> None:
        """Avatar background is #1a56db, text must be white in both light and dark modes."""
        self.assertIn("color: #ffffff;", DARK_STYLESHEET)
        self.assertIn("color: #ffffff;", LIGHT_STYLESHEET)

    def test_brand_text_has_contrasting_colors(self) -> None:
        """Brand text must be bright blue in dark mode and royal blue in light mode."""
        self.assertIn("#60a5fa", DARK_STYLESHEET)
        self.assertIn("#1a56db", LIGHT_STYLESHEET)

    def test_bubble_user_has_proper_contrast(self) -> None:
        """User bubble in dark mode must have visible background and border."""
        self.assertIn("rgba(26, 86, 219, 0.22)", DARK_STYLESHEET)
        self.assertIn("rgba(26, 86, 219, 0.10)", LIGHT_STYLESHEET)

    def test_search_input_is_styled_in_both_themes(self) -> None:
        """Search input should have explicit text color and transparent border/background."""
        self.assertIn("QLineEdit#SearchInput", DARK_STYLESHEET)
        self.assertIn("QLineEdit#SearchInput", LIGHT_STYLESHEET)

    def test_send_icon_renders(self) -> None:
        """Send icon should render white without error."""
        icon = themed_icon("send.svg", "#ffffff", 18)
        self.assertFalse(icon.isNull())


if __name__ == "__main__":
    unittest.main()
