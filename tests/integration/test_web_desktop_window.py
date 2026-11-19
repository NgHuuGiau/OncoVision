from __future__ import annotations

import socket
import unittest

from app.chat_ui.web_desktop_window import find_free_port, is_port_open


class WebDesktopWindowTests(unittest.TestCase):
    def test_is_port_open_true_when_listening(self) -> None:
        port = find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", port))
            server.listen(1)
            self.assertTrue(is_port_open(port))

    def test_is_port_open_false_when_closed(self) -> None:
        port = find_free_port()
        self.assertFalse(is_port_open(port))


if __name__ == "__main__":
    unittest.main()
