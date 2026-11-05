from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from utils import camera_utils


class CameraUtilsTests(unittest.TestCase):
    @patch("utils.camera_utils.open_camera_capture")
    def test_open_camera_capture_with_fallback_returns_first_opened_capture(self, open_capture_mock) -> None:
        closed_capture = SimpleNamespace(isOpened=lambda: False, release=MagicMock())
        open_capture = SimpleNamespace(isOpened=lambda: True, release=MagicMock())
        open_capture_mock.side_effect = [closed_capture, open_capture]

        result = camera_utils.open_camera_capture_with_fallback(2, fallback_indices=(3, 4))

        self.assertIs(result.capture, open_capture)
        self.assertEqual(result.index_used, 3)
        self.assertEqual(result.attempted_indexes, (2, 3))
        closed_capture.release.assert_called_once()

    @patch("utils.camera_utils.open_camera_capture")
    def test_open_camera_capture_with_fallback_returns_none_when_all_fail(self, open_capture_mock) -> None:
        closed_capture_a = SimpleNamespace(isOpened=lambda: False, release=MagicMock())
        closed_capture_b = SimpleNamespace(isOpened=lambda: False, release=MagicMock())
        open_capture_mock.side_effect = [closed_capture_a, closed_capture_b]

        result = camera_utils.open_camera_capture_with_fallback(1, fallback_indices=(2,))

        self.assertIsNone(result.capture)
        self.assertIsNone(result.index_used)
        self.assertEqual(result.attempted_indexes, (1, 2))
        closed_capture_a.release.assert_called_once()
        closed_capture_b.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
