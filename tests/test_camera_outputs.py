from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from core.camera_runner import CameraSessionControls, _load_camera_session_output_config
from core.frame_capture import FrameCapture
from core.recorder import VideoRecorder


class CameraOutputTests(unittest.TestCase):
    @patch("core.frame_capture.cv2.imwrite", return_value=True)
    def test_frame_capture_saves_into_target_directory(self, imwrite_mock) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            capture = FrameCapture(Path(temp_dir) / "caps")
            frame = np.zeros((20, 20, 3), dtype=np.uint8)

            result = capture.save_frame(frame)

            self.assertTrue(result.success)
            self.assertEqual(result.path.parent, Path(temp_dir) / "caps")
            imwrite_mock.assert_called_once()

    @patch("core.recorder.cv2.VideoWriter")
    def test_video_recorder_resizes_frame_before_write(self, writer_cls_mock) -> None:
        writer = MagicMock()
        writer.isOpened.return_value = True
        writer_cls_mock.return_value = writer
        recorder = VideoRecorder("output/recordings", fps=12.0)

        recorder.start((80, 60))
        recorder.write(np.zeros((30, 40, 3), dtype=np.uint8))
        stopped = recorder.stop()

        self.assertIsNotNone(stopped.path)
        writer.write.assert_called_once()
        writer.release.assert_called()

    def test_load_camera_session_output_config_reads_new_settings_blocks(self) -> None:
        payload = {
            "output": {
                "captures_dir": "output/custom_caps",
                "recordings_dir": "output/custom_recs",
            },
            "recording": {
                "codec": "XVID",
                "fps": 15,
            },
        }

        with patch("core.camera_runner.load_yaml_cached", return_value=payload):
            config = _load_camera_session_output_config()

        self.assertEqual(config.capture_dir, Path("output/custom_caps"))
        self.assertEqual(config.recording_dir, Path("output/custom_recs"))
        self.assertEqual(config.recording_codec, "XVID")
        self.assertEqual(config.recording_fps, 15.0)

    def test_camera_session_controls_default_to_visible_overlays(self) -> None:
        controls = CameraSessionControls()
        self.assertTrue(controls.show_overlays)
        self.assertTrue(controls.show_fps)
        self.assertTrue(controls.show_trails)
