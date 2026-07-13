from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from run_medical import build_parser


class ValidateImageCliTests(unittest.TestCase):
    def test_validate_image_success(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "lung_ct_001.jpg"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))

            with patch("medical.validator.DEFAULT_MIN_CONFIDENCE", 0.15):
                parser = build_parser()
                args = parser.parse_args(["validate-image", "--image", str(image_path)])
                self.assertEqual(args.command, "validate-image")
                self.assertEqual(args.image, str(image_path))

    def test_validate_image_requires_image_arg(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["validate-image"])

    def test_calibrate_modality_tuning_command_parses(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["calibrate-modality-tuning", "--apply"])

        self.assertEqual(args.command, "calibrate-modality-tuning")
        self.assertTrue(args.apply)
