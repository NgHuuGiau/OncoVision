from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from medical.pipeline import build_default_medical_analyzer_config


class MedicalConfigTests(unittest.TestCase):
    @patch("medical.pipeline.resolve_trained_model_path", return_value=Path("models/pretrained/yolo11n.pt"))
    @patch(
        "medical.pipeline.load_yaml",
        return_value={
            "medical": {
                "model": "models/trained/best.pt",
                "fallback_model": "yolo11n.pt",
                "image_size": 512,
                "conf_threshold": 0.3,
                "classify_high_risk_threshold": 0.8,
                "classify_medium_risk_threshold": 0.5,
                "output_root": "output/medical-x",
                "reports_dir": "output/medical-x/reports",
                "processed_dir": "output/medical-x/normalized",
                "overlay_dir": "output/medical-x/overlay",
            }
        },
    )
    def test_build_default_medical_analyzer_config_uses_yaml_settings(self, _load_yaml_mock, _resolve_model_mock) -> None:
        config = build_default_medical_analyzer_config()

        self.assertEqual(config.image_size, 512)
        self.assertAlmostEqual(config.conf_threshold, 0.3)
        self.assertAlmostEqual(config.classify_high_risk_threshold, 0.8)
        self.assertAlmostEqual(config.classify_medium_risk_threshold, 0.5)
        self.assertEqual(config.working_dir, Path("output/medical-x"))
        self.assertEqual(config.reports_dir, Path("output/medical-x/reports"))
