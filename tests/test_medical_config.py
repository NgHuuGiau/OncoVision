from __future__ import annotations

import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

from medical.pipeline import build_default_medical_analyzer_config
from medical.pipeline import validate_medical_analyzer_config


class MedicalConfigTests(unittest.TestCase):
    @patch(
        "medical.pipeline.load_yaml",
        return_value={
            "medical": {
                "model": "models/trained/best.pt",
                "fallback_model": "yolo11n.pt",
                "allow_fallback_model": False,
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
    def test_build_default_medical_analyzer_config_uses_yaml_settings(self, _load_yaml_mock) -> None:
        config = build_default_medical_analyzer_config()

        self.assertEqual(config.image_size, 512)
        self.assertAlmostEqual(config.conf_threshold, 0.3)
        self.assertAlmostEqual(config.classify_high_risk_threshold, 0.8)
        self.assertAlmostEqual(config.classify_medium_risk_threshold, 0.5)
        self.assertEqual(config.working_dir, Path("output/medical-x"))
        self.assertEqual(config.reports_dir, Path("output/medical-x/reports"))
        self.assertEqual(config.fallback_model_path, Path("yolo11n.pt"))
        self.assertFalse(config.allow_fallback_model)

    def test_validate_medical_analyzer_config_accepts_valid_config(self) -> None:
        config = build_default_medical_analyzer_config()
        self.assertEqual(validate_medical_analyzer_config(config), [])

    def test_validate_medical_analyzer_config_reports_invalid_thresholds(self) -> None:
        config = build_default_medical_analyzer_config()
        broken = replace(
            config,
            image_size=0,
            conf_threshold=1.2,
            classify_medium_risk_threshold=0.9,
            classify_high_risk_threshold=0.1,
        )
        issues = validate_medical_analyzer_config(broken)
        self.assertTrue(any("image_size" in item for item in issues))
        self.assertTrue(any("conf_threshold" in item for item in issues))
        self.assertTrue(any("nguong nguy co" in item for item in issues))
