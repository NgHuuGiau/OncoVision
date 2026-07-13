from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from medical.modality_calibration import apply_calibrated_modality_tuning, calibrate_modality_tuning
from utils.file_utils import load_yaml, save_yaml


class MedicalModalityCalibrationTests(unittest.TestCase):
    def test_calibrate_modality_tuning_collects_local_stats(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            class_dir = dataset_root / "images" / "train"
            class_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(class_dir / "lung_ct_001.jpg"), np.full((64, 64, 3), 120, dtype=np.uint8))
            cv2.imwrite(str(class_dir / "breast_mammo_001.jpg"), np.full((64, 64, 3), 140, dtype=np.uint8))

            settings_path = Path(temp_dir) / "medical_settings.yaml"
            save_yaml(
                settings_path,
                {
                    "medical": {
                        "certainty_threshold": 0.55,
                        "classify_medium_risk_threshold": 0.45,
                        "modality_tuning": {
                            "default": {
                                "certainty_threshold": 0.55,
                                "medium_threshold": 0.45,
                                "quality_threshold": 0.45,
                                "contrast_boost": 1.0,
                                "normalize": "default",
                            }
                        },
                    }
                },
            )

            report = calibrate_modality_tuning(dataset_root, settings_path=settings_path)

            self.assertEqual(report["sample_count"], 2)
            self.assertIn("ct", report["modality_tuning"])
            self.assertIn("mammogram", report["modality_tuning"])
            self.assertEqual(report["modality_stats"]["ct"]["image_count"], 1)

    def test_apply_calibrated_modality_tuning_updates_yaml(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            class_dir = dataset_root / "images" / "train"
            class_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(class_dir / "lung_ct_001.jpg"), np.full((64, 64, 3), 120, dtype=np.uint8))

            settings_path = Path(temp_dir) / "medical_settings.yaml"
            save_yaml(
                settings_path,
                {
                    "medical": {
                        "certainty_threshold": 0.55,
                        "classify_medium_risk_threshold": 0.45,
                        "modality_tuning": {
                            "default": {
                                "certainty_threshold": 0.55,
                                "medium_threshold": 0.45,
                                "quality_threshold": 0.45,
                                "contrast_boost": 1.0,
                                "normalize": "default",
                            }
                        },
                    }
                },
            )

            report = apply_calibrated_modality_tuning(dataset_root, settings_path=settings_path)
            payload = load_yaml(settings_path)

            self.assertIn("ct", report["modality_tuning"])
            self.assertEqual(payload["medical"]["modality_tuning"]["ct"]["normalize"], "ct")
