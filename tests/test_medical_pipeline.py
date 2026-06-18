from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np

from medical.pipeline import DetectionFinding, MedicalImageAnalyzer, MedicalImageAnalyzerConfig


class _FakeValue:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeTensorRow:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [_FakeValue(cls_id)]
        self.conf = [_FakeValue(conf)]
        self.xyxy = [_FakeTensorRow(xyxy)]


class _FakeDetector:
    def __init__(self, results):
        self._results = results

    def predict(self, source, **kwargs):
        return self._results


class MedicalPipelineTests(unittest.TestCase):
    def test_analyze_image_generates_overlay_and_reports(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            image_path = Path(temp_dir) / "input.jpg"
            cv2.imwrite(str(image_path), np.full((120, 160, 3), 200, dtype=np.uint8))
            config = MedicalImageAnalyzerConfig(
                model_path=Path("models/pretrained/yolo11n.pt"),
                working_dir=Path(temp_dir) / "work",
                reports_dir=Path(temp_dir) / "reports",
                processed_dir=Path(temp_dir) / "normalized",
                overlay_dir=Path(temp_dir) / "overlay",
                image_size=320,
            )
            fake_result = SimpleNamespace(names={0: "lesion"}, boxes=[_FakeBox(0, 0.91, [20, 30, 100, 110])])
            analyzer = MedicalImageAnalyzer(config=config, detector_backend=_FakeDetector([fake_result]))

            result = analyzer.analyze_image(image_path, patient_code="BN100", case_id=7)

            self.assertEqual(result.case_id, 7)
            self.assertTrue(result.processed_image.exists())
            self.assertTrue(result.report_json_path.exists())
            self.assertTrue(result.report_md_path.exists())
            self.assertEqual(result.risk_level, "high")
            self.assertTrue(result.suspected_malignant)
            self.assertEqual(len(result.detections), 1)
            self.assertIsInstance(result.quality_warnings, list)

    def test_classify_findings_returns_low_when_no_detection(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("models/pretrained/yolo11n.pt"),
                working_dir=Path("output/medical"),
                reports_dir=Path("output/medical/reports"),
                processed_dir=Path("output/medical/normalized_images"),
                overlay_dir=Path("output/medical/processed_images"),
            ),
            detector_backend=_FakeDetector([]),
        )

        risk_level, suspected_malignant, recommendation, average_confidence = analyzer._classify_findings([])

        self.assertEqual(risk_level, "low")
        self.assertFalse(suspected_malignant)
        self.assertEqual(average_confidence, 0.0)
        self.assertIn("Khong ghi nhan", recommendation)

    def test_classify_findings_returns_medium_for_mid_confidence(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("models/pretrained/yolo11n.pt"),
                working_dir=Path("output/medical"),
                reports_dir=Path("output/medical/reports"),
                processed_dir=Path("output/medical/normalized_images"),
                overlay_dir=Path("output/medical/processed_images"),
            ),
            detector_backend=_FakeDetector([]),
        )
        findings = [DetectionFinding(label="lesion", confidence=0.6, bbox=(1, 2, 3, 4))]

        risk_level, suspected_malignant, recommendation, average_confidence = analyzer._classify_findings(findings)

        self.assertEqual(risk_level, "medium")
        self.assertTrue(suspected_malignant)
        self.assertAlmostEqual(average_confidence, 0.6)
        self.assertIn("trung binh", recommendation)

    def test_evaluate_image_quality_returns_warning_for_dark_image(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("models/pretrained/yolo11n.pt"),
                working_dir=Path("output/medical"),
                reports_dir=Path("output/medical/reports"),
                processed_dir=Path("output/medical/normalized_images"),
                overlay_dir=Path("output/medical/processed_images"),
            ),
            detector_backend=_FakeDetector([]),
        )

        warnings = analyzer._evaluate_image_quality(np.zeros((128, 128, 3), dtype=np.uint8))

        self.assertTrue(warnings)
        self.assertTrue(any("qua toi" in warning for warning in warnings))
