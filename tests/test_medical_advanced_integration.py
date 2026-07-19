from __future__ import annotations

import unittest

import numpy as np

from medical.pipeline import (
    MedicalImageAnalyzer,
    MedicalImageAnalyzerConfig,
    build_default_medical_analyzer_config,
    validate_medical_analyzer_config,
)
from medical.pipeline import DetectionFinding


def _base_config(**overrides) -> MedicalImageAnalyzerConfig:
    defaults = build_default_medical_analyzer_config()
    return defaults.__class__(**{**defaults.__dict__, **overrides})


class SegmentationROIIntegrationTests(unittest.TestCase):
    def _analyzer(self, **cfg) -> MedicalImageAnalyzer:
        return MedicalImageAnalyzer(config=_base_config(**cfg))

    def test_roi_disabled_returns_original(self) -> None:
        analyzer = self._analyzer(enable_segmentation_roi=False)
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        out, roi_info = analyzer._apply_segmentation_roi(image)
        self.assertIsNone(roi_info)
        self.assertEqual(out.shape, image.shape)

    def test_roi_enabled_crops_with_fallback_otsu(self) -> None:
        analyzer = self._analyzer(enable_segmentation_roi=True, segmentation_roi_margin=4)
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[40:90, 40:90] = 255  # vung sang de Otsu tach ra
        out, roi_info = analyzer._apply_segmentation_roi(image)
        self.assertIsNotNone(roi_info)
        self.assertIn("bbox", roi_info)
        self.assertGreater(out.size, 0)

    def test_roi_failure_returns_original(self) -> None:
        analyzer = self._analyzer(enable_segmentation_roi=True)
        bad_image = np.zeros((0, 0, 3), dtype=np.uint8)
        out, roi_info = analyzer._apply_segmentation_roi(bad_image)
        self.assertIsNone(roi_info)
        self.assertEqual(out.shape, bad_image.shape)


class UncertaintyIntegrationTests(unittest.TestCase):
    def test_uncertainty_disabled_returns_none(self) -> None:
        analyzer = MedicalImageAnalyzer(config=_base_config(enable_mc_dropout=False))
        image = np.full((64, 64, 3), 100, dtype=np.uint8)
        self.assertIsNone(analyzer._estimate_uncertainty(image))

    def test_uncertainty_returns_none_when_no_cnn(self) -> None:
        # Model centroid mac dinh -> _load_cnn_wrapper tra None -> uncertainty None.
        analyzer = MedicalImageAnalyzer(config=_base_config(enable_mc_dropout=True))
        image = np.full((64, 64, 3), 100, dtype=np.uint8)
        self.assertIsNone(analyzer._estimate_uncertainty(image))




class DetectionFrameTests(unittest.TestCase):
    def test_detections_offset_to_source_frame(self) -> None:
        detections = [DetectionFinding(label="x", confidence=0.9, bbox=(1, 2, 3, 4))]
        shifted = MedicalImageAnalyzer._detections_in_source_frame(detections, {"offset": [10, 20]})
        self.assertEqual(shifted[0].bbox, (11, 22, 13, 24))

    def test_no_roi_returns_same_detections(self) -> None:
        detections = [DetectionFinding(label="x", confidence=0.9, bbox=(1, 2, 3, 4))]
        self.assertEqual(
            MedicalImageAnalyzer._detections_in_source_frame(detections, None), detections
        )

    def test_zero_offset_returns_same(self) -> None:
        detections = [DetectionFinding(label="x", confidence=0.9, bbox=(1, 2, 3, 4))]
        shifted = MedicalImageAnalyzer._detections_in_source_frame(detections, {"offset": [0, 0]})
        self.assertEqual(shifted[0].bbox, (1, 2, 3, 4))


class ConfigValidationTests(unittest.TestCase):
    def test_new_fields_validated(self) -> None:
        cfg = _base_config(segmentation_roi_margin=-1, mc_dropout_samples=0)
        issues = validate_medical_analyzer_config(cfg)
        self.assertTrue(any("segmentation_roi_margin" in issue for issue in issues))
        self.assertTrue(any("mc_dropout_samples" in issue for issue in issues))

    def test_valid_config_has_no_new_field_issues(self) -> None:
        cfg = _base_config(segmentation_roi_margin=10, mc_dropout_samples=20)
        issues = validate_medical_analyzer_config(cfg)
        self.assertFalse(any("segmentation_roi_margin" in issue for issue in issues))
        self.assertFalse(any("mc_dropout_samples" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
