from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.filewriter import dcmwrite

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
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "lung_ct_input.jpg"
            model_path = Path(temp_dir) / "medical_model.pt"
            cv2.imwrite(str(image_path), np.full((120, 160, 3), 200, dtype=np.uint8))
            model_path.write_bytes(b"weights")
            config = MedicalImageAnalyzerConfig(
                model_path=model_path,
                working_dir=Path(temp_dir) / "work",
                reports_dir=Path(temp_dir) / "reports",
                processed_dir=Path(temp_dir) / "normalized",
                overlay_dir=Path(temp_dir) / "overlay",
                image_size=320,
                validation_min_confidence=0.10,
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

    def test_analyze_image_accepts_dicom_series_folder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "series"
            root.mkdir()
            model_path = Path(temp_dir) / "medical_model.pt"
            model_path.write_bytes(b"weights")

            def make_dicom(path: Path, value: int) -> None:
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
                dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
                dataset.Rows = 16
                dataset.Columns = 16
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 0
                dataset.Modality = "CT"
                dataset.SeriesDescription = "CT nguc"
                dataset.BodyPartExamined = "CHEST"
                dataset.PixelData = np.full((16, 16), value, dtype=np.uint16).tobytes()
                dcmwrite(path, dataset, little_endian=True, implicit_vr=False)

            make_dicom(root / "slice_001.dcm", 10)
            make_dicom(root / "slice_002.dcm", 20)
            make_dicom(root / "slice_003.dcm", 30)

            config = MedicalImageAnalyzerConfig(
                model_path=model_path,
                working_dir=Path(temp_dir) / "work",
                reports_dir=Path(temp_dir) / "reports",
                processed_dir=Path(temp_dir) / "normalized",
                overlay_dir=Path(temp_dir) / "overlay",
                image_size=320,
                validation_min_confidence=0.10,
            )
            analyzer = MedicalImageAnalyzer(config=config, detector_backend=_FakeDetector([]))

            result = analyzer.analyze_image(root, patient_code="BN100", case_id=7)

            self.assertEqual(result.source_image, root)
            self.assertTrue(result.report_json_path.exists())
            self.assertTrue(result.report_md_path.exists())

    def test_classify_findings_returns_low_when_no_detection(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("medical_7_cancers.pt"),
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
                model_path=Path("medical_7_cancers.pt"),
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

    def test_classify_findings_returns_uncertain_when_below_certainty(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("medical_7_cancers.pt"),
                working_dir=Path("output/medical"),
                reports_dir=Path("output/medical/reports"),
                processed_dir=Path("output/medical/normalized_images"),
                overlay_dir=Path("output/medical/processed_images"),
                certainty_threshold=0.8,
            ),
            detector_backend=_FakeDetector([]),
        )
        findings = [DetectionFinding(label="lesion", confidence=0.5, bbox=(1, 2, 3, 4))]

        risk_level, suspected_malignant, recommendation, average_confidence = analyzer._classify_findings(findings)

        self.assertEqual(risk_level, "uncertain")
        self.assertFalse(suspected_malignant)
        self.assertIn("0.50", recommendation)
        self.assertIn("0.80", recommendation)
        self.assertIn("chuyen khoa", recommendation)

    def test_evaluate_image_quality_returns_warning_for_dark_image(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("medical_7_cancers.pt"),
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

    def test_prepare_image_for_analysis_uses_modality_specific_preprocessing(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("medical_7_cancers.pt"),
                working_dir=Path("output/medical"),
                reports_dir=Path("output/medical/reports"),
                processed_dir=Path("output/medical/normalized_images"),
                overlay_dir=Path("output/medical/processed_images"),
            ),
            detector_backend=_FakeDetector([]),
        )
        base = np.tile(np.linspace(60, 90, 64, dtype=np.uint8), (64, 1)).astype(np.uint8)
        base = np.repeat(base[:, :, None], 3, axis=2)

        processed = analyzer._prepare_image_for_analysis(base, modality="mammogram")

        self.assertEqual(processed.shape, base.shape)
        self.assertFalse(np.array_equal(processed, base))

    def test_get_modality_thresholds_returns_stricter_ultrasound_profile(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("medical_7_cancers.pt"),
                working_dir=Path("output/medical"),
                reports_dir=Path("output/medical/reports"),
                processed_dir=Path("output/medical/normalized_images"),
                overlay_dir=Path("output/medical/processed_images"),
            ),
            detector_backend=_FakeDetector([]),
        )

        ct_profile = analyzer._get_modality_profile("ct")
        ultrasound_profile = analyzer._get_modality_profile("ultrasound")

        self.assertGreater(ultrasound_profile["certainty_threshold"], ct_profile["certainty_threshold"])
        self.assertGreater(ultrasound_profile["medium_threshold"], ct_profile["medium_threshold"])

    def test_validate_medical_model_path_accepts_existing_model_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "medical_7_cancers.pt"
            model_path.write_bytes(b"weights")
            config = MedicalImageAnalyzerConfig(
                model_path=model_path,
                working_dir=Path(temp_dir) / "work",
                reports_dir=Path(temp_dir) / "reports",
                processed_dir=Path(temp_dir) / "normalized",
                overlay_dir=Path(temp_dir) / "overlay",
            )

            resolved = MedicalImageAnalyzer(config=config, detector_backend=_FakeDetector([])).ensure_ready()

        self.assertEqual(resolved, model_path)

    def test_validate_medical_model_path_allows_explicit_fallback_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pretrained_dir = Path(temp_dir) / "models" / "pretrained"
            pretrained_dir.mkdir(parents=True, exist_ok=True)
            fallback_model = pretrained_dir / "medical_fallback.pt"
            fallback_model.write_bytes(b"weights")
            config = MedicalImageAnalyzerConfig(
                model_path=Path(temp_dir) / "missing.pt",
                working_dir=Path(temp_dir) / "work",
                reports_dir=Path(temp_dir) / "reports",
                processed_dir=Path(temp_dir) / "normalized",
                overlay_dir=Path(temp_dir) / "overlay",
                fallback_model_path=fallback_model,
                allow_fallback_model=True,
            )

            resolved = MedicalImageAnalyzer(config=config, detector_backend=_FakeDetector([])).ensure_ready()

        self.assertEqual(resolved, fallback_model)

    def test_detect_findings_returns_topk_cnn_classes(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("medical_7_cancers.pt"),
                working_dir=Path("output/medical"),
                reports_dir=Path("output/medical/reports"),
                processed_dir=Path("output/medical/normalized_images"),
                overlay_dir=Path("output/medical/processed_images"),
                analyze_topk=3,
            ),
        )
        fake_wrapper = SimpleNamespace(
            predict=lambda source, **kw: [
                {"label": "Ung thư gan", "confidence": 0.7, "probabilities": {}},
                {"label": "Ung thư phổi", "confidence": 0.2, "probabilities": {}},
                {"label": "Ung thư vú", "confidence": 0.1, "probabilities": {}},
            ]
        )
        analyzer._load_cnn_wrapper = lambda: fake_wrapper  # type: ignore[assignment]

        findings = analyzer._detect_findings(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(len(findings), 3)
        labels = [item.label for item in findings]
        self.assertIn("Ung thư gan", labels)
        self.assertAlmostEqual(findings[0].confidence, 0.7)
