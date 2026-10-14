from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.filewriter import dcmwrite

from medical.pipeline import (
    DetectionFinding,
    MedicalImageAnalyzer,
    MedicalImageAnalyzerConfig,
)
from medical.validator import ValidationResult
from medical.volume_inference import VolumeData, VolumeFinding, VolumeInferenceResult


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
    def test_segmentation_polygon_is_exported_as_binary_mask(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analyzer = MedicalImageAnalyzer(
                config=MedicalImageAnalyzerConfig(
                    model_path=root / "model.pt",
                    working_dir=root,
                    reports_dir=root / "reports",
                    processed_dir=root / "normalized",
                    overlay_dir=root / "overlay",
                )
            )
            path = analyzer._write_segmentation_mask(
                (40, 40, 3),
                [DetectionFinding("lesion", 0.9, (5, 5, 30, 30), ((5, 5), (30, 5), (15, 30)))],
                patient_code="BNMASK",
            )
            self.assertIsNotNone(path)
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask.shape, (40, 40))
            self.assertGreater(int(mask.sum()), 0)

    def test_video_samples_are_aggregated_and_keep_frame_location(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analyzer = MedicalImageAnalyzer(
                config=MedicalImageAnalyzerConfig(
                    model_path=root / "model.pt",
                    working_dir=root,
                    reports_dir=root / "reports",
                    processed_dir=root / "normalized",
                    overlay_dir=root / "overlay",
                    video_sample_frames=2,
                ),
                detector_backend=_FakeDetector(
                    [SimpleNamespace(names={0: "lesion"}, boxes=[_FakeBox(0, 0.9, [5, 5, 20, 20])])]
                ),
            )
            samples = [
                (0, 0.0, np.zeros((24, 32, 3), dtype=np.uint8)),
                (9, 0.9, np.full((24, 32, 3), 100, dtype=np.uint8)),
            ]
            normalized = root / "representative.jpg"
            validation = ValidationResult(status="success", modality="endoscopy", body_region="stomach")
            with patch("medical.pipeline.sample_medical_video_frames", return_value=samples):
                result = analyzer._run_video_samples(
                    root / "case.mp4",
                    normalized,
                    validation,
                    analyzer._get_modality_profile("endoscopy"),
                )
            detections = result[2]
            self.assertEqual(result[6], 2)
            self.assertEqual(detections[0].frame_index, 0)
            self.assertEqual(detections[0].timestamp_seconds, 0.0)
            self.assertTrue(normalized.is_file())

    def test_segment_detector_preserves_polygon_coordinates(self) -> None:
        analyzer = MedicalImageAnalyzer(
            config=MedicalImageAnalyzerConfig(
                model_path=Path("model.pt"),
                working_dir=Path("output/medical"),
                reports_dir=Path("output/medical/reports"),
                processed_dir=Path("output/medical/normalized_images"),
                overlay_dir=Path("output/medical/processed_images"),
            ),
            detector_backend=_FakeDetector([]),
        )
        result = SimpleNamespace(
            names={0: "lesion"},
            boxes=[_FakeBox(0, 0.9, [1, 2, 20, 30])],
            masks=SimpleNamespace(xy=[np.asarray([[1.2, 2.1], [20.0, 2.0], [10.0, 30.0]])]),
        )
        analyzer._detector_backend = _FakeDetector([result])
        findings = analyzer._detect_with_backend(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertEqual(findings[0].segmentation, ((1, 2), (20, 2), (10, 30)))

    def test_analyze_brain_volume_uses_3d_runtime_and_preserves_coordinates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "brain.nii.gz"
            source.write_bytes(b"volume")
            model_path = root / "brain_volume_model.pt"
            manifest_path = root / "brain_volume_model.json"
            model_path.write_bytes(b"model")
            manifest_path.write_text("{}", encoding="utf-8")
            config = MedicalImageAnalyzerConfig(
                model_path=root / "brain_classifier.pt",
                brain_model_path=root / "brain_classifier.pt",
                working_dir=root / "work",
                reports_dir=root / "reports",
                processed_dir=root / "normalized",
                overlay_dir=root / "overlay",
                volume_model_path=model_path,
                volume_manifest_path=manifest_path,
            )
            analyzer = MedicalImageAnalyzer(config=config)
            analyzer.validate_input = lambda _: ValidationResult(status="success", modality="mri", body_region="brain")
            finding = VolumeFinding(
                label="glioma",
                confidence=0.91,
                bbox_voxel=(2, 3, 4, 8, 9, 10),
                bbox_mm=(2.0, 3.0, 4.0, 8.0, 9.0, 10.0),
                voxel_count=216,
                volume_mm3=216.0,
            )
            inference = VolumeInferenceResult(
                predictions=({"label": "glioma", "confidence": 0.91},),
                findings=(finding,),
                mask=np.ones((12, 12, 12), dtype=bool),
            )
            analyzer._volume_runtime = SimpleNamespace(
                manifest=SimpleNamespace(target="brain", input_modalities=("image",)),
                predict=lambda _: inference,
            )
            volume = VolumeData(
                array=np.ones((12, 12, 12), dtype=np.float32),
                spacing_xyz=(1.0, 1.0, 1.0),
                affine=np.eye(4),
            )
            normalized = root / "normalized.png"
            overlay = root / "overlay.png"
            normalized.write_bytes(b"png")
            overlay.write_bytes(b"png")

            with patch("medical.pipeline.load_medical_volume", return_value=volume), patch(
                "medical.pipeline.render_volume_overlay", return_value=(normalized, overlay)
            ):
                result = analyzer.analyze_image(source, patient_code="BN3D", target_key="brain")

            self.assertEqual(result.input_dimension, "3d")
            self.assertEqual(result.volume_findings[0]["bbox_voxel"], [2, 3, 4, 8, 9, 10])
            self.assertTrue(result.volume_mask_path.is_file())
            with np.load(result.volume_mask_path) as saved_mask:
                self.assertEqual(saved_mask["mask"].shape, (12, 12, 12))
                self.assertEqual(saved_mask["coordinate_system"].item(), "unknown")
            self.assertTrue(result.report_json_path.is_file())

    def test_brain_fallback_rejects_non_brain_input(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "lung_ct_input.jpg"
            source.write_bytes(b"x")
            analyzer = MedicalImageAnalyzer(
                config=MedicalImageAnalyzerConfig(
                    model_path=Path(temp_dir) / "missing.pt",
                    brain_model_path=Path(temp_dir) / "brain.pt",
                    working_dir=Path(temp_dir) / "work",
                    reports_dir=Path(temp_dir) / "reports",
                    processed_dir=Path(temp_dir) / "normalized",
                    overlay_dir=Path(temp_dir) / "overlay",
                ),
            )
            analyzer._brain_fallback_active = True
            analyzer.ensure_ready = lambda *_: Path(temp_dir) / "brain.pt"
            analyzer.validate_input = lambda _: ValidationResult(status="success", body_region="lung")

            with self.assertRaisesRegex(ValueError, "TARGET_MISMATCH"):
                analyzer.analyze_image(source, patient_code="BN100", target_key="brain")

    def test_legacy_brain_mri_model_is_not_used_for_brain_ct(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brain_model = root / "brain_classifier.pt"
            brain_model.write_bytes(b"weights")
            analyzer = MedicalImageAnalyzer(
                config=MedicalImageAnalyzerConfig(
                    model_path=brain_model,
                    brain_model_path=brain_model,
                    working_dir=root / "work",
                    reports_dir=root / "reports",
                    processed_dir=root / "normalized",
                    overlay_dir=root / "overlay",
                    target_model_root=root / "cancers",
                )
            )
            with self.assertRaisesRegex(FileNotFoundError, "brain.*ct"):
                analyzer.ensure_ready("brain", "ct")

    def test_analyze_image_generates_overlay_and_reports(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "lung_ct_input.jpg"
            model_path = Path(temp_dir) / "medical_model.pt"
            target_root = Path(temp_dir) / "cancers"
            (target_root / "lung" / "ct").mkdir(parents=True)
            (target_root / "lung" / "ct" / "classifier.pt").write_bytes(b"weights")
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
                target_model_root=target_root,
            )
            fake_result = SimpleNamespace(names={0: "lesion"}, boxes=[_FakeBox(0, 0.91, [20, 30, 100, 110])])
            analyzer = MedicalImageAnalyzer(config=config, detector_backend=_FakeDetector([fake_result]))

            result = analyzer.analyze_image(image_path, patient_code="BN100", case_id=7)

            self.assertEqual(result.case_id, 7)
            self.assertTrue(result.processed_image.exists())
            self.assertTrue(result.report_json_path.exists())
            self.assertTrue(result.report_md_path.exists())
            self.assertEqual(result.risk_level, "uncertain")
            self.assertFalse(result.suspected_malignant)
            self.assertEqual(len(result.detections), 1)
            self.assertIsInstance(result.quality_warnings, list)

    def test_detect_runs_on_raw_source_and_boxes_map_back_to_canvas(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "ct_input.jpg"
            model_path = Path(temp_dir) / "medical_model.pt"
            target_root = Path(temp_dir) / "cancers"
            (target_root / "lung" / "ct").mkdir(parents=True)
            (target_root / "lung" / "ct" / "classifier.pt").write_bytes(b"weights")
            cv2.imwrite(str(image_path), np.full((120, 160, 3), 180, dtype=np.uint8))
            model_path.write_bytes(b"weights")
            config = MedicalImageAnalyzerConfig(
                model_path=model_path,
                working_dir=Path(temp_dir) / "work",
                reports_dir=Path(temp_dir) / "reports",
                processed_dir=Path(temp_dir) / "normalized",
                overlay_dir=Path(temp_dir) / "overlay",
                image_size=320,
                validation_min_confidence=0.10,
                target_model_root=target_root,
            )
            fake_result = SimpleNamespace(names={0: "lesion"}, boxes=[_FakeBox(0, 0.9, [20, 30, 100, 110])])
            captured: list[tuple[int, ...]] = []

            class _CapturingDetector(_FakeDetector):
                def predict(self, source, **kwargs):
                    array = np.asarray(source)
                    captured.append(array.shape)
                    return super().predict(source, **kwargs)

            analyzer = MedicalImageAnalyzer(config=config, detector_backend=_CapturingDetector([fake_result]))

            result = analyzer.analyze_image(image_path, patient_code="BNRAW", case_id=8)

            self.assertIn((120, 160, 3), captured)
            self.assertEqual(len(result.detections), 1)
            # thumbnail khong phong to (s=1), dan giua canvas 320 -> offset (80, 100)
            self.assertEqual(result.detections[0].bbox, (100, 130, 180, 210))

    def test_detector_slot_warning_points_to_modality_slot(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root_detector = root / "cancers" / "brain" / "detector.pt"
            root_detector.parent.mkdir(parents=True)
            root_detector.write_bytes(b"weights")
            analyzer = MedicalImageAnalyzer(
                config=MedicalImageAnalyzerConfig(
                    model_path=root / "model.pt",
                    working_dir=root / "work",
                    reports_dir=root / "reports",
                    processed_dir=root / "normalized",
                    overlay_dir=root / "overlay",
                    target_model_root=root / "cancers",
                )
            )
            analyzer._active_target_key = "brain"
            analyzer._active_modality = "mri"
            analyzer._detector_backend = None

            warning = analyzer._detector_slot_warning()
            self.assertIsNotNone(warning)
            self.assertIn("cấp cơ quan", warning)
            self.assertIn("mri", warning)

            slot_detector = root / "cancers" / "brain" / "mri" / "detector.pt"
            slot_detector.parent.mkdir()
            slot_detector.write_bytes(b"weights")
            self.assertIsNone(analyzer._detector_slot_warning())

            slot_detector.unlink()
            analyzer._detector_backend = object()
            self.assertIsNone(analyzer._detector_slot_warning())

    def test_dicom_series_requires_matching_3d_model(self) -> None:
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

            with self.assertRaisesRegex(ValueError, "MISSING_3D_MODEL"):
                analyzer.analyze_image(root, patient_code="BN100", case_id=7)

    def test_classify_findings_does_not_treat_no_label_as_negative_diagnosis(self) -> None:
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

        self.assertEqual(risk_level, "uncertain")
        self.assertFalse(suspected_malignant)
        self.assertEqual(average_confidence, 0.0)
        self.assertIn("không phải kết luận", recommendation)

    def test_classify_findings_does_not_infer_malignancy_from_model_score(self) -> None:
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

        self.assertEqual(risk_level, "uncertain")
        self.assertFalse(suspected_malignant)
        self.assertAlmostEqual(average_confidence, 0.6)
        self.assertIn("không phải kết luận", recommendation)

    def test_classify_findings_keeps_clinical_assessment_unreviewed(self) -> None:
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
        self.assertEqual(average_confidence, 0.5)
        self.assertIn("không phải kết luận", recommendation)

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
        self.assertTrue(any("quá tối" in warning for warning in warnings))

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
                {"label": "glioma", "confidence": 0.7, "probabilities": {}},
                {"label": "meningioma", "confidence": 0.2, "probabilities": {}},
                {"label": "pituitary", "confidence": 0.1, "probabilities": {}},
            ]
        )
        analyzer._load_cnn_wrapper = lambda: fake_wrapper  # type: ignore[assignment]

        findings = analyzer._detect_findings(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(len(findings), 3)
        labels = [item.label for item in findings]
        self.assertIn("glioma", labels)
        self.assertAlmostEqual(findings[0].confidence, 0.7)

    def test_detect_findings_auto_loads_configured_detector_and_clamps_bbox(self) -> None:
        with TemporaryDirectory() as temp_dir:
            detector_path = Path(temp_dir) / "brain_detector.pt"
            detector_path.write_bytes(b"weights")
            analyzer = MedicalImageAnalyzer(
                config=MedicalImageAnalyzerConfig(
                    model_path=Path(temp_dir) / "classifier.pt",
                    working_dir=Path(temp_dir) / "work",
                    reports_dir=Path(temp_dir) / "reports",
                    processed_dir=Path(temp_dir) / "normalized",
                    overlay_dir=Path(temp_dir) / "overlay",
                    yolo_model_path=detector_path,
                )
            )
            result = SimpleNamespace(names={0: "lesion"}, boxes=[_FakeBox(0, 0.91, [-5, 4, 90, 80])])
            analyzer._load_default_backend = lambda: _FakeDetector([result])  # type: ignore[method-assign]
            analyzer._load_cnn_wrapper = lambda: None  # type: ignore[method-assign]

            findings = analyzer._detect_findings(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(findings[0].bbox, (0, 4, 64, 64))
        self.assertIsNotNone(analyzer._detector_backend)

    def test_normalize_bbox_rejects_zero_area_box(self) -> None:
        self.assertIsNone(MedicalImageAnalyzer._normalize_bbox([4, 4, 4, 10], (20, 20, 3)))

    def test_detector_backend_rejects_non_localization_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "detect hoặc segment"):
            MedicalImageAnalyzer._validate_detector_backend(SimpleNamespace(task="classify", names={0: "glioma"}))

    def test_detector_backend_rejects_common_object_model(self) -> None:
        names = dict(enumerate(["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"] + [f"class-{index}" for index in range(20)]))
        with self.assertRaisesRegex(ValueError, "vật thể phổ thông"):
            MedicalImageAnalyzer._validate_detector_backend(SimpleNamespace(task="detect", names=names))
