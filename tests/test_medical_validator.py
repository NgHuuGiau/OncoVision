from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.filewriter import dcmwrite

from medical.dashboard import write_training_dashboard
from medical.validator import get_modality_tuning, validate_image


class MedicalValidatorTests(unittest.TestCase):
    def test_valid_ct_lung_image_passes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "lung_ct_001.jpg"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))

            result = validate_image(image_path, min_confidence=0.15)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.modality, "ct")
            self.assertEqual(result.body_region, "lung")
            self.assertGreaterEqual(result.modality_confidence, 0.0)
            self.assertGreaterEqual(result.body_region_confidence, 0.0)

    def test_valid_mammogram_image_passes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "breast_mammo_001.png"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))

            result = validate_image(image_path, min_confidence=0.15)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.modality, "mammogram")
            self.assertEqual(result.body_region, "breast")

    def test_valid_cervical_mri_image_passes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cervical_mri_001.jpg"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))

            result = validate_image(image_path, min_confidence=0.15)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.modality, "mri")
            self.assertEqual(result.body_region, "cervix")

    def test_unsupported_extension_returns_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "scan.pdf"
            image_path.write_bytes(b"%PDF-1.4")

            result = validate_image(image_path)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.error_code, "INVALID_FILE_FORMAT")

    def test_missing_file_returns_error(self) -> None:
        result = validate_image("nonexistent.jpg")

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "IMAGE_READ_FAILED")

    def test_invalid_image_content_returns_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "bad.jpg"
            image_path.write_bytes(b"this is not an image")

            result = validate_image(image_path)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.error_code, "IMAGE_READ_FAILED")

    def test_selfie_returns_unknown_image_type(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "selfie.jpg"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))

            result = validate_image(image_path, min_confidence=0.70)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.error_code, "UNKNOWN_IMAGE_TYPE")

    def test_brain_mri_returns_unsupported_body_region(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "brain_mri.dcm"
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
            dataset = FileDataset(str(image_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
            dataset.Rows = 16
            dataset.Columns = 16
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.Modality = "MR"
            dataset.BodyPartExamined = "BRAIN"
            dataset.PixelData = np.full((16, 16), 128, dtype=np.uint16).tobytes()
            dcmwrite(image_path, dataset, little_endian=True, implicit_vr=False)

            result = validate_image(image_path, min_confidence=0.20)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.error_code, "UNKNOWN_BODY_REGION")

    def test_hand_xray_returns_unknown_image_type(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "hand_xray.jpg"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))

            result = validate_image(image_path, min_confidence=0.10)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.error_code, "UNKNOWN_IMAGE_TYPE")

    def test_cervical_pap_input_returns_non_image_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cervical_pap_colposcopy.jpg"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))

            result = validate_image(image_path, min_confidence=0.15)

            self.assertEqual(result.status, "error")
            self.assertEqual(result.error_code, "NON_IMAGE_CERVICAL_INPUT")

    def test_custom_allowed_extensions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "lung_ct_scan.tif"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))

            result = validate_image(image_path, allowed_extensions=[".tif"], min_confidence=0.15)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.modality, "ct")

    def test_low_contrast_image_reports_quality_warnings(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "lung_ct_001.jpg"
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 20, dtype=np.uint8))

            result = validate_image(image_path, min_confidence=0.15)

            self.assertTrue(result.quality_warnings)
            self.assertIn("contrast", " ".join(result.quality_warnings).lower())

    def test_quality_warning_branch_does_not_crash(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "lung_ct_001.jpg"
            cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))

            result = validate_image(image_path, min_confidence=0.15)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.modality, "ct")
            self.assertEqual(result.body_region, "lung")

    def test_modality_tuning_makes_ultrasound_stricter_than_ct(self) -> None:
        ct = get_modality_tuning("ct")
        ultrasound = get_modality_tuning("ultrasound")

        self.assertGreater(float(ultrasound["certainty_threshold"]), float(ct["certainty_threshold"]))
        self.assertGreater(float(ultrasound["quality_threshold"]), float(ct["quality_threshold"]))

    def test_modality_tuning_can_be_overridden_from_config(self) -> None:
        tuning = get_modality_tuning(
            "ultrasound",
            {
                "default": {
                    "certainty_threshold": 0.50,
                    "medium_threshold": 0.42,
                    "quality_threshold": 0.41,
                },
                "ultrasound": {
                    "certainty_threshold": 0.80,
                    "medium_threshold": 0.60,
                    "quality_threshold": 0.65,
                },
            },
        )

        self.assertEqual(float(tuning["certainty_threshold"]), 0.80)
        self.assertEqual(float(tuning["quality_threshold"]), 0.65)

    def test_write_training_dashboard_adds_summary_sections(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report_path = write_training_dashboard(
                temp_dir,
                {
                    "accuracy": 0.91,
                    "history": {"train_loss": [1.0, 0.6], "val_acc": [0.8, 0.91]},
                    "confusion_matrix": [[2, 0], [1, 3]],
                    "top_k_predictions": [{"label": "lung", "confidence": 0.91}],
                    "low_confidence_cases": [{"label": "lung", "confidence": 0.42}],
                },
            )

            payload = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertIn("summary", payload)
            self.assertEqual(payload["summary"]["accuracy"], 0.91)
            self.assertIn("history", payload)
            self.assertIn("confusion_matrix", payload)

    def test_dicom_series_folder_passes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "lung_ct_series"
            root.mkdir()
            for i in range(3):
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
                dataset = FileDataset(str(root / f"slice_{i:03d}.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
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
                dataset.PixelData = np.full((16, 16), 128, dtype=np.uint16).tobytes()
                dcmwrite(root / f"slice_{i:03d}.dcm", dataset, little_endian=True, implicit_vr=False)

            result = validate_image(root, min_confidence=0.20)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.modality, "ct")
            self.assertEqual(result.body_region, "lung")
