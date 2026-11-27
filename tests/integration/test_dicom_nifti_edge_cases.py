from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from fastapi.testclient import TestClient

import entrypoints.web.shared as web_shared
from app.web_auth import WebAuthDatabase, hash_password
from entrypoints import web_app
from medical.case_payloads import case_detections, case_localization_available
from medical.chat_service import MedicalChatResponse, MedicalChatService
from medical.storage import MedicalCaseDatabase


class DicomNiftiEdgeCaseTests(unittest.TestCase):
    """Integration tests for DICOM/NIfTI upload edge cases."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "onco.db"
        case_db = MedicalCaseDatabase(db_path)

        mock_medical_service = MagicMock(spec=MedicalChatService)
        mock_medical_service.check_ready.return_value = Path("demo_mode")
        mock_medical_service.analyze_attachment.return_value = MagicMock(
            reply_text="Mock analysis result",
            attachment_path=None,
            attachment_kind=None,
            metadata_json="{}"
        )

        patcher_db = patch.object(web_shared, "_db", None)
        patcher_case = patch.object(web_shared, "_case_db", case_db)
        patcher_medical = patch.object(web_shared, "_medical_service", mock_medical_service)
        auth_db = WebAuthDatabase(db_path)
        auth_db.create_user("testadmin", hash_password("ValidAdminPass123!"), "admin")
        auth_db.create_user("clinician01", hash_password("ValidClinicianPass123!"), "clinician")
        patcher_auth = patch.object(web_shared, "_auth_db", auth_db)
        patcher_path = patch.object(web_shared, "CHAT_HISTORY_DB_PATH", db_path)
        for p in (patcher_db, patcher_case, patcher_medical, patcher_auth, patcher_path):
            p.start()
            self.addCleanup(p.stop)

        self.client = TestClient(web_app.app)
        login_page = self.client.get("/login")
        import re
        csrf = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text).group(1)
        self.client.post("/login", data={"username": "testadmin", "password": "ValidAdminPass123!", "csrf_token": csrf})
        root = self.client.get("/")
        self.csrf = re.search(r'name="csrf-token" content="([^"]+)"', root.text).group(1)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _create_dicom_file(self, path: Path, modality: str = "CT", body_part: str = "BRAIN") -> None:
        import pydicom
        from pydicom.dataset import Dataset
        from pydicom.uid import ExplicitVRLittleEndian
        import numpy as np

        ds = Dataset()
        ds.PatientName = "Test^Patient"
        ds.PatientID = "TEST001"
        ds.Modality = modality
        ds.BodyPartExamined = body_part
        ds.Rows = 64
        ds.Columns = 64
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.random.randint(0, 4095, (64, 64), dtype=np.uint16).tobytes()
        ds.file_meta = Dataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9"
        ds.file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9"
        # Use new pydicom v4.0 API to avoid deprecation warnings
        ds.save_as(str(path), enforce_file_format=True, little_endian=True, implicit_vr=False)

    def _create_nifti_file(self, path: Path, shape: tuple = (32, 32, 32)) -> None:
        import nibabel as nib
        import numpy as np
        data = np.random.rand(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, str(path))
    def test_upload_dicom_ct_accepted(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            dicom_path = Path(f.name)
        try:
            self._create_dicom_file(dicom_path, modality="CT", body_part="BRAIN")
            with open(dicom_path, "rb") as f:
                files = {"file": ("test_ct.dcm", f, "application/dicom")}
                resp = self.client.post("/api/upload", files=files, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body["ok"])
            self.assertEqual(body["detected_target"], "brain")
            self.assertEqual(body["detected_modality"].lower(), "ct")
        finally:
            dicom_path.unlink(missing_ok=True)

    def test_upload_dicom_mri_accepted(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            dicom_path = Path(f.name)
        try:
            self._create_dicom_file(dicom_path, modality="MR", body_part="BRAIN")
            with open(dicom_path, "rb") as f:
                files = {"file": ("test_mri.dcm", f, "application/dicom")}
                resp = self.client.post("/api/upload", files=files, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body["ok"])
            self.assertEqual(body["detected_target"], "brain")
            self.assertEqual(body["detected_modality"].lower(), "mri")
        finally:
            dicom_path.unlink(missing_ok=True)

    def test_upload_nifti_accepted(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            nifti_path = Path(f.name)
        try:
            self._create_nifti_file(nifti_path)
            with open(nifti_path, "rb") as f:
                files = {"file": ("brain.nii.gz", f, "application/gzip")}
                resp = self.client.post("/api/upload", files=files, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertTrue(body["ok"])
        finally:
            nifti_path.unlink(missing_ok=True)

    def test_upload_fake_dicom_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            fake_path = Path(f.name)
        try:
            fake_path.write_bytes(b"not a dicom file")
            with open(fake_path, "rb") as f:
                files = {"file": ("fake.dcm", f, "application/dicom")}
                resp = self.client.post("/api/upload", files=files, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(resp.status_code, 400)
            body = resp.json()
            self.assertFalse(body["ok"])
            self.assertIn("không khớp", body["detail"])
        finally:
            fake_path.unlink(missing_ok=True)

    def test_upload_empty_file_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            empty_path = Path(f.name)
        try:
            empty_path.write_bytes(b"")
            with open(empty_path, "rb") as f:
                files = {"file": ("empty.dcm", f, "application/dicom")}
                resp = self.client.post("/api/upload", files=files, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(resp.status_code, 400)
            body = resp.json()
            self.assertFalse(body["ok"])
        finally:
            empty_path.unlink(missing_ok=True)
    def test_upload_dicom_without_dicom_header_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            fake_path = Path(f.name)
        try:
            fake_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
            with open(fake_path, "rb") as f:
                files = {"file": ("fake.dcm", f, "application/dicom")}
                resp = self.client.post("/api/upload", files=files, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(resp.status_code, 400)
            body = resp.json()
            self.assertFalse(body["ok"])
        finally:
            fake_path.unlink(missing_ok=True)

    def test_upload_large_file_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            large_path = Path(f.name)
        try:
            large_path.write_bytes(b"\x00" * (250 * 1024 * 1024))
            with open(large_path, "rb") as f:
                files = {"file": ("large.dcm", f, "application/dicom")}
                resp = self.client.post("/api/upload", files=files, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(resp.status_code, 413)
            body = resp.json()
            # 413 error returns {"detail": "..."} without "ok" key
            self.assertIn("detail", body)
        finally:
            large_path.unlink(missing_ok=True)

    def test_analyze_dicom_upload(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
            dicom_path = Path(f.name)
        try:
            self._create_dicom_file(dicom_path, modality="CT", body_part="BRAIN")
            with open(dicom_path, "rb") as f:
                files = {"file": ("test_ct.dcm", f, "application/dicom")}
                upload_resp = self.client.post("/api/upload", files=files, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(upload_resp.status_code, 200)
            upload_data = upload_resp.json()
            stored_path = upload_data["stored_path"]

            data = {
                "image_path": stored_path,
                "patient_code": "TEST_DICOM_001",
                "target_key": "brain",
                "modality": "ct",
                "patient_age": "50",
                "patient_sex": "M",
                "clinical_note": "Test DICOM upload",
            }
            analyze_resp = self.client.post("/api/analyze", data=data, headers={"X-CSRF-Token": self.csrf})
            self.assertEqual(analyze_resp.status_code, 200)
            analyze_data = analyze_resp.json()
            self.assertTrue(analyze_data["ok"])
            self.assertIn("conversation_id", analyze_data)
        finally:
            dicom_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
