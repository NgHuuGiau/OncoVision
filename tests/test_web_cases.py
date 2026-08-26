from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app
from medical.storage import MedicalCaseDatabase


class WebCaseRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "onco.db"
        case_db = MedicalCaseDatabase(db_path)
        self.case_id = case_db.save_case(
            patient_code="TEST-001",
            image_path="img.png",
            processed_image_path="proc.png",
            report_json_path="r.json",
            report_md_path="r.md",
            suspected_malignant=True,
            risk_level="high",
            recommendation="Theo doi them.",
            metadata={
                "detections": [{"label": "lesion", "confidence": 0.91, "bbox": [1, 2, 3, 4]}],
                "average_confidence": 0.91,
                "model_name": "brain_classifier.pt",
                "quality_warnings": [],
            },
        )
        patcher_db = patch.object(web_app, "_db", None)
        patcher_case = patch.object(web_app, "_case_db", case_db)
        patcher_path = patch.object(web_app, "CHAT_HISTORY_DB_PATH", db_path)
        for p in (patcher_db, patcher_case, patcher_path):
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(web_app.app)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_cases_returns_saved_case(self) -> None:
        resp = self.client.get("/api/cases")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(len(body["cases"]), 1)
        case = body["cases"][0]
        self.assertEqual(case["case_id"], self.case_id)
        self.assertEqual(case["risk_level"], "high")
        self.assertEqual(case["detections"][0]["label"], "lesion")

    def test_get_case_detail(self) -> None:
        resp = self.client.get(f"/api/cases/{self.case_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["case"]["patient_code"], "TEST-001")

    def test_get_missing_case_returns_404(self) -> None:
        self.assertEqual(self.client.get("/api/cases/9999").status_code, 404)

    def test_pdf_export_with_reportlab(self) -> None:
        try:
            import reportlab  # noqa: F401
        except ImportError:
            self.skipTest("reportlab not installed")
        with patch.object(web_app, "OUTPUT_DIR", Path(self._tmp.name)):
            resp = self.client.get(f"/api/cases/{self.case_id}/pdf")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "application/pdf")
        self.assertTrue(resp.content.startswith(b"%PDF"))

    def test_message_endpoint_persists_metadata_json(self) -> None:
        conv_id = web_app.get_db().create_conversation(title="T", subtitle="S")
        meta = json.dumps({"medical_case_id": self.case_id})
        resp = self.client.post(
            f"/api/conversations/{conv_id}/messages",
            data={"sender": "assistant", "text": "x", "metadata_json": meta},
        )
        self.assertEqual(resp.status_code, 200)
        conv = web_app.get_db().get_conversation(conv_id)
        self.assertEqual(conv.messages[0].metadata_json, meta)


if __name__ == "__main__":
    unittest.main()
