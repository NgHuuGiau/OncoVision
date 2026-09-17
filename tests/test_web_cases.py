from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app
from app.web_auth import WebAuthDatabase, hash_password
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
        auth_db = WebAuthDatabase(db_path)
        auth_db.create_user("testadmin", hash_password("ValidAdminPass123!"), "admin")
        auth_db.create_user("clinician01", hash_password("ValidClinicianPass123!"), "clinician")
        auth_db.create_user("clinician02", hash_password("OtherClinicianPass123!"), "clinician")
        auth_db.create_user("viewer01", hash_password("ValidViewerPassword123!"), "viewer")
        patcher_auth = patch.object(web_app, "_auth_db", auth_db)
        patcher_path = patch.object(web_app, "CHAT_HISTORY_DB_PATH", db_path)
        for p in (patcher_db, patcher_case, patcher_auth, patcher_path):
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(web_app.app)
        login_page = self.client.get("/login")
        csrf = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text).group(1)
        self.client.post("/login", data={"username": "testadmin", "password": "ValidAdminPass123!", "csrf_token": csrf})
        self.csrf = re.search(r'name="csrf-token" content="([^"]+)"', self.client.get("/").text).group(1)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def login_as(self, username: str, password: str):
        client = TestClient(web_app.app)
        page = client.get("/login")
        csrf = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
        client.post("/login", data={"username": username, "password": password, "csrf_token": csrf})
        root = client.get("/")
        token = re.search(r'name="csrf-token" content="([^"]+)"', root.text).group(1)
        return client, token

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

    def test_web_ui_lists_new_targets_as_not_model_ready(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Ung thư thận", response.text)
        self.assertIn("Ung thư tụy", response.text)
        self.assertIn("Ung thư tuyến giáp", response.text)
        self.assertIn("model_ready", response.text)
        self.assertIn('id="caseAssignmentModal"', response.text)
        self.assertIn('id="lightBtn"', response.text)

    def test_get_missing_case_returns_404(self) -> None:
        self.assertEqual(self.client.get("/api/cases/9999").status_code, 404)

    def test_admin_assigns_case_clinician_reviews_and_viewer_uses_code(self) -> None:
        assigned = self.client.post(
            f"/api/cases/{self.case_id}/assign",
            data={"username": "clinician01"},
            headers={"X-CSRF-Token": self.csrf},
        )
        self.assertEqual(assigned.status_code, 200)

        clinician, clinician_csrf = self.login_as("clinician01", "ValidClinicianPass123!")
        clinician_page = clinician.get("/")
        self.assertIn('id="workflowCaseList"', clinician_page.text)
        self.assertNotIn('class="app-container"', clinician_page.text)
        cases = clinician.get("/api/cases").json()["cases"]
        self.assertEqual([case["case_id"] for case in cases], [self.case_id])
        self.assertEqual(clinician.get("/api/cases/9999").status_code, 404)
        self.assertEqual(clinician.post(
            "/api/analyze", data={"image_path": "not-used"}, headers={"X-CSRF-Token": clinician_csrf}
        ).status_code, 403)
        self.assertEqual(clinician.get(f"/api/cases/{self.case_id}/image").status_code, 404)
        approved = clinician.post(
            f"/api/cases/{self.case_id}/review",
            data={
                "risk_level": "medium",
                "suspected_malignant": "false",
                "recommendation": "Đã được nhân viên rà soát.",
            },
            headers={"X-CSRF-Token": clinician_csrf},
        )
        self.assertEqual(approved.status_code, 200, approved.text)
        public_code = approved.json()["public_code"]
        self.assertRegex(public_code, r"^[A-Z0-9]{10}$")

        viewer, _ = self.login_as("viewer01", "ValidViewerPassword123!")
        viewer_page = viewer.get("/")
        self.assertIn('id="publicCaseSearch"', viewer_page.text)
        self.assertNotIn('class="app-container"', viewer_page.text)
        self.assertEqual(viewer.get("/api/cases").json()["cases"], [])
        self.assertEqual(viewer.get(f"/api/cases/{self.case_id}").status_code, 403)
        self.assertEqual(viewer.get(f"/api/cases/{self.case_id}/pdf").status_code, 403)
        self.assertEqual(viewer.get("/api/conversations").status_code, 403)
        self.assertEqual(viewer.get("/api/public/cases/AAAAAAAAAA").status_code, 404)
        public_result = viewer.get(f"/api/public/cases/{public_code}")
        self.assertEqual(public_result.status_code, 200)
        self.assertEqual(public_result.json()["case"]["recommendation"], "Đã được nhân viên rà soát.")
        self.assertNotIn("image_path", public_result.json()["case"])

        pdf = Path(self._tmp.name) / "public.pdf"
        pdf.write_bytes(b"%PDF public report")
        with patch.object(web_app, "export_case_pdf", return_value=pdf) as export:
            public_pdf = viewer.get(f"/api/public/cases/{public_code}/pdf")
        self.assertEqual(public_pdf.status_code, 200)
        self.assertEqual(public_pdf.content, b"%PDF public report")
        self.assertEqual(export.call_args.args[1]["source_image"], "")
        self.assertEqual(export.call_args.args[1]["review_status"], "approved")

        reassigned = self.client.post(
            f"/api/cases/{self.case_id}/assign",
            data={"username": "clinician02"},
            headers={"X-CSRF-Token": self.csrf},
        )
        self.assertEqual(reassigned.status_code, 200)
        self.assertEqual(clinician.get(f"/api/cases/{self.case_id}").status_code, 403)
        self.assertEqual(viewer.get(f"/api/public/cases/{public_code}").status_code, 404)

    def test_clinician_cannot_read_case_assigned_to_another_employee(self) -> None:
        self.assertTrue(web_app.get_case_db().assign_case(self.case_id, "clinician02"))
        clinician, _ = self.login_as("clinician01", "ValidClinicianPass123!")
        self.assertEqual(clinician.get(f"/api/cases/{self.case_id}").status_code, 403)

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
            headers={"X-CSRF-Token": self.csrf},
        )
        self.assertEqual(resp.status_code, 200)
        conv = web_app.get_db().get_conversation(conv_id)
        self.assertEqual(conv.messages[0].metadata_json, meta)

    def test_static_files_served(self) -> None:
        css_resp = self.client.get("/static/css/styles.css")
        self.assertEqual(css_resp.status_code, 200)
        self.assertIn("--bg-primary", css_resp.text)
        self.assertIn("color-scheme: light", css_resp.text)
        self.assertIn("min-width: 1024px", css_resp.text)

        js_resp = self.client.get("/static/js/app.js")
        self.assertEqual(js_resp.status_code, 200)
        self.assertIn("OncoVision AI Workstation", js_resp.text)

        fav_resp = self.client.get("/static/favicon.svg")
        self.assertEqual(fav_resp.status_code, 200)
        self.assertIn("svg", fav_resp.headers.get("content-type", ""))

        auth_css = self.client.get("/static/css/auth.css")
        self.assertEqual(auth_css.status_code, 200)
        self.assertIn('[data-theme="light"]', auth_css.text)

        theme_js = self.client.get("/static/js/auth-theme.js")
        self.assertEqual(theme_js.status_code, 200)
        self.assertIn("toggleAuthTheme", theme_js.text)
        workflow_js = self.client.get("/static/js/case-workflow.js")
        self.assertEqual(workflow_js.status_code, 200)
        self.assertIn("searchPublicCase", workflow_js.text)

    def test_settings_get_and_post(self) -> None:
        get_resp = self.client.get("/api/settings")
        self.assertEqual(get_resp.status_code, 200)
        self.assertTrue(get_resp.json()["ok"])

        post_resp = self.client.post(
            "/api/settings",
            data={"language": "en", "theme": "dark"},
            headers={"X-CSRF-Token": self.csrf},
        )
        self.assertEqual(post_resp.status_code, 200)
        self.assertTrue(post_resp.json()["ok"])

        get_resp2 = self.client.get("/api/settings")
        self.assertEqual(get_resp2.json()["language"], "en")
        self.assertEqual(get_resp2.json()["theme"], "dark")

    def test_404_error_page_html(self) -> None:
        resp = self.client.get("/nonexistent-page", headers={"accept": "text/html,application/xhtml+xml"})
        self.assertEqual(resp.status_code, 404)
        self.assertIn("404", resp.text)
        self.assertIn("Không tìm thấy", resp.text)
        self.assertIn("error-theme-toggle", resp.text)

    def test_404_error_json_for_api(self) -> None:
        resp = self.client.get("/api/nonexistent-route")
        self.assertEqual(resp.status_code, 404)
        body = resp.json()
        self.assertFalse(body.get("ok", True))


if __name__ == "__main__":
    unittest.main()
