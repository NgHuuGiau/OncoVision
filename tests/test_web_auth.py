from __future__ import annotations

import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote

from fastapi.testclient import TestClient

import web_app
from app.web_auth import (
    WebAuthDatabase,
    generate_recovery_code,
    hash_password,
    hash_recovery_code,
    verify_password,
)


class WebAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "onco.db"
        self.auth_db = WebAuthDatabase(self.db_path)
        self.admin_id = self.auth_db.create_user(
            "admin01", hash_password("AdminPassword123!"), "admin", hash_recovery_code("ADM123")
        )
        self.viewer_id = self.auth_db.create_user(
            "viewer01", hash_password("ViewerPassword123!"), "viewer", hash_recovery_code("VIEW01")
        )
        for name, value in (
            ("_auth_db", self.auth_db),
            ("_db", None),
            ("_case_db", None),
            ("_medical_service", None),
            ("CHAT_HISTORY_DB_PATH", self.db_path),
        ):
            patcher = patch.object(web_app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(web_app.app, follow_redirects=False)
        self.csrf = self.login("admin01", "AdminPassword123!")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def login(self, username: str, password: str) -> str:
        page = self.client.get("/login")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
        response = self.client.post(
            "/login",
            data={"username": username, "password": password, "csrf_token": token},
        )
        if response.status_code == 303:
            page = self.client.get("/")
            token = re.search(r'name="csrf-token" content="([^"]+)"', page.text).group(1)
        return token

    def test_password_hash_is_salted_and_verifiable(self) -> None:
        first = hash_password("A-secure-password-123")
        second = hash_password("A-secure-password-123")
        self.assertNotEqual(first, second)
        self.assertTrue(verify_password("A-secure-password-123", first))
        self.assertFalse(verify_password("incorrect-password", first))
        with self.assertRaises(ValueError):
            hash_password("short")

    def test_recovery_codes_are_six_alphanumeric_characters(self) -> None:
        self.assertRegex(generate_recovery_code(), r"^[A-Z0-9]{6}$")
        with self.assertRaises(ValueError):
            hash_recovery_code("short")

    def test_recovery_code_resets_password_once(self) -> None:
        new_hash = hash_password("NewViewerPassword123!")
        self.assertTrue(self.auth_db.reset_password_with_recovery("viewer01", "view01", new_hash))
        self.assertTrue(self.auth_db.authenticate("viewer01", "NewViewerPassword123!"))
        self.assertFalse(self.auth_db.reset_password_with_recovery("viewer01", "VIEW01", new_hash))

    def test_auth_database_does_not_seed_a_default_account(self) -> None:
        empty_db = WebAuthDatabase(self.root / "empty.db")
        self.assertEqual(empty_db.list_users(), [])

    def test_auth_pages_use_shared_dark_light_theme(self) -> None:
        anonymous = TestClient(web_app.app, follow_redirects=False)
        for path in ("/login", "/forgot-password"):
            response = anonymous.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn('/static/css/auth.css', response.text)
            self.assertIn("toggleAuthTheme()", response.text)
        admin_page = self.client.get("/admin/users")
        self.assertEqual(admin_page.status_code, 200)
        self.assertIn('/static/css/auth.css', admin_page.text)

    def test_auth_database_adds_recovery_column_to_existing_database(self) -> None:
        legacy_path = self.root / "legacy.db"
        conn = sqlite3.connect(legacy_path)
        try:
            conn.execute(
                "CREATE TABLE web_users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, "
                "role TEXT, is_active INTEGER, created_at TIMESTAMP)"
            )
        finally:
            conn.close()
        WebAuthDatabase(legacy_path)
        conn = sqlite3.connect(legacy_path)
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(web_users)")}
        finally:
            conn.close()
        self.assertIn("recovery_code_hash", columns)

    def test_login_is_required_for_patient_apis_and_output_files(self) -> None:
        anonymous = TestClient(web_app.app, follow_redirects=False)
        self.assertEqual(anonymous.get("/api/cases").status_code, 401)
        self.assertEqual(anonymous.get("/output/private.png").status_code, 303)
        self.assertEqual(anonymous.get("/").status_code, 303)
        blocked_upload_dir = self.root / "blocked-uploads"
        with patch.object(web_app, "WEB_UPLOADS_DIR", blocked_upload_dir):
            response = anonymous.post("/api/upload", files={"file": ("scan.png", b"pixels", "image/png")})
        self.assertEqual(response.status_code, 401)
        self.assertFalse(blocked_upload_dir.exists())

    def test_login_rejects_invalid_password_and_enforces_csrf(self) -> None:
        self.client.post("/logout", headers={"X-CSRF-Token": self.csrf})
        page = self.client.get("/login")
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
        bad = self.client.post("/login", data={"username": "admin01", "password": "wrong", "csrf_token": token})
        self.assertEqual(bad.status_code, 401)
        self.assertEqual(self.client.post("/api/conversations").status_code, 401)

    def test_forgot_password_page_is_public_and_resets_password(self) -> None:
        self.client.post("/logout", headers={"X-CSRF-Token": self.csrf})
        page = self.client.get("/forgot-password")
        self.assertEqual(page.status_code, 200)
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
        response = self.client.post(
            "/forgot-password",
            data={
                "csrf_token": token,
                "username": "viewer01",
                "recovery_code": "VIEW01",
                "password": "ResetViewerPassword123!",
                "confirm_password": "ResetViewerPassword123!",
            },
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login?reset=1")
        self.assertTrue(self.auth_db.authenticate("viewer01", "ResetViewerPassword123!"))

    def test_authenticated_mutations_require_csrf(self) -> None:
        self.assertEqual(self.client.post("/api/conversations").status_code, 403)
        response = self.client.post("/api/conversations", headers={"X-CSRF-Token": self.csrf})
        self.assertEqual(response.status_code, 200)

    def test_viewer_can_read_but_cannot_write(self) -> None:
        self.client.post("/logout", headers={"X-CSRF-Token": self.csrf})
        csrf = self.login("viewer01", "ViewerPassword123!")
        self.assertEqual(self.client.get("/api/cases").status_code, 200)
        page = self.client.get("/")
        self.assertIn('data-role="viewer"', page.text)
        self.assertIn("chế độ chỉ xem", page.text)
        denied = self.client.post("/api/conversations", headers={"X-CSRF-Token": csrf})
        self.assertEqual(denied.status_code, 403)
        settings = self.client.post(
            "/api/settings",
            data={"language": "en", "theme": "light"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(settings.status_code, 200)
        self.assertEqual(web_app.get_db().get_setting("language", "vi"), "vi")
        self.assertEqual(self.client.get("/admin/users").status_code, 403)

    def test_admin_can_create_users_and_last_admin_cannot_be_disabled(self) -> None:
        response = self.client.post(
            "/admin/users/create",
            data={
                "csrf_token": self.csrf,
                "username": "newviewer",
                "password": "NewViewerPassword123!",
                "role": "viewer",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        recovery_code = re.search(r'<strong data-recovery-code[^>]*>([A-Z0-9]{6})</strong>', response.text).group(1)
        self.assertIn("MÃ KHÔI PHỤC MỚI", response.text)
        users = self.auth_db.list_users()
        new_user = next(user for user in users if user.username == "newviewer")
        self.assertEqual(new_user.role, "viewer")
        self.assertTrue(self.auth_db.reset_password_with_recovery(
            "newviewer", recovery_code, hash_password("CreatedViewerPassword123!")
        ))

    def test_admin_can_reissue_recovery_code(self) -> None:
        response = self.client.post(
            f"/admin/users/{self.viewer_id}/recovery-code",
            data={"csrf_token": self.csrf},
        )
        self.assertEqual(response.status_code, 200)
        code = re.search(r'<strong data-recovery-code[^>]*>([A-Z0-9]{6})</strong>', response.text).group(1)
        self.assertTrue(self.auth_db.reset_password_with_recovery(
            "viewer01", code, hash_password("RotatedViewerPassword123!")
        ))

        response = self.client.post(
            f"/admin/users/{self.admin_id}/update",
            data={"csrf_token": self.csrf, "role": "viewer", "is_active": "0"},
        )
        self.assertEqual(response.status_code, 303)
        self.assertIn("admin cuối cùng", unquote(response.headers["location"]))
        self.assertTrue(self.auth_db.get_user(self.admin_id).is_active)

    def test_admin_can_issue_code_for_account_created_without_recovery_code(self) -> None:
        user_id = self.auth_db.create_user("sqladmin", hash_password("SqlAdminPassword123!"), "admin")
        response = self.client.post(
            f"/admin/users/{user_id}/recovery-code",
            data={"csrf_token": self.csrf},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        self.assertRegex(
            re.search(r'<strong data-recovery-code[^>]*>([^<]+)</strong>', response.text).group(1),
            r"^[A-Z0-9]{6}$",
        )

    def test_disabled_account_loses_access_on_next_request(self) -> None:
        self.client.post("/logout", headers={"X-CSRF-Token": self.csrf})
        self.login("viewer01", "ViewerPassword123!")
        self.auth_db.update_user(self.viewer_id, "viewer", False)
        self.assertEqual(self.client.get("/api/cases").status_code, 401)

    def test_login_lock_expires_after_cooldown(self) -> None:
        now = 100_000.0
        for _ in range(5):
            self.auth_db.record_login_failure("admin01", "127.0.0.1", now=now)
        self.assertTrue(self.auth_db.login_locked("admin01", "127.0.0.1", now=now + 1))
        self.assertFalse(self.auth_db.login_locked("admin01", "127.0.0.1", now=now + 901))

    def test_output_files_are_private_and_path_traversal_is_rejected(self) -> None:
        output_dir = self.root / "output"
        output_dir.mkdir()
        (output_dir / "report.txt").write_text("private report", encoding="utf-8")
        with patch.object(web_app, "OUTPUT_DIR", output_dir):
            response = self.client.get("/output/report.txt")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "private report")
            self.assertEqual(self.client.get("/output/../onco.db").status_code, 404)

    def test_upload_filename_cannot_escape_upload_directory(self) -> None:
        uploads_dir = self.root / "uploads"
        with (
            patch.object(web_app, "WEB_UPLOADS_DIR", uploads_dir),
            patch.object(web_app, "infer_medical_upload_context", return_value=(None, None)),
        ):
            response = self.client.post(
                "/api/upload",
                files={"file": ("../../escape.png", b"test image bytes", "image/png")},
                headers={"X-CSRF-Token": self.csrf},
            )
        self.assertEqual(response.status_code, 200)
        stored_path = Path(response.json()["stored_path"])
        self.assertEqual(stored_path.parent, uploads_dir)
        self.assertTrue(stored_path.is_file())


if __name__ == "__main__":
    unittest.main()
