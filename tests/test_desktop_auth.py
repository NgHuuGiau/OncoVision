from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.chat_ui.desktop_auth import (
    authenticate_desktop_user,
    can_analyze,
    create_bootstrap_admin,
    needs_bootstrap,
)
from app.web_auth import WebAuthDatabase, hash_password


class DesktopAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.auth_db = WebAuthDatabase(Path(self._tmp.name) / "onco.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_needs_bootstrap_until_first_user(self) -> None:
        self.assertTrue(needs_bootstrap(self.auth_db))
        self.auth_db.create_user("admin", hash_password("ValidAdminPass123!"), "admin")
        self.assertFalse(needs_bootstrap(self.auth_db))

    def test_can_analyze_matches_web_roles(self) -> None:
        self.assertTrue(can_analyze("admin"))
        self.assertTrue(can_analyze("clinician"))
        self.assertFalse(can_analyze("viewer"))
        self.assertFalse(can_analyze("unknown"))

    def test_create_bootstrap_admin(self) -> None:
        user = create_bootstrap_admin(self.auth_db, "root", "ValidAdminPass123!")
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.role, "admin")
        self.assertIsNone(create_bootstrap_admin(self.auth_db, "root", "ValidAdminPass123!"))
        self.assertIsNone(create_bootstrap_admin(self.auth_db, "x", "short"))

    def test_authenticate_success_and_failure(self) -> None:
        self.auth_db.create_user("staff", hash_password("ValidStaffPass123!"), "clinician")
        user = authenticate_desktop_user(self.auth_db, "staff", "ValidStaffPass123!")
        self.assertIsNotNone(user)
        assert user is not None
        self.assertEqual(user.username, "staff")
        self.assertIsNone(authenticate_desktop_user(self.auth_db, "staff", "wrong-password-123"))
        self.assertIsNone(authenticate_desktop_user(self.auth_db, "", "ValidStaffPass123!"))
        self.assertIsNone(authenticate_desktop_user(self.auth_db, "ghost", "ValidStaffPass123!"))

    def test_authenticate_locked_account_returns_none(self) -> None:
        self.auth_db.create_user("staff", hash_password("ValidStaffPass123!"), "clinician")
        for _ in range(6):
            authenticate_desktop_user(self.auth_db, "staff", "wrong-password-123")
        self.assertIsNone(authenticate_desktop_user(self.auth_db, "staff", "ValidStaffPass123!"))


if __name__ == "__main__":
    unittest.main()
