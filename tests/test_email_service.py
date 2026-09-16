from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.email_service import send_password_recovery_email


class EmailServiceTests(unittest.TestCase):
    @patch.dict(os.environ, {
        "ONCOVISION_SMTP_HOST": "smtp.gmail.com",
        "ONCOVISION_SMTP_PORT": "587",
        "ONCOVISION_SMTP_USERNAME": "sender@example.com",
        "ONCOVISION_SMTP_PASSWORD": "test-app-password",
    }, clear=True)
    @patch("app.email_service.smtplib.SMTP")
    def test_sends_plaintext_code_over_starttls(self, smtp_class) -> None:
        smtp = smtp_class.return_value.__enter__.return_value
        send_password_recovery_email("recipient@example.com", "clinician01", "A1B2C3")
        smtp.starttls.assert_called_once()
        smtp.login.assert_called_once_with("sender@example.com", "test-app-password")
        message = smtp.send_message.call_args.args[0]
        self.assertEqual(message["To"], "recipient@example.com")
        self.assertIn("A1B2C3", message.get_content())
        self.assertIn("10 phút", message.get_content())

    @patch.dict(os.environ, {}, clear=True)
    @patch("app.email_service.smtplib.SMTP")
    def test_missing_smtp_credentials_fail_closed(self, smtp_class) -> None:
        with self.assertRaisesRegex(RuntimeError, "Chưa cấu hình"):
            send_password_recovery_email("recipient@example.com", "clinician01", "A1B2C3")
        smtp_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
