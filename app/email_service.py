from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


def send_password_recovery_email(recipient: str, username: str, code: str) -> None:
    host = os.environ.get("ONCOVISION_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("ONCOVISION_SMTP_PORT", "587"))
    sender = os.environ.get("ONCOVISION_SMTP_USERNAME", "").strip()
    password = os.environ.get("ONCOVISION_SMTP_PASSWORD", "")
    if not sender or not password:
        raise RuntimeError("Chưa cấu hình thông tin SMTP gửi thư.")

    message = EmailMessage()
    message["Subject"] = "Mã khôi phục mật khẩu OncoVision"
    message["From"] = os.environ.get("ONCOVISION_SMTP_FROM", sender).strip()
    message["To"] = recipient
    message.set_content(
        f"Xin chào {username},\n\n"
        f"Mã khôi phục mật khẩu OncoVision của bạn là: {code}\n\n"
        "Mã chỉ dùng một lần và hết hạn sau 10 phút. Nếu bạn không yêu cầu khôi phục, "
        "hãy bỏ qua email này và liên hệ quản trị viên.\n"
    )

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.ehlo()
        server.login(sender, password)
        server.send_message(message)
