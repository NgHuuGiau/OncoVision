from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import re
import secrets
import sqlite3
import string
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from utils.sqlite_utils import create_sqlite_connection

PASSWORD_ITERATIONS = 600_000
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 1024
LOGIN_FAILURE_LIMIT = 5
LOGIN_LOCK_SECONDS = 15 * 60
RECOVERY_CODE_TTL_SECONDS = 10 * 60
RECOVERY_REQUEST_COOLDOWN_SECONDS = 60
ROLES = {"admin", "clinician", "viewer"}
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _encode_bytes(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_bytes(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True)
class WebUser:
    id: int
    username: str
    email: str | None
    role: str
    is_active: bool
    created_at: str


def hash_password(password: str) -> str:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Mật khẩu phải có ít nhất {PASSWORD_MIN_LENGTH} ký tự.")
    if len(password) > PASSWORD_MAX_LENGTH:
        raise ValueError(f"Mật khẩu không được dài quá {PASSWORD_MAX_LENGTH} ký tự.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_encode_bytes(salt)}${_encode_bytes(digest)}"


def generate_recovery_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    code = [secrets.choice(string.ascii_uppercase), secrets.choice(string.digits)]
    code.extend(secrets.choice(alphabet) for _ in range(4))
    secrets.SystemRandom().shuffle(code)
    return "".join(code)


def hash_recovery_code(code: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9]{6}", code):
        raise ValueError("Mã khôi phục phải có đúng 6 chữ cái hoặc chữ số.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", code.upper().encode("ascii"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_encode_bytes(salt)}${_encode_bytes(digest)}"


def normalize_email(email: str | None) -> str | None:
    value = (email or "").strip().lower()
    if not value:
        return None
    if len(value) > 254 or not _EMAIL_RE.fullmatch(value):
        raise ValueError("Địa chỉ email không hợp lệ.")
    return value


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, raw_iterations, raw_salt, raw_digest = encoded.split("$", 3)
        iterations = int(raw_iterations)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= iterations <= 2_000_000:
            return False
        salt, expected = _decode_bytes(raw_salt), _decode_bytes(raw_digest)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except (AttributeError, ValueError, TypeError):
        return False


class WebAuthDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = create_sqlite_connection(self.db_path, enable_foreign_keys=True)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'clinician', 'viewer')),
                    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_login_attempts (
                    username TEXT NOT NULL COLLATE NOCASE,
                    remote_addr TEXT NOT NULL,
                    failures INTEGER NOT NULL,
                    window_started REAL NOT NULL,
                    locked_until REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (username, remote_addr)
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS web_recovery_requests "
                "(request_key TEXT PRIMARY KEY, requested_at REAL NOT NULL)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(web_users)")}
            if "email" not in columns:
                conn.execute("ALTER TABLE web_users ADD COLUMN email TEXT")
            if "recovery_code_hash" not in columns:
                conn.execute("ALTER TABLE web_users ADD COLUMN recovery_code_hash TEXT")
            if "recovery_code_expires_at" not in columns:
                conn.execute("ALTER TABLE web_users ADD COLUMN recovery_code_expires_at REAL")
                # Old admin-issued codes had no expiry and are invalidated by the email-based flow.
                conn.execute("UPDATE web_users SET recovery_code_hash = NULL")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_web_users_email "
                "ON web_users(email COLLATE NOCASE) WHERE email IS NOT NULL AND email <> ''"
            )

    @staticmethod
    def _user(row) -> WebUser | None:
        return WebUser(int(row[0]), row[1], row[2], row[3], bool(row[4]), row[5]) if row else None

    def get_user(self, user_id: int) -> WebUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, email, role, is_active, created_at FROM web_users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return self._user(row)

    def list_users(self) -> list[WebUser]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, username, email, role, is_active, created_at "
                "FROM web_users ORDER BY username COLLATE NOCASE"
            ).fetchall()
        return [user for row in rows if (user := self._user(row)) is not None]

    def create_user(
        self,
        username: str,
        password_hash: str,
        role: str,
        recovery_code_hash: str | None = None,
        email: str | None = None,
    ) -> int:
        username = username.strip()
        email = normalize_email(email)
        if not _USERNAME_RE.fullmatch(username):
            raise ValueError("Tên đăng nhập phải dài 3–32 ký tự, chỉ gồm chữ, số, dấu chấm, gạch ngang hoặc gạch dưới.")
        if role not in ROLES:
            raise ValueError("Vai trò không hợp lệ.")
        if not password_hash.startswith("pbkdf2_sha256$"):
            raise ValueError("Hash mật khẩu không hợp lệ; tạo bằng `python -m app.web_auth hash-password`.")
        if recovery_code_hash is not None and not recovery_code_hash.startswith("pbkdf2_sha256$"):
            raise ValueError("Hash mã khôi phục không hợp lệ.")
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO web_users "
                    "(username, password_hash, email, role, recovery_code_hash, recovery_code_expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        username,
                        password_hash,
                        email,
                        role,
                        recovery_code_hash,
                        time.time() + RECOVERY_CODE_TTL_SECONDS if recovery_code_hash else None,
                    ),
                )
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            if "email" in str(exc).lower():
                raise ValueError("Email đã được dùng cho tài khoản khác.") from exc
            raise ValueError("Tên đăng nhập đã tồn tại.") from exc

    def allow_recovery_request(
        self,
        remote_addr: str,
        username: str,
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else now
        account_key = hashlib.sha256(username.strip().casefold().encode()).hexdigest()
        keys = (f"ip:{remote_addr}", f"account:{account_key}")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM web_recovery_requests WHERE requested_at < ?", (now - 86_400,))
            requested = [
                conn.execute(
                    "SELECT requested_at FROM web_recovery_requests WHERE request_key = ?", (key,)
                ).fetchone()
                for key in keys
            ]
            if any(row and now - row[0] < RECOVERY_REQUEST_COOLDOWN_SECONDS for row in requested):
                return False
            conn.executemany(
                "INSERT OR REPLACE INTO web_recovery_requests (request_key, requested_at) VALUES (?, ?)",
                ((key, now) for key in keys),
            )
        return True

    def store_recovery_code(
        self,
        username: str,
        recovery_code_hash: str,
        expires_at: float,
    ) -> tuple[str, str] | None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            user = conn.execute(
                "SELECT username, email FROM web_users "
                "WHERE username = ? COLLATE NOCASE AND is_active = 1 AND email IS NOT NULL AND email <> ''",
                (username.strip(),),
            ).fetchone()
            if user is None:
                return None
            conn.execute(
                "UPDATE web_users SET recovery_code_hash = ?, recovery_code_expires_at = ? "
                "WHERE username = ? COLLATE NOCASE AND is_active = 1",
                (recovery_code_hash, expires_at, user[0]),
            )
            return user[0], user[1]

    def clear_recovery_code(self, username: str, recovery_code_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE web_users SET recovery_code_hash = NULL, recovery_code_expires_at = NULL "
                "WHERE username = ? COLLATE NOCASE AND recovery_code_hash = ?",
                (username.strip(), recovery_code_hash),
            )

    def reset_password_with_recovery(
        self,
        username: str,
        recovery_code: str,
        new_password_hash: str,
        now: float | None = None,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, is_active, recovery_code_hash, recovery_code_expires_at "
                "FROM web_users WHERE username = ? COLLATE NOCASE",
                (username.strip(),),
            ).fetchone()
        now = time.time() if now is None else now
        code_has_valid_shape = bool(re.fullmatch(r"[A-Za-z0-9]{6}", recovery_code))
        encoded_code = row[2] if row and row[2] else _DUMMY_PASSWORD_HASH
        valid_code = verify_password(recovery_code.upper() if code_has_valid_shape else "INVALID", encoded_code)
        if (
            not row
            or not row[1]
            or not row[2]
            or row[3] is None
            or row[3] < now
            or not code_has_valid_shape
            or not valid_code
        ):
            return False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE web_users SET password_hash = ?, recovery_code_hash = NULL, "
                "recovery_code_expires_at = NULL WHERE id = ? AND is_active = 1 "
                "AND recovery_code_hash = ? AND recovery_code_expires_at >= ?",
                (new_password_hash, row[0], row[2], now),
            )
            return cursor.rowcount == 1

    def update_user(self, user_id: int, role: str, is_active: bool, email: str | None = None) -> None:
        if role not in ROLES:
            raise ValueError("Vai trò không hợp lệ.")
        normalized_email = normalize_email(email) if email is not None else None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT role, is_active FROM web_users WHERE id = ?", (user_id,)).fetchone()
            if current is None:
                raise ValueError("Không tìm thấy tài khoản.")
            remains_admin = role == "admin" and is_active
            if current[0] == "admin" and current[1] and not remains_admin:
                admins = conn.execute(
                    "SELECT COUNT(*) FROM web_users WHERE role = 'admin' AND is_active = 1"
                ).fetchone()[0]
                if admins <= 1:
                    raise ValueError("Không thể khóa hoặc hạ quyền admin cuối cùng.")
            try:
                if email is None:
                    conn.execute(
                        "UPDATE web_users SET role = ?, is_active = ? WHERE id = ?",
                        (role, int(is_active), user_id),
                    )
                else:
                    conn.execute(
                        "UPDATE web_users SET role = ?, is_active = ?, email = ? WHERE id = ?",
                        (role, int(is_active), normalized_email, user_id),
                    )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Email đã được dùng cho tài khoản khác.") from exc

    def authenticate(self, username: str, password: str) -> WebUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, email, role, is_active, created_at "
                "FROM web_users WHERE username = ? COLLATE NOCASE",
                (username.strip(),),
            ).fetchone()
        password_ok = verify_password(password, row[2] if row else _DUMMY_PASSWORD_HASH)
        if row is None or not bool(row[5]) or not password_ok:
            return None
        return WebUser(int(row[0]), row[1], row[3], row[4], True, row[6])

    def login_locked(self, username: str, remote_addr: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._connect() as conn:
            row = conn.execute(
                "SELECT locked_until FROM web_login_attempts WHERE username = ? AND remote_addr = ?",
                (username.strip(), remote_addr),
            ).fetchone()
        return bool(row and row[0] > now)

    def record_login_failure(self, username: str, remote_addr: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT failures, window_started FROM web_login_attempts WHERE username = ? AND remote_addr = ?",
                (username.strip(), remote_addr),
            ).fetchone()
            if row is None or now - row[1] > LOGIN_LOCK_SECONDS:
                failures, started = 1, now
            else:
                failures, started = int(row[0]) + 1, row[1]
            locked_until = now + LOGIN_LOCK_SECONDS if failures >= LOGIN_FAILURE_LIMIT else 0
            conn.execute(
                "INSERT OR REPLACE INTO web_login_attempts "
                "(username, remote_addr, failures, window_started, locked_until) VALUES (?, ?, ?, ?, ?)",
                (username.strip(), remote_addr, failures, started, locked_until),
            )

    def clear_login_failures(self, username: str, remote_addr: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM web_login_attempts WHERE username = ? AND remote_addr = ?",
                (username.strip(), remote_addr),
            )


_DUMMY_PASSWORD_HASH = hash_password("OncoVision-invalid-account-placeholder")


def main() -> int:
    if sys.argv[1:] != ["hash-password"]:
        print("Dùng: python -m app.web_auth hash-password", file=sys.stderr)
        return 2
    password = getpass.getpass("Nhập mật khẩu (ít nhất 12 ký tự): ")
    confirmation = getpass.getpass("Nhập lại mật khẩu: ")
    if password != confirmation:
        print("Mật khẩu nhập lại không khớp.", file=sys.stderr)
        return 1
    try:
        print(hash_password(password))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
