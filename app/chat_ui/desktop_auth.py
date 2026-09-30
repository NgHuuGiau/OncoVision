from __future__ import annotations

from app.web_auth import WebAuthDatabase, WebUser, hash_password

DESKTOP_REMOTE_ADDR = "desktop"

ANALYZE_ROLES = frozenset({"admin", "clinician"})


def needs_bootstrap(auth_db: WebAuthDatabase) -> bool:
    """Chưa có tài khoản nào thì desktop phải tạo admin đầu tiên."""
    return len(auth_db.list_users()) == 0


def can_analyze(role: str) -> bool:
    """Viewer chỉ xem, không được chạy phân tích ảnh y khoa."""
    return role in ANALYZE_ROLES


def create_bootstrap_admin(auth_db: WebAuthDatabase, username: str, password: str) -> WebUser | None:
    """Tạo admin đầu tiên khi DB trống. Trả về user hoặc None nếu lỗi."""
    try:
        user_id = auth_db.create_user(username.strip(), hash_password(password), "admin")
    except ValueError:
        return None
    return auth_db.get_user(user_id)


def authenticate_desktop_user(
    auth_db: WebAuthDatabase, username: str, password: str, *, remote_addr: str = DESKTOP_REMOTE_ADDR
) -> WebUser | None:
    """Đăng nhập desktop, tái dùng chính sách khóa của web. None = sai hoặc bị khóa."""
    username = (username or "").strip()
    if not username or not password:
        return None
    if auth_db.login_locked(username, remote_addr):
        return None
    user = auth_db.authenticate(username, password)
    if user is None:
        auth_db.record_login_failure(username, remote_addr)
        return None
    auth_db.clear_login_failures(username, remote_addr)
    return user
