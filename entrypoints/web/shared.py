from __future__ import annotations

import os
import secrets
import sys
import threading
import time
from pathlib import Path

# Add project root to sys.path when running from entrypoints/
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import Depends, HTTPException, Request
from starlette.datastructures import UploadFile

from app.chat_ui.models import ChatMessage  # noqa: F401 — re-exported
from app.chat_ui.paths import CHAT_HISTORY_DB_PATH, OUTPUT_DIR, PROJECT_ROOT
from app.chat_ui.storage import ChatDatabase
from app.email_service import send_password_recovery_email
from app.web_auth import (
    WebAuthDatabase,
    WebUser,
    generate_recovery_code,  # noqa: F401 — re-exported
    hash_password,  # noqa: F401 — re-exported
    hash_recovery_code,  # noqa: F401 — re-exported
)
from medical.cancer_catalog import get_cancer_target
from medical.case_payloads import (
    build_case_export_payload,  # noqa: F401 — re-exported
    case_detections,
    case_localization_available,
)
from medical.chat_service import MedicalChatService
from medical.compliance import (
    MEDICAL_DISCLAIMER,  # noqa: F401 — re-exported
    get_audit_trail,
)
from medical.reporting import export_case_pdf  # noqa: F401 — re-exported
from medical.storage import MedicalCaseDatabase
from medical.system_status import get_medical_system_status
from utils.logger import get_logger

logger = get_logger(__name__)
audit = get_audit_trail()

WEB_UPLOADS_DIR = OUTPUT_DIR / "web_uploads"
WEB_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_FORM_BYTES = 1024 * 1024


def _max_concurrent_analyses() -> int:
    try:
        return max(1, int(os.environ.get("ONCOVISION_MAX_CONCURRENT_ANALYSIS", "1")))
    except ValueError:
        return 1


ANALYSIS_SEMAPHORE = threading.BoundedSemaphore(_max_concurrent_analyses())

TEMPLATES_DIR = PROJECT_ROOT / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

IS_PRODUCTION = os.environ.get("ONCOVISION_ENV", "").lower() in {"prod", "production"}
COOKIE_SECURE = os.environ.get("ONCOVISION_COOKIE_SECURE", "0") == "1"
if (env_secret := os.environ.get("ONCOVISION_SESSION_SECRET")):
    SESSION_SECRET = env_secret
else:
    if IS_PRODUCTION:
        raise RuntimeError("Thiếu ONCOVISION_SESSION_SECRET ở môi trường production.")
    SESSION_SECRET = secrets.token_urlsafe(32)
if len(SESSION_SECRET) < 32:
    raise RuntimeError("ONCOVISION_SESSION_SECRET phải dài ít nhất 32 ký tự.")
if IS_PRODUCTION and not COOKIE_SECURE:
    raise RuntimeError("Production yêu cầu ONCOVISION_COOKIE_SECURE=1 để bảo vệ cookie phiên.")

# ---------------------------------------------------------------------------
# Database singletons
# ---------------------------------------------------------------------------

_db: ChatDatabase | None = None
_medical_service: MedicalChatService | None = None
_case_db: MedicalCaseDatabase | None = None
_auth_db: WebAuthDatabase | None = None


def get_db() -> ChatDatabase:
    global _db
    if _db is None:
        _db = ChatDatabase(str(CHAT_HISTORY_DB_PATH))
    return _db


def get_case_db() -> MedicalCaseDatabase:
    global _case_db
    if _case_db is None:
        _case_db = MedicalCaseDatabase(CHAT_HISTORY_DB_PATH)
    return _case_db


def get_auth_db() -> WebAuthDatabase:
    global _auth_db
    if _auth_db is None:
        _auth_db = WebAuthDatabase(CHAT_HISTORY_DB_PATH)
    return _auth_db


def get_medical_service() -> MedicalChatService:
    global _medical_service
    if _medical_service is None:
        _medical_service = MedicalChatService()
    return _medical_service


# ---------------------------------------------------------------------------
# CSRF helpers
# ---------------------------------------------------------------------------

def _request_csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


# ---------------------------------------------------------------------------
# Auth & role helpers
# ---------------------------------------------------------------------------

_ROLE_HIERARCHY = {"admin": 3, "clinician": 2, "viewer": 1}


async def require_authenticated(request: Request) -> WebUser | None:
    if request.url.path in {"/login", "/forgot-password", "/forgot-password/request", "/change-credentials", "/favicon.svg", "/favicon.ico", "/healthz", "/health", "/v1/models"}:
        return None

    user_id = request.session.get("user_id")
    user = get_auth_db().get_user(user_id) if isinstance(user_id, int) else None
    if user is None or not user.is_active:
        request.session.clear()
        if request.url.path.startswith("/api/"):
            raise HTTPException(status_code=401, detail="Vui lòng đăng nhập.", headers={"X-Login-Required": "true"})
        raise HTTPException(status_code=303, headers={"Location": "/login"})

    # Ép đổi tên đăng nhập/mật khẩu nếu là lần đầu (admin/admin)
    if user.must_change_credentials and request.url.path != "/change-credentials":
        raise HTTPException(status_code=303, headers={"Location": "/change-credentials"})

    request.state.current_user = user

    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        provided_token = request.headers.get("x-csrf-token")
        _csrf_form_paths = {
            "/logout", "/admin/users/create",
        }
        if not provided_token and (
            request.url.path in _csrf_form_paths
            or (
                request.url.path.startswith("/admin/users/")
                and request.url.path.endswith("/update")
            )
        ):
            form = await request.form()
            provided_token = str(form.get("csrf_token", ""))
            await form.close()
        expected_token = request.session.get("csrf_token", "")
        if not expected_token or not secrets.compare_digest(str(provided_token or ""), str(expected_token)):
            raise HTTPException(status_code=403, detail="CSRF token không hợp lệ hoặc đã hết hạn.")

    allowed_viewer_write = request.url.path == "/logout"
    if user.role == "viewer" and request.method not in {"GET", "HEAD", "OPTIONS"} and not allowed_viewer_write:
        raise HTTPException(status_code=403, detail="Tài khoản chỉ có quyền xem.")
    return user


def require_role(*roles: str):
    def dependency(request: Request) -> WebUser:
        user = getattr(request.state, "current_user", None)
        if user is None:
            raise HTTPException(status_code=403, detail="Bạn không có quyền thực hiện thao tác này.")
        user_level = _ROLE_HIERARCHY.get(user.role, 0)
        required_level = min(_ROLE_HIERARCHY.get(r, 0) for r in roles)
        if user_level < required_level:
            raise HTTPException(status_code=403, detail="Bạn không có quyền thực hiện thao tác này.")
        return user

    return dependency


ADMIN_REQUIRED = Depends(require_role("admin"))
STAFF_REQUIRED = Depends(require_role("admin", "clinician"))


# ---------------------------------------------------------------------------
# Path & file helpers
# ---------------------------------------------------------------------------

def _safe_path(base: Path, path: str) -> Path | None:
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        return None
    return resolved if resolved.exists() else None


# ---------------------------------------------------------------------------
# Medical status cache
# ---------------------------------------------------------------------------

_MEDICAL_STATUS_CACHE_TTL_SECONDS = 60.0
_medical_status_cache: tuple[float, object] | None = None


def _cached_medical_status():
    global _medical_status_cache
    if _medical_status_cache is not None and time.monotonic() - _medical_status_cache[0] < _MEDICAL_STATUS_CACHE_TTL_SECONDS:
        return _medical_status_cache[1]
    status = get_medical_system_status()
    _medical_status_cache = (time.monotonic(), status)
    return status


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

async def _save_upload(file: UploadFile, dest: Path) -> int:
    import aiofiles
    size = 0
    try:
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File quá lớn (tối đa {MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
                    )
                await out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return size


def _has_allowed_magic(path: Path, filename: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(600)
    except OSError:
        return False
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return head[:2] == b"\x1f\x8b"
    if lower.endswith(".nii"):
        return head[344:348] in (b"n+1\x00", b"ni1\x00")
    if lower.endswith(".dcm") or head[128:132] == b"DICM":
        return True
    img_magic = (b"\xff\xd8\xff", b"\x89PNG", b"BM", b"II*\x00", b"MM\x00*", b"RIFF", b"\x49\x49\x2b\x00")
    return head.startswith(img_magic) or (lower.endswith((".mp4", ".mov")) and head[4:8] == b"ftyp")


# ---------------------------------------------------------------------------
# Target validation helpers
# ---------------------------------------------------------------------------

def _validate_target_input(target_key: str, modality: str):
    """Kiểm tra input nhóm bệnh/modality — lỗi request thì 400."""
    from medical.validator import _canonical_modality

    target = get_cancer_target(target_key)
    if target is None:
        raise HTTPException(status_code=400, detail="Nhóm bệnh không hợp lệ.")

    if modality:
        modality_canonical = _canonical_modality(modality)
        if modality_canonical is None:
            raise HTTPException(status_code=400, detail=f"Modality không hợp lệ: {modality}")

        target_modalities_canonical = set()
        for m in target.modalities:
            canon = _canonical_modality(m)
            if canon:
                target_modalities_canonical.add(canon)

        if modality_canonical not in target_modalities_canonical:
            raise HTTPException(status_code=400, detail="Modality không phù hợp với nhóm bệnh đã chọn.")

    return target


def _require_ready_capability(target) -> None:
    """Kiểm tra model đã sẵn sàng — thiếu model thì 503 (trừ khi demo_mode bật)."""
    from medical.pipeline import build_default_medical_analyzer_config
    config = build_default_medical_analyzer_config()
    if config.demo_mode:
        return

    capabilities = get_medical_system_status().target_capabilities.get(target.key, {})
    if not capabilities.get("runtime_ready", False):
        mod_caps = capabilities.get("modality_capabilities", {})
        ready_modalities = [m for m, info in mod_caps.items() if info.get("runtime_ready", False)]
        if ready_modalities:
            detail = (
                f"Chưa có model suy luận cho {target.label}. "
                f"Modality có sẵn: {', '.join(ready_modalities)}. "
                f"Hãy chọn modality khác hoặc bật demo_mode trong config để chạy thử."
            )
        else:
            detail = (
                f"Chưa có model suy luận cho {target.label}. "
                f"Hãy bổ sung model vào models/pretrained/cancers/{target.key}/ hoặc bật demo_mode trong config để chạy thử."
            )
        raise HTTPException(status_code=503, detail=detail)


# ---------------------------------------------------------------------------
# Case summary helper
# ---------------------------------------------------------------------------

def _case_summary(record) -> dict:
    return {
        "case_id": record.case_id,
        "patient_code": record.patient_code,
        "risk_level": record.risk_level,
        "suspected_malignant": record.suspected_malignant,
        "clinical_assessment_available": record.review_status in {"submitted", "approved"},
        "image_path": record.image_path,
        "processed_image_path": record.processed_image_path,
        "recommendation": record.recommendation,
        "created_at": record.created_at,
        "detections": case_detections(record.metadata),
        "localization_available": case_localization_available(record.metadata),
        "average_confidence": record.metadata.get("average_confidence", 0),
        "model_name": record.metadata.get("model_name", "-"),
        "quality_warnings": record.metadata.get("quality_warnings", []),
        "assigned_to": record.assigned_to,
        "review_status": record.review_status,
        "reviewed_by": record.reviewed_by,
        "reviewed_at": record.reviewed_at,
        "public_code": record.public_code,
        "patient_username": record.patient_username,
        "modality": record.metadata.get("modality"),
        "body_region": record.metadata.get("body_region"),
        "segmentation_mask_path": record.metadata.get("segmentation_mask_path"),
        "volume_mask_path": record.metadata.get("volume_mask_path"),
        "analyzed_frame_count": record.metadata.get("analyzed_frame_count", 1),
        "representative_frame_index": record.metadata.get("representative_frame_index"),
    }


def _can_read_case(record, user) -> bool:
    if user.role == "admin":
        return True
    if user.role == "clinician":
        return record.assigned_to == user.username
    return (
        record.review_status == "approved"
        and record.patient_username is not None
        and record.patient_username.casefold() == user.username.casefold()
    )


# ---------------------------------------------------------------------------
# Recovery email helper
# ---------------------------------------------------------------------------

def _deliver_recovery_email(auth_db: WebAuthDatabase, username: str, email: str, code: str, code_hash: str) -> None:
    try:
        send_password_recovery_email(email, username, code)
    except Exception:
        auth_db.clear_recovery_code(username, code_hash)
        logger.exception("Không gửi được email khôi phục mật khẩu.")
