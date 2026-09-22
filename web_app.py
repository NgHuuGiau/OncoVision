from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, cast
from urllib.parse import quote

import aiofiles
from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from app.chat_ui.models import ChatMessage
from app.chat_ui.paths import CHAT_HISTORY_DB_PATH, OUTPUT_DIR, PROJECT_ROOT
from app.chat_ui.storage import ChatDatabase
from app.email_service import send_password_recovery_email
from app.web_auth import (
    RECOVERY_CODE_TTL_SECONDS,
    WebAuthDatabase,
    WebUser,
    generate_recovery_code,
    hash_password,
    hash_recovery_code,
)
from medical.cancer_catalog import COMMON_CANCER_TARGETS, get_cancer_target
from medical.case_payloads import build_case_export_payload
from medical.chat_service import MedicalChatResponse, MedicalChatService
from medical.compliance import MEDICAL_DISCLAIMER
from medical.dataset import infer_medical_upload_context
from medical.reporting import export_case_pdf
from medical.storage import MedicalCaseDatabase
from medical.system_status import get_medical_system_status
from utils.logger import get_logger

logger = get_logger(__name__)

WEB_UPLOADS_DIR = OUTPUT_DIR / "web_uploads"
WEB_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_FORM_BYTES = 1024 * 1024
ANALYSIS_SEMAPHORE = threading.BoundedSemaphore(1)

TEMPLATES_DIR = PROJECT_ROOT / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("OncoVision Web Chat khoi dong...")
    get_auth_db()
    try:
        service = get_medical_service()
        service.check_ready()
        logger.info("Medical service san sang.")
    except Exception as exc:
        logger.warning("Medical service chua san sang: %s", exc)
    yield


def _request_csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


async def require_authenticated(request: Request) -> WebUser | None:
    if request.url.path in {"/login", "/forgot-password", "/forgot-password/request", "/favicon.svg"}:
        return None

    user_id = request.session.get("user_id")
    user = get_auth_db().get_user(user_id) if isinstance(user_id, int) else None
    if user is None or not user.is_active:
        request.session.clear()
        if request.url.path.startswith("/api/"):
            raise HTTPException(status_code=401, detail="Vui lòng đăng nhập.", headers={"X-Login-Required": "true"})
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    request.state.current_user = user

    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        provided_token = request.headers.get("x-csrf-token")
        if not provided_token and (
            request.url.path in {"/logout", "/admin/users/create"}
            or (
                request.url.path.startswith("/admin/users/")
                and request.url.path.endswith("/update")
            )
        ):
            form = await request.form()
            provided_token = str(form.get("csrf_token", ""))
        expected_token = request.session.get("csrf_token", "")
        if not expected_token or not secrets.compare_digest(str(provided_token or ""), str(expected_token)):
            raise HTTPException(status_code=403, detail="CSRF token không hợp lệ hoặc đã hết hạn.")

    allowed_viewer_write = request.url.path in {"/logout", "/api/settings"}
    if user.role == "viewer" and request.method not in {"GET", "HEAD", "OPTIONS"} and not allowed_viewer_write:
        raise HTTPException(status_code=403, detail="Tài khoản chỉ có quyền xem.")
    return user


def require_role(*roles: str):
    def dependency(request: Request) -> WebUser:
        user = getattr(request.state, "current_user", None)
        if user is None or user.role not in roles:
            raise HTTPException(status_code=403, detail="Bạn không có quyền thực hiện thao tác này.")
        return user

    return dependency


ADMIN_REQUIRED = Depends(require_role("admin"))
STAFF_REQUIRED = Depends(require_role("admin", "clinician"))
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


app = FastAPI(
    title="OncoVision Web Chat",
    lifespan=lifespan,
    dependencies=[Depends(require_authenticated)],
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="oncovision_session",
    max_age=8 * 60 * 60,
    same_site="lax",
    https_only=COOKIE_SECURE,
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    raw_length = request.headers.get("content-length")
    if not raw_length and request.headers.get("transfer-encoding"):
        return JSONResponse({"detail": "Yêu cầu truyền theo luồng không được hỗ trợ."}, status_code=411)
    if raw_length:
        try:
            request_length = int(raw_length)
        except ValueError:
            return JSONResponse({"detail": "Content-Length không hợp lệ."}, status_code=400)
        max_length = MAX_UPLOAD_BYTES + MAX_FORM_BYTES if request.url.path == "/api/upload" else MAX_FORM_BYTES
        if request_length > max_length:
            return JSONResponse({"detail": "Yêu cầu vượt quá dung lượng cho phép."}, status_code=413)
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if request.url.path.startswith(("/api/", "/output/")):
        response.headers.setdefault("Cache-Control", "no-store")
    if IS_PRODUCTION:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

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


def _safe_path(base: Path, path: str) -> Path | None:
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        return None
    return resolved if resolved.exists() else None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cancer_targets = [
        {"key": t.key, "label": t.label, "modalities": list(t.modalities), "model_ready": t.model_ready}
        for t in COMMON_CANCER_TARGETS
    ]
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "cancer_targets": json.dumps(cancer_targets, ensure_ascii=False),
        "disclaimer": MEDICAL_DISCLAIMER,
        "current_user": request.state.current_user,
        "csrf_token": _request_csrf_token(request),
    })


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = "", reset: str = ""):
    user_id = request.session.get("user_id")
    user = get_auth_db().get_user(user_id) if isinstance(user_id, int) else None
    if user and user.is_active:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {
        "request": request,
        "csrf_token": _request_csrf_token(request),
        "error": error,
        "reset": reset == "1",
    })


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(""), password: str = Form(""), csrf_token: str = Form("")):
    expected_token = request.session.get("csrf_token", "")
    if not expected_token or not secrets.compare_digest(csrf_token, expected_token):
        raise HTTPException(status_code=403, detail="CSRF token không hợp lệ hoặc đã hết hạn.")

    auth_db = get_auth_db()
    remote_addr = request.client.host if request.client else "unknown"
    if auth_db.login_locked(username, remote_addr):
        raise HTTPException(status_code=429, detail="Đăng nhập tạm khóa 15 phút do nhập sai quá nhiều lần.")
    user = auth_db.authenticate(username, password)
    if user is None:
        auth_db.record_login_failure(username, remote_addr)
        return templates.TemplateResponse(request, "login.html", {
            "request": request,
            "csrf_token": _request_csrf_token(request),
            "error": "Tên đăng nhập hoặc mật khẩu không đúng.",
        }, status_code=401)

    auth_db.clear_login_failures(username, remote_addr)
    request.session.clear()
    request.session["user_id"] = user.id
    _request_csrf_token(request)
    return RedirectResponse("/", status_code=303)


@app.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request, error: str = "", sent: str = ""):
    user_id = request.session.get("user_id")
    user = get_auth_db().get_user(user_id) if isinstance(user_id, int) else None
    if user and user.is_active:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "forgot_password.html", {
        "request": request,
        "csrf_token": _request_csrf_token(request),
        "error": error,
        "sent": sent == "1",
    })


def _deliver_recovery_email(auth_db: WebAuthDatabase, username: str, email: str, code: str, code_hash: str) -> None:
    try:
        send_password_recovery_email(email, username, code)
    except Exception:
        auth_db.clear_recovery_code(username, code_hash)
        logger.exception("Không gửi được email khôi phục mật khẩu.")


@app.post("/forgot-password/request")
async def request_password_recovery(
    request: Request,
    background_tasks: BackgroundTasks,
    username: str = Form(""),
    csrf_token: str = Form(""),
):
    expected_token = request.session.get("csrf_token", "")
    if not expected_token or not secrets.compare_digest(csrf_token, expected_token):
        raise HTTPException(status_code=403, detail="CSRF token không hợp lệ hoặc đã hết hạn.")

    auth_db = get_auth_db()
    remote_addr = request.client.host if request.client else "unknown"
    request.session["password_reset_username"] = username.strip() if len(username.strip()) <= 32 else ""
    if auth_db.allow_recovery_request(remote_addr, username):
        recovery_code = generate_recovery_code()
        recovery_hash = hash_recovery_code(recovery_code)
        try:
            stored = auth_db.store_recovery_code(
                username,
                recovery_hash,
                time.time() + RECOVERY_CODE_TTL_SECONDS,
            )
        except ValueError:
            stored = None
        if stored:
            stored_username, recipient = stored
            request.session["password_reset_username"] = stored_username
            background_tasks.add_task(
                _deliver_recovery_email, auth_db, stored_username, recipient, recovery_code, recovery_hash
            )
    return RedirectResponse("/forgot-password?sent=1", status_code=303)


@app.post("/forgot-password", response_class=HTMLResponse)
async def reset_forgotten_password(
    request: Request,
    recovery_code: str = Form(""),
    password: str = Form(""),
    confirm_password: str = Form(""),
    csrf_token: str = Form(""),
):
    expected_token = request.session.get("csrf_token", "")
    if not expected_token or not secrets.compare_digest(csrf_token, expected_token):
        raise HTTPException(status_code=403, detail="CSRF token không hợp lệ hoặc đã hết hạn.")

    username = str(request.session.get("password_reset_username", ""))
    auth_db = get_auth_db()
    remote_addr = request.client.host if request.client else "unknown"
    attempt_key = f"recovery:{username.strip()}"
    if auth_db.login_locked(attempt_key, remote_addr):
        raise HTTPException(status_code=429, detail="Khôi phục tạm khóa 15 phút do nhập sai quá nhiều lần.")
    if password != confirm_password:
        return templates.TemplateResponse(request, "forgot_password.html", {
            "request": request,
            "csrf_token": _request_csrf_token(request),
            "error": "Mật khẩu nhập lại không khớp.",
        }, status_code=400)
    try:
        new_password_hash = hash_password(password)
    except ValueError as exc:
        return templates.TemplateResponse(request, "forgot_password.html", {
            "request": request,
            "csrf_token": _request_csrf_token(request),
            "error": str(exc),
        }, status_code=400)

    if not auth_db.reset_password_with_recovery(username, recovery_code, new_password_hash):
        auth_db.record_login_failure(attempt_key, remote_addr)
        return templates.TemplateResponse(request, "forgot_password.html", {
            "request": request,
            "csrf_token": _request_csrf_token(request),
            "error": "Tên đăng nhập hoặc mã khôi phục không đúng.",
        }, status_code=400)
    auth_db.clear_login_failures(attempt_key, remote_addr)
    request.session.pop("password_reset_username", None)
    return RedirectResponse("/login?reset=1", status_code=303)


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


def _admin_users_response(
    request: Request,
    *,
    notice: str = "",
    error: str = "",
):
    return templates.TemplateResponse(request, "admin_users.html", {
        "request": request,
        "users": get_auth_db().list_users(),
        "current_user": request.state.current_user,
        "csrf_token": _request_csrf_token(request),
        "notice": notice,
        "error": error,
    })


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(request: Request, _: WebUser = ADMIN_REQUIRED):
    return _admin_users_response(
        request,
        notice=request.query_params.get("notice", ""),
        error=request.query_params.get("error", ""),
    )


@app.post("/admin/users/create")
async def admin_create_user(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    _: WebUser = ADMIN_REQUIRED,
):
    try:
        get_auth_db().create_user(
            username,
            hash_password(password),
            role,
            email=email,
        )
    except ValueError as exc:
        return RedirectResponse(f"/admin/users?error={quote(str(exc))}", status_code=303)
    return _admin_users_response(request, notice="created")


@app.post("/admin/users/{user_id}/update")
async def admin_update_user(
    user_id: int,
    role: str = Form(...),
    is_active: str = Form("0"),
    email: str = Form(""),
    _: WebUser = ADMIN_REQUIRED,
):
    try:
        get_auth_db().update_user(user_id, role, is_active == "1", email)
    except ValueError as exc:
        return RedirectResponse(f"/admin/users?error={quote(str(exc))}", status_code=303)
    return RedirectResponse("/admin/users?notice=updated", status_code=303)


_STATUS_CACHE_TTL_SECONDS = 60.0
_status_cache: tuple[float, dict] | None = None


@app.get("/api/status")
def api_status():
    global _status_cache
    if _status_cache is not None and time.monotonic() - _status_cache[0] < _STATUS_CACHE_TTL_SECONDS:
        return _status_cache[1]
    medical = get_medical_system_status()
    db_stats = get_db().get_db_stats()
    payload = {
        "ok": True,
        "model_ready": medical.model_ready,
        "model_message": medical.model_message,
        "case_count": medical.case_count,
        "analyzed_cancers": list(medical.analyzed_cancers),
        "analyzed_modalities": list(medical.analyzed_modalities),
        "db_stats": db_stats,
        "disclaimer": MEDICAL_DISCLAIMER,
    }
    _status_cache = (time.monotonic(), payload)
    return payload


async def _save_upload(file: UploadFile, dest: Path) -> int:
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
    # ponytail: sniff vài magic bytes, không thay thế validator y khoa đầy đủ
    try:
        with open(path, "rb") as f:
            head = f.read(280)
    except OSError:
        return False
    lower = filename.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return head[:2] == b"\x1f\x8b" or len(head) > 0
    if lower.endswith(".dcm") or head[128:132] == b"DICM":
        return True
    img_magic = (b"\xff\xd8\xff", b"\x89PNG", b"BM", b"II*\x00", b"MM\x00*", b"RIFF", b"\x49\x49\x2b\x00")
    return head.startswith(img_magic)


def _require_ready_target(target_key: str, modality: str):
    target = get_cancer_target(target_key)
    if target is None:
        raise HTTPException(status_code=400, detail="Nhóm bệnh không hợp lệ.")
    if not target.model_ready:
        raise HTTPException(status_code=409, detail=f"{target.label} chưa có model suy luận tích hợp.")
    if modality and modality not in target.modalities:
        raise HTTPException(status_code=400, detail="Modality không phù hợp với nhóm bệnh đã chọn.")
    return target


@app.post("/api/upload", dependencies=[STAFF_REQUIRED])
async def upload_file(request: Request):
    raw_length = request.headers.get("content-length")
    if raw_length is None:
        raise HTTPException(status_code=411, detail="Cần Content-Length để giới hạn dung lượng tải lên.")
    if int(raw_length) > MAX_UPLOAD_BYTES + MAX_FORM_BYTES:
        raise HTTPException(status_code=413, detail="File vượt quá dung lượng cho phép.")

    form = await request.form(max_files=1, max_fields=1)
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile) or not file.filename:
            raise HTTPException(status_code=400, detail="Không có file được chọn.")
        filename = Path(file.filename.replace("\\", "/")).name
        if not filename or filename in {".", ".."}:
            raise HTTPException(status_code=400, detail="Tên file không hợp lệ.")
        lower_name = filename.lower()
        allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dcm"}
        if Path(filename).suffix.lower() not in allowed_ext and not lower_name.endswith((".nii", ".nii.gz")):
            raise HTTPException(status_code=400, detail=f"Định dạng file không được hỗ trợ: {filename}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        token = uuid.uuid4().hex[:10]
        safe_name = f"{timestamp}_{token}_{filename}"
        WEB_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        stored_path = WEB_UPLOADS_DIR / safe_name
        size_bytes = await _save_upload(file, stored_path)
        if not _has_allowed_magic(stored_path, filename):
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Nội dung file không khớp định dạng cho phép.")
        upload_id = get_db().add_web_upload(
            filename=filename,
            stored_path=str(stored_path),
            size_bytes=size_bytes,
            mime_type=file.content_type or "application/octet-stream",
        )

        target_key, modality = infer_medical_upload_context(str(stored_path))
        logger.info("Da upload medical file (%d bytes), context: %s / %s", size_bytes, target_key, modality)
        return {
            "ok": True,
            "upload_id": upload_id,
            "filename": filename,
            "stored_path": str(stored_path),
            "size_bytes": size_bytes,
            "detected_target": target_key,
            "detected_modality": modality,
        }
    finally:
        await form.close()


@app.post("/api/analyze", dependencies=[STAFF_REQUIRED])
def analyze_image(
    image_path: str = Form(""),
    patient_code: str = Form("WEB"),
    user_prompt: str = Form(""),
    conversation_id: int = Form(0),
    target_key: str = Form("brain"),
    modality: str = Form(""),
):
    target = _require_ready_target(target_key, modality)
    if not image_path or not image_path.strip():
        raise HTTPException(status_code=400, detail="Thieu file anh.")
    stored = _safe_path(OUTPUT_DIR, image_path)
    if stored is None:
        raise HTTPException(status_code=400, detail=f"Không tìm thấy file hợp lệ: {image_path}")
    pc = patient_code or f"WEB-{uuid.uuid4().hex[:8].upper()}"
    if not ANALYSIS_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Hệ thống đang phân tích một ca khác. Vui lòng thử lại sau.")
    try:
        service = get_medical_service()
        response: MedicalChatResponse = service.analyze_attachment(
            image_path=str(stored),
            patient_code=pc,
            user_prompt=user_prompt,
            target_key=target.key,
            modality=modality or None,
        )
    except Exception:
        logger.exception("Phân tích ảnh thất bại.")
        raise HTTPException(status_code=500, detail="Không thể phân tích ảnh. Vui lòng thử lại hoặc kiểm tra nhật ký hệ thống.")
    finally:
        ANALYSIS_SEMAPHORE.release()
    metadata = json.loads(response.metadata_json) if response.metadata_json else {}

    if conversation_id and get_db().conversation_exists(conversation_id):
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "reply_text": response.reply_text,
            "attachment_path": response.attachment_path,
            "attachment_kind": response.attachment_kind,
            "metadata": metadata,
            "metadata_json": response.metadata_json,
        }
    fname = Path(image_path).name
    title = f"Phân tích {fname}"
    db = get_db()
    conv_id = db.create_conversation(title=title, subtitle=pc)
    user_msg = ChatMessage(sender="user", text=user_prompt or f"Phân tích ảnh: {fname}")
    db.add_message(conv_id, user_msg)
    assistant_msg = ChatMessage(
        sender="assistant",
        text=response.reply_text,
        attachment_path=response.attachment_path,
        attachment_kind=response.attachment_kind,
        metadata_json=response.metadata_json,
    )
    db.add_message(conv_id, assistant_msg)
    return {
        "ok": True,
        "conversation_id": conv_id,
        "reply_text": response.reply_text,
        "attachment_path": response.attachment_path,
        "attachment_kind": response.attachment_kind,
        "metadata": metadata,
        "metadata_json": response.metadata_json,
    }


@app.post("/api/conversations", dependencies=[ADMIN_REQUIRED])
async def create_conversation():
    db = get_db()
    conv_id = db.create_conversation(title="Cuoc tro chuyen moi", subtitle="Hom nay")
    return {"ok": True, "conversation_id": conv_id}


def _case_summary(record) -> dict:
    return {
        "case_id": record.case_id,
        "patient_code": record.patient_code,
        "risk_level": record.risk_level,
        "suspected_malignant": record.suspected_malignant,
        "image_path": record.image_path,
        "processed_image_path": record.processed_image_path,
        "recommendation": record.recommendation,
        "created_at": record.created_at,
        "detections": record.metadata.get("detections", []),
        "average_confidence": record.metadata.get("average_confidence", 0),
        "model_name": record.metadata.get("model_name", "-"),
        "quality_warnings": record.metadata.get("quality_warnings", []),
        "assigned_to": record.assigned_to,
        "review_status": record.review_status,
        "reviewed_by": record.reviewed_by,
        "reviewed_at": record.reviewed_at,
        "public_code": record.public_code,
        "modality": record.metadata.get("modality"),
        "body_region": record.metadata.get("body_region"),
    }


@app.get("/api/cases")
def list_cases(request: Request, limit: int = 50, offset: int = 0):
    user = request.state.current_user
    if user.role == "viewer":
        return {"ok": True, "cases": []}
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    records = get_case_db().list_cases(assigned_to=user.username if user.role == "clinician" else None, limit=limit, offset=offset)
    cases = [_case_summary(record) for record in records]
    return {"ok": True, "cases": cases}


@app.get("/api/cases/{case_id}")
def get_case(case_id: int, request: Request):
    record = get_case_db().get_case(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
    user = request.state.current_user
    if user.role == "viewer" or (user.role == "clinician" and record.assigned_to != user.username):
        raise HTTPException(status_code=403, detail="Bạn không được phân quyền xem ca bệnh này.")
    return {"ok": True, "case": _case_summary(record)}


@app.get("/api/clinicians", dependencies=[ADMIN_REQUIRED])
def list_clinicians():
    return {"ok": True, "clinicians": [
        {"username": user.username}
        for user in get_auth_db().list_users()
        if user.role == "clinician" and user.is_active
    ]}


@app.post("/api/cases/{case_id}/assign", dependencies=[ADMIN_REQUIRED])
def assign_case(case_id: int, username: str = Form(...)):
    clinician = next(
        (user for user in get_auth_db().list_users() if user.username.casefold() == username.strip().casefold()),
        None,
    )
    if clinician is None or clinician.role != "clinician" or not clinician.is_active:
        raise HTTPException(status_code=400, detail="Hãy chọn tài khoản nhân viên y tế đang hoạt động.")
    if not get_case_db().assign_case(case_id, clinician.username):
        raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
    return {"ok": True}


@app.post("/api/cases/{case_id}/review", dependencies=[Depends(require_role("clinician"))])
def approve_case(
    case_id: int,
    request: Request,
    risk_level: str = Form(...),
    suspected_malignant: bool = Form(...),
    recommendation: str = Form(...),
):
    if risk_level not in {"low", "medium", "high", "uncertain"}:
        raise HTTPException(status_code=400, detail="Mức nguy cơ không hợp lệ.")
    recommendation = recommendation.strip()
    if not recommendation or len(recommendation) > 5000:
        raise HTTPException(status_code=400, detail="Khuyến nghị phải có từ 1 đến 5000 ký tự.")
    db = get_case_db()
    record = db.get_case(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
    user = request.state.current_user
    if record.assigned_to != user.username:
        raise HTTPException(status_code=403, detail="Ca bệnh chưa được phân công cho bạn.")
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for _ in range(5):
        public_code = "".join(secrets.choice(alphabet) for _ in range(10))
        try:
            if db.approve_case(
                case_id,
                reviewer=user.username,
                risk_level=risk_level,
                suspected_malignant=suspected_malignant,
                recommendation=recommendation,
                public_code=public_code,
            ):
                return {"ok": True, "public_code": public_code}
        except sqlite3.IntegrityError:
            continue
    raise HTTPException(status_code=500, detail="Không thể tạo mã đọc kết quả, vui lòng thử lại.")


@app.get("/api/public/cases/{public_code}", dependencies=[Depends(require_role("viewer"))])
def get_public_case(public_code: str):
    if not re.fullmatch(r"[A-Z0-9]{10}", public_code.upper(), flags=re.ASCII):
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả đã duyệt.")
    record = get_case_db().get_case_by_public_code(public_code)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả đã duyệt.")
    return {
        "ok": True,
        "case": {
            "patient_code": record.patient_code,
            "risk_level": record.risk_level,
            "suspected_malignant": record.suspected_malignant,
            "recommendation": record.recommendation,
            "created_at": record.created_at,
            "reviewed_at": record.reviewed_at,
        },
    }


@app.get("/api/public/cases/{public_code}/pdf", dependencies=[Depends(require_role("viewer"))])
def download_public_case_pdf(public_code: str):
    record = get_case_db().get_case_by_public_code(public_code)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy kết quả đã duyệt.")
    try:
        pdf_path = export_case_pdf(OUTPUT_DIR / "medical" / "reports", {
            "case_id": record.patient_code,
            "patient_code": record.patient_code,
            "source_image": "",
            "processed_image": "",
            "risk_level": record.risk_level,
            "suspected_malignant": record.suspected_malignant,
            "recommendation": record.recommendation,
            "review_status": "approved",
            "reviewed_by": "Nhân viên y tế",
            "reviewed_at": record.reviewed_at,
            "quality_warnings": [],
            "detections": [],
            "model_name": "OncoVision AI",
            "disclaimer": MEDICAL_DISCLAIMER,
        })
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return FileResponse(pdf_path, filename=f"ket-qua-{public_code.upper()}.pdf", media_type="application/pdf")


@app.get("/api/cases/{case_id}/image")
def get_case_image(case_id: int, request: Request):
    record = get_case_db().get_case(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
    user = request.state.current_user
    if user.role == "viewer" or (user.role == "clinician" and record.assigned_to != user.username):
        raise HTTPException(status_code=403, detail="Bạn không được xem ảnh của ca bệnh này.")
    image_path = Path(record.processed_image_path).resolve()
    if not image_path.is_relative_to(OUTPUT_DIR.resolve()) or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh đã xử lý.")
    return FileResponse(image_path)


@app.get("/api/cases/{case_id}/pdf")
def download_case_pdf(case_id: int, request: Request):
    record = get_case_db().get_case(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
    user = request.state.current_user
    if user.role == "viewer" or (user.role == "clinician" and record.assigned_to != user.username):
        raise HTTPException(status_code=403, detail="Bạn không được xuất ca bệnh này.")
    try:
        pdf_path = export_case_pdf(OUTPUT_DIR / "medical" / "reports", build_case_export_payload(record))
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return FileResponse(pdf_path, filename=pdf_path.name, media_type="application/pdf")


@app.get("/api/conversations", dependencies=[ADMIN_REQUIRED])
async def list_conversations(limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    convs = get_db().get_all_conversations(limit=limit, offset=offset)
    result = []
    for conv in convs:
        msgs = []
        for msg in conv.messages:
            msgs.append({
                "id": msg.id,
                "sender": msg.sender,
                "text": msg.text[:500] if msg.text else "",
                "attachment_path": msg.attachment_path,
                "attachment_kind": msg.attachment_kind,
                "metadata_json": msg.metadata_json,
            })
        result.append({
            "id": conv.id,
            "title": conv.title or "Hoi thoai",
            "subtitle": conv.subtitle or "",
            "messages": msgs,
            "message_count": len(msgs),
        })
    return {"ok": True, "conversations": result}


@app.get("/api/conversations/{conv_id}", dependencies=[ADMIN_REQUIRED])
async def get_conversation(conv_id: int):
    conv = get_db().get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")
    msgs = []
    for msg in conv.messages:
        msgs.append({
            "id": msg.id,
            "sender": msg.sender,
            "text": msg.text,
            "attachment_path": msg.attachment_path,
            "attachment_kind": msg.attachment_kind,
            "metadata_json": msg.metadata_json,
        })
    return {
        "ok": True,
        "conversation": {
            "id": conv.id,
            "title": conv.title or "Hoi thoai",
            "subtitle": conv.subtitle or "",
            "messages": msgs,
        },
    }


@app.post("/api/conversations/{conv_id}/messages", dependencies=[ADMIN_REQUIRED])
async def add_message(conv_id: int, sender: str = Form(...), text: str = Form(""), attachment_path: str = Form(""), attachment_kind: str = Form(""), metadata_json: str = Form("")):
    db = get_db()
    if sender not in {"user", "assistant"}:
        raise HTTPException(status_code=422, detail="Người gửi không hợp lệ.")
    normalized_attachment_kind = attachment_kind or None
    if normalized_attachment_kind not in {None, "image", "text", "camera"}:
        raise HTTPException(status_code=422, detail="Loại tệp đính kèm không hợp lệ.")
    conv = db.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")
    msg = ChatMessage(
        sender=cast(Literal["user", "assistant"], sender),
        text=text,
        attachment_path=attachment_path or None,
        attachment_kind=cast(Literal["image", "text", "camera"] | None, normalized_attachment_kind),
        metadata_json=metadata_json or None,
    )
    msg_id = db.add_message(conv_id, msg)

    if sender == "user" and conv.title in ("Cuoc tro chuyen moi", "New chat", ""):
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if first_line and len(first_line) > 2:
            db.update_conversation_title(conv_id, first_line[:28])
    return {"ok": True, "message_id": msg_id}


@app.delete("/api/conversations/{conv_id}", dependencies=[ADMIN_REQUIRED])
async def delete_conversation(conv_id: int):
    db = get_db()
    if not db.conversation_exists(conv_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")
    db.delete_conversation(conv_id)
    return {"ok": True}


@app.get("/api/settings")
async def get_settings():
    db = get_db()
    return {
        "ok": True,
        "language": db.get_setting("language", "vi"),
        "theme": db.get_setting("theme", "system"),
    }


@app.post("/api/settings")
async def save_settings(request: Request, language: str = Form("vi"), theme: str = Form("system")):
    if request.state.current_user.role == "viewer":
        raise HTTPException(status_code=403, detail="Tài khoản chỉ có quyền xem.")
    db = get_db()
    db.set_setting("language", language)
    db.set_setting("theme", theme)
    return {"ok": True}


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/output/{file_path:path}", dependencies=[ADMIN_REQUIRED])
async def serve_output_file(file_path: str):
    path = _safe_path(OUTPUT_DIR, file_path)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp.")
    return FileResponse(path)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.svg", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        if "text/html" in request.headers.get("accept", "") and not request.url.path.startswith("/api/"):
            return templates.TemplateResponse(request, "404.html", {"request": request}, status_code=404)
        return JSONResponse(status_code=404, content={"ok": False, "detail": exc.detail or "Not Found"})
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def custom_500_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error: %s", exc)
    if "text/html" in request.headers.get("accept", "") and not request.url.path.startswith("/api/"):
        return templates.TemplateResponse(request, "500.html", {"request": request}, status_code=500)
    return JSONResponse(status_code=500, content={"ok": False, "detail": "Internal Server Error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
