from __future__ import annotations

"""Web page routes: index, login, forgot-password, change-credentials, admin-users, logout, favicon."""

import json
import secrets
import time
from pathlib import Path

from fastapi import BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from entrypoints.web.shared import (
    ADMIN_REQUIRED,
    MEDICAL_DISCLAIMER,
    OUTPUT_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    _deliver_recovery_email,
    _request_csrf_token,
    audit,
    generate_recovery_code,
    get_auth_db,
    hash_password,
    hash_recovery_code,
    require_authenticated,
    require_role,
)
from medical.cancer_catalog import COMMON_CANCER_TARGETS

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def register_pages_routes(app):
    """Register all page routes onto the FastAPI app."""

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        from entrypoints.web.shared import _cached_medical_status
        medical = _cached_medical_status()
        from medical.dataset import available_medical_modalities_for_target
        cancer_targets = [
            {
                "key": t.key,
                "label": t.label,
                "modalities": list(t.modalities),
                "available_modalities": list(available_medical_modalities_for_target(t.key)),
                "model_ready": medical.target_capabilities.get(t.key, {}).get("runtime_ready", False),
                "capabilities": medical.target_capabilities.get(t.key, {}),
            }
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
            audit.log_login(user=username, success=False, ip_address=remote_addr)
            return templates.TemplateResponse(request, "login.html", {
                "request": request,
                "csrf_token": _request_csrf_token(request),
                "error": "Tên đăng nhập hoặc mật khẩu không đúng.",
            }, status_code=401)

        auth_db.clear_login_failures(username, remote_addr)
        request.session.clear()
        request.session["user_id"] = user.id
        _request_csrf_token(request)
        audit.log_login(user=username, success=True, ip_address=remote_addr)
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
                    time.time() + 3600,  # RECOVERY_CODE_TTL_SECONDS
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

    @app.get("/change-credentials", response_class=HTMLResponse)
    async def change_credentials_page(request: Request, error: str = ""):
        user_id = request.session.get("user_id")
        user = get_auth_db().get_user(user_id) if isinstance(user_id, int) else None
        if user is None or not user.is_active:
            request.session.clear()
            return RedirectResponse("/login", status_code=303)
        if not user.must_change_credentials:
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(request, "change_credentials.html", {
            "request": request,
            "csrf_token": _request_csrf_token(request),
            "error": error,
            "current_username": user.username,
        })

    @app.post("/change-credentials", response_class=HTMLResponse)
    async def change_credentials(
        request: Request,
        new_username: str = Form(""),
        new_password: str = Form(""),
        confirm_password: str = Form(""),
        csrf_token: str = Form(""),
    ):
        expected_token = request.session.get("csrf_token", "")
        if not expected_token or not secrets.compare_digest(csrf_token, expected_token):
            raise HTTPException(status_code=403, detail="CSRF token không hợp lệ hoặc đã hết hạn.")

        user_id = request.session.get("user_id")
        user = get_auth_db().get_user(user_id) if isinstance(user_id, int) else None
        if user is None or not user.is_active or not user.must_change_credentials:
            return RedirectResponse("/login", status_code=303)

        new_username = new_username.strip()
        if new_password != confirm_password:
            return templates.TemplateResponse(request, "change_credentials.html", {
                "request": request,
                "csrf_token": _request_csrf_token(request),
                "error": "Mật khẩu nhập lại không khớp.",
                "current_username": user.username,
            }, status_code=400)
        if not new_username:
            return templates.TemplateResponse(request, "change_credentials.html", {
                "request": request,
                "csrf_token": _request_csrf_token(request),
                "error": "Tên đăng nhập không được để trống.",
                "current_username": user.username,
            }, status_code=400)
        try:
            new_password_hash = hash_password(new_password)
        except ValueError as exc:
            return templates.TemplateResponse(request, "change_credentials.html", {
                "request": request,
                "csrf_token": _request_csrf_token(request),
                "error": str(exc),
                "current_username": user.username,
            }, status_code=400)

        auth_db = get_auth_db()
        try:
            auth_db.update_credentials(user.id, new_username, new_password_hash)
        except ValueError as exc:
            return templates.TemplateResponse(request, "change_credentials.html", {
                "request": request,
                "csrf_token": _request_csrf_token(request),
                "error": str(exc),
                "current_username": user.username,
            }, status_code=400)

        request.session.clear()
        return RedirectResponse("/login?reset=1", status_code=303)

    # --- Admin users ---

    def _admin_users_response(request: Request, *, notice: str = "", error: str = ""):
        from urllib.parse import quote as url_quote
        return templates.TemplateResponse(request, "admin_users.html", {
            "request": request,
            "users": get_auth_db().list_users(),
            "current_user": request.state.current_user,
            "csrf_token": _request_csrf_token(request),
            "notice": notice,
            "error": error,
        })

    @app.get("/admin/users", response_class=HTMLResponse)
    async def admin_users_page(request: Request, _: object = ADMIN_REQUIRED):
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
        _: object = ADMIN_REQUIRED,
    ):
        from urllib.parse import quote as url_quote
        try:
            get_auth_db().create_user(
                username,
                hash_password(password),
                role,
                email=email,
            )
        except ValueError as exc:
            return RedirectResponse(f"/admin/users?error={url_quote(str(exc))}", status_code=303)
        return _admin_users_response(request, notice="created")

    @app.post("/admin/users/{user_id}/update")
    async def admin_update_user(
        user_id: int,
        role: str = Form(...),
        is_active: str = Form("0"),
        email: str = Form(""),
        _: object = ADMIN_REQUIRED,
    ):
        from urllib.parse import quote as url_quote
        try:
            get_auth_db().update_user(user_id, role, is_active == "1", email)
        except ValueError as exc:
            return RedirectResponse(f"/admin/users?error={url_quote(str(exc))}", status_code=303)
        return RedirectResponse("/admin/users?notice=updated", status_code=303)

    # --- Favicon ---

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon_ico():
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")
