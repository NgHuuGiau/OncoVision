from __future__ import annotations

"""OncoVision Web App — slim entry point that registers blueprint routes.

All route handlers have been moved to entrypoints/web/ for maintainability.
This file only sets up the FastAPI app, middleware, and exception handlers.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to sys.path when running from entrypoints/
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from entrypoints.web.shared import (
    COOKIE_SECURE,
    IS_PRODUCTION,
    OUTPUT_DIR,
    SESSION_SECRET,
    get_auth_db,
    get_medical_service,
    logger,
    require_authenticated,
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    from utils.logger import setup_production_logging
    setup_production_logging()
    logger.info("OncoVision Web Chat khoi dong...")
    get_auth_db().ensure_default_admin()
    try:
        service = get_medical_service()
        service.check_ready()
        logger.info("Medical service san sang.")
    except Exception as exc:
        logger.warning("Medical service chua san sang: %s", exc)
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OncoVision Web Chat",
    description="AI y khoa chuyên về phân tích ảnh y tế — sàng lọc ung thư từ 11 nhóm.",
    version="2.0.0",
    lifespan=lifespan,
    dependencies=[Depends(require_authenticated)],
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url="/redoc" if not IS_PRODUCTION else None,
    openapi_url="/openapi.json" if not IS_PRODUCTION else None,
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="oncovision_session",
    max_age=8 * 60 * 60,
    same_site="lax",
    https_only=COOKIE_SECURE,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

from entrypoints.web.shared import MAX_FORM_BYTES, MAX_UPLOAD_BYTES


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
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; connect-src 'self' https://cdn.jsdelivr.net blob:; worker-src 'self' blob:")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if request.url.path.startswith(("/api/", "/output/")):
        response.headers.setdefault("Cache-Control", "no-store")
    if IS_PRODUCTION:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


# ---------------------------------------------------------------------------
# Register blueprint routes
# ---------------------------------------------------------------------------

from entrypoints.web.cases import register_cases_routes
from entrypoints.web.conversations import register_conversations_routes
from entrypoints.web.health import register_health_routes
from entrypoints.web.pages import register_pages_routes
from entrypoints.web.settings import register_settings_routes
from entrypoints.web.upload_analyze import register_upload_analyze_routes
from entrypoints.web.utils_routes import register_utils_routes
from entrypoints.web.anatomy_routes import register_anatomy_routes

register_pages_routes(app)
register_health_routes(app)
register_upload_analyze_routes(app)
register_cases_routes(app)
register_conversations_routes(app)
register_settings_routes(app)
register_utils_routes(app)
register_anatomy_routes(app)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

from fastapi.templating import Jinja2Templates

from entrypoints.web.shared import TEMPLATES_DIR

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("ONCOVISION_HOST", "127.0.0.1"), port=8000)
