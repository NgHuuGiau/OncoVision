from __future__ import annotations

"""Settings routes: get and save user settings."""

from fastapi import Form, HTTPException, Request

from entrypoints.web.shared import get_db


def register_settings_routes(app):
    """Register settings routes."""

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
