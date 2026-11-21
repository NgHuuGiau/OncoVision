from __future__ import annotations

"""Utility routes: crop image, serve output files, static files mount."""

import io
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from entrypoints.web.shared import (
    ADMIN_REQUIRED,
    OUTPUT_DIR,
    STATIC_DIR,
    _safe_path,
    logger,
)


def register_utils_routes(app):
    """Register utility routes: crop, output files, static mount."""

    @app.get("/api/crop")
    async def crop_image(image_path: str, bbox: str):
        try:
            coords = [int(c.strip()) for c in bbox.split(",")]
            if len(coords) != 4:
                raise HTTPException(status_code=400, detail="bbox phải có 4 số: x1,y1,x2,y2")
            x1, y1, x2, y2 = coords
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox không hợp lệ")

        clean_path = image_path
        clean_path = clean_path.removeprefix('/output/')

        image_file = Path(clean_path)
        if image_file.is_absolute():
            try:
                resolved = image_file.resolve()
                if not resolved.is_relative_to(OUTPUT_DIR.resolve()):
                    raise HTTPException(status_code=403, detail="Đường dẫn ảnh không hợp lệ")
                stored = resolved
            except (ValueError, OSError):
                raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")
        else:
            stored = _safe_path(OUTPUT_DIR, clean_path)

        if stored is None or not stored.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")

        try:
            with Image.open(stored) as img:
                arr = np.array(img)
                h, w = arr.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                margin_x = max(0, (x2 - x1) // 20)
                margin_y = max(0, (y2 - y1) // 20)
                cx1 = max(0, x1 - margin_x)
                cy1 = max(0, y1 - margin_y)
                cx2 = min(w, x2 + margin_x)
                cy2 = min(h, y2 + margin_y)
                cropped = arr[cy1:cy2, cx1:cx2]

                if cropped.ndim == 2 or cropped.shape[2] == 1:
                    pil_img = Image.fromarray(cropped if cropped.ndim == 2 else cropped[:, :, 0], mode="L")
                else:
                    pil_img = Image.fromarray(cropped[:, :, :3], mode="RGB")

                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                return Response(content=buf.read(), media_type="image/png")
        except Exception as exc:
            logger.exception("Không thể crop ảnh: %s", exc)
            raise HTTPException(status_code=500, detail="Không thể xử lý ảnh")

    @app.get("/output/{file_path:path}", dependencies=[ADMIN_REQUIRED])
    async def serve_output_file(file_path: str):
        path = _safe_path(OUTPUT_DIR, file_path)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp.")
        return FileResponse(path)

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
