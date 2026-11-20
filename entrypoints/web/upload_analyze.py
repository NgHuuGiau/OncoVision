from __future__ import annotations

"""Upload & analyze routes: api/upload, api/analyze."""

import json
import time
import uuid
from pathlib import Path

from fastapi import Form, HTTPException, Request
from starlette.datastructures import UploadFile

from entrypoints.web.shared import (
    ANALYSIS_SEMAPHORE,
    MAX_FORM_BYTES,
    MAX_UPLOAD_BYTES,
    OUTPUT_DIR,
    STAFF_REQUIRED,
    WEB_UPLOADS_DIR,
    _has_allowed_magic,
    _require_ready_capability,
    _safe_path,
    _save_upload,
    _validate_target_input,
    get_auth_db,
    get_case_db,
    get_db,
    get_medical_service,
    logger,
)
from app.chat_ui.models import ChatMessage
from medical.chat_service import MedicalChatResponse


def register_upload_analyze_routes(app):
    """Register upload and analyze routes."""

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
            allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dcm", ".mp4", ".avi", ".mov", ".mkv"}
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

            from medical.dataset import infer_medical_upload_context
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
        request: Request,
        image_path: str = Form(""),
        patient_code: str = Form("WEB"),
        user_prompt: str = Form(""),
        conversation_id: int = Form(0),
        target_key: str = Form("brain"),
        modality: str = Form(""),
        patient_age: str = Form(""),
        patient_sex: str = Form(""),
        clinical_note: str = Form(""),
    ):
        target = _validate_target_input(target_key, modality)
        if not image_path or not image_path.strip():
            raise HTTPException(status_code=400, detail="Thieu file anh.")
        stored = _safe_path(OUTPUT_DIR, image_path)
        if stored is None:
            raise HTTPException(status_code=400, detail=f"Không tìm thấy file hợp lệ: {image_path}")
        _require_ready_capability(target)
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
                patient_age=patient_age or None,
                patient_sex=patient_sex or None,
                clinical_note=clinical_note or None,
            )
        except ValueError as exc:
            detail = str(exc)
            if detail.startswith("Cấu hình medical không hợp lệ"):
                logger.exception("Cấu hình medical sai.")
                raise HTTPException(status_code=500, detail="Cấu hình hệ thống chưa đúng. Vui lòng kiểm tra nhật ký hệ thống.")
            _, _, message = detail.partition(":")
            raise HTTPException(status_code=400, detail=message.strip() or detail)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception:
            logger.exception("Phân tích ảnh thất bại.")
            raise HTTPException(status_code=500, detail="Không thể phân tích ảnh. Vui lòng thử lại hoặc kiểm tra nhật ký hệ thống.")
        finally:
            ANALYSIS_SEMAPHORE.release()
        metadata = json.loads(response.metadata_json) if response.metadata_json else {}
        case_id = metadata.get("medical_case_id")
        actor = request.state.current_user
        if actor.role == "clinician" and isinstance(case_id, int):
            if not get_case_db().assign_case(case_id, actor.username):
                logger.error("Không thể gán ca #%s cho nhân viên %s.", case_id, actor.username)

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
        conv_id = db.create_conversation(title=title, subtitle=pc, owner=actor.username)
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
