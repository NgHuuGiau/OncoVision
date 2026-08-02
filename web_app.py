from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.chat_ui.models import ChatMessage
from app.chat_ui.paths import PROJECT_ROOT, OUTPUT_DIR, CHAT_HISTORY_DB_PATH
from app.chat_ui.storage import ChatDatabase
from medical.cancer_catalog import COMMON_CANCER_TARGETS
from medical.chat_service import MedicalChatService, MedicalChatResponse
from medical.compliance import MEDICAL_DISCLAIMER
from medical.dataset import infer_medical_upload_context
from medical.system_status import get_medical_system_status
from utils.logger import get_logger


logger = get_logger(__name__)

WEB_UPLOADS_DIR = OUTPUT_DIR / "web_uploads"
WEB_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = PROJECT_ROOT / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OncoVision Web Chat")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_db: ChatDatabase | None = None
_medical_service: MedicalChatService | None = None


def get_db() -> ChatDatabase:
    global _db
    if _db is None:
        _db = ChatDatabase(str(CHAT_HISTORY_DB_PATH))
    return _db


def get_medical_service() -> MedicalChatService:
    global _medical_service
    if _medical_service is None:
        _medical_service = MedicalChatService()
    return _medical_service


def _safe_path(base: Path, path: str) -> Path | None:
    resolved = (base / path).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        return None
    return resolved if resolved.exists() else None


@app.on_event("startup")
async def startup():
    logger.info("OncoVision Web Chat khoi dong...")
    try:
        service = get_medical_service()
        service.check_ready()
        logger.info("Medical service san sang.")
    except Exception as exc:
        logger.warning("Medical service chua san sang: %s", exc)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cancer_targets = [
        {"key": t.key, "label": t.label, "modalities": list(t.modalities)}
        for t in COMMON_CANCER_TARGETS
    ]
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "cancer_targets": json.dumps(cancer_targets, ensure_ascii=False),
        "disclaimer": MEDICAL_DISCLAIMER,
    })


@app.get("/api/status")
async def api_status():
    medical = get_medical_system_status()
    db_stats = get_db().get_db_stats()
    return {
        "ok": True,
        "model_ready": medical.model_ready,
        "model_message": medical.model_message,
        "dataset_initialized": medical.dataset_initialized,
        "total_images": medical.total_images,
        "case_count": medical.case_count,
        "analyzed_cancers": list(medical.analyzed_cancers),
        "analyzed_modalities": list(medical.analyzed_modalities),
        "db_stats": db_stats,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Khong co file duoc chon.")
    filename = file.filename
    lower_name = filename.lower()
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dcm"}
    if Path(filename).suffix.lower() not in allowed_ext and not lower_name.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail=f"Dinh dang file khong duoc ho tro: {filename}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    token = uuid.uuid4().hex[:10]
    safe_name = f"{timestamp}_{token}_{filename}"
    stored_path = WEB_UPLOADS_DIR / safe_name
    content = await file.read()
    stored_path.write_bytes(content)
    upload_id = get_db().add_web_upload(
        filename=filename,
        stored_path=str(stored_path),
        size_bytes=len(content),
        mime_type=file.content_type or "application/octet-stream",
    )
    # Auto-detect medical context
    target_key, modality = infer_medical_upload_context(str(stored_path))
    logger.info("Da upload file: %s (%d bytes), context: %s / %s", filename, len(content), target_key, modality)
    return {
        "ok": True,
        "upload_id": upload_id,
        "filename": filename,
        "stored_path": str(stored_path),
        "size_bytes": len(content),
        "detected_target": target_key,
        "detected_modality": modality,
    }


@app.post("/api/analyze")
async def analyze_image(
    image_path: str = Form(""),
    patient_code: str = Form("WEB"),
    user_prompt: str = Form(""),
):
    if not image_path or not image_path.strip():
        raise HTTPException(status_code=400, detail="Thieu file anh.")
    stored = _safe_path(PROJECT_ROOT, image_path)
    if stored is None:
        raise HTTPException(status_code=400, detail=f"Khong tim thay file: {image_path}")
    pc = patient_code or f"WEB-{uuid.uuid4().hex[:8].upper()}"
    try:
        service = get_medical_service()
        response: MedicalChatResponse = service.analyze_attachment(
            image_path=str(stored),
            patient_code=pc,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.exception("Phan tich anh that bai: %s", stored)
        raise HTTPException(status_code=500, detail=f"Loi phan tich: {exc}")
    metadata = json.loads(response.metadata_json) if response.metadata_json else {}
    fname = Path(image_path).name
    title = f"Phan tich {fname}"
    db = get_db()
    conv_id = db.create_conversation(title=title, subtitle=pc)
    user_msg = ChatMessage(sender="user", text=user_prompt or f"Phan tich anh: {fname}")
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
    }


@app.post("/api/conversations")
async def create_conversation():
    db = get_db()
    conv_id = db.create_conversation(title="Cuoc tro chuyen moi", subtitle="Hom nay")
    return {"ok": True, "conversation_id": conv_id}


@app.get("/api/conversations")
async def list_conversations():
    convs = get_db().get_all_conversations()
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


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: int):
    convs = get_db().get_all_conversations()
    for conv in convs:
        if conv.id == conv_id:
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
    raise HTTPException(status_code=404, detail="Khong tim thay hoi thoai.")


@app.post("/api/conversations/{conv_id}/messages")
async def add_message(conv_id: int, sender: str = Form(...), text: str = Form(""), attachment_path: str = Form(""), attachment_kind: str = Form("")):
    db = get_db()
    convs = db.get_all_conversations()
    if not any(c.id == conv_id for c in convs):
        raise HTTPException(status_code=404, detail="Khong tim thay hoi thoai.")
    msg = ChatMessage(
        sender=sender,
        text=text,
        attachment_path=attachment_path or None,
        attachment_kind=attachment_kind or None,
    )
    msg_id = db.add_message(conv_id, msg)
    # Auto-title from first user message
    conv = next(c for c in convs if c.id == conv_id)
    if sender == "user" and conv.title in ("Cuoc tro chuyen moi", "New chat", ""):
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if first_line and len(first_line) > 2:
            db.update_conversation_title(conv_id, first_line[:28])
    return {"ok": True, "message_id": msg_id}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: int):
    db = get_db()
    if not any(c.id == conv_id for c in db.get_all_conversations()):
        raise HTTPException(status_code=404, detail="Khong tim thay hoi thoai.")
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
async def save_settings(language: str = Form("vi"), theme: str = Form("system")):
    db = get_db()
    db.set_setting("language", language)
    db.set_setting("theme", theme)
    return {"ok": True}


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
