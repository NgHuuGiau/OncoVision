from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.chat_ui.models import ChatMessage
from app.chat_ui.paths import CHAT_HISTORY_DB_PATH, OUTPUT_DIR, PROJECT_ROOT
from app.chat_ui.storage import ChatDatabase
from medical.cancer_catalog import COMMON_CANCER_TARGETS
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

TEMPLATES_DIR = PROJECT_ROOT / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("OncoVision Web Chat khoi dong...")
    try:
        service = get_medical_service()
        service.check_ready()
        logger.info("Medical service san sang.")
    except Exception as exc:
        logger.warning("Medical service chua san sang: %s", exc)
    yield


app = FastAPI(title="OncoVision Web Chat", lifespan=lifespan)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_db: ChatDatabase | None = None
_medical_service: MedicalChatService | None = None
_case_db: MedicalCaseDatabase | None = None


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
        {"key": t.key, "label": t.label, "modalities": list(t.modalities)}
        for t in COMMON_CANCER_TARGETS
    ]
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "cancer_targets": json.dumps(cancer_targets, ensure_ascii=False),
        "disclaimer": MEDICAL_DISCLAIMER,
    })


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
        "dataset_initialized": medical.dataset_initialized,
        "total_images": medical.total_images,
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


@app.post("/api/upload")
async def upload_file(file: Annotated[UploadFile, File()]):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Không có file được chọn.")
    filename = file.filename
    lower_name = filename.lower()
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dcm"}
    if Path(filename).suffix.lower() not in allowed_ext and not lower_name.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail=f"Định dạng file không được hỗ trợ: {filename}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    token = uuid.uuid4().hex[:10]
    safe_name = f"{timestamp}_{token}_{filename}"
    stored_path = WEB_UPLOADS_DIR / safe_name
    size_bytes = await _save_upload(file, stored_path)
    upload_id = get_db().add_web_upload(
        filename=filename,
        stored_path=str(stored_path),
        size_bytes=size_bytes,
        mime_type=file.content_type or "application/octet-stream",
    )

    target_key, modality = infer_medical_upload_context(str(stored_path))
    logger.info("Da upload file: %s (%d bytes), context: %s / %s", filename, size_bytes, target_key, modality)
    return {
        "ok": True,
        "upload_id": upload_id,
        "filename": filename,
        "stored_path": str(stored_path),
        "size_bytes": size_bytes,
        "detected_target": target_key,
        "detected_modality": modality,
    }


@app.post("/api/analyze")
def analyze_image(
    image_path: str = Form(""),
    patient_code: str = Form("WEB"),
    user_prompt: str = Form(""),
    conversation_id: int = Form(0),
):
    if not image_path or not image_path.strip():
        raise HTTPException(status_code=400, detail="Thieu file anh.")
    stored = _safe_path(OUTPUT_DIR, image_path)
    if stored is None:
        raise HTTPException(status_code=400, detail=f"Không tìm thấy file hợp lệ: {image_path}")
    pc = patient_code or f"WEB-{uuid.uuid4().hex[:8].upper()}"
    try:
        service = get_medical_service()
        response: MedicalChatResponse = service.analyze_attachment(
            image_path=str(stored),
            patient_code=pc,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.exception("Phân tích ảnh thất bại: %s", stored)
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {exc}")
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


@app.post("/api/conversations")
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
    }


@app.get("/api/cases")
def list_cases():
    cases = [_case_summary(r) for r in get_case_db().list_cases()]
    return {"ok": True, "cases": cases}


@app.get("/api/cases/{case_id}")
def get_case(case_id: int):
    record = get_case_db().get_case(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
    return {"ok": True, "case": _case_summary(record)}


@app.get("/api/cases/{case_id}/pdf")
def download_case_pdf(case_id: int):
    record = get_case_db().get_case(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
    try:
        pdf_path = export_case_pdf(OUTPUT_DIR / "medical" / "reports", build_case_export_payload(record))
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return FileResponse(pdf_path, filename=pdf_path.name, media_type="application/pdf")


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


@app.post("/api/conversations/{conv_id}/messages")
async def add_message(conv_id: int, sender: str = Form(...), text: str = Form(""), attachment_path: str = Form(""), attachment_kind: str = Form(""), metadata_json: str = Form("")):
    db = get_db()
    conv = db.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")
    msg = ChatMessage(
        sender=sender,
        text=text,
        attachment_path=attachment_path or None,
        attachment_kind=attachment_kind or None,
        metadata_json=metadata_json or None,
    )
    msg_id = db.add_message(conv_id, msg)

    if sender == "user" and conv.title in ("Cuoc tro chuyen moi", "New chat", ""):
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if first_line and len(first_line) > 2:
            db.update_conversation_title(conv_id, first_line[:28])
    return {"ok": True, "message_id": msg_id}


@app.delete("/api/conversations/{conv_id}")
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
async def save_settings(language: str = Form("vi"), theme: str = Form("system")):
    db = get_db()
    db.set_setting("language", language)
    db.set_setting("theme", theme)
    return {"ok": True}


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
