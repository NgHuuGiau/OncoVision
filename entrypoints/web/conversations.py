from __future__ import annotations

"""Admin conversation routes: list, get, add message, delete."""

from typing import Literal, cast

from fastapi import Depends, Form, HTTPException

from entrypoints.web.shared import (
    ADMIN_REQUIRED,
    ChatMessage,
    get_db,
)


def register_conversations_routes(app):
    """Register admin conversation routes."""

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
    async def add_message(
        conv_id: int,
        sender: str = Form(...),
        text: str = Form(""),
        attachment_path: str = Form(""),
        attachment_kind: str = Form(""),
        metadata_json: str = Form(""),
    ):
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
