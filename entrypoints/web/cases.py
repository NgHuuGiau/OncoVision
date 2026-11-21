from __future__ import annotations

"""Case management routes: list, get, assign, review, approve, link-patient, public, image, pdf."""

import re
import secrets

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse

from entrypoints.web.shared import (
    ADMIN_REQUIRED,
    MEDICAL_DISCLAIMER,
    OUTPUT_DIR,
    _can_read_case,
    _case_summary,
    build_case_export_payload,
    export_case_pdf,
    get_auth_db,
    get_case_db,
    require_role,
)


def register_cases_routes(app):
    """Register all case-related routes."""

    @app.get("/api/cases")
    def list_cases(request: Request, limit: int = 50, offset: int = 0):
        user = request.state.current_user
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        if user.role == "viewer":
            records = get_case_db().list_cases_for_patient(user.username, limit=limit, offset=offset)
        else:
            records = get_case_db().list_cases(assigned_to=user.username if user.role == "clinician" else None, limit=limit, offset=offset)
        cases = [_case_summary(record) for record in records]
        return {"ok": True, "cases": cases}

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: int, request: Request):
        record = get_case_db().get_case(case_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
        user = request.state.current_user
        if not _can_read_case(record, user):
            raise HTTPException(status_code=403, detail="Bạn không được phân quyền xem ca bệnh này.")
        return {"ok": True, "case": _case_summary(record)}

    @app.get("/api/clinicians", dependencies=[ADMIN_REQUIRED])
    def list_clinicians():
        return {"ok": True, "clinicians": [
            {"username": user.username}
            for user in get_auth_db().list_users()
            if user.role == "clinician" and user.is_active
        ]}

    @app.get("/api/viewers", dependencies=[ADMIN_REQUIRED])
    def list_viewers():
        return {"ok": True, "viewers": [
            {"username": user.username}
            for user in get_auth_db().list_users()
            if user.role == "viewer" and user.is_active
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
    def submit_case_review(
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
        if not db.submit_review(
            case_id,
            reviewer=user.username,
            risk_level=risk_level,
            suspected_malignant=suspected_malignant,
            recommendation=recommendation,
        ):
            raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
        return {"ok": True, "review_status": "submitted"}

    @app.post("/api/cases/{case_id}/approve", dependencies=[ADMIN_REQUIRED])
    def approve_case(case_id: int):
        db = get_case_db()
        record = db.get_case(case_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
        if record.review_status not in {"submitted", "approved"}:
            raise HTTPException(status_code=409, detail="Ca bệnh chưa được nhân viên gửi duyệt.")
        if record.review_status == "approved" and record.public_code:
            return {"ok": True, "public_code": record.public_code}
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        for _ in range(5):
            public_code = "".join(secrets.choice(alphabet) for _ in range(10))
            try:
                if db.approve_case(
                    case_id,
                    reviewer=record.reviewed_by or "admin",
                    risk_level=record.risk_level,
                    suspected_malignant=record.suspected_malignant,
                    recommendation=record.recommendation,
                    public_code=public_code,
                ):
                    return {"ok": True, "public_code": public_code}
            except Exception:  # noqa: S112 — retry loop for public code generation
                continue
        raise HTTPException(status_code=500, detail="Không thể tạo mã đọc kết quả, vui lòng thử lại.")

    @app.post("/api/cases/{case_id}/link-patient", dependencies=[ADMIN_REQUIRED])
    def link_case_patient(case_id: int, username: str = Form(...)):
        patient = next(
            (user for user in get_auth_db().list_users() if user.username.casefold() == username.strip().casefold()),
            None,
        )
        if patient is None or patient.role != "viewer" or not patient.is_active:
            raise HTTPException(status_code=400, detail="Hãy chọn tài khoản người dùng đang hoạt động.")
        db = get_case_db()
        record = db.get_case(case_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
        if record.review_status != "approved":
            raise HTTPException(status_code=409, detail="Chỉ gán tài khoản cho ca đã được admin duyệt.")
        if not db.link_patient(case_id, patient.username):
            raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
        return {"ok": True}

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
                "clinical_assessment_available": True,
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
                "clinical_assessment_available": True,
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
        if not _can_read_case(record, user):
            raise HTTPException(status_code=403, detail="Bạn không được xem ảnh của ca bệnh này.")
        image_path = record.processed_image_path
        from pathlib import Path as PathLib
        image_path_obj = PathLib(image_path).resolve()
        if not image_path_obj.is_relative_to(OUTPUT_DIR.resolve()) or not image_path_obj.is_file():
            raise HTTPException(status_code=404, detail="Không tìm thấy ảnh đã xử lý.")
        return FileResponse(image_path_obj)

    @app.get("/api/cases/{case_id}/pdf")
    def download_case_pdf(case_id: int, request: Request):
        record = get_case_db().get_case(case_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy ca bệnh.")
        user = request.state.current_user
        if not _can_read_case(record, user):
            raise HTTPException(status_code=403, detail="Bạn không được xuất ca bệnh này.")
        try:
            pdf_path = export_case_pdf(OUTPUT_DIR / "medical" / "reports", build_case_export_payload(record))
        except ImportError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        return FileResponse(pdf_path, filename=pdf_path.name, media_type="application/pdf")
