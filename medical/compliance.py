from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent

logger = logging.getLogger(__name__)

MEDICAL_DISCLAIMER = dedent(
    """
    Cảnh báo y khoa:
    Hệ thống này chỉ hỗ trợ sàng lọc và nghiên cứu hình ảnh y khoa.
    Kết quả phân tích tự động không được dùng để tự chẩn đoán, kê đơn hoặc thay thế ý kiến bác sĩ.
    Mọi trường hợp nghi ngờ ung thư cần được đánh giá bởi bác sĩ chuyên khoa và xét nghiệm bổ sung.
    """
).strip()


from app.chat_ui.paths import PROJECT_ROOT

# Audit trail cho inference requests
AUDIT_LOG_FILE = PROJECT_ROOT / "output/logs/audit.json.log"


class AuditTrail:
    """Audit trail cho mọi inference request — phục vụ compliance và debugging.

    Mỗi entry bao gồm:
    - timestamp: Thời gian request
    - user: Người dùng (nếu có auth)
    - action: Hành động (inference, export, login, v.v.)
    - target_key: Nhóm ung thư được phân tích
    - model_version: Version của model dùng
    - result_summary: Tóm tắt kết quả
    - metadata: Thông tin bổ sung
    """

    def __init__(self, log_file: Path | None = None) -> None:
        self._log_file = log_file or AUDIT_LOG_FILE
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        action: str,
        user: str | None = None,
        target_key: str | None = None,
        model_version: str | None = None,
        result_summary: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Ghi một entry vào audit trail."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "user": user or "anonymous",
            "action": action,
        }
        if target_key:
            entry["target_key"] = target_key
        if model_version:
            entry["model_version"] = model_version
        if result_summary:
            entry["result_summary"] = result_summary
        if metadata:
            entry["metadata"] = metadata

        try:
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning("Không thể ghi audit log: %s", e)

    def log_inference(
        self,
        *,
        user: str | None = None,
        target_key: str,
        model_path: str,
        classification: str | None = None,
        confidence: float | None = None,
        detections_count: int | None = None,
        image_path: str | None = None,
    ) -> None:
        """Ghi audit log cho inference request."""
        summary_parts = []
        if classification:
            summary_parts.append(f"classified={classification}")
        if confidence is not None:
            summary_parts.append(f"confidence={confidence:.3f}")
        if detections_count is not None:
            summary_parts.append(f"detections={detections_count}")

        self.log(
            action="inference",
            user=user,
            target_key=target_key,
            model_version=Path(model_path).name,
            result_summary=", ".join(summary_parts) if summary_parts else None,
            metadata={"image_path": image_path, "model_path": model_path},
        )

    def log_login(self, *, user: str, success: bool, ip_address: str | None = None) -> None:
        """Ghi audit log cho login attempt."""
        self.log(
            action="login",
            user=user,
            result_summary="success" if success else "failed",
            metadata={"ip_address": ip_address} if ip_address else None,
        )

    def log_export(self, *, user: str | None = None, case_id: int | None = None, format: str = "pdf") -> None:
        """Ghi audit log cho export action."""
        self.log(
            action="export",
            user=user,
            metadata={"case_id": case_id, "format": format},
        )


# Global audit trail instance
_audit_trail: AuditTrail | None = None


def get_audit_trail() -> AuditTrail:
    """Lấy global audit trail instance (singleton)."""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail


_DICOM_PII_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "MilitaryRank",
    "EthnicGroup",
    "Occupation",
    "PatientComments",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "ReferringPhysicianIdentificationSequence",
    "StudyDescription",
    "SeriesDescription",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "AccessionNumber",
    "StudyID",
    "RequestingPhysician",
    "RequestingService",
    "RequestedProcedureDescription",
    "ScheduledProcedureStepDescription",
    "ScheduledPerformingPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "StudyComments",
    "DeviceUID",
    "IssuerOfPatientID",
    "IssuerOfPatientIDQualifiersSequence",
    "OrderEnteredBy",
    "OrderEntererLocation",
    "ReasonForTheRequestedProcedure",
    "ReasonForStudy",
    "RequestAttributesSequence",
    "ConfidentialityConstraintOnPatientDataDescription",
    "OtherPatientIDs",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
    "PatientBirthName",
    "PatientInsurancePlanCodeSequence",
    "PatientPrimaryLanguageCodeSequence",
    "PatientAddress",
    "PatientTelecomInformation",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientSize",
    "PatientWeight",
    "PatientComments",
    "IssuerOfPatientIDQualifiersSequence",
    "SourcePatientGroupIdentificationSequence",
    "GroupOfPatientsIdentificationSequence",
    "RequestingService",
    "RequestedProcedurePriority",
    "ScheduledProcedureStepID",
    "ScheduledProcedureStepLocation",
    "ScheduledProcedureStepStartDateTime",
    "ScheduledPerformingPhysicianIdentificationSequence",
    "PerformingPhysicianIdentificationSequence",
    "ResultsComments",
    "RequestingServiceCodeSequence",
    "RequestedProcedureCodeSequence",
    "RequestedContrastAgent",
    "RetrieveURI",
    "StorageMediaFileSetUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "VerifyingObserverName",
    "VerifyingObserverSequence",
    "VerifyingObserverIdentificationCodeSequence",
    "AuthorObserverSequence",
    "ParticipantSequence",
    "CustodialOrganizationSequence",
    "ConsentForClinicalTrialUseSequence",
    "PersonIdentificationCodeSequence",
    "PersonName",
    "PersonAddress",
    "PersonTelephoneNumbers",
    "PersonTelecomInformation",
    "InstitutionAddress",
    "ReferringPhysicianIdentificationSequence",
    "ConsultingPhysicianName",
    "ConsultingPhysicianIdentificationSequence",
    "RequestingServiceCodeSequence",
    "ReasonForStudy",
    "ReasonForRequestedProcedureCodeSequence",
    "RequestAttributesSequence",
    "ScheduledProcedureStepSequence",
    "ScheduledProcedureStepID",
    "ScheduledProcedureStepStartDateTime",
    "ScheduledProcedureStepDescription",
    "ScheduledProcedureStepLocation",
    "ScheduledPerformingPhysicianName",
    "ScheduledPerformingPhysicianIdentificationSequence",
    "ScheduledStationName",
    "ScheduledStationGeographicLocationCodeSequence",
    "PerformedProcedureStepID",
    "PerformedProcedureStepStartDateTime",
    "PerformedProcedureStepEndDateTime",
    "PerformedStationName",
    "PerformedStationGeographicLocationCodeSequence",
    "RequestAttributesSequence",
    "CommentsOnThePerformedProcedureStep",
    "RequestingService",
    "RequestedProcedureID",
    "RequestedProcedureDescription",
    "RequestedProcedureCodeSequence",
    "RequestedContrastAgent",
    "StudyInstanceUID",
    "RequestAttributesSequence",
    "ConfidentialityCode",
    "RetrieveAETitle",
    "StorageMediaFileSetUID",
]


def _blank_tag(ds, tag_name: str) -> None:
    if tag_name in ds:
        tag = ds[tag_name]
        vr = tag.VR if hasattr(tag, "VR") else None
        if vr in {"PN", "LO", "LT", "SH", "ST", "UT"}:
            ds[tag_name].value = "ANONYMOUS"
        elif vr == "DA":
            ds[tag_name].value = "00010101"
        elif vr == "TM":
            ds[tag_name].value = "000000"
        elif vr == "DT":
            ds[tag_name].value = "00010101000000"
        elif vr == "AS":
            ds[tag_name].value = ""
        elif vr == "US" or vr == "UL":
            ds[tag_name].value = 0
        elif vr == "IS":
            ds[tag_name].value = "0"
        else:
            ds[tag_name].value = ""


def deidentify_dicom(ds) -> None:
    """Remove or blank PII tags from a pydicom Dataset in-place.

    UID định danh (study/series/instance) không được để trống: DICOM yêu cầu UID hợp lệ,
    nên thay bằng UID mới. SOPClassUID là loại ảnh, không phải PII → giữ nguyên.
    """
    from pydicom.uid import generate_uid

    for tag_name in _DICOM_PII_TAGS:
        _blank_tag(ds, tag_name)
    for tag_name in ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID"):
        if tag_name in ds:
            ds[tag_name].value = generate_uid()
    file_meta = getattr(ds, "file_meta", None)
    if file_meta is not None and "SOPInstanceUID" in ds:
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID


def deidentify_dicom_file(input_path: str | Path, output_path: str | Path) -> None:
    import pydicom

    ds = pydicom.dcmread(str(input_path), force=True)
    deidentify_dicom(ds)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path), enforce_file_format=True)


def deidentify_dicom_series(input_dir: str | Path, output_dir: str | Path) -> ComplianceReport:
    report = ComplianceReport()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for dcm_file in sorted(input_path.rglob("*.dcm")):
        relative = dcm_file.relative_to(input_path)
        target = output_path / relative
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            deidentify_dicom_file(dcm_file, target)
            report.record_deidentified(target)
        except Exception as exc:
            report.record_error(f"{dcm_file}: {exc}")
            report.record_skipped(dcm_file)

    return report


class ComplianceReport:
    def __init__(self) -> None:
        self.deidentified_files: list[Path] = []
        self.skipped_files: list[Path] = []
        self.errors: list[str] = []

    def record_deidentified(self, path: Path) -> None:
        self.deidentified_files.append(path)

    def record_skipped(self, path: Path) -> None:
        self.skipped_files.append(path)

    def record_error(self, message: str) -> None:
        self.errors.append(message)

    @property
    def summary(self) -> str:
        return (
            f"De-identification: {len(self.deidentified_files)} files processed, "
            f"{len(self.skipped_files)} skipped, {len(self.errors)} errors."
        )
