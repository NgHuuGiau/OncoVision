from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent

MEDICAL_DISCLAIMER = dedent(
    """
    Cảnh báo y khoa:
    Hệ thống này chỉ hỗ trợ sàng lọc và nghiên cứu hình ảnh y khoa.
    Kết quả phân tích tự động không được dùng để tự chẩn đoán, kê đơn hoặc thay thế ý kiến bác sĩ.
    Mọi trường hợp nghi ngờ ung thư cần được đánh giá bởi bác sĩ chuyên khoa và xét nghiệm bổ sung.
    """
).strip()


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
    "DeviceAddress",
    "InstitutionalEntityName",
    "InstitutionalEntityType",
    "InstitutionalEntityIdentificationSequence",
    "IssuerOfPatientID",
    "IssuerOfPatientIDQualifiersSequence",
    "OrderEnteredBy",
    "OrderEntererLocation",
    "OrderCallbackTelephoneNumber",
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
    "PatientTelephoneNumber",
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
    "ScheduledProcedureStepModality",
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
    "SOPClassUID",
    "VerifyingObserverName",
    "VerifyingObserverSequence",
    "VerifyingObserverIdentificationCodeSequence",
    "AuthorObserverSequence",
    "ParticipantSequence",
    "CustodialOrganizationSequence",
    "ProcedureExpiryDate",
    "ConsentForClinicalTrialUseSequence",
    "PersonIdentificationCodeSequence",
    "PersonName",
    "PersonAddress",
    "PersonTelephoneNumbers",
    "PersonTelecomInformation",
    "InstitutionAddress",
    "InstitutionalEntityName",
    "InstitutionalEntityType",
    "InstitutionalEntityIdentificationSequence",
    "ReferringPhysicianIdentificationSequence",
    "ConsultingPhysicianName",
    "ConsultingPhysicianIdentificationSequence",
    "ProcedureIdentifierCodeSequence",
    "RequestingServiceCodeSequence",
    "ReasonForStudy",
    "ReasonForRequestedProcedureCodeSequence",
    "RequestAttributesSequence",
    "ScheduledProcedureStepSequence",
    "ScheduledProcedureStepID",
    "ScheduledProcedureStepStartDateTime",
    "ScheduledProcedureStepEndDateTime",
    "ScheduledProcedureStepModality",
    "ScheduledProcedureStepDescription",
    "ScheduledProcedureStepLocation",
    "ScheduledPerformingPhysicianName",
    "ScheduledPerformingPhysicianIdentificationSequence",
    "ScheduledStationName",
    "ScheduledStationClassTitle",
    "ScheduledStationGeographicLocationCodeSequence",
    "PerformedProcedureStepID",
    "PerformedProcedureStepStartDateTime",
    "PerformedProcedureStepEndDateTime",
    "PerformedProcedureStepType",
    "PerformedStationName",
    "PerformedStationClassTitle",
    "PerformedStationGeographicLocationCodeSequence",
    "RequestAttributesSequence",
    "CommentsOnThePerformedProcedureStep",
    "ProcedureStepSequence",
    "RequestingService",
    "RequestedProcedureID",
    "RequestedProcedureDescription",
    "RequestedProcedureCodeSequence",
    "RequestedContrastAgent",
    "StudyInstanceUID",
    "RequestAttributesSequence",
    "ConfidentialityCode",
    "EntryQualificationString",
    "RetrieveAETitle",
    "StorageMediaFileSetUID",
]


def _blank_tag(ds, tag_name: str) -> None:
    if tag_name in ds:
        tag = ds[tag_name]
        vr = tag.VR if hasattr(tag, "VR") else None
        if vr in {"PN", "LO", "LT", "SH", "ST", "UT", "PN"}:
            ds[tag_name].value = "ANONYMOUS"
        elif vr == "DA":
            ds[tag_name].value = "00010101"
        elif vr == "TM":
            ds[tag_name].value = "000000"
        elif vr == "DT":
            ds[tag_name].value = "00010101000000"
        elif vr == "AS":
            ds[tag_name].value = ""
        elif vr == "US":
            ds[tag_name].value = 0
        elif vr == "UL":
            ds[tag_name].value = 0
        elif vr == "IS":
            ds[tag_name].value = "0"
        else:
            ds[tag_name].value = ""


def deidentify_dicom(ds) -> None:
    """Remove or blank PII tags from a pydicom Dataset in-place."""
    for tag_name in _DICOM_PII_TAGS:
        _blank_tag(ds, tag_name)


def deidentify_dicom_file(input_path: str | Path, output_path: str | Path) -> None:
    import pydicom

    ds = pydicom.dcmread(str(input_path), force=True)
    deidentify_dicom(ds)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path), write_like_original=False)


def deidentify_dicom_series(input_dir: str | Path, output_dir: str | Path) -> "ComplianceReport":
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


def compute_dataset_hash(dataset_root: str | Path) -> str:
    root = Path(dataset_root)
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hasher.update(str(path.relative_to(root)).encode("utf-8"))
            hasher.update(b":")
            hasher.update(str(path.stat().st_size).encode("utf-8"))
            hasher.update(b"\n")
    return hasher.hexdigest()[:16]


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
