from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from medical.cancer_catalog import COMMON_CANCER_TARGETS
from medical.cancer_model_registry import (
    modality_key,
    probe_runtime_capability,
    target_model_paths,
)
from medical.clinical_validation import clinical_validation_status
from medical.model_policy import resolve_medical_runtime_model_path
from medical.pipeline import build_default_medical_analyzer_config
from medical.status_helpers import count_files
from medical.volume_inference import TorchVolumeRuntime


@dataclass(frozen=True)
class MedicalSystemStatus:
    configured_model_path: Path
    resolved_model_path: Path | None
    allow_fallback_model: bool
    using_fallback_model: bool
    model_ready: bool
    model_message: str
    report_files: int
    normalized_files: int
    overlay_files: int
    export_files: int
    case_db_path: Path
    case_count: int
    screening_targets: tuple[tuple[str, bool], ...]
    analyzed_cancers: tuple[str, ...]
    analyzed_modalities: tuple[str, ...]
    classification_ready: bool = False
    localization_ready: bool = False
    volume_ready: bool = False
    clinical_validation_ready: bool = False
    all_targets_ready: bool = False
    target_capabilities: dict[str, dict[str, bool]] = field(default_factory=dict)


def _count_cases(case_db_path: Path) -> int:
    if not case_db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(case_db_path)
        try:
            row = conn.execute("SELECT COUNT(*) FROM medical_cases").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return 0


def _ready_modalities(ready_targets) -> tuple[str, ...]:
    return tuple(dict.fromkeys(modality for target in ready_targets for modality in target.modalities))


def get_medical_system_status() -> MedicalSystemStatus:
    config = build_default_medical_analyzer_config()
    report_dir = Path(config.working_dir) / "reports"
    normalized_dir = Path(config.working_dir) / "normalized_images"
    overlay_dir = Path(config.working_dir) / "processed_images"
    export_dir = Path(config.working_dir) / "exports"
    case_db_path = Path(config.working_dir).parent / "onco.db"
    try:
        resolved_model_path = resolve_medical_runtime_model_path(config)
        resolved_model_abs = resolved_model_path.resolve(strict=False)
        configured_model_abs = config.model_path.resolve(strict=False)
        fallback_model_abs = (
            Path(config.fallback_model_path).resolve(strict=False)
            if config.allow_fallback_model and config.fallback_model_path is not None
            else None
        )
        using_fallback_model = fallback_model_abs is not None and resolved_model_abs == fallback_model_abs
        model_ready = True
        if resolved_model_abs == configured_model_abs:
            model_message = f"Sẵn sàng với model: {resolved_model_path}"
        elif using_fallback_model:
            model_message = f"Đang dùng model dự phòng: {resolved_model_path}"
        else:
            model_message = f"Đang dùng model y khoa: {resolved_model_path}"
    except Exception as exc:
        resolved_model_path = None
        using_fallback_model = False
        brain_model_path = Path(config.brain_model_path) if config.brain_model_path else None
        if brain_model_path is not None and brain_model_path.exists():
            resolved_model_path = brain_model_path
            model_ready = True
            model_message = f"Chỉ phân tích ảnh vùng não bằng model: {brain_model_path.name}"
        else:
            model_ready = False
            model_message = str(exc)

    target_root = Path(getattr(config, "target_model_root", "models/pretrained/cancers"))
    target_capabilities: dict[str, dict[str, bool]] = {}
    resolved_target_paths: dict[str, dict[str, Path]] = {}
    target_messages: list[str] = []
    for target in COMMON_CANCER_TARGETS:
        candidates = [target_model_paths(target.key, target_root)]
        candidates.extend(target_model_paths(target.key, target_root, modality) for modality in target.modalities)
        unique_candidates = {str(paths["directory"]): paths for paths in candidates}
        paths = next(iter(unique_candidates.values()))
        if target.key == "brain":
            legacy_paths = {
                "classifier": getattr(config, "brain_model_path", None) or resolved_model_path,
                "detector": getattr(config, "yolo_model_path", None),
                "volume_model": getattr(config, "volume_model_path", None),
                "volume_manifest": getattr(config, "volume_manifest_path", None),
                "clinical_validation": getattr(config, "clinical_validation_report_path", None),
            }
            for name, candidate in legacy_paths.items():
                if not paths[name].is_file() and candidate is not None:
                    paths[name] = Path(candidate)

        candidate_paths = list(unique_candidates.values())
        if paths not in candidate_paths:
            candidate_paths.append(paths)
        classification = False
        localization = False
        organ_root_ready = False
        modality_slot_ready = False
        for index, item in enumerate(candidate_paths):
            for name in ("classifier", "detector"):
                path = item[name]
                if not path.is_file():
                    continue
                # File ngoài target_root (model legacy của config) giữ kiểm theo tồn tại;
                # file trong cancers/ phải nạp được runtime thật mới tính sẵn sàng.
                if path.is_relative_to(target_root):
                    ok, reason = probe_runtime_capability(path, name)
                else:
                    ok, reason = True, ""
                if ok:
                    if name == "classifier":
                        classification = True
                    else:
                        localization = True
                    if index == 0:
                        organ_root_ready = True
                    else:
                        modality_slot_ready = True
                elif reason:
                    target_messages.append(f"{target.label}: {path.name} không chạy được runtime ({reason})")
        # Model đặt ở cấp cơ quan không được runtime dùng khi ảnh xác định được modality
        # (trừ não: có fallback legacy theo config — xem pipeline._target_paths).
        if organ_root_ready and not modality_slot_ready and target.key != "brain" and target.modalities:
            target_messages.append(
                f"{target.label}: model đang ở cấp cơ quan; ảnh có modality sẽ không dùng tới "
                f"(runtime tìm trong {target.key}/<modality>/, vd {target.key}/{modality_key(target.modalities[0])}/) — "
                "dùng scripts/fetch_model.py --modality để đặt đúng ô."
            )
        ready_2d = next(
            (item for item in candidate_paths if item["classifier"].is_file() or item["detector"].is_file()),
            None,
        )
        if ready_2d is not None:
            paths = ready_2d
        volume = False
        for item in candidate_paths:
            if not (item["volume_model"].is_file() and item["volume_manifest"].is_file()):
                continue
            try:
                runtime = TorchVolumeRuntime(item["volume_model"], item["volume_manifest"], device="cpu")
                if runtime.manifest.target == target.key:
                    volume = True
                    paths = item
                    break
            except Exception as exc:
                target_messages.append(f"{target.label}: model 3D không hợp lệ ({exc})")

        clinical = False
        for item in candidate_paths:
            for validation_model in (item["volume_model"], item["classifier"], item["detector"]):
                if not validation_model.is_file():
                    continue
                clinical, _ = clinical_validation_status(item["clinical_validation"], validation_model)
                if clinical:
                    break
            if clinical:
                break
        target_capabilities[target.key] = {
            "runtime_ready": classification or localization or volume,
            "classification_ready": classification,
            "localization_ready": localization,
            "volume_ready": volume,
            "clinical_validation_ready": clinical,
        }
        resolved_target_paths[target.key] = paths

    ready_targets = [target for target in COMMON_CANCER_TARGETS if target_capabilities[target.key]["runtime_ready"]]
    classification_ready = any(item["classification_ready"] for item in target_capabilities.values())
    localization_ready = any(item["localization_ready"] for item in target_capabilities.values())
    volume_ready = any(item["volume_ready"] for item in target_capabilities.values())
    clinical_ready = any(item["clinical_validation_ready"] for item in target_capabilities.values())
    model_ready = bool(ready_targets)
    if resolved_model_path is None and ready_targets:
        first_paths = resolved_target_paths[ready_targets[0].key]
        resolved_model_path = next(
            path for path in (first_paths["classifier"], first_paths["detector"], first_paths["volume_model"])
            if path.is_file()
        )
    model_message = (
        f"{model_message}; {len(ready_targets)}/{len(COMMON_CANCER_TARGETS)} nhóm có model runtime; "
        f"định vị {'có' if localization_ready else 'thiếu'}; volume 3D {'có' if volume_ready else 'thiếu'}; "
        f"kiểm định độc lập {'có' if clinical_ready else 'thiếu'}"
    )
    if target_messages:
        model_message += "; " + "; ".join(target_messages)

    return MedicalSystemStatus(
        configured_model_path=config.model_path,
        resolved_model_path=resolved_model_path,
        allow_fallback_model=config.allow_fallback_model,
        using_fallback_model=using_fallback_model,
        model_ready=model_ready,
        model_message=model_message,
        report_files=count_files(report_dir),
        normalized_files=count_files(normalized_dir),
        overlay_files=count_files(overlay_dir),
        export_files=count_files(export_dir),
        case_db_path=case_db_path,
        case_count=_count_cases(case_db_path),
        screening_targets=tuple(
            (target.label, target_capabilities[target.key]["runtime_ready"]) for target in COMMON_CANCER_TARGETS
        ),
        analyzed_cancers=tuple(target.label for target in ready_targets),
        analyzed_modalities=_ready_modalities(ready_targets),
        classification_ready=classification_ready,
        localization_ready=localization_ready,
        volume_ready=volume_ready,
        clinical_validation_ready=clinical_ready,
        all_targets_ready=len(ready_targets) == len(COMMON_CANCER_TARGETS),
        target_capabilities=target_capabilities,
    )


def recommended_medical_commands(status: MedicalSystemStatus) -> list[str]:
    commands: list[str] = []
    if not status.model_ready:
        commands.append(f"Bổ sung model suy luận vào models/pretrained/ (đang cấu hình: {status.configured_model_path.name})")
    if any((status.report_files, status.normalized_files, status.overlay_files, status.export_files)):
        commands.append("python run_chat.py --cleanup-output --older-than-days 30")
    if hasattr(status, "localization_ready") and not status.localization_ready:
        commands.append("Bổ sung detector.pt cho từng nhóm cần tọa độ trong models/pretrained/cancers/<key>/")
    if hasattr(status, "volume_ready") and not status.volume_ready:
        commands.append("Bổ sung volume.pt2 và volume.json cho từng nhóm MRI/CT 3D")
    if hasattr(status, "clinical_validation_ready") and not status.clinical_validation_ready:
        commands.append("Chạy validate-cohort và gửi báo cáo cho đơn vị lâm sàng độc lập phê duyệt")
    return commands
