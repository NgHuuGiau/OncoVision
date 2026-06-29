from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from medical.output_management import medical_output_directories
from medical.cancer_catalog import list_common_cancer_targets, supported_cancer_labels
from medical.model_policy import resolve_medical_runtime_model_path
from medical.pipeline import build_default_medical_analyzer_config
from medical.status_helpers import count_files
from medical.training import medical_training_paths


@dataclass(frozen=True)
class MedicalSystemStatus:
    configured_model_path: Path
    resolved_model_path: Path | None
    allow_fallback_model: bool
    using_fallback_model: bool
    model_ready: bool
    model_message: str
    dataset_root: Path
    data_yaml_path: Path
    raw_images: int
    raw_labels: int
    train_images: int
    val_images: int
    test_images: int
    report_files: int
    normalized_files: int
    overlay_files: int
    export_files: int
    case_db_path: Path
    case_count: int
    screening_targets: tuple[tuple[str, bool], ...]
    analyzed_cancers: tuple[str, ...]

    @property
    def dataset_initialized(self) -> bool:
        return self.data_yaml_path.exists()

    @property
    def raw_dataset_ready(self) -> bool:
        return self.raw_images > 0 and self.raw_labels > 0

    @property
    def processed_dataset_ready(self) -> bool:
        return self.train_images > 0 and self.val_images > 0


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


def get_medical_system_status() -> MedicalSystemStatus:
    config = build_default_medical_analyzer_config()
    training_paths = medical_training_paths()
    report_dir, normalized_dir, overlay_dir, export_dir = medical_output_directories()
    case_db_path = config.working_dir / "medical_cases.db"

    try:
        resolved_model_path = resolve_medical_runtime_model_path(config)
        using_fallback_model = resolved_model_path.resolve(strict=False) != config.model_path.resolve(strict=False)
        model_ready = True
        if using_fallback_model:
            model_message = f"Dang dung fallback model: {resolved_model_path.name}"
        else:
            model_message = f"Da san sang voi model: {resolved_model_path.name}"
    except Exception as exc:
        resolved_model_path = None
        using_fallback_model = False
        model_ready = False
        model_message = str(exc)

    return MedicalSystemStatus(
        configured_model_path=config.model_path,
        resolved_model_path=resolved_model_path,
        allow_fallback_model=config.allow_fallback_model,
        using_fallback_model=using_fallback_model,
        model_ready=model_ready,
        model_message=model_message,
        dataset_root=training_paths.dataset_root,
        data_yaml_path=training_paths.data_yaml_path,
        raw_images=count_files(training_paths.raw_images_dir),
        raw_labels=count_files(training_paths.raw_labels_dir),
        train_images=count_files(training_paths.processed_images_dir / "train"),
        val_images=count_files(training_paths.processed_images_dir / "val"),
        test_images=count_files(training_paths.processed_images_dir / "test"),
        report_files=count_files(report_dir),
        normalized_files=count_files(normalized_dir),
        overlay_files=count_files(overlay_dir),
        export_files=count_files(export_dir),
        case_db_path=case_db_path,
        case_count=_count_cases(case_db_path),
        screening_targets=tuple((target.label, target.model_ready) for target in list_common_cancer_targets()),
        analyzed_cancers=tuple(supported_cancer_labels()),
    )


def recommended_medical_commands(status: MedicalSystemStatus) -> list[str]:
    commands: list[str] = []
    if not status.dataset_initialized:
        commands.append("python run_medical.py init-dataset")
    if status.raw_dataset_ready and not status.processed_dataset_ready:
        commands.append("python run_medical.py split-dataset")
    elif not status.raw_dataset_ready:
        commands.append("python run_medical.py audit-dataset")
    if not status.model_ready:
        commands.append("python run_medical.py train-all")
    else:
        commands.append("python run_medical.py validate")
    if any((status.report_files, status.normalized_files, status.overlay_files, status.export_files)):
        commands.append("python run_chat.py --cleanup-output --older-than-days 30")
    return list(dict.fromkeys(commands))
