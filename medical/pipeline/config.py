"""Section 4-5: Configuration builders & validation.

Xây dựng và kiểm tra cấu hình MedicalImageAnalyzer từ YAML settings.
"""
from __future__ import annotations

from pathlib import Path

from app.chat_ui.paths import PROJECT_ROOT
from medical.pipeline.dataclasses import MedicalImageAnalyzerConfig
from utils.file_utils import load_yaml


def _project_path(value: str | Path) -> Path:
    """Đường dẫn tương đối trong config được gắn với project root.

    Nếu không, khi app chạy với CWD khác (shortcut desktop, service) model và
    kết quả phân tích sẽ bị ghi nhầm chỗ.
    """
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_default_medical_analyzer_config() -> MedicalImageAnalyzerConfig:
    try:
        settings = load_yaml(_project_path("config/medical_settings.yaml")).get("medical", {})
    except FileNotFoundError:
        raise FileNotFoundError(
            "Không tìm thấy config/medical_settings.yaml. "
            "Hay chay tu thu muc goc cua project OncoVision."
        )
    if not isinstance(settings, dict):
        settings = {}
    advanced = settings.get("advanced", {})
    if not isinstance(advanced, dict):
        advanced = {}
    configured_model = _project_path(settings.get("model", "models/pretrained/brain_classifier.pt"))
    brain_model = settings.get("brain_model")
    detector_model = settings.get("yolo_model_path")
    if detector_model is None:
        detector_model = "models/pretrained/brain_detector.pt"
    volume_model = _project_path(settings.get("volume_model_path", "models/pretrained/brain_volume_model.pt2"))
    legacy_volume_model = volume_model.with_suffix(".pt")
    if volume_model.suffix.lower() == ".pt2" and not volume_model.is_file() and legacy_volume_model.is_file():
        volume_model = legacy_volume_model
    volume_manifest = _project_path(settings.get("volume_manifest_path", "models/pretrained/brain_volume_model.json"))
    return MedicalImageAnalyzerConfig(
        model_path=configured_model,
        working_dir=_project_path(settings.get("output_root", "output/medical")),
        reports_dir=_project_path(settings.get("reports_dir", "output/medical/reports")),
        processed_dir=_project_path(settings.get("processed_dir", "output/medical/normalized_images")),
        overlay_dir=_project_path(settings.get("overlay_dir", "output/medical/processed_images")),
        brain_model_path=_project_path(brain_model) if brain_model else None,
        modality_model_path=_project_path(settings["modality_model"]) if settings.get("modality_model") else None,
        fallback_model_path=_project_path(settings["fallback_model"]) if settings.get("fallback_model") else None,
        allow_fallback_model=bool(settings.get("allow_fallback_model", False)),
        image_size=int(settings.get("image_size", 320)),
        conf_threshold=float(settings.get("conf_threshold", 0.25)),
        classify_high_risk_threshold=float(settings.get("classify_high_risk_threshold", 0.75)),
        classify_medium_risk_threshold=float(settings.get("classify_medium_risk_threshold", 0.45)),
        certainty_threshold=float(settings.get("certainty_threshold", 0.55)),
        validation_min_confidence=float(settings.get("validation_min_confidence", 0.70)),
        validation_allowed_extensions=tuple(
            settings.get("validation_allowed_extensions", [".jpg", ".jpeg", ".png", ".dcm", ".nii", ".nii.gz"])
        ),
        cnn_image_size=int(settings.get("cnn_image_size", 224)),
        cnn_tta=bool(settings.get("cnn_tta", True)),
        analyze_topk=int(settings.get("analyze_topk", 3)),
        yolo_model_path=_project_path(detector_model),
        volume_model_path=volume_model,
        volume_manifest_path=volume_manifest,
        max_volume_voxels=int(settings.get("max_volume_voxels", 256_000_000)),
        clinical_validation_report_path=(
            _project_path(settings["clinical_validation_report"])
            if settings.get("clinical_validation_report")
            else None
        ),
        target_model_root=(
            None
            if not settings.get("target_model_root")
            else _project_path(settings["target_model_root"])
        ),
        ensemble_yolo_weight=float(settings.get("ensemble_yolo_weight", 0.4)),
        ensemble_cnn_weight=float(settings.get("ensemble_cnn_weight", 0.6)),
        detection_consistency_threshold=float(settings.get("detection_consistency_threshold", 0.5)),
        modality_tuning=settings.get("modality_tuning", {}) if isinstance(settings.get("modality_tuning", {}), dict) else {},
        enable_segmentation_roi=bool(advanced.get("enable_segmentation_roi", False)),
        segmentation_roi_margin=int(advanced.get("segmentation_roi_margin", 12)),
        enable_mc_dropout=bool(advanced.get("enable_mc_dropout", False)),
        mc_dropout_samples=int(advanced.get("mc_dropout_samples", 20)),
        enable_advanced_preprocessing=bool(advanced.get("enable_advanced_preprocessing", False)),
        video_sample_frames=int(settings.get("video_sample_frames", 12)),
        demo_mode=bool(settings.get("demo_mode", False)),
    )


def validate_medical_analyzer_config(config: MedicalImageAnalyzerConfig) -> list[str]:
    issues: list[str] = []
    if config.image_size <= 0:
        issues.append("image_size phai lon hon 0.")
    if not 0.0 < config.conf_threshold < 1.0:
        issues.append("conf_threshold phai nam trong khoang (0, 1).")
    if not 0.0 < config.classify_medium_risk_threshold <= config.classify_high_risk_threshold <= 1.0:
        issues.append("ngưỡng nguy cơ không hợp lệ.")
    if not 0.0 < config.certainty_threshold <= 1.0:
        issues.append("certainty_threshold phải nằm trong khoảng (0, 1].")
    if not 0.0 < config.validation_min_confidence <= 1.0:
        issues.append("validation_min_confidence phải nằm trong khoảng (0, 1].")
    if not config.validation_allowed_extensions:
        issues.append("validation_allowed_extensions không được để trống.")
    if not config.cnn_image_size > 0:
        issues.append("cnn_image_size phải lớn hơn 0.")
    if not 0.0 <= config.detection_consistency_threshold <= 1.0:
        issues.append("detection_consistency_threshold phải nằm trong khoảng [0, 1].")
    if config.segmentation_roi_margin < 0:
        issues.append("segmentation_roi_margin không được âm.")
    if config.mc_dropout_samples <= 0:
        issues.append("mc_dropout_samples phải lớn hơn 0.")
    if config.max_volume_voxels <= 0:
        issues.append("max_volume_voxels phải lớn hơn 0.")
    if config.video_sample_frames <= 0:
        issues.append("video_sample_frames không được âm.")
    total_ensemble_weight = config.ensemble_yolo_weight + config.ensemble_cnn_weight
    if not config.ensemble_yolo_weight >= 0.0 or not config.ensemble_cnn_weight >= 0.0:
        issues.append("ensemble weights không được âm.")
    if total_ensemble_weight <= 0.0:
        issues.append("Tổng ensemble_yolo_weight và ensemble_cnn_weight phải lớn hơn 0.")
    if not config.model_path:
        issues.append("model_path không được để trống.")
    if config.allow_fallback_model and config.fallback_model_path is None:
        issues.append("allow_fallback_model dang bat nhung fallback_model_path bi thieu.")
    return issues
