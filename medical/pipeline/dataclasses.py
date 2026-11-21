"""Section 2: Data classes & protocols.

Định nghĩa các dataclass và Protocol dùng trong pipeline y khoa:
- DetectorBackend: Protocol cho backend detector (YOLO, v.v.)
- MedicalImageAnalyzerConfig: Cấu hình cho MedicalImageAnalyzer
- DetectionFinding: Kết quả phát hiện đơn lẻ
- PipelineStageResult: Trạng thái từng stage trong pipeline
- MedicalAnalysisResult: Kết quả phân tích hoàn chỉnh
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class DetectorBackend(Protocol):
    def predict(self, source: Any, **kwargs) -> list[Any]:
        ...


@dataclass(frozen=True)
class MedicalImageAnalyzerConfig:
    model_path: Path
    working_dir: Path
    reports_dir: Path
    processed_dir: Path
    overlay_dir: Path
    brain_model_path: Path | None = None
    modality_model_path: Path | None = None
    fallback_model_path: Path | None = None
    allow_fallback_model: bool = False
    image_size: int = 320
    conf_threshold: float = 0.25
    classify_high_risk_threshold: float = 0.75
    classify_medium_risk_threshold: float = 0.45
    certainty_threshold: float = 0.55
    validation_min_confidence: float = 0.70
    validation_allowed_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".dcm",
        ".nii",
        ".nii.gz",
    )
    cnn_image_size: int = 224
    cnn_tta: bool = True
    analyze_topk: int = 3
    yolo_model_path: Path | None = None
    volume_model_path: Path | None = None
    volume_manifest_path: Path | None = None
    max_volume_voxels: int = 256_000_000
    clinical_validation_report_path: Path | None = None
    target_model_root: Path | None = None  # None = auto-detect (works with exe)
    ensemble_yolo_weight: float = 0.4
    ensemble_cnn_weight: float = 0.6
    detection_consistency_threshold: float = 0.5
    modality_tuning: dict[str, Any] = field(default_factory=dict)
    enable_segmentation_roi: bool = False
    segmentation_roi_margin: int = 12
    enable_mc_dropout: bool = False
    mc_dropout_samples: int = 20
    enable_advanced_preprocessing: bool = False
    max_cached_images: int = 100
    video_sample_frames: int = 12
    # Demo mode: allows analysis without real models (returns simulated results)
    demo_mode: bool = False

    def with_model_path(self, model_path: str | Path) -> MedicalImageAnalyzerConfig:
        from dataclasses import replace
        return replace(self, model_path=Path(model_path))


@dataclass(frozen=True)
class DetectionFinding:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int] | None
    segmentation: tuple[tuple[int, int], ...] | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    # Đánh dấu bbox có đáng tin không. False khi tọa độ bị clamp nhiều (dấu hiệu lỗi scale).
    bbox_reliable: bool = True
    # Kích thước ảnh mà bbox áp dụng (để phân biệt hệ canvas vs hệ ảnh gốc)
    bbox_image_size: tuple[int, int] | None = None
    # Tọa độ bbox trong hệ ảnh gốc (nếu khác bbox canvas)
    bbox_source: tuple[int, int, int, int] | None = None
    bbox_source_image_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class PipelineStageResult:
    stage: str
    status: str
    confidence: float
    message: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class MedicalAnalysisResult:
    case_id: int | None
    patient_code: str
    source_image: Path
    normalized_image: Path
    processed_image: Path
    report_json_path: Path
    report_md_path: Path
    detections: list[DetectionFinding]
    risk_level: str
    suspected_malignant: bool
    recommendation: str
    disclaimer: str
    average_confidence: float
    model_name: str
    quality_warnings: list[str]
    stage_confidences: dict[str, float] = field(default_factory=dict)
    stage_messages: dict[str, str] = field(default_factory=dict)
    gradcam_overlays: list[str] = field(default_factory=list)
    deidentified_source: Path | None = None
    dicom_info: dict[str, str] | None = None
    analysis_time_seconds: float = 0.0
    modality: str | None = None
    body_region: str | None = None
    input_dimension: str = "2d"
    volume_findings: list[dict[str, Any]] = field(default_factory=list)
    volume_shape: tuple[int, ...] | None = None
    volume_spacing_xyz_mm: tuple[float, float, float] | None = None
    volume_coordinate_system: str | None = None
    segmentation_mask_path: Path | None = None
    analyzed_frame_count: int = 1
    representative_frame_index: int | None = None
    volume_mask_path: Path | None = None
