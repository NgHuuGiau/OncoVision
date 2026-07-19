from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol


import cv2
import numpy as np
from PIL import Image

from medical.classifier import MedicalClassifierModel, load_medical_classifier
from medical.cnn_classifier import MedicalCNNClassifierWrapper, is_cnn_classifier_path, load_cnn_classifier
from medical.compliance import MEDICAL_DISCLAIMER
from medical.dashboard import write_inference_dashboard
from medical.dataset import normalize_uploaded_image
from medical.model_policy import resolve_medical_runtime_model_path
from medical.model_versioning import read_model_manifest
from medical.reporting import build_artifact_stamp, write_case_report
from medical.validator import (
    ValidationResult,
    _DICOM_MODALITY_MAP,
    _load_modality_tuning_from_config,
    get_modality_tuning,
    validate_image,
)
from utils.draw_utils import draw_detection_results
from utils.file_utils import load_yaml


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
    cnn_backbone: str = "resnet50"
    cnn_image_size: int = 320
    cnn_batch_size: int = 16
    cnn_num_epochs: int = 30
    cnn_learning_rate: float = 0.0001
    cnn_dropout: float = 0.3
    cnn_early_stopping_patience: int = 7
    cnn_label_smoothing: float = 0.1
    cnn_mixed_precision: bool = True
    cnn_warmup_epochs: int = 3
    cnn_tta: bool = True
    analyze_topk: int = 3
    yolo_model_path: Path | None = None
    ensemble_yolo_weight: float = 0.4
    ensemble_cnn_weight: float = 0.6
    detection_consistency_threshold: float = 0.5
    modality_tuning: dict[str, Any] = field(default_factory=dict)
    enable_segmentation_roi: bool = False
    segmentation_roi_margin: int = 12
    enable_mc_dropout: bool = False
    mc_dropout_samples: int = 20
    enable_advanced_preprocessing: bool = False

    def with_model_path(self, model_path: str | Path) -> "MedicalImageAnalyzerConfig":
        return replace(self, model_path=Path(model_path))


@dataclass(frozen=True)
class DetectionFinding:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


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


def _coerce_detections(items: list[Any]) -> list[DetectionFinding]:
    detections: list[DetectionFinding] = []
    for item in items:
        if isinstance(item, DetectionFinding):
            detections.append(item)
            continue
        if isinstance(item, dict) and "label" in item and "confidence" in item:
            raw_bbox = item.get("bbox") or (0, 0, 0, 0)
            bbox = tuple(int(value) for value in raw_bbox)
            if len(bbox) != 4:
                bbox = (0, 0, 0, 0)
            detections.append(
                DetectionFinding(
                    label=str(item["label"]),
                    confidence=float(item["confidence"]),
                    bbox=bbox,
                )
            )
    return detections


def build_default_medical_analyzer_config() -> MedicalImageAnalyzerConfig:
    settings = load_yaml("config/medical_settings.yaml").get("medical", {})
    if not isinstance(settings, dict):
        settings = {}
    advanced = settings.get("advanced", {})
    if not isinstance(advanced, dict):
        advanced = {}
    configured_model = Path(settings.get("model", "medical_7_cancers.pt"))
    return MedicalImageAnalyzerConfig(
        model_path=configured_model,
        working_dir=Path(settings.get("output_root", "output/medical")),
        reports_dir=Path(settings.get("reports_dir", "output/medical/reports")),
        processed_dir=Path(settings.get("processed_dir", "output/medical/normalized_images")),
        overlay_dir=Path(settings.get("overlay_dir", "output/medical/processed_images")),
        fallback_model_path=Path(settings["fallback_model"]) if settings.get("fallback_model") else None,
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
        cnn_backbone=str(settings.get("cnn_backbone", "resnet50")),
        cnn_image_size=int(settings.get("cnn_image_size", 320)),
        cnn_batch_size=int(settings.get("cnn_batch_size", 16)),
        cnn_num_epochs=int(settings.get("cnn_num_epochs", 30)),
        cnn_learning_rate=float(settings.get("cnn_learning_rate", 0.0001)),
        cnn_dropout=float(settings.get("cnn_dropout", 0.3)),
        cnn_early_stopping_patience=int(settings.get("cnn_early_stopping_patience", 7)),
        cnn_label_smoothing=float(settings.get("cnn_label_smoothing", 0.1)),
        cnn_mixed_precision=bool(settings.get("cnn_mixed_precision", True)),
        cnn_warmup_epochs=int(settings.get("cnn_warmup_epochs", 3)),
        cnn_tta=bool(settings.get("cnn_tta", True)),
        analyze_topk=int(settings.get("analyze_topk", 3)),
        yolo_model_path=Path(settings["yolo_model_path"]) if settings.get("yolo_model_path") else None,
        ensemble_yolo_weight=float(settings.get("ensemble_yolo_weight", 0.4)),
        ensemble_cnn_weight=float(settings.get("ensemble_cnn_weight", 0.6)),
        detection_consistency_threshold=float(settings.get("detection_consistency_threshold", 0.5)),
        modality_tuning=settings.get("modality_tuning", {}) if isinstance(settings.get("modality_tuning", {}), dict) else {},
        enable_segmentation_roi=bool(advanced.get("enable_segmentation_roi", False)),
        segmentation_roi_margin=int(advanced.get("segmentation_roi_margin", 12)),
        enable_mc_dropout=bool(advanced.get("enable_mc_dropout", False)),
        mc_dropout_samples=int(advanced.get("mc_dropout_samples", 20)),
        enable_advanced_preprocessing=bool(advanced.get("enable_advanced_preprocessing", False)),
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
        issues.append("certainty_threshold phai nam trong khoang (0, 1].")
    if not 0.0 < config.validation_min_confidence <= 1.0:
        issues.append("validation_min_confidence phai nam trong khoang (0, 1].")
    if not config.validation_allowed_extensions:
        issues.append("validation_allowed_extensions khong duoc de trong.")
    if not config.cnn_image_size > 0:
        issues.append("cnn_image_size phai lon hon 0.")
    if not config.cnn_batch_size > 0:
        issues.append("cnn_batch_size phai lon hon 0.")
    if not config.cnn_num_epochs > 0:
        issues.append("cnn_num_epochs phai lon hon 0.")
    if not 0.0 < config.cnn_learning_rate <= 1.0:
        issues.append("cnn_learning_rate khong hop le.")
    if not 0.0 <= config.cnn_dropout < 1.0:
        issues.append("cnn_dropout khong hop le.")
    if config.cnn_early_stopping_patience <= 0:
        issues.append("cnn_early_stopping_patience phai lon hon 0.")
    if not 0.0 <= config.cnn_label_smoothing < 1.0:
        issues.append("cnn_label_smoothing khong hop le.")
    if config.cnn_warmup_epochs < 0:
        issues.append("cnn_warmup_epochs khong duoc am.")
    if not 0.0 <= config.detection_consistency_threshold <= 1.0:
        issues.append("detection_consistency_threshold phai nam trong khoang [0, 1].")
    if config.segmentation_roi_margin < 0:
        issues.append("segmentation_roi_margin khong duoc am.")
    if config.mc_dropout_samples <= 0:
        issues.append("mc_dropout_samples phai lon hon 0.")
    total_ensemble_weight = config.ensemble_yolo_weight + config.ensemble_cnn_weight
    if not config.ensemble_yolo_weight >= 0.0 or not config.ensemble_cnn_weight >= 0.0:
        issues.append("ensemble weights khong duoc am.")
    if total_ensemble_weight <= 0.0:
        issues.append("tong ensemble_yolo_weight va ensemble_cnn_weight phai lon hon 0.")
    if not config.model_path:
        issues.append("model_path khong duoc de trong.")
    if config.allow_fallback_model and config.fallback_model_path is None:
        issues.append("allow_fallback_model dang bat nhung fallback_model_path bi thieu.")
    return issues


class MedicalImageAnalyzer:
    def __init__(self, config: MedicalImageAnalyzerConfig | None = None, detector_backend: DetectorBackend | None = None) -> None:
        self.config = config or build_default_medical_analyzer_config()
        self._detector_backend = detector_backend
        self._classifier_model: MedicalClassifierModel | None = None
        self._dicom_modality_cache: dict[Path, str | None] = {}
        self._cnn_wrapper_cache: MedicalCNNClassifierWrapper | None = None
        self._cnn_wrapper_cache_path: Path | None = None
        self._roi_extractor: Any = None

    def analyze_image(self, image_path: str | Path, *, patient_code: str, case_id: int | None = None) -> MedicalAnalysisResult:
        self.ensure_ready()
        if self._detector_backend is None and self.config.yolo_model_path and Path(self.config.yolo_model_path).exists():
            try:
                self._detector_backend = self._load_default_backend()
            except Exception as exc:
                print(f"[Ensemble] Bo qua YOLO backend: {exc}")
        resolved_source = Path(image_path)
        validation = self.validate_input(resolved_source)
        if validation.status == "error":
            raise ValueError(f"{validation.error_code}: {validation.message}")

        deidentified_source = None
        if resolved_source.suffix.lower() == ".dcm" or resolved_source.is_dir():
            deidentified_source = self._deidentify_input(resolved_source)

        normalized_path = normalize_uploaded_image(deidentified_source or resolved_source, self.config.processed_dir, image_size=self.config.image_size)
        image = cv2.imread(str(normalized_path))
        if image is None:
            raise RuntimeError(f"Khong doc duoc anh: {normalized_path}")
        modality_profile = self._get_modality_profile(validation.modality)
        prepared_image = self._prepare_image_for_analysis(image, modality=validation.modality)
        prepared_image, roi_info = self._apply_segmentation_roi(prepared_image)
        quality_warnings = sorted(
            set(self._evaluate_image_quality(prepared_image) + list(validation.quality_warnings))
        )
        stage_results = self._run_pipeline_stages(prepared_image, normalized_path, validation, modality_profile=modality_profile)
        detect_stage = stage_results["detect"]
        detect_details = detect_stage.details or {}
        raw_detections = detect_details.get("detections", [])
        detections = _coerce_detections(raw_detections) if detect_stage.status == "success" else []
        report_detections = self._detections_in_source_frame(detections, roi_info)
        consistency_score, conflicting_pairs = self._compute_detection_consistency(detections)
        if consistency_score < self.config.detection_consistency_threshold:
            quality_warnings.append(
                f"Phat hien khong nhat quan giua cac vung (consistency={consistency_score:.2f} "
                f"< {self.config.detection_consistency_threshold:.2f}, "
                f"conflicting_pairs={len(conflicting_pairs)})."
            )
        risk_level, suspected_malignant, recommendation, average_confidence = self._classify_findings(
            detections,
            modality=validation.modality,
            modality_profile=modality_profile,
        )
        uncertainty_info = self._estimate_uncertainty(prepared_image)
        if uncertainty_info and uncertainty_info.get("high_uncertainty"):
            quality_warnings.append(
                f"Do bat dinh cao (entropy={uncertainty_info['entropy']:.2f}, "
                f"confidence={uncertainty_info['confidence']:.2f}). Can bac si xem lai."
            )
        processed_path = self._render_overlay(prepared_image.copy(), detections, patient_code=patient_code)
        gradcam_overlays = self._run_gradcam_if_possible(prepared_image, detections, validation)
        payload = {
            "case_id": case_id,
            "patient_code": patient_code,
            "source_image": str(resolved_source),
            "normalized_image": str(normalized_path),
            "processed_image": str(processed_path),
            "model_name": self.config.model_path.name,
            "risk_level": risk_level,
            "suspected_malignant": suspected_malignant,
            "average_confidence": average_confidence,
            "recommendation": recommendation,
            "detections": [
                {"label": item.label, "confidence": item.confidence, "bbox": list(item.bbox)} for item in report_detections
            ],
            "quality_warnings": quality_warnings,
            "stages": {name: {"status": item.status, "confidence": item.confidence, "message": item.message} for name, item in stage_results.items()},
            "disclaimer": MEDICAL_DISCLAIMER,
            "gradcam_overlays": gradcam_overlays,
            "deidentified": deidentified_source is not None,
            "roi": roi_info,
            "uncertainty": uncertainty_info,
        }
        report_json_path, report_md_path, _ = write_case_report(self.config.reports_dir, payload)
        dashboard_payload = {
            "patient_code": patient_code,
            "source_image": str(resolved_source),
            "risk_level": risk_level,
            "suspected_malignant": suspected_malignant,
            "average_confidence": average_confidence,
            "detection_count": len(detections),
            "quality_warnings": quality_warnings,
            "model_name": self.config.model_path.name,
        }
        write_inference_dashboard(self.config.reports_dir, dashboard_payload)
        return MedicalAnalysisResult(
            case_id=case_id,
            patient_code=patient_code,
            source_image=resolved_source,
            normalized_image=normalized_path,
            processed_image=processed_path,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            detections=report_detections,
            risk_level=risk_level,
            suspected_malignant=suspected_malignant,
            recommendation=recommendation,
            disclaimer=MEDICAL_DISCLAIMER,
            average_confidence=average_confidence,
            model_name=self.config.model_path.name,
            quality_warnings=quality_warnings,
            stage_confidences={name: item.confidence for name, item in stage_results.items()},
            stage_messages={name: item.message for name, item in stage_results.items()},
            gradcam_overlays=gradcam_overlays,
            deidentified_source=deidentified_source,
        )

    def _run_pipeline_stages(
        self,
        image: np.ndarray,
        normalized_path: Path,
        validation: ValidationResult,
        *,
        modality_profile: dict[str, Any] | None = None,
    ) -> dict[str, PipelineStageResult]:
        validate_stage = PipelineStageResult(
            stage="validate",
            status=validation.status,
            confidence=max(validation.modality_confidence, validation.body_region_confidence),
            message=validation.message or "Đã kiểm tra đầu vào ảnh.",
            details={"modality": validation.modality, "body_region": validation.body_region, "quality_warnings": list(validation.quality_warnings)},
        )

        preprocess_stage = PipelineStageResult(
            stage="preprocess",
            status="success" if validation.status in {"success", "uncertain"} else "error",
            confidence=0.85 if validation.status in {"success", "uncertain"} else 0.0,
            message="Đã chuẩn hóa ảnh và chuẩn bị cho phân tích.",
            details={"normalized_path": str(normalized_path)},
        )

        detect_stage = PipelineStageResult(
            stage="detect",
            status="success",
            confidence=0.0,
            message="Đợi kết quả phát hiện.",
            details={},
        )

        if validation.status == "error":
            detect_stage = PipelineStageResult(
                stage="detect",
                status="error",
                confidence=0.0,
                message="Bỏ qua phát hiện vì ảnh không đạt điều kiện kiểm tra đầu vào.",
                details={},
            )
        else:
            detections = self._detect_findings(image)
            if detections:
                confidence = float(np.mean([item.confidence for item in detections]))
                detect_stage = PipelineStageResult(
                    stage="detect",
                    status="success",
                    confidence=confidence,
                    message="Phát hiện vùng quan tâm thành công.",
                    details={"detections": [
                        {"label": item.label, "confidence": item.confidence, "bbox": list(item.bbox)} for item in detections
                    ]},
                )
            else:
                detect_stage = PipelineStageResult(
                    stage="detect",
                    status="uncertain",
                    confidence=0.0,
                    message="Không phát hiện được vùng quan tâm rõ ràng trên ảnh.",
                    details={},
                )

        classify_stage = PipelineStageResult(
            stage="classify",
            status="success",
            confidence=0.0,
            message="Đợi kết quả phân loại.",
            details={},
        )
        detect_details = detect_stage.details or {}
        if detection_details := detect_details.get("detections"):
            average_confidence = float(np.mean([item["confidence"] for item in detection_details]))
            certainty_threshold = (modality_profile or {}).get("certainty_threshold", self.config.certainty_threshold)
            classify_stage = PipelineStageResult(
                stage="classify",
                status="success" if average_confidence >= certainty_threshold else "uncertain",
                confidence=average_confidence,
                message="Phân loại rủi ro dựa trên vùng phát hiện.",
                details={"average_confidence": average_confidence, "certainty_threshold": certainty_threshold},
            )

        report_stage = PipelineStageResult(
            stage="report",
            status="success",
            confidence=min(validate_stage.confidence, preprocess_stage.confidence, detect_stage.confidence, classify_stage.confidence),
            message="Đã tạo báo cáo và dashboard cho ca phân tích.",
            details={},
        )

        return {
            "validate": validate_stage,
            "preprocess": preprocess_stage,
            "detect": detect_stage,
            "classify": classify_stage,
            "report": report_stage,
        }

    def _get_modality_profile(self, modality: str | None) -> dict[str, Any]:
        modality_key = (modality or "").lower()
        tuning_source = self.config.modality_tuning or _load_modality_tuning_from_config()
        has_specific = isinstance(tuning_source, dict) and modality_key in tuning_source
        tuning = get_modality_tuning(modality, self.config.modality_tuning)
        if tuning["normalize"] == "default" and not has_specific:
            return {
                "certainty_threshold": self.config.certainty_threshold,
                "medium_threshold": self.config.classify_medium_risk_threshold,
                "contrast_boost": 1.0,
                "normalize": "default",
                "quality_threshold": tuning["quality_threshold"],
            }
        return {
            "certainty_threshold": tuning["certainty_threshold"],
            "medium_threshold": tuning["medium_threshold"],
            "contrast_boost": tuning["contrast_boost"],
            "normalize": tuning["normalize"],
            "quality_threshold": tuning["quality_threshold"],
        }

    def _prepare_image_for_analysis(self, image: np.ndarray, *, modality: str | None) -> np.ndarray:
        if self.config.enable_advanced_preprocessing:
            advanced = self._advanced_preprocess(image, modality=modality)
            if advanced is not None:
                image = advanced
        profile = self._get_modality_profile(modality)
        img = image.astype(np.float32)
        if profile["normalize"] == "ct":
            img = np.clip(img * 1.08, 0, 255).astype(np.uint8)
        elif profile["normalize"] == "mri":
            img = np.clip(img * 1.04, 0, 255).astype(np.uint8)
        elif profile["normalize"] == "mammogram":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)  # type: ignore[assignment]
        elif profile["normalize"] == "ultrasound":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            img = cv2.cvtColor(cv2.equalizeHist(blurred), cv2.COLOR_GRAY2BGR)  # type: ignore[assignment]
        else:
            img = image

        if profile["contrast_boost"] != 1.0:
            img_uint8 = img.astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=profile["contrast_boost"], tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            img = cv2.merge((l_channel, a, b))  # type: ignore[assignment]
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)  # type: ignore[assignment]
        return img.astype(np.uint8)

    def _apply_segmentation_roi(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None]:
        """Cat vung quan tam (ROI) bang segmentation truoc khi phan tich.

        Dung SAMROIExtractor voi fallback Otsu (khong can tai trong so mang).
        Neu tat hoac loi, tra ve anh goc va roi_info=None.

        roi_info chua 'bbox' (he toa do anh truoc crop) va 'offset' (x1, y1) de
        cac stage sau co the doi bbox detections ve cung he toa do anh truoc crop.
        """
        if not self.config.enable_segmentation_roi:
            return image, None
        try:
            from medical.segmentation import crop_to_roi

            extractor = self._load_roi_extractor()
            result = extractor.extract_roi(image)
            if result is None:
                return image, None
            cropped = crop_to_roi(image, result.bbox, margin=self.config.segmentation_roi_margin)
            if cropped is None or cropped.size == 0:
                return image, None
            # Goc crop thuc te sau khi tinh margin/clamp (giong logic crop_to_roi).
            height, width = image.shape[:2]
            x1 = max(0, result.bbox[0] - self.config.segmentation_roi_margin)
            y1 = max(0, result.bbox[1] - self.config.segmentation_roi_margin)
            x1 = min(x1, max(0, width - 1))
            y1 = min(y1, max(0, height - 1))
            roi_info = {
                "bbox": list(result.bbox),
                "area": float(result.area),
                "confidence": float(result.confidence),
                "offset": [int(x1), int(y1)],
            }
            return cropped, roi_info
        except Exception as exc:
            print(f"[Segmentation] Bo qua ROI crop: {exc}")
            return image, None

    def _load_roi_extractor(self):
        if self._roi_extractor is None:
            from medical.segmentation import SAMROIExtractor

            self._roi_extractor = SAMROIExtractor()
        return self._roi_extractor

    def _advanced_preprocess(self, image: np.ndarray, *, modality: str | None) -> np.ndarray | None:
        """Tien xu ly theo modality bang medical.preprocessing (tuy chon).

        Tra ve anh BGR uint8 da xu ly, hoac None neu loi/khong ap dung.
        """
        try:
            from medical.preprocessing import preprocess_image

            result = preprocess_image(image, target_size=self.config.image_size, modality=modality)
            processed = getattr(result, "image", None)
            if processed is None or getattr(processed, "size", 0) == 0:
                return None
            return processed
        except Exception as exc:
            print(f"[Preprocess] Bo qua advanced preprocessing: {exc}")
            return None

    def _estimate_uncertainty(self, image: np.ndarray) -> dict[str, Any] | None:
        """Uoc luong do bat dinh bang MC Dropout khi backend la CNN.

        Chi chay khi enable_mc_dropout va model la CNN classifier. Khong tai
        trong so mang. Neu loi, tra ve None de khong pha vo luong chinh.
        """
        if not self.config.enable_mc_dropout:
            return None
        cnn_wrapper = self._load_cnn_wrapper()
        if cnn_wrapper is None:
            return None
        try:
            from medical.cnn_classifier import _load_image_as_tensor
            from medical.uncertainty import MCDropoutUncertainty

            tensor = _load_image_as_tensor(image, image_size=self.config.cnn_image_size, assume_bgr=True)
            tensor = tensor.unsqueeze(0)
            device = getattr(cnn_wrapper, "device", "cpu")
            estimator = MCDropoutUncertainty(
                cnn_wrapper.model,
                num_samples=self.config.mc_dropout_samples,
                device=device,
            )
            result = estimator.predict(tensor, class_labels=list(cnn_wrapper.class_labels))
            entropy = float(result.entropy[0]) if len(result.entropy) else 0.0
            confidence = float(result.confidence)
            high_uncertainty = confidence < self.config.certainty_threshold
            return {
                "predicted_label": result.predicted_label,
                "confidence": confidence,
                "entropy": entropy,
                "mutual_information": float(result.mutual_information[0]) if len(result.mutual_information) else 0.0,
                "num_samples": self.config.mc_dropout_samples,
                "high_uncertainty": bool(high_uncertainty),
            }
        except Exception as exc:
            print(f"[Uncertainty] Bo qua MC Dropout: {exc}")
            return None

    def _evaluate_image_quality(self, image: np.ndarray) -> list[str]:
        warnings: list[str] = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        height, width = gray.shape[:2]
        if min(height, width) < 256:
            warnings.append("Anh co do phan giai thap, ket qua co the kem on dinh.")
        if blur_score < 45:
            warnings.append("Anh co dau hieu mo, nen chup lai ro hon va lay net vao vung ton thuong.")
        if brightness < 45:
            warnings.append("Anh qua toi, nen bo sung anh sang deu truoc khi phan tich.")
        if brightness > 220:
            warnings.append("Anh qua sang, co nguy co mat chi tiet ton thuong.")
        return warnings

    def _detect_findings(self, image: np.ndarray) -> list[DetectionFinding]:
        if self._detector_backend is not None:
            yolo_detections = self._detect_with_backend(image)
            cnn_wrapper = self._load_cnn_wrapper()
            if cnn_wrapper is not None and yolo_detections:
                return self._ensemble_detections(image, yolo_detections)
            return yolo_detections
        cnn_wrapper = self._load_cnn_wrapper()
        if cnn_wrapper is not None:
            cnn_predictions = cnn_wrapper.predict(image, top_k=max(1, self.config.analyze_topk), tta=self.config.cnn_tta)
            height, width = image.shape[:2]
            bbox = (max(0, width // 8), max(0, height // 8), max(1, width - width // 8), max(1, height - height // 8))
            return [
                DetectionFinding(
                    label=str(item.get("label", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    bbox=bbox,
                )
                for item in cnn_predictions
            ]
        classifier = self._load_classifier()
        classifier_predictions = classifier.predict(image, top_k=max(1, self.config.analyze_topk))
        height, width = image.shape[:2]
        bbox = (max(0, width // 8), max(0, height // 8), max(1, width - width // 8), max(1, height - height // 8))
        return [
            DetectionFinding(
                label=str(item.label),
                confidence=float(item.confidence),
                bbox=bbox,
            )
            for item in classifier_predictions
        ]

    def _detect_with_backend(self, image: np.ndarray) -> list[DetectionFinding]:
        backend = self._detector_backend or self._load_default_backend()
        results = backend.predict(
            source=image,
            imgsz=self.config.image_size,
            conf=self.config.conf_threshold,
            verbose=False,
            stream=False,
        )
        findings: list[DetectionFinding] = []
        for result in results:
            if hasattr(result, "boxes"):
                names = getattr(result, "names", {})
                for box in getattr(result, "boxes", []):
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                    findings.append(
                        DetectionFinding(
                            label=str(names.get(cls_id, cls_id)),
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                        )
                    )
            elif hasattr(result, "label") and hasattr(result, "confidence"):
                raw_bbox = getattr(result, "bbox", None) or (0, 0, image.shape[1] - 1, image.shape[0] - 1)
                bbox = tuple(int(value) for value in raw_bbox)
                if len(bbox) != 4:
                    bbox = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
                findings.append(
                    DetectionFinding(
                        label=str(result.label),
                        confidence=float(result.confidence),
                        bbox=bbox,
                    )
                )
            elif isinstance(result, dict) and "label" in result and "confidence" in result:
                raw_bbox = result.get("bbox") or (0, 0, image.shape[1] - 1, image.shape[0] - 1)
                bbox = tuple(int(value) for value in raw_bbox)
                if len(bbox) != 4:
                    bbox = (0, 0, image.shape[1] - 1, image.shape[0] - 1)
                findings.append(
                    DetectionFinding(
                        label=str(result["label"]),
                        confidence=float(result["confidence"]),
                        bbox=bbox,
                    )
                )
        return findings

    def ensure_ready(self) -> Path:
        model_path = resolve_medical_runtime_model_path(self.config)
        issues = validate_medical_analyzer_config(self.config)
        if issues:
            raise ValueError("Cau hinh medical khong hop le: " + "; ".join(issues))
        self.config = self.config.with_model_path(model_path)
        self._log_model_manifest(model_path)
        return model_path

    def _log_model_manifest(self, model_path: Path) -> None:
        try:
            manifest = read_model_manifest(model_path)
            if manifest is not None:
                print(
                    f"[ModelManifest] {manifest.model_name} v{manifest.version} "
                    f"(backbone={manifest.backbone}, classes={manifest.num_classes}, "
                    f"size={manifest.image_size}, git={manifest.git_commit[:8]})"
                )
            else:
                print(
                    f"[ModelManifest] Warning: no manifest found for {model_path.name} "
                    f"(old model without versioning)"
                )
        except Exception:
            pass

    def _ensemble_detections(self, image: np.ndarray, yolo_detections: list[DetectionFinding]) -> list[DetectionFinding]:
        if not yolo_detections:
            return yolo_detections

        cnn_wrapper = self._load_cnn_wrapper()
        if cnn_wrapper is None:
            return yolo_detections

        try:
            cnn_prediction = cnn_wrapper.predict(image, top_k=1, tta=self.config.cnn_tta)[0]
            cnn_label = str(cnn_prediction.get("label", ""))
            cnn_confidence = float(cnn_prediction.get("confidence", 0.0))
        except Exception:
            cnn_label = ""
            cnn_confidence = 0.0

        return self._weighted_ensemble_detections(yolo_detections, cnn_label, cnn_confidence)

    def _weighted_ensemble_detections(
        self,
        yolo_detections: list[DetectionFinding],
        cnn_label: str,
        cnn_confidence: float,
    ) -> list[DetectionFinding]:
        if not yolo_detections:
            return []

        yolo_weight = float(self.config.ensemble_yolo_weight)
        cnn_weight = float(self.config.ensemble_cnn_weight)
        total_weight = yolo_weight + cnn_weight
        if total_weight <= 0.0:
            yolo_weight, cnn_weight, total_weight = 0.4, 0.6, 1.0
        if not np.isclose(total_weight, 1.0):
            yolo_weight = yolo_weight / total_weight
            cnn_weight = cnn_weight / total_weight

        combined: list[DetectionFinding] = []
        for detection in yolo_detections:
            ensemble_confidence = float(yolo_weight * detection.confidence + cnn_weight * cnn_confidence)
            combined.append(
                DetectionFinding(
                    label=cnn_label or detection.label,
                    confidence=ensemble_confidence,
                    bbox=detection.bbox,
                )
            )
        return combined

    @staticmethod
    def _bbox_iou(
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _compute_detection_consistency(
        self,
        detections: list[DetectionFinding],
    ) -> tuple[float, list[tuple[int, int]]]:
        if len(detections) < 2:
            return 1.0, []

        iou_threshold = 0.5
        conflicting_pairs: list[tuple[int, int]] = []
        total_pairs = 0
        overlapping_pairs = 0
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                total_pairs += 1
                iou = self._bbox_iou(detections[i].bbox, detections[j].bbox)
                if iou > iou_threshold:
                    overlapping_pairs += 1
                    conflicting_pairs.append((i, j))

        if total_pairs == 0:
            return 1.0, conflicting_pairs

        consistency_score = 1.0 - (overlapping_pairs / total_pairs)
        return float(consistency_score), conflicting_pairs

    def log_ensemble_metrics(
        self,
        yolo_detections: list[DetectionFinding],
        cnn_detections: list[DetectionFinding],
        ensemble_detections: list[DetectionFinding],
    ) -> dict[str, Any]:
        yolo_count = len(yolo_detections)
        cnn_count = len(cnn_detections)
        ensemble_count = len(ensemble_detections)

        yolo_avg = float(np.mean([item.confidence for item in yolo_detections])) if yolo_detections else 0.0
        cnn_avg = float(np.mean([item.confidence for item in cnn_detections])) if cnn_detections else 0.0
        ensemble_avg = float(np.mean([item.confidence for item in ensemble_detections])) if ensemble_detections else 0.0

        print(
            f"[EnsembleMetrics] yolo={yolo_count} (avg_conf={yolo_avg:.3f}), "
            f"cnn={cnn_count} (avg_conf={cnn_avg:.3f}), "
            f"ensemble={ensemble_count} (avg_conf={ensemble_avg:.3f})"
        )

        return {
            "yolo_count": yolo_count,
            "cnn_count": cnn_count,
            "ensemble_count": ensemble_count,
            "avg_confidences": {
                "yolo": yolo_avg,
                "cnn": cnn_avg,
                "ensemble": ensemble_avg,
            },
        }

    def _read_dicom_header_modality(self, image_path: str | Path) -> str | None:
        source = Path(image_path)
        if source.suffix.lower() != ".dcm":
            return None
        if source in self._dicom_modality_cache:
            return self._dicom_modality_cache[source]

        modality: str | None = None
        try:
            import pydicom

            ds = pydicom.dcmread(str(source), stop_before_pixels=True, force=True)
            dicom_modality = str(getattr(ds, "Modality", "")).strip()
            if dicom_modality:
                modality = dicom_modality.upper()
        except Exception:
            modality = None

        self._dicom_modality_cache[source] = modality
        return modality

    def validate_input(self, image_path: str | Path) -> ValidationResult:
        source = Path(image_path)
        base_result = validate_image(
            image_path,
            allowed_extensions=list(self.config.validation_allowed_extensions),
            min_confidence=self.config.validation_min_confidence,
        )

        dicom_modality = self._read_dicom_header_modality(source)
        if dicom_modality is not None:
            canonical = _DICOM_MODALITY_MAP.get(dicom_modality.upper(), dicom_modality.lower())
            if base_result.modality is None:
                return replace(
                    base_result,
                    modality=canonical,
                    modality_confidence=max(base_result.modality_confidence, 0.9),
                )
            if base_result.modality_confidence < self.config.validation_min_confidence:
                return replace(
                    base_result,
                    modality=canonical,
                    modality_confidence=max(base_result.modality_confidence, 0.9),
                )
        return base_result

    def _load_default_backend(self) -> DetectorBackend:
        if self.config.yolo_model_path and Path(self.config.yolo_model_path).exists():
            try:
                from ultralytics import YOLO

                return YOLO(str(self.config.yolo_model_path))
            except Exception as exc:
                print(f"[Ensemble] Khong the tai YOLO backend: {exc}")
        raise RuntimeError("Medical pipeline hien tai dung classifier local, khong can backend YOLO.")

    def _load_classifier(self) -> MedicalClassifierModel:
        if self._classifier_model is None:
            self._classifier_model = load_medical_classifier(self.config.model_path)
        return self._classifier_model

    def _load_cnn_wrapper(self) -> MedicalCNNClassifierWrapper | None:
        model_path = Path(self.config.model_path)
        if self._cnn_wrapper_cache is not None and self._cnn_wrapper_cache_path == model_path:
            return self._cnn_wrapper_cache
        if is_cnn_classifier_path(model_path):
            wrapper = load_cnn_classifier(model_path)
            self._cnn_wrapper_cache = wrapper
            self._cnn_wrapper_cache_path = model_path
            return wrapper
        return None

    @staticmethod
    def _detections_in_source_frame(
        detections: list[DetectionFinding],
        roi_info: dict[str, Any] | None,
    ) -> list[DetectionFinding]:
        """Doi bbox detections (dang o he toa do anh da crop) ve he toa do anh
        truoc crop, dung offset ROI. Neu khong crop, tra ve nguyen ven."""
        if not roi_info:
            return detections
        offset = roi_info.get("offset")
        if not offset or len(offset) != 2:
            return detections
        off_x, off_y = int(offset[0]), int(offset[1])
        if off_x == 0 and off_y == 0:
            return detections
        shifted: list[DetectionFinding] = []
        for item in detections:
            x1, y1, x2, y2 = item.bbox
            shifted.append(
                DetectionFinding(
                    label=item.label,
                    confidence=item.confidence,
                    bbox=(x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y),
                )
            )
        return shifted

    def _classify_findings(
        self,
        detections: list[DetectionFinding],
        *,
        modality: str | None = None,
        modality_profile: dict[str, Any] | None = None,
    ) -> tuple[str, bool, str, float]:
        if not detections:
            return (
                "low",
                False,
                "Khong ghi nhan vung ton thuong ro rang tren anh nay. Neu benh nhan co trieu chung hoac ton thuong ton tai, van nen kham chuyen khoa.",
                0.0,
            )
        average_confidence = sum(item.confidence for item in detections) / len(detections)
        max_confidence = max(item.confidence for item in detections)
        profile = modality_profile or self._get_modality_profile(modality)
        certainty_threshold = profile.get("certainty_threshold", self.config.certainty_threshold)
        medium_threshold = profile.get("medium_threshold", self.config.classify_medium_risk_threshold)
        if max_confidence < certainty_threshold:
            return (
                "uncertain",
                False,
                f"Ket qua chua du tin tuong (max_confidence={max_confidence:.2f} < {certainty_threshold:.2f}). Khong duoc phan loai benh nhan. Can kham chuyen khoa de bao dam doanh nghiep.",
                average_confidence,
            )
        if max_confidence >= self.config.classify_high_risk_threshold or len(detections) >= 3:
            return (
                "high",
                True,
                "Phat hien vung ton thuong co nguy co cao. Nen chuyen benh nhan den bac si da lieu/ung buou de danh gia tiep va sinh thiet neu can.",
                average_confidence,
            )
        if max_confidence >= medium_threshold:
            return (
                "medium",
                True,
                "Phat hien vung ton thuong co nguy co trung binh. Nen tai kham som va doi chieu voi kham lam sang chuyen khoa.",
                average_confidence,
            )
        return (
            "low",
            False,
            "Co mot vai vung ton thuong nguy co thap. Nen theo doi va kham chuyen khoa neu ton thuong thay doi kich thuoc, mau sac hoac hinh dang.",
            average_confidence,
        )

    def _deidentify_input(self, source: Path) -> Path | None:
        from medical.compliance import ComplianceReport, deidentify_dicom_series, deidentify_dicom_file
        from medical.reporting import build_artifact_stamp

        try:
            working = self.config.working_dir / "deidentified"
            working.mkdir(parents=True, exist_ok=True)
            stamp = build_artifact_stamp()
            if source.is_dir():
                target = working / f"series_{stamp}"
                target.mkdir(parents=True, exist_ok=True)
                report = deidentify_dicom_series(source, target)
            else:
                target = working / f"{source.stem}_{stamp}{source.suffix}"
                report = ComplianceReport()
                try:
                    deidentify_dicom_file(source, target)
                    report.record_deidentified(target)
                except Exception as exc:
                    report.record_error(str(exc))
                    report.record_skipped(source)
                    return None
            print(f"[Compliance] {report.summary}")
            return target if target.exists() else None
        except Exception as exc:
            print(f"[Compliance] De-identification failed: {exc}")
            return None

    def _run_gradcam_if_possible(self, prepared_image: np.ndarray, detections: list[DetectionFinding], validation: ValidationResult) -> list[str]:
        overlays: list[str] = []
        try:
            cnn_wrapper = self._load_cnn_wrapper()
            if cnn_wrapper is None or not detections:
                return overlays
            explainer = None
            try:
                from medical.explainability import MedicalGradCAMExplainer

                explainer = MedicalGradCAMExplainer(cnn_wrapper, image_size=self.config.image_size, device=getattr(cnn_wrapper, "device", None))
            except Exception:
                return overlays
            if explainer is None or not explainer.is_supported:
                return overlays
            top_label = detections[0].label if detections else ""
            results = explainer.explain(prepared_image, top_k=min(3, len(detections) or 1), tta=self.config.cnn_tta)
            for idx, result in enumerate(results):
                self.config.overlay_dir.mkdir(parents=True, exist_ok=True)
                stamp = build_artifact_stamp()
                out_path = self.config.overlay_dir / f"gradcam_{stamp}_{idx}_{top_label}.jpg"
                Image.fromarray(result.overlay).save(out_path, format="JPEG", quality=95)
                overlays.append(str(out_path))
        except Exception:
            pass
        return overlays

    def _render_overlay(self, image: np.ndarray, detections: list[DetectionFinding], *, patient_code: str) -> Path:
        overlay = draw_detection_results(
            image=image,
            detections=detections,
            box_thickness=3,
            label_font_scale=0.9,
            show_fps=False,
        )
        self.config.overlay_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.overlay_dir / f"{patient_code}_{build_artifact_stamp()}.jpg"
        cv2.imwrite(str(path), overlay)
        return path
