from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol


import cv2
import numpy as np

from medical.classifier import MedicalClassifierModel, load_medical_classifier
from medical.cnn_classifier import MedicalCNNClassifierWrapper, is_cnn_classifier_path, load_cnn_classifier
from medical.compliance import MEDICAL_DISCLAIMER
from medical.dashboard import write_inference_dashboard
from medical.dataset import normalize_uploaded_image
from medical.model_policy import resolve_medical_runtime_model_path
from medical.reporting import build_artifact_stamp, write_case_report
from medical.validator import ValidationResult, get_modality_tuning, validate_image
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
    modality_tuning: dict[str, Any] = field(default_factory=dict)

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
        modality_tuning=settings.get("modality_tuning", {}) if isinstance(settings.get("modality_tuning", {}), dict) else {},
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

    def analyze_image(self, image_path: str | Path, *, patient_code: str, case_id: int | None = None) -> MedicalAnalysisResult:
        self.ensure_ready()
        resolved_source = Path(image_path)
        validation = self.validate_input(resolved_source)
        if validation.status == "error":
            raise ValueError(f"{validation.error_code}: {validation.message}")
        normalized_path = normalize_uploaded_image(resolved_source, self.config.processed_dir, image_size=self.config.image_size)
        image = cv2.imread(str(normalized_path))
        if image is None:
            raise RuntimeError(f"Khong doc duoc anh: {normalized_path}")
        modality_profile = self._get_modality_profile(validation.modality)
        prepared_image = self._prepare_image_for_analysis(image, modality=validation.modality)
        quality_warnings = sorted(
            set(self._evaluate_image_quality(prepared_image) + list(validation.quality_warnings))
        )
        stage_results = self._run_pipeline_stages(prepared_image, normalized_path, validation, modality_profile=modality_profile)
        detect_stage = stage_results["detect"]
        detect_details = detect_stage.details or {}
        raw_detections = detect_details.get("detections", [])
        detections = _coerce_detections(raw_detections) if detect_stage.status == "success" else []
        risk_level, suspected_malignant, recommendation, average_confidence = self._classify_findings(
            detections,
            modality=validation.modality,
            modality_profile=modality_profile,
        )
        processed_path = self._render_overlay(prepared_image.copy(), detections, patient_code=patient_code)
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
                {"label": item.label, "confidence": item.confidence, "bbox": list(item.bbox)} for item in detections
            ],
            "quality_warnings": quality_warnings,
            "stages": {name: {"status": item.status, "confidence": item.confidence, "message": item.message} for name, item in stage_results.items()},
            "disclaimer": MEDICAL_DISCLAIMER,
        }
        report_json_path, report_md_path = write_case_report(self.config.reports_dir, payload)
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
            detections=detections,
            risk_level=risk_level,
            suspected_malignant=suspected_malignant,
            recommendation=recommendation,
            disclaimer=MEDICAL_DISCLAIMER,
            average_confidence=average_confidence,
            model_name=self.config.model_path.name,
            quality_warnings=quality_warnings,
            stage_confidences={name: item.confidence for name, item in stage_results.items()},
            stage_messages={name: item.message for name, item in stage_results.items()},
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
            if detection_details:
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
            confidence=max(validate_stage.confidence, preprocess_stage.confidence, detect_stage.confidence, classify_stage.confidence),
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
        tuning = get_modality_tuning(modality, self.config.modality_tuning)
        if tuning["normalize"] == "default":
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
            img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        elif profile["normalize"] == "ultrasound":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            img = cv2.cvtColor(cv2.equalizeHist(blurred), cv2.COLOR_GRAY2BGR)
        else:
            img = image

        if profile["contrast_boost"] != 1.0:
            img_uint8 = img.astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=profile["contrast_boost"], tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            img = cv2.merge((l_channel, a, b))
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        return img.astype(np.uint8)

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
            return self._detect_with_backend(image)
        cnn_wrapper = self._load_cnn_wrapper()
        if cnn_wrapper is not None:
            prediction = cnn_wrapper.predict(image, top_k=1, tta=self.config.cnn_tta)[0]
            if isinstance(prediction, dict):
                label = str(prediction.get("label", ""))
                confidence = float(prediction.get("confidence", 0.0))
            else:
                label = str(prediction.label)
                confidence = float(prediction.confidence)
        else:
            classifier = self._load_classifier()
            prediction = classifier.predict(image, top_k=1)[0]
            label = str(prediction.label)
            confidence = float(prediction.confidence)
        height, width = image.shape[:2]
        bbox = (max(0, width // 8), max(0, height // 8), max(1, width - width // 8), max(1, height - height // 8))
        return [DetectionFinding(label=label, confidence=confidence, bbox=bbox)]

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
        return model_path

    def validate_input(self, image_path: str | Path) -> ValidationResult:
        return validate_image(
            image_path,
            allowed_extensions=list(self.config.validation_allowed_extensions),
            min_confidence=self.config.validation_min_confidence,
        )

    def _load_default_backend(self) -> DetectorBackend:
        raise RuntimeError("Medical pipeline hien tai dung classifier local, khong can backend YOLO.")

    def _load_classifier(self) -> MedicalClassifierModel:
        if self._classifier_model is None:
            self._classifier_model = load_medical_classifier(self.config.model_path)
        return self._classifier_model

    def _load_cnn_wrapper(self) -> MedicalCNNClassifierWrapper | None:
        if is_cnn_classifier_path(self.config.model_path):
            return load_cnn_classifier(self.config.model_path)
        return None

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
