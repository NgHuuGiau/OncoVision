from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from medical.classifier import MedicalClassifierModel, load_medical_classifier
from medical.compliance import MEDICAL_DISCLAIMER
from medical.dataset import is_supported_medical_upload_path, normalize_uploaded_image
from medical.model_policy import resolve_medical_runtime_model_path
from medical.reporting import build_artifact_stamp, write_case_report
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

    def with_model_path(self, model_path: str | Path) -> "MedicalImageAnalyzerConfig":
        return replace(self, model_path=Path(model_path))


@dataclass(frozen=True)
class DetectionFinding:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


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
    )


def validate_medical_analyzer_config(config: MedicalImageAnalyzerConfig) -> list[str]:
    issues: list[str] = []
    if config.image_size <= 0:
        issues.append("image_size phai lon hon 0.")
    if not 0.0 < config.conf_threshold < 1.0:
        issues.append("conf_threshold phai nam trong khoang (0, 1).")
    if not 0.0 < config.classify_medium_risk_threshold <= config.classify_high_risk_threshold <= 1.0:
        issues.append("ngưỡng nguy cơ không hợp lệ.")
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
        if not is_supported_medical_upload_path(resolved_source):
            raise ValueError(f"Khong ho tro file upload: {resolved_source}")
        normalized_path = normalize_uploaded_image(resolved_source, self.config.processed_dir, image_size=self.config.image_size)
        image = cv2.imread(str(normalized_path))
        if image is None:
            raise RuntimeError(f"Khong doc duoc anh: {normalized_path}")
        quality_warnings = self._evaluate_image_quality(image)
        detections = self._detect_findings(image)
        risk_level, suspected_malignant, recommendation, average_confidence = self._classify_findings(detections)
        processed_path = self._render_overlay(image.copy(), detections, patient_code=patient_code)
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
            "disclaimer": MEDICAL_DISCLAIMER,
        }
        report_json_path, report_md_path = write_case_report(self.config.reports_dir, payload)
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
        )

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
        classifier = self._load_classifier()
        prediction = classifier.predict(image, top_k=1)[0]
        height, width = image.shape[:2]
        bbox = (max(0, width // 8), max(0, height // 8), max(1, width - width // 8), max(1, height - height // 8))
        return [DetectionFinding(label=prediction.label, confidence=prediction.confidence, bbox=bbox)]

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
                bbox = getattr(result, "bbox", None) or (0, 0, image.shape[1] - 1, image.shape[0] - 1)
                findings.append(
                    DetectionFinding(
                        label=str(result.label),
                        confidence=float(result.confidence),
                        bbox=tuple(int(value) for value in bbox),
                    )
                )
            elif isinstance(result, dict) and "label" in result and "confidence" in result:
                bbox = result.get("bbox") or (0, 0, image.shape[1] - 1, image.shape[0] - 1)
                findings.append(
                    DetectionFinding(
                        label=str(result["label"]),
                        confidence=float(result["confidence"]),
                        bbox=tuple(int(value) for value in bbox),
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

    def _load_default_backend(self) -> DetectorBackend:
        raise RuntimeError("Medical pipeline hien tai dung classifier local, khong can backend YOLO.")

    def _load_classifier(self) -> MedicalClassifierModel:
        if self._classifier_model is None:
            self._classifier_model = load_medical_classifier(self.config.model_path)
        return self._classifier_model

    def _classify_findings(self, detections: list[DetectionFinding]) -> tuple[str, bool, str, float]:
        if not detections:
            return (
                "low",
                False,
                "Khong ghi nhan vung ton thuong ro rang tren anh nay. Neu benh nhan co trieu chung hoac ton thuong ton tai, van nen kham chuyen khoa.",
                0.0,
            )
        average_confidence = sum(item.confidence for item in detections) / len(detections)
        max_confidence = max(item.confidence for item in detections)
        if max_confidence >= self.config.classify_high_risk_threshold or len(detections) >= 3:
            return (
                "high",
                True,
                "Phat hien vung ton thuong co nguy co cao. Nen chuyen benh nhan den bac si da lieu/ung buou de danh gia tiep va sinh thiet neu can.",
                average_confidence,
            )
        if max_confidence >= self.config.classify_medium_risk_threshold:
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
