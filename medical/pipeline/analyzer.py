"""Section 6: MedicalImageAnalyzer — Main class.

Trục chính của pipeline y khoa: validate → preprocess → detect → classify → report.
Hỗ trợ 2D images, videos, và 3D volumes (NIfTI/DICOM).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image

from medical.cancer_catalog import get_cancer_target
from medical.cancer_model_registry import modality_key, target_model_paths
from medical.classifier import MedicalClassifierModel, load_medical_classifier
from medical.cnn_classifier import (
    MedicalCNNClassifierWrapper,
    is_cnn_classifier_path,
    load_cnn_classifier,
)
from medical.compliance import MEDICAL_DISCLAIMER
from medical.dashboard import write_inference_dashboard
from medical.dataset import (
    infer_medical_upload_context,
    normalize_uploaded_image,
    sample_medical_video_frames,
)
from medical.model_policy import (
    iter_medical_runtime_model_paths,
    resolve_medical_runtime_model_path,
)
from medical.model_versioning import read_model_manifest
from medical.pipeline.config import (
    build_default_medical_analyzer_config,
    validate_medical_analyzer_config,
)
from medical.pipeline.dataclasses import (
    DetectionFinding,
    DetectorBackend,
    MedicalAnalysisResult,
    MedicalImageAnalyzerConfig,
    PipelineStageResult,
)
from medical.pipeline.detection import (
    _coerce_detections,
    _decode_raw_image,
    _map_detections_to_canvas,
)
from medical.pipeline.helpers import (
    _canonical_body_region_key,
    _sanitize_filename_part,
    _validate_output_path,
)
from medical.reporting import build_artifact_stamp, write_case_report
from medical.validator import (
    _DICOM_MODALITY_MAP,
    ValidationResult,
    _canonical_modality,
    _load_modality_tuning_from_config,
    detect_non_medical_image,
    get_modality_tuning,
    validate_image,
)
from medical.volume_inference import (
    TorchVolumeRuntime,
    load_medical_volume,
    render_volume_overlay,
)
from utils.draw_utils import draw_detection_results

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None] | None


class MedicalImageAnalyzer:
    def __init__(self, config: MedicalImageAnalyzerConfig | None = None, detector_backend: DetectorBackend | None = None) -> None:
        self.config = config or build_default_medical_analyzer_config()
        self._detector_backend = detector_backend
        self._injected_detector_backend = detector_backend is not None
        self._detector_backend_path: Path | None = None
        self._invalid_detector_paths: set[Path] = set()
        self._classifier_model: MedicalClassifierModel | None = None
        self._classifier_model_path: Path | None = None
        self._dicom_modality_cache: dict[Path, str | None] = {}
        self._cnn_wrapper_cache: MedicalCNNClassifierWrapper | None = None
        self._cnn_wrapper_cache_path: Path | None = None
        self._brain_wrapper_cache: MedicalCNNClassifierWrapper | None = None
        self._brain_wrapper_cache_path: Path | None = None
        self._modality_wrapper_cache: MedicalCNNClassifierWrapper | None = None
        self._modality_wrapper_cache_path: Path | None = None
        self._roi_extractor: Any = None
        self._volume_runtime: TorchVolumeRuntime | None = None
        self._volume_runtime_path: tuple[Path, Path] | None = None
        self._runtime_model_path: Path | None = None
        self._active_target_key: str | None = None
        self._active_modality: str | None = None
        self._brain_fallback_active = False

        # --- 6.1: Initialization & Progress ---

    @staticmethod
    def _call_progress(cb: ProgressCallback, stage: str, pct: float) -> None:
        if cb is not None:
            try:
                cb(stage, pct)
            except Exception:
                pass

        # --- 6.2: Model Path Resolution ---

    def _active_model_path(self, body_region: str | None = None) -> Path:
        if self._runtime_model_path is not None:
            return self._runtime_model_path
        target_key = self._active_target_key or self._target_key_from_body_region(body_region)
        if target_key is not None:
            classifier = self._target_paths(target_key, self._active_modality)["classifier"]
            if classifier.is_file():
                return classifier
        if self._uses_brain_model(body_region):
            brain_model_path = self.config.brain_model_path
            if brain_model_path is not None:
                return Path(brain_model_path)
        return Path(self.config.model_path)

    @staticmethod
    def _target_key_from_body_region(body_region: str | None) -> str | None:
        if body_region == "cervix":
            return "cervical"
        return body_region if body_region and get_cancer_target(body_region) is not None else None

    def _target_paths(self, target_key: str, modality: str | None = None) -> dict[str, Path]:
        paths = target_model_paths(target_key, self.config.target_model_root, modality)
        if target_key != "brain":
            return paths
        if modality is not None and modality_key(modality) != "mri":
            return paths
        legacy = {
            "classifier": self.config.brain_model_path,
            "detector": self.config.yolo_model_path,
            "volume_model": self.config.volume_model_path,
            "volume_manifest": self.config.volume_manifest_path,
            "clinical_validation": self.config.clinical_validation_report_path,
        }
        for name, candidate in legacy.items():
            if not paths[name].is_file() and candidate is not None:
                paths[name] = Path(candidate)
        return paths

        # --- 6.3: Image Analysis Entry Point ---

    def cleanup_cached_images(self) -> int:
        """Xoa normalized image cu neu vuot qua max_cached_images."""
        processed = self.config.processed_dir
        if not processed.exists():
            return 0
        images = sorted(processed.iterdir(), key=lambda p: p.stat().st_mtime if p.is_file() else 0)
        max_keep = max(self.config.max_cached_images, 10)
        removed = 0
        for f in images[:-max_keep] if len(images) > max_keep else []:
            try:
                f.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
        return removed

        # --- 6.3: Image Analysis Entry Point ---

    def analyze_image(
        self,
        image_path: str | Path,
        *,
        patient_code: str,
        case_id: int | None = None,
        target_key: str | None = None,
        requested_modality: str | None = None,
        progress_callback: ProgressCallback = None,
    ) -> MedicalAnalysisResult:
        _start_t = time.monotonic()
        self._call_progress(progress_callback, "Validating image...", 0.05)
        requested_target_key = target_key.strip().lower() if target_key else None
        if requested_target_key and get_cancer_target(requested_target_key) is None:
            raise ValueError("UNKNOWN_TARGET: Nhóm bệnh được chọn không hợp lệ.")
        requested_body_region = _canonical_body_region_key(requested_target_key) if requested_target_key else None
        resolved_source = Path(image_path)
        validation = self.validate_input(resolved_source)
        if validation.status == "error":
            raise ValueError(f"{validation.error_code}: {validation.message}")
        requested_modality_key: str | None = None
        if requested_modality:
            requested_modality_key = _canonical_modality(requested_modality)
            if requested_modality_key is None:
                raise ValueError("INVALID_MODALITY: Loại ảnh được chọn không hợp lệ.")
        strict_medical_gate = not self.config.demo_mode and self.config.validation_min_confidence >= 0.5
        if strict_medical_gate:
            non_medical_reason = detect_non_medical_image(
                resolved_source,
                expected_modality=requested_modality_key or validation.modality,
                expected_target=requested_target_key,
            )
            if non_medical_reason is not None:
                raise ValueError(f"NON_MEDICAL_IMAGE: {non_medical_reason}")
            if validation.status == "uncertain" and (
                validation.modality is None or validation.error_code == "UNKNOWN_IMAGE_TYPE"
            ):
                raise ValueError(
                    "NON_MEDICAL_IMAGE: Không nhận diện được đây là ảnh y khoa hợp lệ "
                    "(CT, MRI, X-quang, siêu âm, nội soi...). Vui lòng tải lên đúng ảnh y khoa."
                )
        if requested_modality_key is not None:
            # Chỉ kiểm tra strict modality khi validation_min_confidence đủ lớn (>= 0.5)
            # và classifier có confidence cao. Nếu threshold nhỏ thì coi như user override.
            if (self.config.validation_min_confidence >= 0.5
                    and validation.modality_confidence >= self.config.validation_min_confidence):
                if validation.modality != requested_modality_key:
                    raise ValueError(
                        "MODALITY_MISMATCH: Loại ảnh bạn chọn không khớp với modality nhận diện từ tệp "
                        f"({requested_modality} ≠ {validation.modality}). Hãy chọn đúng loại ảnh hoặc tải tệp khác."
                    )
            # Chỉ thêm warning khi hệ thống không tự xác định được modality
            # Nếu validation đã có modality với confidence cao, không cần warning
            modality_verified = (
                validation.modality_confidence >= self.config.validation_min_confidence
                and validation.modality == requested_modality_key
            )
            warnings_to_add = validation.quality_warnings
            if not modality_verified:
                warnings_to_add = (
                    *validation.quality_warnings,
                    (
                        f"Modality {requested_modality} được chọn bởi người dùng; "
                        "hệ thống không tự xác minh được từ thông tin ảnh."
                    ),
                )
            validation = replace(
                validation,
                modality=requested_modality_key,
                quality_warnings=warnings_to_add,
            )
        if requested_body_region is not None:
            if (
                validation.body_region is not None
                and (
                    validation.status == "success"
                    or validation.body_region_confidence >= self.config.validation_min_confidence
                )
                and validation.body_region != requested_body_region
            ):
                raise ValueError(
                    "TARGET_MISMATCH: Vùng cơ thể nhận diện từ ảnh không khớp nhóm ung thư đã chọn."
                )
            validation = replace(validation, body_region=requested_body_region)
        # Xác định target_key: ưu tiên requested > validation > auto-detect từ pixel
        
        if requested_target_key is not None:
            self._active_target_key = requested_target_key
        elif validation.body_region is not None:
            self._active_target_key = self._target_key_from_body_region(validation.body_region)
        else:
            # Auto-detect: dùng pixel analysis + classifier để tìm cancer type
            try:
                detected_target, detected_modality = infer_medical_upload_context(str(resolved_source))
                if detected_target is not None:
                    # Xác minh model tồn tại trước khi dùng auto-detect result
                    target_obj = get_cancer_target(detected_target)
                    if target_obj is not None:
                        paths = self._target_paths(detected_target, detected_modality)
                        if paths["classifier"].is_file() or paths["detector"].is_file():
                            self._active_target_key = detected_target
                            logger.info("Auto-detect cancer type: %s (modality: %s) — model xác minh OK",
                                        detected_target, detected_modality)
                        else:
                            logger.warning("Auto-detect tìm thấy %s nhưng model không tồn tại tại %s — bỏ qua",
                                           detected_target, paths["directory"])
                    else:
                        logger.warning("Auto-detect tìm thấy %s nhưng không có trong catalog — bỏ qua", detected_target)
                else:
                    logger.warning("Auto-detect không xác định được cancer type từ ảnh")
            except Exception as e:
                logger.warning("Auto-detect fail: %s", e)

        if strict_medical_gate and self._active_target_key is None and requested_target_key is None:
            raise ValueError(
                "NON_MEDICAL_IMAGE: Không xác định được vùng cơ thể hoặc nhóm ung thư phù hợp từ ảnh. "
                "Vui lòng tải lên ảnh y khoa hợp lệ hoặc chọn đúng nhóm bệnh."
            )

        self._active_modality = validation.modality
        self._runtime_model_path = None
        self._brain_fallback_active = False
        is_volume_input = resolved_source.is_dir() or resolved_source.name.lower().endswith(
            (".nii", ".nii.gz", ".mha", ".mhd")
        )
        
        # Demo mode: return simulated results without running real models
        if self.config.demo_mode:
            return self._analyze_demo(
                resolved_source,
                validation=validation,
                patient_code=patient_code,
                case_id=case_id,
                requested_target_key=requested_target_key,
                progress_callback=progress_callback,
                started_at=_start_t,
            )
        
        if self._active_target_key is not None and is_volume_input:
            return self._analyze_volume(
                resolved_source,
                validation=validation,
                patient_code=patient_code,
                case_id=case_id,
                requested_target_key=requested_target_key,
                progress_callback=progress_callback,
                started_at=_start_t,
            )
        try:
            self.ensure_ready(self._active_target_key, validation.modality)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Chưa có model medical để phân tích. Hãy bổ sung model suy luận đã chuẩn bị bên ngoài vào "
                "models/pretrained/ hoặc kiểm tra config/medical_settings.yaml "
                "(brain_model, modality_model).\n"
                f"Chi tiet: {exc}"
            ) from exc
        self._call_progress(progress_callback, "Normalizing image...", 0.10)
        if self._brain_fallback_active:
            if requested_body_region not in {None, "brain"}:
                raise ValueError("UNSUPPORTED_TARGET: Runtime hiện tại chỉ hỗ trợ phân tích u não.")
            if validation.body_region not in {None, "brain"}:
                raise ValueError("TARGET_MISMATCH: Ảnh không khớp với nhóm u não đã chọn.")
            validation = replace(validation, body_region="brain")

        deidentified_source = None
        dicom_info = {}
        if resolved_source.suffix.lower() == ".dcm":
            try:
                import pydicom
                ds = pydicom.dcmread(str(resolved_source), stop_before_pixels=True, force=True)
                dicom_info = {
                    "Modality": str(getattr(ds, "Modality", "")),
                    "BodyPart": str(getattr(ds, "BodyPartExamined", "")),
                    "WindowCenter": str(getattr(ds, "WindowCenter", "")),
                    "WindowWidth": str(getattr(ds, "WindowWidth", "")),
                    "RescaleIntercept": str(getattr(ds, "RescaleIntercept", "")),
                    "RescaleSlope": str(getattr(ds, "RescaleSlope", "")),
                }
            except Exception:
                pass
            deidentified_source = self._deidentify_input(resolved_source)
        elif resolved_source.is_dir():
            deidentified_source = self._deidentify_input(resolved_source)

        self._call_progress(progress_callback, "Running CNN inference...", 0.35)
        normalized_path = normalize_uploaded_image(deidentified_source or resolved_source, self.config.processed_dir, image_size=self.config.image_size)
        image = cv2.imdecode(np.frombuffer(normalized_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Không đọc được ảnh: {normalized_path}")
        modality_profile = self._get_modality_profile(validation.modality)
        is_video = resolved_source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        analyzed_frame_count = 1
        representative_frame_index = None
        raw_image: np.ndarray | None = None
        source_frame_detections: list[DetectionFinding] | None = None
        if is_video:
            (
                image,
                prepared_image,
                detections,
                overlay_detections,
                stage_results,
                quality_warnings,
                analyzed_frame_count,
                representative_frame_index,
            ) = self._run_video_samples(resolved_source, normalized_path, validation, modality_profile)
            report_detections = detections
        else:
            analyzed_image, roi_info = self._apply_segmentation_roi(image)
            prepared_image = self._prepare_image_for_analysis(analyzed_image, modality=validation.modality)
            quality_warnings = sorted(
                set(self._evaluate_image_quality(prepared_image, modality=validation.modality) + list(validation.quality_warnings))
            )
            # Detector chạy trên pixel ảnh gốc: model train trên ảnh thô, cho qua
            # thumbnail/canvas + chuẩn hóa modality làm rơi phát hiện (thực tế trên ảnh CT thận).
            raw_image = _decode_raw_image(deidentified_source or resolved_source)
            detect_input = analyzed_image
            if raw_image is not None and roi_info is None:
                detect_input = raw_image
            # Debug logging: in kích thước ở các bước để phát hiện lỗi scale
            raw_size = f"{raw_image.shape[1]}x{raw_image.shape[0]}" if raw_image is not None else "None"
            logger.info(
                "[Pipeline] Ảnh canvas: %dx%d, ảnh raw: %s, detect_input: %dx%d",
                image.shape[1], image.shape[0],
                raw_size,
                detect_input.shape[1], detect_input.shape[0],
            )
            stage_results = self._run_pipeline_stages(
                prepared_image,
                normalized_path,
                validation,
                modality_profile=modality_profile,
                detect_image=detect_input,
            )
            detect_stage = stage_results["detect"]
            detect_details = detect_stage.details or {}
            raw_detections = detect_details.get("detections", [])
            detections = _coerce_detections(raw_detections) if detect_stage.status == "success" else []
            if detect_input is raw_image:
                source_frame_detections = self._detections_in_source_frame(detections, roi_info)
                detections = _map_detections_to_canvas(detections, raw_image.shape[:2], image.shape[:2])
            report_detections = self._detections_in_source_frame(detections, roi_info)
            overlay_detections = detections
        segmentation_mask_path = self._write_segmentation_mask(
            image.shape,
            overlay_detections if is_video else report_detections,
            patient_code=patient_code,
        )
        # Cảnh báo khi có detection với bbox không đáng tin (do clamp tọa độ)
        unreliable_bbox_detections = [
            det for det in detections
            if det.bbox is not None and not getattr(det, "bbox_reliable", True)
        ]
        if unreliable_bbox_detections:
            labels_unreliable = ", ".join({det.label for det in unreliable_bbox_detections})
            quality_warnings.append(
                f"Tọa độ vùng phát hiện của '{labels_unreliable}' có thể không chính xác do lỗi scale "
                "giữa kích thước huấn luyện mô hình và ảnh đầu vào. "
                "Vui lòng kiểm tra bằng mắt và không dùng tọa độ này cho mục đích lâm sàng."
            )
        consistency_score, conflicting_pairs = self._compute_detection_consistency(detections)
        slot_warning = self._detector_slot_warning()
        if slot_warning:
            quality_warnings.append(slot_warning)
        if self._detector_backend is not None and consistency_score < self.config.detection_consistency_threshold:
            quality_warnings.append(
                f"Phát hiện không nhất quán giữa các vùng (consistency={consistency_score:.2f} "
                f"< {self.config.detection_consistency_threshold:.2f}, "
                f"conflicting_pairs={len(conflicting_pairs)})."
            )
        if len(detections) >= 2:
            top1_conf = detections[0].confidence
            top1_label = detections[0].label.lower()
            # Chỉ tính gap giữa các lớp khác nhau, không phải giữa 2 box cùng lớp
            top2_conf_for_gap = None
            for det in detections[1:]:
                if det.label.lower() != top1_label:
                    top2_conf_for_gap = det.confidence
                    break
            if top2_conf_for_gap is not None:
                gap = top1_conf - top2_conf_for_gap
                if top1_conf < 0.7 or gap < 0.2:
                    quality_warnings.append(
                        f"Độ tin cậy thấp giữa các lớp khác nhau (top1={top1_conf:.2f}, "
                        f"top2_other_class={top2_conf_for_gap:.2f}, gap={gap:.2f}). "
                        f"Kết quả có thể không chính xác."
                    )
            elif top1_conf < 0.7:
                quality_warnings.append(
                    f"Điểm dự đoán cao nhất thấp (top1={top1_conf:.2f} < 0.70). "
                    f"Kết quả có thể không chính xác."
                )
        risk_level, suspected_malignant, recommendation, average_confidence = self._classify_findings(detections)
        uncertainty_info = self._estimate_uncertainty(prepared_image)
        if uncertainty_info and uncertainty_info.get("high_uncertainty"):
            quality_warnings.append(
                f"Do bat dinh cao (entropy={uncertainty_info['entropy']:.2f}, "
                f"confidence={uncertainty_info['confidence']:.2f}). Can bac si xem lai."
            )
        self._call_progress(progress_callback, "Generating heatmap...", 0.65)
        gradcam_overlays = self._run_gradcam_if_possible(prepared_image, overlay_detections, validation)
        self._call_progress(progress_callback, "Analyzing results...", 0.85)
        processed_path = self._render_overlay(prepared_image.copy(), overlay_detections, patient_code=patient_code, gradcam_overlays=gradcam_overlays)
        coordinate_frame, source_size = self._report_coordinate_frame(
            image, raw_image, source_frame_detections is not None
        )
        detection_payloads = self._detection_payloads(
            report_detections, source_frame_detections, coordinate_frame,
            source_size=source_size, canvas_size=[int(image.shape[1]), int(image.shape[0])],
        )
        payload = {
            "case_id": case_id,
            "patient_code": patient_code,
            "source_image": str(resolved_source),
            "normalized_image": str(normalized_path),
            "processed_image": str(processed_path),
            "model_name": self._active_model_path(validation.body_region).name,
            "risk_level": risk_level,
            "suspected_malignant": suspected_malignant,
            "clinical_assessment_available": False,
            "average_confidence": average_confidence,
            "recommendation": recommendation,
            "detections": detection_payloads,
            "segmentation_mask_path": str(segmentation_mask_path) if segmentation_mask_path else None,
            "coordinate_frame": coordinate_frame,
            "source_size": source_size,
            "input_dimension": "video" if is_video else "2d",
            "analyzed_frame_count": analyzed_frame_count,
            "representative_frame_index": representative_frame_index,
            "localization_available": any(item.bbox or item.segmentation for item in report_detections),
            "quality_warnings": quality_warnings,
            "stages": {name: {"status": item.status, "confidence": item.confidence, "message": item.message} for name, item in stage_results.items()},
            "disclaimer": MEDICAL_DISCLAIMER,
            "gradcam_overlays": gradcam_overlays,
            "deidentified": deidentified_source is not None,
            "roi": roi_info,
            "uncertainty": uncertainty_info,
            "dicom_info": dicom_info if dicom_info else None,
            "modality": validation.modality,
            "body_region": validation.body_region,
            "analysis_time_seconds": time.monotonic() - _start_t,
        }
        self._call_progress(progress_callback, "Writing report...", 0.95)
        report_json_path, report_md_path, _ = write_case_report(self.config.reports_dir, payload)
        dashboard_payload = {
            "patient_code": patient_code,
            "source_image": str(resolved_source),
            "risk_level": risk_level,
            "suspected_malignant": suspected_malignant,
            "clinical_assessment_available": False,
            "average_confidence": average_confidence,
            "detection_count": len(detections),
            "quality_warnings": quality_warnings,
            "model_name": self._active_model_path(validation.body_region).name,
        }
        write_inference_dashboard(self.config.reports_dir, dashboard_payload)
        removed = self.cleanup_cached_images()
        if removed:
            logger.info("cleanup removed %d old normalized images", removed)
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
            model_name=self._active_model_path(validation.body_region).name,
            quality_warnings=quality_warnings,
            stage_confidences={name: item.confidence for name, item in stage_results.items()},
            stage_messages={name: item.message for name, item in stage_results.items()},
            gradcam_overlays=gradcam_overlays,
            deidentified_source=deidentified_source,
            dicom_info=dicom_info,
            analysis_time_seconds=time.monotonic() - _start_t,
            modality=validation.modality,
            body_region=validation.body_region,
            segmentation_mask_path=segmentation_mask_path,
            input_dimension="video" if is_video else "2d",
            analyzed_frame_count=analyzed_frame_count,
            representative_frame_index=representative_frame_index,
        )

        # --- 6.4: Volume 3D Analysis ---

    def _analyze_volume(
        self,
        source: Path,
        *,
        validation: ValidationResult,
        patient_code: str,
        case_id: int | None,
        requested_target_key: str | None,
        progress_callback: ProgressCallback,
        started_at: float,
    ) -> MedicalAnalysisResult:
        target_key = requested_target_key or self._target_key_from_body_region(validation.body_region)
        if target_key is None:
            raise ValueError("UNKNOWN_3D_TARGET: Không xác định được nhóm ung thư cho volume 3D.")
        paths = self._target_paths(target_key, validation.modality)
        model_path = paths["volume_model"]
        manifest_path = paths["volume_manifest"]
        if model_path is None or manifest_path is None or not Path(model_path).is_file() or not Path(manifest_path).is_file():
            raise ValueError(
                f"MISSING_3D_MODEL: Cần cả volume.pt2 (hoặc volume.pt cũ) và volume.json trong "
                f"{paths['directory']} để phân tích {target_key} trên MRI/CT 3D."
            )
        runtime_paths = (Path(model_path), Path(manifest_path))
        if self._volume_runtime is None or (
            self._volume_runtime_path is not None and self._volume_runtime_path != runtime_paths
        ):
            self._volume_runtime = TorchVolumeRuntime(model_path, manifest_path)
            self._volume_runtime_path = runtime_paths
        if self._volume_runtime.manifest.target != target_key:
            raise ValueError(
                "TARGET_MODEL_MISMATCH: target trong volume.json không khớp nhóm ung thư đã chọn."
            )

        deidentified_source = self._deidentify_input(source) if source.is_dir() else None
        analysis_source = deidentified_source or source
        self._call_progress(progress_callback, "Loading 3D volume...", 0.20)
        volume = load_medical_volume(
            analysis_source,
            max_voxels=self.config.max_volume_voxels,
            input_modalities=self._volume_runtime.manifest.input_modalities,
        )
        self._call_progress(progress_callback, "Running 3D inference...", 0.50)
        inference = self._volume_runtime.predict(volume)
        volume_mask_path = self._write_volume_mask(
            inference.mask,
            volume,
            patient_code=patient_code,
        )
        normalized_path, processed_path = render_volume_overlay(
            volume,
            inference,
            self.config.processed_dir,
            self.config.overlay_dir,
            patient_code=patient_code,
        )
        detections = [
            DetectionFinding(label=str(item["label"]), confidence=float(item["confidence"]), bbox=None)
            for item in inference.predictions
        ]
        if not detections and inference.findings:
            detections = [
                DetectionFinding(label=item.label, confidence=item.confidence, bbox=None)
                for item in inference.findings
            ]
        volume_findings = [
            {
                "label": item.label,
                "confidence": item.confidence,
                "bbox_voxel": list(item.bbox_voxel),
                "bbox_mm": [round(value, 3) for value in item.bbox_mm],
                "coordinate_system": volume.coordinate_system,
                "voxel_count": item.voxel_count,
                "volume_mm3": round(item.volume_mm3, 3),
            }
            for item in inference.findings
        ]
        risk_level, suspected_malignant, recommendation, average_confidence = self._classify_findings(detections)
        warnings = sorted(set(validation.quality_warnings))
        if not volume_findings:
            warnings.append("Model 3D không trả về vùng tổn thương vượt ngưỡng segmentation.")
        elapsed = time.monotonic() - started_at
        payload = {
            "case_id": case_id,
            "patient_code": patient_code,
            "source_image": str(source),
            "normalized_image": str(normalized_path),
            "processed_image": str(processed_path),
            "model_name": Path(model_path).name,
            "input_dimension": "3d",
            "volume_shape": list(volume.array.shape),
            "spacing_xyz_mm": list(volume.spacing_xyz),
            "coordinate_system": volume.coordinate_system,
            "risk_level": risk_level,
            "suspected_malignant": suspected_malignant,
            "clinical_assessment_available": False,
            "average_confidence": average_confidence,
            "recommendation": recommendation,
            "detections": [
                {"label": item.label, "confidence": item.confidence, "bbox": None} for item in detections
            ],
            "volume_findings": volume_findings,
            "volume_mask_path": str(volume_mask_path) if volume_mask_path else None,
            "localization_available": bool(volume_findings),
            "quality_warnings": warnings,
            "stages": {
                "validate": {"status": validation.status, "confidence": max(validation.modality_confidence, validation.body_region_confidence), "message": validation.message},
                "volume_inference": {"status": "success", "confidence": average_confidence, "message": "Đã chạy model 3D trên toàn volume."},
            },
            "disclaimer": MEDICAL_DISCLAIMER,
            "gradcam_overlays": [],
            "deidentified": deidentified_source is not None,
            "dicom_info": None,
            "modality": validation.modality or volume.modality,
            "body_region": validation.body_region or target_key,
            "analysis_time_seconds": elapsed,
        }
        self._call_progress(progress_callback, "Writing 3D report...", 0.90)
        report_json_path, report_md_path, _ = write_case_report(self.config.reports_dir, payload)
        write_inference_dashboard(
            self.config.reports_dir,
            {
                "patient_code": patient_code,
                "source_image": str(source),
                "risk_level": risk_level,
                "suspected_malignant": suspected_malignant,
                "clinical_assessment_available": False,
                "average_confidence": average_confidence,
                "detection_count": len(volume_findings),
                "quality_warnings": warnings,
                "model_name": Path(model_path).name,
            },
        )
        return MedicalAnalysisResult(
            case_id=case_id,
            patient_code=patient_code,
            source_image=source,
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
            model_name=Path(model_path).name,
            quality_warnings=warnings,
            stage_confidences={"validate": max(validation.modality_confidence, validation.body_region_confidence), "volume_inference": average_confidence},
            stage_messages={"validate": validation.message or "", "volume_inference": "Đã chạy model 3D trên toàn volume."},
            deidentified_source=deidentified_source,
            analysis_time_seconds=elapsed,
            modality=validation.modality or volume.modality,
            body_region=validation.body_region or target_key,
            input_dimension="3d",
            volume_findings=volume_findings,
            volume_shape=tuple(int(value) for value in volume.array.shape),
            volume_spacing_xyz_mm=volume.spacing_xyz,
            volume_coordinate_system=volume.coordinate_system,
            volume_mask_path=volume_mask_path,
        )

        # --- 6.5: Pipeline Stages ---

    def _run_pipeline_stages(
        self,
        image: np.ndarray,
        normalized_path: Path,
        validation: ValidationResult,
        *,
        modality_profile: dict[str, Any] | None = None,
        detect_image: np.ndarray | None = None,
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
            # Detector chạy trên ảnh đúng kích thước gốc, chưa qua chuẩn hóa modality:
            # model YOLO train trên pixel thô, preprocessing CT/US bóp mạnh recall của nó.
            detector_input = image if detect_image is None else detect_image
            detections = self._detect_findings(detector_input, body_region=validation.body_region)
            if detections:
                confidence = float(np.mean([item.confidence for item in detections]))
                detect_stage = PipelineStageResult(
                    stage="detect",
                    status="success",
                    confidence=confidence,
                    message="Đã tạo dự đoán từ model.",
                    details={"detections": [
                        {
                            "label": item.label,
                            "confidence": item.confidence,
                            "bbox": list(item.bbox) if item.bbox else None,
                            "segmentation": [list(point) for point in item.segmentation] if item.segmentation else None,
                            "frame_index": item.frame_index,
                            "timestamp_seconds": item.timestamp_seconds,
                        }
                        for item in detections
                    ]},
                )
            else:
                detect_stage = PipelineStageResult(
                    stage="detect",
                    status="uncertain",
                    confidence=0.0,
                    message="Model không trả về nhãn dự đoán đủ điều kiện.",
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
                message="Đã tính điểm dự đoán của model.",
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

    @staticmethod
    def _fit_video_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(image_size / max(width, 1), image_size / max(height, 1))
        resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        canvas: np.ndarray = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        x = (image_size - resized.shape[1]) // 2
        y = (image_size - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return canvas

    @staticmethod
    def _aggregate_video_detections(items: list[DetectionFinding]) -> list[DetectionFinding]:
        grouped: dict[str, list[DetectionFinding]] = {}
        for item in items:
            grouped.setdefault(item.label, []).append(item)
        aggregated: list[DetectionFinding] = []
        for label_items in grouped.values():
            best = max(label_items, key=lambda item: item.confidence)
            aggregated.append(
                replace(
                    best,
                    confidence=float(np.mean([item.confidence for item in label_items])),
                )
            )
        return sorted(aggregated, key=lambda item: item.confidence, reverse=True)

        # --- 6.6: Video Analysis ---

    def _run_video_samples(
        self,
        source: Path,
        normalized_path: Path,
        validation: ValidationResult,
        modality_profile: dict[str, Any],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[DetectionFinding],
        list[DetectionFinding],
        dict[str, PipelineStageResult],
        list[str],
        int,
        int,
    ]:
        samples = sample_medical_video_frames(source, max_frames=self.config.video_sample_frames)
        analyzed: list[tuple[int, np.ndarray, np.ndarray, list[DetectionFinding], dict[str, PipelineStageResult]]] = []
        warnings = list(validation.quality_warnings)
        for frame_index, timestamp, frame in samples:
            image = self._fit_video_frame(frame, self.config.image_size)
            prepared = self._prepare_image_for_analysis(image, modality=validation.modality)
            warnings.extend(self._evaluate_image_quality(prepared, modality=validation.modality))
            stages = self._run_pipeline_stages(
                prepared,
                normalized_path,
                validation,
                modality_profile=modality_profile,
                detect_image=image,
            )
            raw = (stages["detect"].details or {}).get("detections", [])
            detections = [
                replace(item, frame_index=frame_index, timestamp_seconds=timestamp)
                for item in _coerce_detections(raw)
            ]
            analyzed.append((frame_index, image, prepared, detections, stages))
        representative = max(
            analyzed,
            key=lambda item: max((finding.confidence for finding in item[3]), default=-1.0),
        )
        representative_index, image, prepared, overlay_detections, stages = representative
        detections = self._aggregate_video_detections(
            [finding for _index, _image, _prepared, findings, _stages in analyzed for finding in findings]
        )
        detect_confidence = float(np.mean([item.confidence for item in detections])) if detections else 0.0
        stages["detect"] = replace(
            stages["detect"],
            status="success" if detections else "uncertain",
            confidence=detect_confidence,
            message=f"Đã tổng hợp dự đoán từ {len(samples)} khung hình trải đều video.",
            details={
                "detections": [
                    {
                        "label": item.label,
                        "confidence": item.confidence,
                        "bbox": list(item.bbox) if item.bbox else None,
                        "segmentation": [list(point) for point in item.segmentation] if item.segmentation else None,
                        "frame_index": item.frame_index,
                        "timestamp_seconds": item.timestamp_seconds,
                    }
                    for item in detections
                ]
            },
        )
        certainty_threshold = modality_profile.get("certainty_threshold", self.config.certainty_threshold)
        stages["classify"] = replace(
            stages["classify"],
            status="success" if detect_confidence >= certainty_threshold else "uncertain",
            confidence=detect_confidence,
            message="Đã tính điểm tổng hợp trên các khung video được lấy mẫu.",
            details={"average_confidence": detect_confidence, "certainty_threshold": certainty_threshold},
        )
        stages["report"] = replace(
            stages["report"],
            confidence=min(stages["validate"].confidence, stages["preprocess"].confidence, detect_confidence),
        )
        warnings.append(f"Video được lấy mẫu {len(samples)} khung hình trải đều; overlay dùng khung {representative_index}.")
        if not cv2.imwrite(str(normalized_path), image):
            raise OSError(f"Không thể ghi khung video đại diện: {normalized_path}")
        return (
            image,
            prepared,
            detections,
            overlay_detections,
            stages,
            sorted(set(warnings)),
            len(samples),
            representative_index,
        )

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
        img: np.ndarray = image.astype(np.float32)
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
            img_uint8: np.ndarray = img.astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=profile["contrast_boost"], tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            img = cv2.merge((l_channel, a, b))
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        return img.astype(np.uint8)

    def _apply_segmentation_roi(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None]:
        """Cắt vùng quan tâm (ROI) bằng segmentation trước khi phân tích.

        Dùng SAMROIExtractor với fallback Otsu (không cần tải trọng số mạng).
        Nếu tắt hoặc lỗi, trả về ảnh gốc và roi_info=None.

        roi_info chứa 'bbox' (hệ tọa độ ảnh trước crop) và 'offset' (x1, y1) để
        các stage sau có thể đổi bbox detections về cùng hệ tọa độ ảnh trước crop.
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
        except Exception:
            logger.warning("[Segmentation] Bo qua ROI crop", exc_info=True)
            return image, None

    def _load_roi_extractor(self):
        if self._roi_extractor is None:
            from medical.segmentation import SAMROIExtractor

            self._roi_extractor = SAMROIExtractor()
        return self._roi_extractor

    def _advanced_preprocess(self, image: np.ndarray, *, modality: str | None) -> np.ndarray | None:
        """Tiền xử lý theo modality bằng medical.preprocessing (tùy chọn).

        Trả về ảnh BGR uint8 đã xử lý, hoặc None nếu lỗi/không áp dụng.
        """
        try:
            from medical.preprocessing import preprocess_image

            result = preprocess_image(image, target_size=self.config.image_size, modality=modality)
            processed = getattr(result, "image", None)
            if processed is None or getattr(processed, "size", 0) == 0:
                return None
            return processed
        except Exception:
            logger.warning("[Preprocess] Bo qua advanced preprocessing", exc_info=True)
            return None

    def _estimate_uncertainty(self, image: np.ndarray) -> dict[str, Any] | None:
        """Ước lượng độ bất định bằng MC Dropout khi backend là CNN.

        Chỉ chạy khi enable_mc_dropout và model là CNN classifier. Không tải
        trọng số mạng. Nếu lỗi, trả về None để không phá vỡ luồng chính.
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
        except Exception:
            logger.warning("[Uncertainty] Bo qua MC Dropout", exc_info=True)
            return None

    # Ngưỡng blur (Laplacian variance) theo loại ảnh.
    # Lưu ý quan trọng: các ngưỡng này là giá trị khởi tạo. Để chính xác, cần tính phân phối
    # blur score trên tập dữ liệu thật (cả ảnh tốt và ảnh mờ) rồi chọn ngưỡng theo percentile.
    # Không nên hạ ngưỡng liên tục để "vừa" một vài ảnh mẫu — điều này làm mất khả năng
    # phát hiện ảnh mờ thật. Thay vào đó, hãy tính blur trên vùng có mô (tissue mask)
    # để loại ảnh hưởng của nền đen.
    #
    # Tham khảo: Laplacian variance của ảnh MRI não T1 (toàn ảnh) thường 5-50,
    # nhưng trên vùng có mô thường 20-100. Ảnh camera y khoa (endoscopy) thường 100-500.
    _BLUR_THRESHOLD_CROSS_SECTIONAL = 20.0
    _BLUR_THRESHOLD_CAMERA = 45.0

    def _evaluate_image_quality(self, image: np.ndarray, *, modality: str | None = None) -> list[str]:
        """Đánh giá chất lượng ảnh, tách riêng logic cho CT/MRI/PET và camera images.

        Với CT/MRI/PET: ảnh dùng thang Hounsfield Units (HU) hoặc grayscale chuyên biệt,
        vùng nền xung quanh cơ thể luôn màu đen — không áp dụng kiểm tra độ sáng toàn ảnh.
        Cả blur và brightness đều tính trên vùng có mô (tissue mask) để loại nền đen.

        Với camera images (endoscopy, ultrasound, mammogram, colposcopy, xray):
        áp dụng đầy đủ kiểm tra độ sáng, độ tương phản, độ mờ trên toàn ảnh.

        Lưu ý: ngưỡng blur là giá trị khởi tạo. Nên hiệu chỉnh dựa trên phân phối thực tế
        của tập dữ liệu để tránh overfit vào một vài ảnh mẫu.
        """
        warnings: list[str] = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        height, width = gray.shape[:2]

        # Xác định loại ảnh để áp dụng threshold phù hợp
        modality_lower = (modality or "").lower()
        is_cross_sectional = modality_lower in {"ct", "mri", "pet_ct", "pet"}

        if min(height, width) < 256:
            warnings.append("Ảnh có độ phân giải thấp, kết quả có thể kém ổn định.")

        # Độ mờ: tính trên vùng có mô cho CT/MRI để loại ảnh hưởng của nền đen
        if is_cross_sectional:
            # Mask vùng có mô: loại nền đen (threshold 20 để bắt các pixel có tín hiệu)
            tissue_mask = gray > 20
            if tissue_mask.sum() > 0:
                # Tạo ảnh chỉ có vùng mô, nền đen — tính Laplacian trên ảnh này
                # để loại ảnh hưởng của nền đen lên variance
                masked_gray = np.zeros_like(gray)
                masked_gray[tissue_mask] = gray[tissue_mask]
                blur_score = cv2.Laplacian(masked_gray, cv2.CV_64F).var()
            else:
                # Toàn ảnh đen — không có mô, dùng toàn ảnh
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        else:
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        blur_threshold = (
            self._BLUR_THRESHOLD_CROSS_SECTIONAL if is_cross_sectional else self._BLUR_THRESHOLD_CAMERA
        )
        if blur_score < blur_threshold:
            if is_cross_sectional:
                warnings.append(
                    f"Ảnh có dấu hiệu mờ (blur_score={blur_score:.1f} < {blur_threshold:.0f}), "
                    f"nên kiểm tra lại chất lượng chụp. "
                    f"(Blur được tính trên vùng có mô, ngưỡng có thể cần hiệu chỉnh dựa trên "
                    f"phân phối của tập dữ liệu thực tế.)"
                )
            else:
                warnings.append(
                    f"Ảnh có dấu hiệu mờ (blur_score={blur_score:.1f} < {blur_threshold:.0f}), "
                    f"nên chụp lại rõ hơn và lấy nét vào vùng tổn thương."
                )

        # Độ sáng: CHỈ áp dụng cho camera images, KHÔNG áp dụng cho CT/MRI/PET
        # Vì ảnh CT/MRI có vùng nền đen (không khí) là bình thường
        if not is_cross_sectional:
            if brightness < 45:
                warnings.append("Ảnh quá tối, nên bổ sung ánh sáng đều trước khi phân tích.")
            if brightness > 220:
                warnings.append("Ảnh quá sáng, có nguy cơ mất chi tiết tổn thương.")
        else:
            # Với CT/MRI: tính độ sáng trên vùng có mô (loại nền đen) để đánh giá chính xác hơn
            # Nền đen của MRI/CT là bình thường; chỉ cảnh báo nếu vùng có mô cũng quá tối
            non_black_mask = gray > 20
            if non_black_mask.sum() > 0:
                tissue_brightness = float(np.mean(gray[non_black_mask]))
                if tissue_brightness < 30:
                    warnings.append(
                        f"Vùng có mô trong ảnh quá tối (brightness={tissue_brightness:.1f}), "
                        f"nên kiểm tra lại tham số chụp."
                    )

        return warnings

    def _uses_brain_model(self, body_region: str | None) -> bool:
        if self.config.brain_model_path is None:
            return False
        brain_model_path = Path(self.config.brain_model_path)
        if not brain_model_path.exists():
            return False
        return self._brain_fallback_active or body_region == "brain"

        # --- 6.8: Detection & Classification ---

    def _detect_findings(
        self, image: np.ndarray, *, body_region: str | None = None
    ) -> list[DetectionFinding]:
        cnn_wrapper = self._load_cnn_wrapper()
        detector_path = self._active_detector_path()
        if (
            not self._injected_detector_backend
            and self._detector_backend is not None
            and self._detector_backend_path != detector_path
        ):
            self._detector_backend = None
        detector_file = detector_path is not None and Path(detector_path).is_file()
        if detector_file and detector_path is not None and Path(detector_path) in self._invalid_detector_paths:
            detector_file = False
        if self._detector_backend is not None or detector_file:
            yolo_detections = self._detect_with_backend(image)
            if yolo_detections:
                if cnn_wrapper is not None:
                    return self._ensemble_detections(image, yolo_detections, body_region=body_region)
                # Lọc detections có confidence thấp
                min_conf = getattr(self.config, 'detection_min_confidence', 0.3)
                yolo_detections = [d for d in yolo_detections if d.confidence >= min_conf]
                return yolo_detections
            if cnn_wrapper is None:
                return yolo_detections or []
        if cnn_wrapper is not None:
            cnn_predictions = cnn_wrapper.predict(image, top_k=max(1, self.config.analyze_topk), tta=self.config.cnn_tta)
            # Lọc predictions có confidence thấp
            min_conf = getattr(self.config, 'detection_min_confidence', 0.3)
            cnn_predictions = [p for p in cnn_predictions if float(p.get("confidence", 0.0)) >= min_conf]
            return [
                DetectionFinding(
                    label=str(item.get("label", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    bbox=None,
                )
                for item in cnn_predictions
            ]
        classifier = self._load_classifier()
        classifier_predictions = classifier.predict(image, top_k=max(1, self.config.analyze_topk))
        # Lọc predictions có confidence thấp
        min_conf = getattr(self.config, 'detection_min_confidence', 0.3)
        classifier_predictions = [p for p in classifier_predictions if p.confidence >= min_conf]
        return [
            DetectionFinding(
                label=str(item.label),
                confidence=float(item.confidence),
                bbox=None,
            )
            for item in classifier_predictions
        ]

    def _detect_with_backend(self, image: np.ndarray) -> list[DetectionFinding] | None:
        backend = self._detector_backend
        if backend is None:
            backend = self._load_default_backend()
            if backend is None:
                detector_path = self._active_detector_path()
                if detector_path is not None:
                    self._invalid_detector_paths.add(Path(detector_path))
                return None
            self._detector_backend = backend
        # Debug logging: in kích thước ảnh và thông tin YOLO để phát hiện lỗi scale
        img_h, img_w = image.shape[:2]
        logger.info(
            "[Detect] Ảnh đầu vào YOLO: %dx%d (channels=%d)",
            img_w, img_h, image.shape[2] if image.ndim == 3 else 1,
        )
        results = backend.predict(
            source=image,
            conf=self.config.conf_threshold,
            verbose=False,
            stream=False,
        )
        # Log orig_shape từ kết quả YOLO để so sánh với ảnh đầu vào
        for result in results:
            orig_shape = getattr(result, "orig_shape", None)
            if orig_shape is not None:
                logger.info(
                    "[Detect] YOLO orig_shape: %sx%s, ảnh truyền vào: %dx%d",
                    orig_shape[0], orig_shape[1], img_h, img_w,
                )
        findings: list[DetectionFinding] = []
        for result in results:
            if hasattr(result, "boxes"):
                names = getattr(result, "names", {})
                mask_polygons = list(getattr(getattr(result, "masks", None), "xy", ()) or ())
                for box_index, box in enumerate(getattr(result, "boxes", [])):
                    raw_xyxy = box.xyxy[0].tolist()
                    logger.info(
                        "[Detect] Raw bbox từ YOLO xyxy: (%.1f, %.1f, %.1f, %.1f) trên ảnh %dx%d",
                        raw_xyxy[0], raw_xyxy[1], raw_xyxy[2], raw_xyxy[3], img_w, img_h,
                    )
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    bbox, bbox_reliable = self._normalize_bbox(
                        raw_xyxy, image.shape
                    )
                    if bbox:
                        logger.info(
                            "[Detect] Bbox sau normalize: (%d, %d, %d, %d), reliable=%s",
                            bbox[0], bbox[1], bbox[2], bbox[3], bbox_reliable,
                        )
                    segmentation = None
                    if box_index < len(mask_polygons):
                        height, width = image.shape[:2]
                        points = tuple(
                            (max(0, min(width - 1, round(float(point[0])))), max(0, min(height - 1, round(float(point[1])))))
                            for point in mask_polygons[box_index]
                        )
                        segmentation = points if len(points) >= 3 else None
                    findings.append(
                        DetectionFinding(
                            label=str(names.get(cls_id, cls_id)),
                            confidence=confidence,
                            bbox=bbox,
                            segmentation=segmentation,
                            bbox_reliable=bbox_reliable,
                        )
                    )
            elif hasattr(result, "label") and hasattr(result, "confidence"):
                raw_bbox = getattr(result, "bbox", None)
                bbox, bbox_reliable = self._normalize_bbox(raw_bbox, image.shape)
                findings.append(
                    DetectionFinding(
                        label=str(result.label),
                        confidence=float(result.confidence),
                        bbox=bbox,
                        bbox_reliable=bbox_reliable,
                    )
                )
            elif isinstance(result, dict) and "label" in result and "confidence" in result:
                raw_bbox = result.get("bbox")
                bbox, bbox_reliable = self._normalize_bbox(raw_bbox, image.shape)
                findings.append(
                    DetectionFinding(
                        label=str(result["label"]),
                        confidence=float(result["confidence"]),
                        bbox=bbox,
                        bbox_reliable=bbox_reliable,
                    )
                )
        return findings

    @staticmethod
    def _normalize_bbox(
        raw_bbox: Any,
        image_shape: tuple[int, ...],
    ) -> tuple[tuple[int, int, int, int] | None, bool]:
        """Chuẩn hóa bbox và trả về (bbox, is_reliable).

        Lưu ý quan trọng: Ultralytics YOLO trả về boxes.xyxy đã ở tọa độ ảnh gốc
        (original-image pixels), không phải không gian imgsz. Hàm này chỉ clamp
        tọa độ vượt khỏi ảnh và kiểm tra bbox chiếm quá 95% diện tích.

        is_reliable = False khi:
        - Tọa độ bị clamp (vượt khỏi ảnh) — dấu hiệu lỗi scale từ nguồn khác YOLO
        - Bbox chiếm quá 95% diện tích ảnh

        HEURISTIC SCALE ĐÃ BỊ LOẠI BỎ: trước đây hàm này cố gắng phát hiện bbox
        lệch scale bằng cách so sánh imgsz với kích thước ảnh, nhưng Ultralytics
        đã scale về ảnh gốc tự động. Heuristic cũ gây cảnh báo oan cho hầu hết ảnh.
        Để debug tọa độ sai, hãy in ra raw bbox từ model và so với ảnh gốc.
        """
        try:
            if raw_bbox is None or len(raw_bbox) != 4:
                return None, True
            coordinates = tuple(round(float(value)) for value in raw_bbox)
        except (TypeError, ValueError, OverflowError):
            return None, True
        height, width = image_shape[:2]
        orig_x1, orig_y1, orig_x2, orig_y2 = coordinates
        # Clamp tọa độ vào giới hạn ảnh để tránh bbox vượt khỏi ảnh
        # Dùng width-1 và height-1 vì tọa độ pixel tối đa là width-1, height-1
        clamped_x1, clamped_x2 = sorted((max(0, min(width - 1, orig_x1)), max(0, min(width - 1, orig_x2))))
        clamped_y1, clamped_y2 = sorted((max(0, min(height - 1, orig_y1)), max(0, min(height - 1, orig_y2))))
        # Kiểm tra bbox hợp lệ: width và height phải > 0
        if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
            return None, True
        # --- Đánh giá độ đáng tin của bbox ---
        is_reliable = True

        # 1. Kiểm tra clamp: tọa độ bị cắt nghĩa là sai scale
        clamp_distance = (
            abs(orig_x1 - clamped_x1)
            + abs(orig_y1 - clamped_y1)
            + abs(orig_x2 - clamped_x2)
            + abs(orig_y2 - clamped_y2)
        )
        if clamp_distance > 0:
            is_reliable = False
            logger.warning(
                "[BBox] Tọa độ bbox bị clamp: (%d,%d,%d,%d) -> (%d,%d,%d,%d) trên ảnh %dx%d. "
                "Có thể do lỗi scale từ nguồn dữ liệu.",
                orig_x1, orig_y1, orig_x2, orig_y2,
                clamped_x1, clamped_y1, clamped_x2, clamped_y2,
                width, height,
            )

        # 2. Kiểm tra bbox không vượt quá 95% diện tích ảnh (có thể là lỗi scale)
        bbox_area = (clamped_x2 - clamped_x1) * (clamped_y2 - clamped_y1)
        image_area = width * height
        if bbox_area > 0.95 * image_area:
            is_reliable = False
            logger.warning(
                "[BBox] Bbox chiếm %.1f%% diện tích ảnh, có thể do lỗi scale.",
                bbox_area / image_area * 100,
            )
        return ((clamped_x1, clamped_y1, clamped_x2, clamped_y2), is_reliable)

        # --- 6.9: Model Loading & Readiness ---

    def ensure_ready(self, target_key: str | None = None, modality: str | None = None) -> Path:
        issues = validate_medical_analyzer_config(self.config)
        if issues:
            raise ValueError("Cấu hình medical không hợp lệ: " + "; ".join(issues))
        
        # Demo mode: bypass model requirements and return a dummy path
        if self.config.demo_mode:
            logger.info("[Medical] Demo mode enabled - skipping model loading")
            self._runtime_model_path = Path("demo_mode")
            self._active_target_key = target_key or "demo"
            self._active_modality = modality or "demo"
            return self._runtime_model_path
        
        if target_key is not None:
            target = get_cancer_target(target_key)
            if target is None:
                raise FileNotFoundError(f"Nhóm ung thư không có trong catalog: {target_key}")
            paths = self._target_paths(target.key, modality)
            model_path = paths["classifier"] if paths["classifier"].is_file() else paths["detector"]
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Thiếu classifier.pt hoặc detector.pt cho {target.label} trong {paths['directory']}."
                )
            if self._runtime_model_path != model_path:
                self._classifier_model = None
                self._classifier_model_path = None
                self._cnn_wrapper_cache = None
                self._cnn_wrapper_cache_path = None
            self._runtime_model_path = model_path
            self._active_target_key = target.key
            self._active_modality = modality
            self._log_model_manifest(model_path)
            return model_path
        try:
            model_path = resolve_medical_runtime_model_path(self.config)
        except FileNotFoundError:
            brain_model_path = Path(self.config.brain_model_path) if self.config.brain_model_path else None
            if brain_model_path is not None and brain_model_path.exists():
                logger.warning(
                    "[Medical] Chua co model tong quat, chi co brain model (%s). "
                    "Chỉ phân tích được ảnh vùng đầu (body_region=brain).",
                    brain_model_path.name,
                )
                self._brain_fallback_active = True
                self._runtime_model_path = brain_model_path
                self._active_target_key = "brain"
                return brain_model_path
            candidates = ", ".join(str(p) for p in iter_medical_runtime_model_paths(self.config))
            raise FileNotFoundError(
                "Thiếu model medical để phân tích. Đã thử các đường dẫn: "
                f"{candidates}. Hãy cập nhật model_path, fallback_model_path hoặc brain_model trong config/medical_settings.yaml. "
                "Hoặc bật demo_mode trong config để chạy không cần model."
            )
        self.config = self.config.with_model_path(model_path)
        self._runtime_model_path = model_path
        self._log_model_manifest(model_path)
        return model_path

    def _log_model_manifest(self, model_path: Path) -> None:
        try:
            manifest = read_model_manifest(model_path)
            if manifest is not None:
                logger.info(
                    "[ModelManifest] %s v%s (backbone=%s, classes=%s, size=%s, git=%s)",
                    manifest.model_name,
                    manifest.version,
                    manifest.backbone,
                    manifest.num_classes,
                    manifest.image_size,
                    manifest.git_commit[:8],
                )
            else:
                logger.warning(
                    "[ModelManifest] Warning: no manifest found for %s (old model without versioning)",
                    model_path.name,
                )
        except Exception:
            logger.warning("[ModelManifest] Không đọc được manifest", exc_info=True)

    def _ensemble_detections(
        self,
        image: np.ndarray,
        yolo_detections: list[DetectionFinding],
        *,
        body_region: str | None = None,
    ) -> list[DetectionFinding]:
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
            logger.warning("[Ensemble] Không thể predict CNN cho ensemble", exc_info=True)
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
                    segmentation=detection.segmentation,
                    frame_index=detection.frame_index,
                    timestamp_seconds=detection.timestamp_seconds,
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
                bbox_i, bbox_j = detections[i].bbox, detections[j].bbox
                if bbox_i is None or bbox_j is None:
                    total_pairs -= 1
                    continue
                iou = self._bbox_iou(bbox_i, bbox_j)
                if iou > iou_threshold:
                    overlapping_pairs += 1
                    conflicting_pairs.append((i, j))

        if total_pairs == 0:
            return 1.0, conflicting_pairs

        consistency_score = 1.0 - (overlapping_pairs / total_pairs)
        return float(consistency_score), conflicting_pairs

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
            logger.warning("[DICOM] Không đọc được Modality từ %s", source, exc_info=True)
            modality = None

        self._dicom_modality_cache[source] = modality
        return modality

        # --- 6.10: Input Validation ---

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



        if base_result.modality is None or base_result.modality_confidence < self.config.validation_min_confidence:
            cnn_modality = self._predict_modality_with_cnn(source)
            if cnn_modality is not None:
                updated = replace(
                    base_result,
                    modality=cnn_modality,
                    modality_confidence=max(base_result.modality_confidence, 0.9),
                )


                if updated.body_region is None:
                    inferred_target, _ = infer_medical_upload_context(source)
                    if inferred_target is not None:
                        canonical_body = _canonical_body_region_key(inferred_target)
                        if canonical_body is not None:
                            updated = replace(
                                updated,
                                body_region=canonical_body,
                                body_region_confidence=max(updated.body_region_confidence, 0.8),
                            )
                return updated
        return base_result

    def _predict_modality_with_cnn(self, image_path: str | Path) -> str | None:
        wrapper = self._load_modality_wrapper()
        if wrapper is None:
            return None
        try:
            predictions = wrapper.predict(image_path, top_k=1, tta=self.config.cnn_tta)
            if not predictions:
                return None
            top = predictions[0]
            label = str(top.get("label", "")).lower()
            confidence = float(top.get("confidence", 0.0))
            if confidence < self.config.validation_min_confidence:
                return None
            return label
        except Exception:
            logger.warning("[ModalityCNN] Không dự đoán được modality", exc_info=True)
            return None

    def _load_default_backend(self) -> DetectorBackend | None:
        detector_path = self._active_detector_path()
        if detector_path and detector_path.is_file():
            try:
                from ultralytics import YOLO

                backend = YOLO(str(detector_path))
                self._validate_detector_backend(backend)
                self._detector_backend_path = detector_path
                return backend
            except Exception as exc:
                # Detector hỏng/sai loại không nên làm sập cả ca: bỏ định vị, dùng classifier.
                logger.warning("Bỏ qua detector không dùng được %s: %s", detector_path, exc)
                return None
        return None

    def _active_detector_path(self) -> Path | None:
        if self._active_target_key is not None:
            return self._target_paths(self._active_target_key, self._active_modality)["detector"]
        return Path(self.config.yolo_model_path) if self.config.yolo_model_path is not None else None

    def _detector_slot_warning(self) -> str | None:
        """Báo khi detector đang nằm ở cấp cơ quan nhưng ảnh dùng ô modality trống."""
        if self._active_target_key is None or self._detector_backend is not None:
            return None
        active = self._active_detector_path()
        if active is None or Path(active).is_file():
            return None
        root_detector = target_model_paths(self._active_target_key, self.config.target_model_root)["detector"]
        if not root_detector.is_file():
            return None
        return (
            f"Không dùng detector định vị: {active} không tồn tại, trong khi detector đang đặt ở cấp "
            f"cơ quan ({root_detector}). Ảnh có modality={self._active_modality} nên runtime tìm model "
            f"trong đúng ô {self._active_target_key}/<modality>/ — đặt lại bằng "
            "scripts/fetch_model.py --modality để định vị hoạt động."
        )

    @staticmethod
    def _validate_detector_backend(backend: Any) -> None:
        task = getattr(backend, "task", None)
        if task not in {"detect", "segment"}:
            raise ValueError(f"task={task!r}; cần model detect hoặc segment")
        names = getattr(backend, "names", {})
        labels = {str(value).lower() for value in (names.values() if isinstance(names, dict) else names)}
        coco_markers = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"}
        if len(labels) >= 20 and len(labels & coco_markers) >= 5:
            raise ValueError("đây là model vật thể phổ thông, không phải model định vị y khoa")

    def _load_classifier(self) -> MedicalClassifierModel:
        model_path = self._active_model_path()
        if self._classifier_model is None or self._classifier_model_path != model_path:
            self._classifier_model = load_medical_classifier(model_path)
            self._classifier_model_path = model_path
        return self._classifier_model

    def _load_cnn_wrapper(self) -> MedicalCNNClassifierWrapper | None:
        model_path = self._active_model_path()
        if self._cnn_wrapper_cache is not None and self._cnn_wrapper_cache_path == model_path:
            return self._cnn_wrapper_cache
        if is_cnn_classifier_path(model_path):
            wrapper = load_cnn_classifier(model_path)
            self._cnn_wrapper_cache = wrapper
            self._cnn_wrapper_cache_path = model_path
            return wrapper
        return None

    def _load_brain_wrapper(self) -> MedicalCNNClassifierWrapper | None:
        model_path = self._target_paths("brain")["classifier"]
        if not model_path.is_file():
            return None
        if self._brain_wrapper_cache is not None and self._brain_wrapper_cache_path == model_path:
            return self._brain_wrapper_cache
        if is_cnn_classifier_path(model_path):
            wrapper = load_cnn_classifier(model_path)
            self._brain_wrapper_cache = wrapper
            self._brain_wrapper_cache_path = model_path
            return wrapper
        return None

    def _load_modality_wrapper(self) -> MedicalCNNClassifierWrapper | None:
        if self.config.modality_model_path is None:
            return None
        model_path = Path(self.config.modality_model_path)
        if self._modality_wrapper_cache is not None and self._modality_wrapper_cache_path == model_path:
            return self._modality_wrapper_cache
        if is_cnn_classifier_path(model_path):
            wrapper = load_cnn_classifier(model_path)
            self._modality_wrapper_cache = wrapper
            self._modality_wrapper_cache_path = model_path
            return wrapper
        return None

    @staticmethod
    def _detections_in_source_frame(
        detections: list[DetectionFinding],
        roi_info: dict[str, Any] | None,
    ) -> list[DetectionFinding]:
        """Đổi bbox detections (đang ở hệ tọa độ ảnh đã crop) về hệ tọa độ ảnh
        trước crop, dùng offset ROI. Nếu không crop, trả về nguyên ven."""
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
            if item.bbox is None:
                shifted.append(item)
                continue
            x1, y1, x2, y2 = item.bbox
            shifted.append(
                DetectionFinding(
                    label=item.label,
                    confidence=item.confidence,
                    bbox=(x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y),
                    segmentation=(
                        tuple((x + off_x, y + off_y) for x, y in item.segmentation)
                        if item.segmentation
                        else None
                    ),
                    frame_index=item.frame_index,
                    timestamp_seconds=item.timestamp_seconds,
                )
            )
        return shifted

    @staticmethod
    def _report_coordinate_frame(
        canvas_image: np.ndarray,
        raw_image: np.ndarray | None,
        detections_in_source: bool,
    ) -> tuple[str, list[int]]:
        """Hệ tọa độ báo cáo: ưu tiên ảnh gốc người dùng tải lên khi có pixel gốc."""
        if detections_in_source and raw_image is not None:
            return "source_image", [int(raw_image.shape[1]), int(raw_image.shape[0])]
        return "normalized_canvas", [int(canvas_image.shape[1]), int(canvas_image.shape[0])]

    @staticmethod
    def _detection_payloads(
        report_detections: list[DetectionFinding],
        source_frame_detections: list[DetectionFinding] | None,
        coordinate_frame: str,
        source_size: list[int] | None = None,
        canvas_size: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Dựng detections cho báo cáo: giữ bbox canvas + bổ sung bbox hệ ảnh gốc.

        source_size: kích thước ảnh gốc [w, h] (dùng cho bbox_source).
        canvas_size: kích thước canvas chuẩn hóa [w, h] (dùng cho bbox).
        """
        payloads: list[dict[str, Any]] = []
        for index, item in enumerate(report_detections):
            source_item = (
                source_frame_detections[index]
                if source_frame_detections is not None and index < len(source_frame_detections)
                else item
            )
            # coordinate_frame mô tả hệ của "bbox" (canvas), không phải của "bbox_source".
            bbox_frame = (
                "source_image"
                if coordinate_frame == "source_image" and item.bbox == source_item.bbox
                else "normalized_canvas"
            )
            bbox_reliable = getattr(item, "bbox_reliable", True)
            # bbox_image_size ghi rõ kích thước ảnh mà 'bbox' áp dụng
            # để báo cáo không nhầm lẫn giữa hệ canvas và hệ ảnh gốc
            bbox_image_size = None
            if item.bbox is not None:
                if bbox_frame == "source_image":
                    bbox_image_size = source_size
                else:
                    bbox_image_size = canvas_size
            # bbox_source_image_size ghi rõ kích thước ảnh mà 'bbox_source' áp dụng
            bbox_source_image_size = None
            if source_item.bbox is not None:
                bbox_source_image_size = source_size
            payloads.append(
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "bbox": list(item.bbox) if item.bbox else None,
                    "bbox_source": list(source_item.bbox) if source_item.bbox else None,
                    "bbox_image_size": bbox_image_size,
                    "bbox_source_image_size": bbox_source_image_size,
                    "segmentation": [list(point) for point in item.segmentation] if item.segmentation else None,
                    "segmentation_source": (
                        [list(point) for point in source_item.segmentation] if source_item.segmentation else None
                    ),
                    "frame_index": item.frame_index,
                    "timestamp_seconds": item.timestamp_seconds,
                    "coordinate_frame": bbox_frame,
                    "bbox_reliable": bbox_reliable,
                }
            )
        return payloads

        # --- 6.11: Risk Classification ---

    # Các loại u thường gặp phần lớn là lành tính (WHO grade I), không nên gắn trực tiếp với "ác tính".
    # Lưu ý: ngay cả các loại u này cũng có thể có độ II/III (ví dụ meningioma WHO grade II/III),
    # nên không nên hạ mức ưu tiên khám xuống thấp. Chỉ tách riêng "loại u" khỏi "mức độ ưu tiên".
    _BENIGN_PREDOMINANT_LABELS: ClassVar[set[str]] = {
        "meningioma",
        "pituitary_adenoma",
        "pituitary adenoma",
        "pituitary",
        "adenoma",
        "hemangioma",
        "lipoma",
        "neurofibroma",
        "schwannoma",
        "vestibular_schwannoma",
        "acoustic_neuroma",
        "craniopharyngioma",
        "choroid_plexus papilloma",
    }

    def _classify_findings(self, detections: list[DetectionFinding]) -> tuple[str, bool, str, float]:
        """Phân loại mức độ ưu tiên khám dựa trên confidence threshold 3 mức và loại nhãn.

        Ngưỡng:
        - < 35%: Low suspicion (ít cần ưu tiên)
        - 35% - 65%: Indeterminate (chưa rõ ràng, cần hội chẩn)
        - > 65%: High suspicion (cần ưu tiên khám)

        Lưu ý quan trọng:
        - confidence cao không đồng nghĩa với "ác tính". Một số loại u (ví dụ meningioma)
          phần lớn là lành tính dù điểm dự đoán cao.
        - Ngay cả các loại u lành tính cũng cần được bác sĩ đánh giá; không hạ mức ưu tiên
          khám xuống thấp chỉ vì nhãn thường lành tính (meningioma có thể WHO grade II/III).
        - Hàm này trả về mức độ cần ưu tiên khám, không phải kết luận về tính ác tính.
        - Chỉ có giải phẫu bệnh mới kết luận được lành tính hay ác tính.

        Trả về: (risk_level, suspected_malignant, recommendation, average_confidence)
        """
        if not detections:
            return (
                "uncertain",
                False,
                "Không có phát hiện nào. Cần được nhân viên y tế đánh giá trước khi đưa ra nhận định lâm sàng.",
                0.0,
            )

        average_confidence = sum(item.confidence for item in detections) / len(detections)
        top_detection = detections[0]
        top_confidence = top_detection.confidence
        top_label = top_detection.label.lower().replace(" ", "_").replace("-", "_")

        # Kiểm tra xem nhãn có thuộc nhóm thường lành tính không
        is_benign_predominant = top_label in self._BENIGN_PREDOMINANT_LABELS or any(
            benign in top_label for benign in ("meningioma", "adenoma", "hemangioma", "lipoma", "schwannoma")
        )

        # Xác định risk level dựa trên 3 mức ngưỡng
        if top_confidence < 0.35:
            risk_level = "low_suspicion"
            suspected_malignant = False
            recommendation = (
                "Điểm dự đoán thấp, mức độ cần ưu tiên khám thấp. "
                "Đây là nhãn và điểm dự đoán của model, không phải kết luận về vị trí tổn thương hay mức độ nghiêm trọng. "
                "Cần được nhân viên y tế đánh giá trước khi đưa ra nhận định lâm sàng."
            )
        elif top_confidence < 0.65:
            risk_level = "indeterminate"
            suspected_malignant = False
            recommendation = (
                "Điểm dự đoán ở mức chưa rõ ràng (vùng xám). "
                "Không thể kết luận mức độ nghiêm trọng dựa trên kết quả này. "
                "Cần hội chẩn và chụp bổ sung để xác nhận. "
                "Đây là nhãn và điểm dự đoán của model, không phải kết luận về vị trí tổn thương hay mức độ nghiêm trọng. "
                "Cần được nhân viên y tế đánh giá trước khi đưa ra nhận định lâm sàng."
            )
        else:
            # Confidence cao (> 65%): cần ưu tiên khám cao cho mọi loại nhãn.
            # Ngay cả nhãn thường lành tính cũng cần được bác sĩ đánh giá sớm
            # vì có thể có ngoại lệ (meningioma WHO grade II/III, v.v.).
            if is_benign_predominant:
                risk_level = "high_suspicion"
                suspected_malignant = False
                recommendation = (
                    f"Điểm dự đoán cao cho nhãn '{top_detection.label}', cần được bác sĩ đánh giá sớm. "
                    f"Loại u này phần lớn là lành tính, nhưng vẫn có thể có trường hợp ngoại lệ "
                    f"(ví dụ meningioma độ II/III theo WHO). Điểm dự đoán cao chỉ phản ánh mô hình "
                    f"nhận diện rõ ràng, không đồng nghĩa với tính ác tính. "
                    f"Chỉ có bác sĩ và giải phẫu bệnh mới kết luận được lành tính hay ác tính. "
                    f"Cần được nhân viên y tế đánh giá và xét nghiệm bổ sung trước khi đưa ra nhận định lâm sàng."
                )
            else:
                risk_level = "high_suspicion"
                suspected_malignant = True
                recommendation = (
                    "Điểm dự đoán cao, mức độ cần ưu tiên khám cao. "
                    "Tuy nhiên, đây là nhãn và điểm dự đoán của model, không phải kết luận về vị trí tổn thương hay tính ác tính. "
                    "Chỉ có bác sĩ và giải phẫu bệnh mới kết luận được lành tính hay ác tính. "
                    "Cần được nhân viên y tế đánh giá và xét nghiệm bổ sung trước khi đưa ra nhận định lâm sàng."
                )

        return risk_level, suspected_malignant, recommendation, average_confidence

    def _deidentify_input(self, source: Path) -> Path | None:
        from medical.compliance import (
            ComplianceReport,
            deidentify_dicom_file,
            deidentify_dicom_series,
        )
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
            logger.info("[Compliance] %s", report.summary)
            return target if target.exists() else None
        except Exception:
            logger.warning("[Compliance] De-identification failed", exc_info=True)
            return None

        # --- 6.12: Grad-CAM & Explainability ---

    def _run_gradcam_if_possible(
        self, prepared_image: np.ndarray, detections: list[DetectionFinding], validation: ValidationResult
    ) -> list[str]:
        overlays: list[str] = []
        try:
            cnn_wrapper = self._load_brain_wrapper() if self._uses_brain_model(validation.body_region) else self._load_cnn_wrapper()
            if cnn_wrapper is None or not detections:
                return overlays
            explainer = None
            try:
                from medical.explainability import MedicalGradCAMExplainer

                explainer = MedicalGradCAMExplainer(cnn_wrapper, image_size=self.config.image_size, device=getattr(cnn_wrapper, "device", None))
            except Exception:
                logger.warning("[GradCAM] Không khởi tạo được explainer", exc_info=True)
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

    def _render_overlay(self, image: np.ndarray, detections: list[DetectionFinding], *, patient_code: str, gradcam_overlays: list[str] | None = None) -> Path:
        overlay = draw_detection_results(
            image=image,
            detections=detections,
            box_thickness=1,
            label_font_scale=0.6,
            show_fps=False,
            label_alpha=0.85,
        )
        if gradcam_overlays:
            try:
                gcam_path = Path(gradcam_overlays[0])
                gcam_bytes = gcam_path.read_bytes()
                gcam = cv2.imdecode(np.frombuffer(gcam_bytes, np.uint8), cv2.IMREAD_COLOR)
                if gcam is not None:
                    h, w = gcam.shape[:2]
                    if overlay.shape[:2] != (h, w):
                        gcam = cv2.resize(gcam, (overlay.shape[1], overlay.shape[0]))
                    alpha = 0.3
                    overlay = cv2.addWeighted(overlay, 1.0 - alpha, gcam, alpha, 0)
            except Exception:
                pass
        self.config.overlay_dir.mkdir(parents=True, exist_ok=True)
        safe_code = _sanitize_filename_part(patient_code)
        path = self.config.overlay_dir / f"{safe_code}_{build_artifact_stamp()}.jpg"
        path = _validate_output_path(self.config.overlay_dir, path)
        cv2.imwrite(str(path), overlay)
        return path

    def _write_segmentation_mask(
        self,
        image_shape: tuple[int, ...],
        detections: list[DetectionFinding],
        *,
        patient_code: str,
    ) -> Path | None:
        polygons = [item.segmentation for item in detections if item.segmentation and len(item.segmentation) >= 3]
        if not polygons:
            return None
        mask: np.ndarray = np.zeros(image_shape[:2], dtype=np.uint8)
        for polygon in polygons:
            cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        self.config.overlay_dir.mkdir(parents=True, exist_ok=True)
        safe_code = _sanitize_filename_part(patient_code)
        path = self.config.overlay_dir / f"{safe_code}_{build_artifact_stamp()}_mask.png"
        path = _validate_output_path(self.config.overlay_dir, path)
        if not cv2.imwrite(str(path), mask):
            raise OSError(f"Không thể ghi segmentation mask: {path}")
        return path

    def _write_volume_mask(
        self,
        mask: np.ndarray | None,
        volume: Any,
        *,
        patient_code: str,
    ) -> Path | None:
        if mask is None:
            return None
        self.config.overlay_dir.mkdir(parents=True, exist_ok=True)
        safe_code = _sanitize_filename_part(patient_code)
        path = self.config.overlay_dir / f"{safe_code}_{build_artifact_stamp()}_volume_mask.npz"
        path = _validate_output_path(self.config.overlay_dir, path)
        np.savez_compressed(
            path,
            mask=np.asarray(mask, dtype=np.uint8),
            affine=np.asarray(volume.affine, dtype=np.float64),
            spacing_xyz_mm=np.asarray(volume.spacing_xyz, dtype=np.float64),
            coordinate_system=np.asarray(volume.coordinate_system),
        )
        return path

        # --- 6.13: Demo Mode ---

    def _analyze_demo(
        self,
        source: Path,
        *,
        validation: ValidationResult,
        patient_code: str,
        case_id: int | None,
        requested_target_key: str | None,
        progress_callback: ProgressCallback,
        started_at: float,
    ) -> MedicalAnalysisResult:
        """Demo mode: returns simulated analysis results without real models."""
        import cv2
        import numpy as np
        
        self._call_progress(progress_callback, "Demo mode: Simulating analysis...", 0.2)
        
        # Determine target for demo
        target_key = requested_target_key or self._active_target_key or "brain"
        get_cancer_target(target_key)
        
        # Simulate modality and body region
        modality = validation.modality or "mri"
        body_region = validation.body_region or target_key
        
        # Generate mock detections based on target
        mock_detections = self._generate_mock_detections(target_key, validation)
        
        self._call_progress(progress_callback, "Demo mode: Generating report...", 0.5)
        
        # Create a simple processed image (copy of normalized)
        normalized_path = normalize_uploaded_image(source, self.config.processed_dir, image_size=self.config.image_size)
        
        # Create a processed image with mock overlay
        try:
            image = cv2.imdecode(np.frombuffer(normalized_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                # Draw mock bounding boxes
                overlay = image.copy()
                for det in mock_detections:
                    if det.bbox:
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(overlay, f"{det.label} {det.confidence:.0%}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.config.overlay_dir.mkdir(parents=True, exist_ok=True)
                safe_code = _sanitize_filename_part(patient_code)
                stamp = build_artifact_stamp()
                processed_path = self.config.overlay_dir / f"{safe_code}_{stamp}_demo.jpg"
                processed_path = _validate_output_path(self.config.overlay_dir, processed_path)
                cv2.imwrite(str(processed_path), overlay)
            else:
                processed_path = normalized_path
        except Exception:
            processed_path = normalized_path
        
        self._call_progress(progress_callback, "Demo mode: Finalizing...", 0.8)
        
        # Classify findings
        risk_level, suspected_malignant, recommendation, average_confidence = self._classify_findings(mock_detections)
        
        # Build payload
        payload = {
            "case_id": case_id,
            "patient_code": patient_code,
            "source_image": str(source),
            "normalized_image": str(normalized_path),
            "processed_image": str(processed_path),
            "model_name": f"demo_{target_key}_classifier",
            "risk_level": risk_level,
            "suspected_malignant": suspected_malignant,
            "clinical_assessment_available": False,
            "average_confidence": average_confidence,
            "recommendation": recommendation,
            "detections": [
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "bbox": list(item.bbox) if item.bbox else None,
                    "segmentation": [list(point) for point in item.segmentation] if item.segmentation else None,
                    "frame_index": item.frame_index,
                    "timestamp_seconds": item.timestamp_seconds,
                }
                for item in mock_detections
            ],
            "segmentation_mask_path": None,
            "coordinate_frame": "demo",
            "source_size": (512, 512),
            "input_dimension": "2d",
            "analyzed_frame_count": 1,
            "representative_frame_index": 0,
            "localization_available": any(item.bbox or item.segmentation for item in mock_detections),
            "quality_warnings": list(validation.quality_warnings) + ["⚠️ DEMO MODE: Kết quả mô phỏng - không dùng cho chẩn đoán thực tế"],
            "stages": {
                "validate": {"status": validation.status, "confidence": max(validation.modality_confidence, validation.body_region_confidence), "message": validation.message or "Đã kiểm tra đầu vào ảnh."},
                "demo_inference": {"status": "success", "confidence": average_confidence, "message": "Chạy mô phỏng (demo mode) - không dùng model thực tế."},
            },
            "disclaimer": MEDICAL_DISCLAIMER,
            "gradcam_overlays": [],
            "deidentified": False,
            "roi": None,
            "uncertainty": {"entropy": 0.5, "confidence": average_confidence, "high_uncertainty": True},
            "dicom_info": None,
            "modality": modality,
            "body_region": body_region,
            "analysis_time_seconds": time.monotonic() - started_at,
        }
        
        report_json_path, report_md_path, _ = write_case_report(self.config.reports_dir, payload)
        
        elapsed = time.monotonic() - started_at
        return MedicalAnalysisResult(
            case_id=case_id,
            patient_code=patient_code,
            source_image=source,
            normalized_image=normalized_path,
            processed_image=processed_path,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
            detections=mock_detections,
            risk_level=risk_level,
            suspected_malignant=suspected_malignant,
            recommendation=recommendation,
            disclaimer=MEDICAL_DISCLAIMER,
            average_confidence=average_confidence,
            model_name=f"demo_{target_key}_classifier",
            quality_warnings=list(validation.quality_warnings) + ["⚠️ DEMO MODE: Kết quả mô phỏng - không dùng cho chẩn đoán thực tế"],
            stage_confidences={"validate": max(validation.modality_confidence, validation.body_region_confidence), "demo_inference": average_confidence},
            stage_messages={"validate": validation.message or "", "demo_inference": "Chạy mô phỏng (demo mode) - không dùng model thực tế."},
            gradcam_overlays=[],
            deidentified_source=None,
            dicom_info=None,
            analysis_time_seconds=elapsed,
            modality=modality,
            body_region=body_region,
            segmentation_mask_path=None,
            input_dimension="2d",
            analyzed_frame_count=1,
            representative_frame_index=0,
        )
    
    def _generate_mock_detections(self, target_key: str, validation: ValidationResult) -> list[DetectionFinding]:
        """Generate mock detections for demo mode based on target."""
        import random

        
        # Define mock labels per target
        mock_labels = {
            "brain": ["glioma", "meningioma", "pituitary", "no_tumor"],
            "lung": ["adenocarcinoma", "squamous_cell", "small_cell", "no_cancer"],
            "breast": ["invasive_ductal", "invasive_lobular", "dcis", "no_cancer"],
            "liver": ["hepatocellular", "cholangiocarcinoma", "metastasis", "no_cancer"],
            "stomach": ["adenocarcinoma", "gist", "lymphoma", "no_cancer"],
            "colorectal": ["adenocarcinoma", "carcinoid", "gist", "no_cancer"],
            "prostate": ["adenocarcinoma", "pin", "no_cancer"],
            "cervical": ["squamous_cell", "adenocarcinoma", "no_cancer"],
            "kidney": ["clear_cell", "papillary", "chromophobe", "no_cancer"],
            "pancreas": ["adenocarcinoma", "neuroendocrine", "no_cancer"],
            "thyroid": ["papillary", "follicular", "medullary", "no_cancer"],
        }
        
        labels = mock_labels.get(target_key, ["no_tumor"])
        num_detections = random.randint(1, 3)
        
        detections = []
        h, w = 512, 512  # Assume 512x512 image
        
        for i in range(num_detections):
            label = random.choice(labels)
            confidence = random.uniform(0.6, 0.95)
            
            # Generate random bbox
            x1 = random.randint(50, w - 150)
            y1 = random.randint(50, h - 150)
            x2 = x1 + random.randint(50, 100)
            y2 = y1 + random.randint(50, 100)
            
            # Generate simple segmentation (rectangle polygon)
            segmentation = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            
            detections.append(DetectionFinding(
                label=label,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                segmentation=tuple(segmentation),
                frame_index=0,
                timestamp_seconds=0.0,
            ))
        
        # Sort by confidence descending
        detections.sort(key=lambda x: x.confidence, reverse=True)
        return detections
