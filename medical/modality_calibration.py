from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

from medical.dataset import infer_medical_upload_context
from medical.training import MedicalTrainingPaths, medical_training_paths
from medical.validator import DEFAULT_MEDICAL_SETTINGS_PATH, assess_image_quality, get_modality_tuning, validate_image
from utils.file_utils import load_yaml, save_yaml


@dataclass(frozen=True)
class ModalityCalibrationSummary:
    modality: str
    image_count: int
    average_modality_confidence: float
    median_modality_confidence: float
    average_body_confidence: float
    median_body_confidence: float
    average_quality_score: float
    median_quality_score: float
    warning_rate: float
    proposed_tuning: dict[str, Any]


def _score_quantile(values: list[float], fraction: float, fallback: float) -> float:
    if not values:
        return fallback
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _propose_tuning(modality: str, modality_scores: list[float], body_scores: list[float], quality_scores: list[float], warning_rate: float) -> dict[str, Any]:
    base = get_modality_tuning(modality)
    certainty = _score_quantile(modality_scores, 0.25, float(base["certainty_threshold"]))
    body_certainty = _score_quantile(body_scores, 0.25, float(base["medium_threshold"]))
    quality = _score_quantile(quality_scores, 0.25, float(base["quality_threshold"]))

    certainty = round(max(0.50, min(0.95, certainty - 0.03)), 2)
    medium = round(max(0.35, min(certainty - 0.08, body_certainty - 0.03)), 2)
    if medium >= certainty:
        medium = round(max(0.35, certainty - 0.12), 2)
    quality = round(max(0.35, min(0.75, quality - (0.02 if warning_rate > 0.4 else 0.0))), 2)
    contrast_boost = float(base["contrast_boost"])
    if average_or_zero(quality_scores) < 0.50 or warning_rate > 0.35:
        contrast_boost = round(min(1.45, contrast_boost + 0.05), 2)
    elif average_or_zero(quality_scores) > 0.70:
        contrast_boost = round(max(1.0, contrast_boost - 0.03), 2)

    return {
        "certainty_threshold": certainty,
        "medium_threshold": medium,
        "quality_threshold": quality,
        "contrast_boost": contrast_boost,
        "normalize": base["normalize"],
    }


def average_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def calibrate_modality_tuning(
    dataset_root: str | Path | None = None,
    *,
    settings_path: str | Path = DEFAULT_MEDICAL_SETTINGS_PATH,
) -> dict[str, Any]:
    settings = load_yaml(settings_path).get("medical", {})
    if not isinstance(settings, dict):
        settings = {}
    tuning_settings = settings.get("modality_tuning", {})
    default_tuning = tuning_settings.get("default") if isinstance(tuning_settings, dict) else None
    if not isinstance(default_tuning, dict):
        default_tuning = {
            "certainty_threshold": settings.get("certainty_threshold", 0.55),
            "medium_threshold": settings.get("classify_medium_risk_threshold", 0.45),
            "quality_threshold": 0.45,
            "contrast_boost": 1.0,
            "normalize": "default",
        }
    paths = medical_training_paths() if dataset_root is None else MedicalTrainingPaths(dataset_root=dataset_root)

    modality_samples: dict[str, dict[str, Any]] = {}
    images = [path for path in sorted(paths.dataset_root.rglob("*")) if path.is_file()]

    for image_path in images:
        _, modality = infer_medical_upload_context(image_path)
        if not modality:
            continue
        validation = validate_image(image_path, min_confidence=0.0)
        if validation.status == "error":
            continue
        warnings, quality_score = assess_image_quality(image_path)
        bucket = modality_samples.setdefault(
            validation.modality or modality.lower(),
            {
                "modality_confidences": [],
                "body_confidences": [],
                "quality_scores": [],
                "warning_count": 0,
                "image_count": 0,
            },
        )
        bucket["image_count"] = int(cast(Any, bucket["image_count"])) + 1
        bucket["modality_confidences"].append(float(validation.modality_confidence))
        bucket["body_confidences"].append(float(validation.body_region_confidence))
        bucket["quality_scores"].append(float(quality_score))
        bucket["warning_count"] = int(cast(Any, bucket["warning_count"])) + (1 if warnings else 0)

    summaries: dict[str, ModalityCalibrationSummary] = {}
    for modality, stats in modality_samples.items():
        modality_confidences = cast(list[float], list(stats["modality_confidences"]))
        body_confidences = cast(list[float], list(stats["body_confidences"]))
        quality_scores = cast(list[float], list(stats["quality_scores"]))
        image_count = int(cast(Any, stats["image_count"]))
        warning_count = int(cast(Any, stats["warning_count"]))
        warning_rate = float(warning_count) / max(1, image_count)
        summaries[modality] = ModalityCalibrationSummary(
            modality=modality,
            image_count=image_count,
            average_modality_confidence=round(average_or_zero(modality_confidences), 4),
            median_modality_confidence=round(median(modality_confidences), 4) if modality_confidences else 0.0,
            average_body_confidence=round(average_or_zero(body_confidences), 4),
            median_body_confidence=round(median(body_confidences), 4) if body_confidences else 0.0,
            average_quality_score=round(average_or_zero(quality_scores), 4),
            median_quality_score=round(median(quality_scores), 4) if quality_scores else 0.0,
            warning_rate=round(warning_rate, 4),
            proposed_tuning=_propose_tuning(modality, modality_confidences, body_confidences, quality_scores, warning_rate),
        )

    report = {
        "dataset_root": str(paths.dataset_root),
        "sample_count": sum(item.image_count for item in summaries.values()),
        "modality_tuning": {
            "default": default_tuning,
            **{name: summary.proposed_tuning for name, summary in summaries.items()},
        },
        "modality_stats": {name: summary.__dict__ for name, summary in summaries.items()},
    }
    return report


def apply_calibrated_modality_tuning(
    dataset_root: str | Path | None = None,
    *,
    settings_path: str | Path = DEFAULT_MEDICAL_SETTINGS_PATH,
) -> dict[str, Any]:
    report = calibrate_modality_tuning(dataset_root, settings_path=settings_path)
    settings_path = Path(settings_path)
    payload = load_yaml(settings_path)
    if not isinstance(payload, dict):
        payload = {}
    medical = payload.get("medical", {})
    if not isinstance(medical, dict):
        medical = {}
    medical["modality_tuning"] = report["modality_tuning"]
    payload["medical"] = medical
    save_yaml(settings_path, payload)
    return report
