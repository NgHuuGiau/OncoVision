from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from medical.classifier import iter_medical_image_paths, load_medical_classifier


@dataclass(frozen=True)
class ActiveLearningConfig:
    uncertainty_threshold: float = 0.15
    max_samples_per_batch: int = 10
    output_dir: Path = field(default_factory=lambda: Path("output/medical/active_learning"))
    enabled: bool = True


@dataclass(frozen=True)
class PredictionResult:
    image_path: Path
    confidence: float


def _extract_confidence(result: Any) -> float:
    if hasattr(result, "confidence"):
        return float(result.confidence)
    if isinstance(result, dict):
        return float(result.get("confidence", 0.0))
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        return float(result[1])
    raise TypeError(f"Unsupported prediction result type: {type(result)}")


def _extract_image_path(result: Any) -> Path:
    if hasattr(result, "image_path"):
        return Path(result.image_path)
    if isinstance(result, dict):
        return Path(result.get("image_path", ""))
    if isinstance(result, (tuple, list)) and len(result) >= 1:
        return Path(result[0])
    raise TypeError(f"Unsupported prediction result type: {type(result)}")


def select_uncertain_samples(
    predictions: Iterable[Any],
    *,
    config: ActiveLearningConfig,
) -> list[tuple[Path, float, float]]:
    if not config.enabled:
        return []

    candidates: list[tuple[float, Path, float]] = []
    for result in predictions:
        confidence = _extract_confidence(result)
        uncertainty_score = abs(confidence - 0.5)
        if uncertainty_score <= config.uncertainty_threshold:
            candidates.append((uncertainty_score, _extract_image_path(result), confidence))

    candidates.sort(key=lambda item: item[0])
    limited = candidates[: config.max_samples_per_batch]
    return [(path, confidence, uncertainty_score) for uncertainty_score, path, confidence in limited]


class ActiveLearningLogger:
    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._total_reviewed = 0
        self._flagged_count = 0
        self._confirmed_count = 0
        self._file_handle = open(self.log_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file_handle)
        if self.log_path.stat().st_size == 0:
            self._writer.writerow(["timestamp", "image_path", "confidence", "uncertainty_score", "status"])

    def log_uncertain_sample(self, image_path: str | Path, confidence: float, uncertainty_score: float) -> None:
        self._total_reviewed += 1
        self._flagged_count += 1
        self._writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            str(Path(image_path)),
            f"{confidence:.6f}",
            f"{uncertainty_score:.6f}",
            "flagged",
        ])
        self._file_handle.flush()

    def confirm_sample(self, image_path: str | Path, *, correct_label: str | None = None) -> None:
        self._confirmed_count += 1
        self._writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            str(Path(image_path)),
            "",
            "",
            "confirmed",
        ])
        if correct_label is not None:
            self._writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                str(Path(image_path)),
                correct_label,
                "",
                "labeled",
            ])
        self._file_handle.flush()

    def get_stats(self) -> dict[str, int]:
        return {
            "total_samples_reviewed": self._total_reviewed,
            "samples_flagged": self._flagged_count,
            "samples_confirmed": self._confirmed_count,
        }

    def close(self) -> None:
        self._file_handle.close()

    def __enter__(self) -> ActiveLearningLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def suggest_active_learning_samples(
    image_dir: str | Path,
    *,
    config: ActiveLearningConfig,
    model_path: str | Path,
) -> list[tuple[Path, float, float]]:
    if not config.enabled:
        return []

    model = load_medical_classifier(model_path)
    predictions: list[PredictionResult] = []
    for image_path in iter_medical_image_paths(image_dir):
        try:
            top_results = model.predict(image_path, top_k=1)
            if top_results:
                top = top_results[0]
                predictions.append(PredictionResult(image_path=image_path, confidence=top.confidence))
        except Exception as error:
            logging.getLogger("medical.active_learning").warning("Failed inference on %s: %s", image_path, error)

    uncertain = select_uncertain_samples(predictions, config=config)

    output_path = config.output_dir / "active_learning_candidates.csv"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "confidence", "uncertainty_score"])
        for path, confidence, uncertainty_score in uncertain:
            writer.writerow([str(path), f"{confidence:.6f}", f"{uncertainty_score:.6f}"])

    return uncertain
