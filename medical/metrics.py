from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MedicalMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    accuracy: float
    f1_score: float


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_medical_metrics(truths: list[bool], predictions: list[bool]) -> MedicalMetrics:
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must have the same length")
    tp = tn = fp = fn = 0
    for truth, prediction in zip(truths, predictions):
        if truth and prediction:
            tp += 1
        elif truth and not prediction:
            fn += 1
        elif not truth and prediction:
            fp += 1
        else:
            tn += 1
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    recall = sensitivity
    accuracy = _safe_divide(tp + tn, len(truths))
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    return MedicalMetrics(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1_score=f1_score,
    )
