from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PerClassMetrics:
    label: str
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class MedicalMetrics:
    accuracy: float
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    per_class: list[PerClassMetrics] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    sensitivity: float = 0.0
    specificity: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


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


def compute_multiclass_metrics(
    truths: list[int],
    predictions: list[int],
    class_labels: list[str],
) -> dict[str, Any]:
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must have the same length")
    if not truths:
        raise ValueError("truths and predictions cannot be empty")

    num_classes = len(class_labels)
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for truth, pred in zip(truths, predictions):
        confusion[truth][pred] += 1

    per_class = []
    macro_p = macro_r = macro_f1 = 0.0
    total_support = 0
    correct = 0

    for i, label in enumerate(class_labels):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(num_classes)) - tp
        fn = sum(confusion[i]) - tp
        support = sum(confusion[i])
        total_support += support
        correct += tp

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        per_class.append(
            PerClassMetrics(
                label=label,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=support,
            )
        )

    accuracy = correct / len(truths)
    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes

    micro_p = _safe_divide(correct, correct + sum(confusion[i][j] for i in range(num_classes) for j in range(num_classes) if i != j))
    micro_r = accuracy
    micro_f1 = _safe_divide(2 * micro_p * micro_r, micro_p + micro_r)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "support": total_support,
        "correct": correct,
        "total": len(truths),
    }


__all__ = [
    "MedicalMetrics",
    "PerClassMetrics",
    "compute_medical_metrics",
    "compute_multiclass_metrics",
]
