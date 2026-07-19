from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PerClassMetrics:
    label: str
    precision: float
    recall: float
    f1_score: float
    support: int
    sensitivity: float = 0.0
    specificity: float = 0.0


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
    balanced_accuracy: float = 0.0
    matthews_correlation: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    ece: float = 0.0
    average_confidence: float = 0.0


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
    balanced_acc = (sensitivity + specificity) / 2.0
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-6))
    mcc = mcc_num / mcc_den
    return MedicalMetrics(
        true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn,
        sensitivity=sensitivity, specificity=specificity, precision=precision, recall=recall,
        accuracy=accuracy, f1_score=f1_score, balanced_accuracy=balanced_acc, matthews_correlation=mcc,
    )


def compute_multiclass_metrics(truths: list[int], predictions: list[int], class_labels: list[str]) -> dict[str, Any]:
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must have the same length")
    if not truths:
        raise ValueError("truths and predictions cannot be empty")
    num_classes = len(class_labels)
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for truth, pred in zip(truths, predictions):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
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
        sensitivity = recall
        specificity = _safe_divide(confusion[i][i], max(confusion[i][i] + sum(confusion[j][i] for j in range(num_classes) if j != i), 1e-6))
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        per_class.append(PerClassMetrics(label=label, precision=precision, recall=recall, f1_score=f1, support=support, sensitivity=sensitivity, specificity=specificity))
    accuracy = correct / len(truths)
    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes
    micro_p = _safe_divide(correct, correct + sum(confusion[i][j] for i in range(num_classes) for j in range(num_classes) if i != j))
    micro_r = accuracy
    micro_f1 = _safe_divide(2 * micro_p * micro_r, micro_p + micro_r)
    return {
        "accuracy": accuracy, "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "per_class": per_class, "confusion_matrix": confusion, "support": total_support, "correct": correct, "total": len(truths),
    }


ArrayLike = Sequence[float] | Sequence[int] | np.ndarray
ScoreMatrix = Sequence[Sequence[float]] | np.ndarray


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x))


def _as_score_matrix(scores: ScoreMatrix) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("scores must be a 2D array of shape (n_samples, n_classes)")
    return arr


def _validate_inputs(y_true: ArrayLike, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(y_true, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("y_true must be a 1D sequence of integer class indices")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same number of samples")
    if labels.shape[0] == 0:
        raise ValueError("y_true and scores cannot be empty")
    return labels


def _binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = float(np.count_nonzero(labels == 1))
    negatives = float(labels.shape[0] - positives)
    if positives == 0.0 or negatives == 0.0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    tpr = np.concatenate(([0.0], tps / positives))
    fpr = np.concatenate(([0.0], fps / negatives))
    return _trapezoid(tpr, fpr)


def _binary_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = float(np.count_nonzero(labels == 1))
    if positives == 0.0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    recall = tps / positives
    precision = np.where((tps + fps) > 0, tps / np.maximum(tps + fps, 1), 1.0)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([precision[0]], precision))
    return _trapezoid(precision, recall)


def compute_roc_auc(y_true: ArrayLike, y_scores: ScoreMatrix, class_labels: list[str] | None = None) -> dict[str, Any]:
    scores = _as_score_matrix(y_scores)
    labels = _validate_inputs(y_true, scores)
    num_classes = scores.shape[1]
    if class_labels is not None and len(class_labels) != num_classes:
        raise ValueError("class_labels length must match the number of score columns")
    names = class_labels if class_labels is not None else [str(i) for i in range(num_classes)]
    per_class: dict[str, float] = {}
    for c in range(num_classes):
        binary_labels = (labels == c).astype(np.int64)
        per_class[names[c]] = _binary_roc_auc(binary_labels, scores[:, c])
    defined = [v for v in per_class.values() if not np.isnan(v)]
    macro = float(np.mean(defined)) if defined else float("nan")
    return {"per_class": per_class, "macro": macro, "num_classes": num_classes}


def compute_pr_auc(y_true: ArrayLike, y_scores: ScoreMatrix, class_labels: list[str] | None = None) -> dict[str, Any]:
    scores = _as_score_matrix(y_scores)
    labels = _validate_inputs(y_true, scores)
    num_classes = scores.shape[1]
    if class_labels is not None and len(class_labels) != num_classes:
        raise ValueError("class_labels length must match the number of score columns")
    names = class_labels if class_labels is not None else [str(i) for i in range(num_classes)]
    per_class: dict[str, float] = {}
    for c in range(num_classes):
        binary_labels = (labels == c).astype(np.int64)
        per_class[names[c]] = _binary_pr_auc(binary_labels, scores[:, c])
    defined = [v for v in per_class.values() if not np.isnan(v)]
    macro = float(np.mean(defined)) if defined else float("nan")
    return {"per_class": per_class, "macro": macro, "num_classes": num_classes}


def compute_calibration_curve(y_true: ArrayLike, y_scores: ScoreMatrix, n_bins: int = 10) -> dict[str, Any]:
    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    scores = _as_score_matrix(y_scores)
    labels = _validate_inputs(y_true, scores)
    n_samples = labels.shape[0]
    confidences = scores.max(axis=1)
    predictions = scores.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, bin_edges[1:-1], right=False), 0, n_bins - 1)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_confidences = np.full(n_bins, np.nan, dtype=np.float64)
    bin_accuracies = np.full(n_bins, np.nan, dtype=np.float64)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.count_nonzero(mask))
        bin_counts[b] = count
        if count == 0:
            continue
        mean_conf = float(confidences[mask].mean())
        mean_acc = float(correct[mask].mean())
        bin_confidences[b] = mean_conf
        bin_accuracies[b] = mean_acc
        ece += (count / n_samples) * abs(mean_acc - mean_conf)
    return {
        "bin_counts": bin_counts, "bin_confidences": bin_confidences, "bin_accuracies": bin_accuracies,
        "bin_edges": bin_edges, "ece": float(ece), "n_bins": n_bins,
    }


__all__ = [
    "MedicalMetrics", "PerClassMetrics", "compute_medical_metrics", "compute_multiclass_metrics",
    "compute_roc_auc", "compute_pr_auc", "compute_calibration_curve",
]
