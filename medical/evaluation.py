"""Danh gia model medical tren TEST SET giu rieng, cong bo metric per-class.

Module nay tao bao cao day du cho model phan loai ung thu:
- Metric tong hop: accuracy, macro/micro precision/recall/F1, macro ROC-AUC, PR-AUC.
- Metric per-class: precision, recall (=sensitivity), specificity, F1, support,
  ROC-AUC, PR-AUC.
- Confusion matrix.
- Xuat JSON + Markdown vao output/medical/reports/.

Danh gia CHI dung split "test" (khong dung train/val) de tranh ro ri.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from medical.classifier import load_medical_classifier
from medical.cnn_classifier import is_cnn_classifier_path, load_cnn_classifier
from medical.metrics import compute_multiclass_metrics, compute_pr_auc, compute_roc_auc
from medical.training import MedicalTrainingPaths, _samples_for_split, medical_training_paths


def _resolve_model(model_path: Path):
    if is_cnn_classifier_path(model_path):
        return load_cnn_classifier(model_path), True
    return load_medical_classifier(model_path), False


def _predict_scores(model: Any, image_path: Path, class_labels: tuple[str, ...]) -> tuple[int, np.ndarray]:
    """Tra ve (predicted_index, score_vector) do dai = so lop."""
    predictions = model.predict(image_path, top_k=len(class_labels))
    scores = np.zeros(len(class_labels), dtype=np.float64)
    label_to_index = {name: idx for idx, name in enumerate(class_labels)}
    top_label = ""
    top_conf = -1.0
    for item in predictions:
        if isinstance(item, dict):
            label = str(item.get("label", ""))
            conf = float(item.get("confidence", 0.0))
        else:
            label = str(getattr(item, "label", ""))
            conf = float(getattr(item, "confidence", 0.0))
        if label in label_to_index:
            scores[label_to_index[label]] = conf
        if conf > top_conf:
            top_conf = conf
            top_label = label
    predicted_index = label_to_index.get(top_label, int(np.argmax(scores)))
    return predicted_index, scores


def evaluate_on_test_set(
    paths: MedicalTrainingPaths | None = None,
    *,
    model_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Danh gia model tren split giu rieng va tra ve bao cao metric per-class."""
    paths = paths or medical_training_paths()
    resolved_model_path = Path(model_path) if model_path else paths.trained_model_path
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model: {resolved_model_path}")

    samples = _samples_for_split(paths, split)
    if not samples:
        raise FileNotFoundError(
            f"Khong co du lieu '{split}' de danh gia. Hay chay split-dataset truoc."
        )

    model, _is_cnn = _resolve_model(resolved_model_path)
    class_labels = paths.class_names

    truths: list[int] = []
    preds: list[int] = []
    score_rows: list[np.ndarray] = []
    for image_path, class_index in samples:
        predicted_index, scores = _predict_scores(model, image_path, class_labels)
        truths.append(class_index)
        preds.append(predicted_index)
        score_rows.append(scores)

    score_matrix = np.vstack(score_rows) if score_rows else np.zeros((0, len(class_labels)))
    multiclass = compute_multiclass_metrics(truths, preds, list(class_labels))

    try:
        roc = compute_roc_auc(truths, score_matrix, list(class_labels))
    except Exception:
        roc = {"per_class": {}, "macro": float("nan")}
    try:
        pr = compute_pr_auc(truths, score_matrix, list(class_labels))
    except Exception:
        pr = {"per_class": {}, "macro": float("nan")}

    per_class_report: list[dict[str, Any]] = []
    for entry in multiclass["per_class"]:
        label = entry.label
        per_class_report.append({
            "label": label,
            "support": int(entry.support),
            "precision": float(entry.precision),
            "recall": float(entry.recall),
            "sensitivity": float(entry.sensitivity),
            "specificity": float(entry.specificity),
            "f1_score": float(entry.f1_score),
            "roc_auc": float(roc["per_class"].get(label, float("nan"))),
            "pr_auc": float(pr["per_class"].get(label, float("nan"))),
        })

    return {
        "model_path": str(resolved_model_path),
        "split": split,
        "num_samples": len(samples),
        "class_labels": list(class_labels),
        "accuracy": float(multiclass["accuracy"]),
        "macro_precision": float(multiclass["macro_precision"]),
        "macro_recall": float(multiclass["macro_recall"]),
        "macro_f1": float(multiclass["macro_f1"]),
        "micro_f1": float(multiclass["micro_f1"]),
        "macro_roc_auc": float(roc["macro"]),
        "macro_pr_auc": float(pr["macro"]),
        "per_class": per_class_report,
        "confusion_matrix": multiclass["confusion_matrix"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Bao Cao Danh Gia Model Medical (Test Set)")
    lines.append("")
    lines.append(f"- Model: `{report['model_path']}`")
    lines.append(f"- Split: `{report['split']}`")
    lines.append(f"- So mau: {report['num_samples']}")
    lines.append(f"- Danh gia luc: {report['evaluated_at']}")
    lines.append("")
    lines.append("## Metric Tong Hop")
    lines.append("")
    lines.append("| Metric | Gia tri |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {report['accuracy']:.4f} |")
    lines.append(f"| Macro Precision | {report['macro_precision']:.4f} |")
    lines.append(f"| Macro Recall (Sensitivity) | {report['macro_recall']:.4f} |")
    lines.append(f"| Macro F1 | {report['macro_f1']:.4f} |")
    lines.append(f"| Micro F1 | {report['micro_f1']:.4f} |")
    lines.append(f"| Macro ROC-AUC | {report['macro_roc_auc']:.4f} |")
    lines.append(f"| Macro PR-AUC | {report['macro_pr_auc']:.4f} |")
    lines.append("")
    lines.append("## Metric Per-Class")
    lines.append("")
    lines.append("| Lop | Support | Precision | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for entry in report["per_class"]:
        lines.append(
            f"| {entry['label']} | {entry['support']} | {entry['precision']:.4f} | "
            f"{entry['sensitivity']:.4f} | {entry['specificity']:.4f} | {entry['f1_score']:.4f} | "
            f"{entry['roc_auc']:.4f} | {entry['pr_auc']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_evaluation_report(
    report: dict[str, Any],
    reports_dir: str | Path = "output/medical/reports",
) -> tuple[Path, Path]:
    """Ghi bao cao ra JSON + Markdown, tra ve (json_path, md_path)."""
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"evaluation_{stamp}.json"
    md_path = out_dir / f"evaluation_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_format_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = ["evaluate_on_test_set", "write_evaluation_report"]
