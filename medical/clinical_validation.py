from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

from medical.metrics import compute_multiclass_metrics


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * ((proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def plan_binary_validation_sample_size(
    *,
    sensitivity: float,
    specificity: float,
    prevalence: float,
    margin: float = 0.05,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Ước lượng số ca cho CI hai phía của sensitivity/specificity."""
    for name, value in {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "prevalence": prevalence,
        "confidence": confidence,
    }.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} phải nằm trong khoảng (0, 1).")
    if not 0.0 < margin < 0.5:
        raise ValueError("margin phải nằm trong khoảng (0, 0.5).")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    positive_cases = max(1, math.ceil(z * z * sensitivity * (1.0 - sensitivity) / (margin * margin)))
    negative_cases = max(1, math.ceil(z * z * specificity * (1.0 - specificity) / (margin * margin)))
    total = max(
        math.ceil(positive_cases / prevalence),
        math.ceil(negative_cases / (1.0 - prevalence)),
    )
    return {
        "method": "Normal approximation for two-sided binomial confidence intervals",
        "confidence": confidence,
        "margin": margin,
        "expected_sensitivity": sensitivity,
        "expected_specificity": specificity,
        "expected_prevalence": prevalence,
        "minimum_positive_cases": positive_cases,
        "minimum_negative_cases": negative_cases,
        "minimum_total_cases": total,
        "planned_positive_cases": math.ceil(total * prevalence),
        "planned_negative_cases": total - math.ceil(total * prevalence),
        "note": "Ước lượng cho một bài toán nhị phân; cần statistician phê duyệt protocol chính thức.",
    }


def build_independent_validation_report(
    csv_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    cohort_id: str,
) -> Path:
    if not cohort_id.strip():
        raise ValueError("cohort_id không được để trống.")
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"patient_id", "truth", "prediction"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV kiểm định phải có patient_id, truth và prediction.")
    patient_ids = [str(row["patient_id"]).strip() for row in rows]
    if any(not patient_id for patient_id in patient_ids):
        raise ValueError("patient_id không được để trống.")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Mỗi bệnh nhân chỉ được xuất hiện một lần trong cohort kiểm định.")
    truths_text = [str(row["truth"]).strip() for row in rows]
    predictions_text = [str(row["prediction"]).strip() for row in rows]
    if any(not value for value in truths_text + predictions_text):
        raise ValueError("truth và prediction không được để trống.")
    labels = sorted(set(truths_text) | set(predictions_text))
    if len(labels) < 2:
        raise ValueError("Cohort kiểm định phải có ít nhất hai lớp.")
    label_to_index = {label: index for index, label in enumerate(labels)}
    metrics = compute_multiclass_metrics(
        [label_to_index[value] for value in truths_text],
        [label_to_index[value] for value in predictions_text],
        labels,
    )
    serializable_metrics = {
        **{key: value for key, value in metrics.items() if key != "per_class"},
        "per_class": [vars(item) for item in metrics["per_class"]],
    }
    confusion = metrics["confusion_matrix"]
    per_class_intervals: dict[str, dict[str, list[float]]] = {}
    for index, label in enumerate(labels):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(labels))) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        true_negative = len(rows) - true_positive - false_positive - false_negative
        per_class_intervals[label] = {
            "sensitivity_95_ci": _wilson_interval(true_positive, true_positive + false_negative),
            "specificity_95_ci": _wilson_interval(true_negative, true_negative + false_positive),
        }
    confidence_intervals = {
        "method": "Wilson score, 95%",
        "accuracy_95_ci": _wilson_interval(metrics["correct"], metrics["total"]),
        "per_class": per_class_intervals,
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_id": cohort_id.strip(),
        "cohort_csv_sha256": sha256_file(csv_path),
        "independent_cohort": True,
        "patient_level": True,
        "patient_count": len(rows),
        "class_labels": labels,
        "model_path": str(Path(model_path)),
        "model_sha256": sha256_file(model_path),
        "metrics": serializable_metrics,
        "confidence_intervals": confidence_intervals,
        "approved_by": None,
        "approved_at": None,
        "dataset_provenance": None,
        "ground_truth_method": None,
        "sample_size_justification": None,
        "site_count": None,
        "approval_note": "Cần đơn vị lâm sàng độc lập xác nhận nguồn cohort và phê duyệt báo cáo.",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def clinical_validation_status(report_path: str | Path | None, model_path: str | Path) -> tuple[bool, str]:
    if report_path is None or not Path(report_path).is_file():
        return False, "chưa có báo cáo kiểm định cohort độc lập"
    try:
        report: dict[str, Any] = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "báo cáo kiểm định không đọc được"
    try:
        current_hash = sha256_file(model_path)
    except OSError:
        return False, "không đọc được model để đối chiếu hash kiểm định"
    if report.get("model_sha256") != current_hash:
        return False, "báo cáo kiểm định không khớp hash model đang chạy"
    if not report.get("independent_cohort") or not report.get("patient_level"):
        return False, "báo cáo không xác nhận cohort độc lập theo bệnh nhân"
    if not report.get("approved_by") or not report.get("approved_at"):
        return False, "báo cáo chưa được đơn vị lâm sàng độc lập phê duyệt"
    required_protocol_fields = ("dataset_provenance", "ground_truth_method", "sample_size_justification")
    try:
        site_count = int(report.get("site_count") or 0)
    except (TypeError, ValueError):
        site_count = 0
    if any(not report.get(field) for field in required_protocol_fields) or site_count < 1:
        return False, "báo cáo thiếu nguồn dữ liệu, chuẩn tham chiếu hoặc biện minh cỡ mẫu"
    return True, "đã có báo cáo cohort độc lập được phê duyệt và khớp hash model"


__all__ = [
    "build_independent_validation_report",
    "clinical_validation_status",
    "plan_binary_validation_sample_size",
    "sha256_file",
]
