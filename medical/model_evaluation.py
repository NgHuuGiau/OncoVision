"""Đánh giá model classifier trên bộ ảnh có nhãn (thư mục/lớp).

Đây là số mức dataset, không thay thế kiểm định lâm sàng theo bệnh nhân
(run_medical.py validate-cohort); dùng để phát hiện model rỗng/sai trước
khi đưa vào sử dụng.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from medical.metrics import compute_multiclass_metrics

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def evaluate_classifier(
    wrapper: Any,
    data_dir: str | Path,
    *,
    per_class_limit: int | None = None,
) -> dict[str, Any]:
    """Chạy wrapper.predict top-1 trên data_dir/<label>/* và trả báo cáo chỉ số.

    - data_dir phải có đúng các thư mục con trùng class_labels của model;
      thư mục lạ gây lỗi để không lệch tỷ lệ.
    - per_class_limit: giới hạn số ảnh mỗi lớp (đo nhanh trên máy CPU).
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Thư mục dữ liệu không tồn tại: {root}")
    labels = [str(item) for item in wrapper.class_labels]
    subdirs = sorted(entry for entry in root.iterdir() if entry.is_dir())
    unknown = [entry.name for entry in subdirs if entry.name not in labels]
    if unknown:
        raise ValueError(
            f"Thư mục lớp không khớp nhãn model {labels}: {', '.join(unknown)}"
        )
    missing = [label for label in labels if label not in {entry.name for entry in subdirs}]
    if missing:
        raise ValueError(f"Thiếu thư mục lớp: {', '.join(missing)}")

    label_to_index = {label: index for index, label in enumerate(labels)}
    truths: list[int] = []
    predictions: list[int] = []
    for subdir in subdirs:
        files = sorted(
            path for path in subdir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if per_class_limit is not None:
            files = files[: max(0, per_class_limit)]
        for path in files:
            result = wrapper.predict(path, top_k=1)[0]
            truths.append(label_to_index[subdir.name])
            predictions.append(label_to_index[str(result["label"])])
    if not truths:
        raise ValueError(f"Không tìm thấy ảnh hợp lệ nào trong {root}")

    metrics = compute_multiclass_metrics(truths, predictions, labels)
    metrics["per_class"] = [asdict(item) for item in metrics["per_class"]]
    return {
        **metrics,
        "model_labels": labels,
        "image_size": getattr(wrapper, "image_size", None),
        "data_dir": str(root),
        "per_class_limit": per_class_limit,
        "note": (
            "Chỉ số mức dataset trên bộ ảnh có nhãn; chưa phải kiểm định "
            "lâm sàng theo bệnh nhân (dùng validate-cohort cho mục đó)."
        ),
    }


__all__ = ["evaluate_classifier"]
