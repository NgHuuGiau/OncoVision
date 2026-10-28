"""Đánh giá YOLO detector trên bộ bbox có nhãn dạng YOLO.

Bố cục bắt buộc của --data:
    images/*.jpg|png...   ảnh
    labels/*.txt          nhãn YOLO (class cx cy w h, chuẩn hóa 0-1)

Số mAP lấy từ ultralytics val (không tự viết lại AP); đây là số mức dataset,
chưa thay kiểm định lâm sàng theo bệnh nhân.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _validate_layout(data_dir: Path) -> tuple[Path, Path, list[Path]]:
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"Thiếu thư mục images/ hoặc labels/ trong {data_dir}")
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"Không có ảnh nào trong {images_dir}")
    missing_labels = [p.stem for p in images if not (labels_dir / f"{p.stem}.txt").is_file()]
    if missing_labels:
        raise ValueError(
            f"Thiếu nhãn cho {len(missing_labels)} ảnh, ví dụ: {', '.join(missing_labels[:5])}"
        )
    return images_dir, labels_dir, images


def _dataset_class_names(model: Any, label_files: list[Path]) -> dict[int, str]:
    """Tên lớp theo id nhãn, liên tục 0..max (ultralytics yêu cầu index liên tục)."""
    model_names = dict(getattr(model, "names", {}) or {})
    seen: set[int] = set()
    for path in label_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                seen.add(int(float(parts[0])))
    if not seen:
        raise ValueError("Không đọc được class id nào từ các file nhãn YOLO.")
    return {class_id: str(model_names.get(class_id, class_id)) for class_id in range(max(seen) + 1)}


def _run_yolo_val(model: Any, data_yaml: str, *, conf: float, iou: float, imgsz: int) -> Any:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return model.val(data=data_yaml, conf=conf, iou=iou, imgsz=imgsz, verbose=False, plots=False)


def evaluate_detector(
    model_path: str | Path,
    data_dir: str | Path,
    *,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
) -> dict[str, Any]:
    from ultralytics import YOLO

    root = Path(data_dir)
    images_dir, _labels_dir, images = _validate_layout(root)
    label_files = [root / "labels" / f"{image.stem}.txt" for image in images]

    model = YOLO(str(model_path))
    names = _dataset_class_names(model, label_files)
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "dataset.yaml"
        yaml_path.write_text(
            json.dumps({"path": str(root), "train": "images", "val": "images", "names": names}, ensure_ascii=False),
            encoding="utf-8",
        )
        metrics = _run_yolo_val(model, str(yaml_path), conf=conf, iou=iou, imgsz=imgsz)

    box = metrics.box
    per_class = []
    # box.p/r/ap50/ap theo vị trí trong ap_class_index (GT-present classes),
    # không phải class id — bộ thiếu lớp không GT sẽ lệch. Fallback: vị trí theo id xếp tăng dần.
    raw_class_index = getattr(box, "ap_class_index", None)
    ap_class_index = [int(class_id) for class_id in raw_class_index] if raw_class_index is not None else []
    rows: Iterable[tuple[int, Any, Any, Any, Any]]
    if ap_class_index and len(ap_class_index) == len(box.p):
        rows = zip(ap_class_index, box.p, box.r, box.ap50, box.ap)
    else:
        rows = (
            (class_id, box.p[i], box.r[i], box.ap50[i], box.ap[i])
            for i, (class_id, _label) in enumerate(sorted(names.items()))
            if i < len(box.p)
        )
    for class_id, precision, recall, ap50, ap in rows:
        per_class.append(
            {
                "label": names.get(int(class_id), str(class_id)),
                "precision": float(precision),
                "recall": float(recall),
                "ap50": float(ap50),
                "ap": float(ap),
            }
        )
    return {
        "map50": float(box.map50),
        "map": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "conf_threshold": conf,
        "match_iou": iou,
        "imgsz": imgsz,
        "image_count": len(images),
        "class_names": names,
        "model_path": str(model_path),
        "data_dir": str(root),
        "note": (
            "mAP tính bằng ultralytics val trên bộ bbox có nhãn; đây là số mức "
            "dataset, chưa phải kiểm định lâm sàng theo bệnh nhân."
        ),
    }


__all__ = ["evaluate_detector"]
