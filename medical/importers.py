from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image


def import_isic_2016_part3b_to_yolo(
    source_dir: str | Path,
    *,
    target_images_dir: str | Path,
    target_labels_dir: str | Path,
    diagnosis_csv_path: str | Path | None = None,
    metadata_output_path: str | Path | None = None,
) -> dict[str, int]:
    source_root = Path(source_dir)
    images_dir = Path(target_images_dir)
    labels_dir = Path(target_labels_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_map = _load_diagnosis_csv(diagnosis_csv_path) if diagnosis_csv_path else {}
    imported = 0
    skipped = 0
    metadata_rows: list[dict[str, str]] = []

    for image_path in sorted(source_root.glob("*.jpg")):
        mask_path = source_root / f"{image_path.stem}_Segmentation.png"
        if not mask_path.exists():
            skipped += 1
            continue
        bbox = _bbox_from_mask(mask_path)
        if bbox is None:
            skipped += 1
            continue
        image_target = images_dir / image_path.name
        label_target = labels_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, image_target)
        with Image.open(image_path) as image:
            label_target.write_text(bbox_to_yolo_line(0, bbox, image.size), encoding="utf-8")
        metadata_rows.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_target),
                "label_path": str(label_target),
                "diagnosis": diagnosis_map.get(image_path.stem, ""),
            }
        )
        imported += 1

    if metadata_output_path is not None:
        _write_metadata_csv(Path(metadata_output_path), metadata_rows)
    return {"imported": imported, "skipped": skipped}


def _load_diagnosis_csv(csv_path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2:
                result[row[0].strip()] = row[1].strip()
    return result


def _bbox_from_mask(mask_path: Path) -> tuple[int, int, int, int] | None:
    with Image.open(mask_path) as mask:
        gray = mask.convert("L")
        pixels = gray.load()
        width, height = gray.size
        xs: list[int] = []
        ys: list[int] = []
        for y in range(height):
            for x in range(width):
                if pixels[x, y] > 0:
                    xs.append(x)
                    ys.append(y)
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def bbox_to_yolo_line(class_id: int, bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> str:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    x_center = (x1 + (box_width / 2.0)) / width
    y_center = (y1 + (box_height / 2.0)) / height
    normalized_w = box_width / width
    normalized_h = box_height / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {normalized_w:.6f} {normalized_h:.6f}\n"


def _write_metadata_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "label_path", "diagnosis"])
        writer.writeheader()
        writer.writerows(rows)
