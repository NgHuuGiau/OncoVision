from __future__ import annotations

import argparse
import importlib
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

try:
    from training.model_paths import resolve_model_source, resolve_trained_model_path
except ModuleNotFoundError:
    from model_paths import resolve_model_source, resolve_trained_model_path

try:
    from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
except ModuleNotFoundError:
    from terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section

from PIL import Image

from utils.file_utils import ensure_project_directories
from utils.logger import get_logger


YOLO = None
ULTRALYTICS_IMPORT_ERROR = None
logger = get_logger(__name__)
RAW_IMAGES_DIR = Path("dataset/object_detection/raw/images")
RAW_LABELS_DIR = Path("dataset/object_detection/raw/labels")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _require_yolo():
    global YOLO, ULTRALYTICS_IMPORT_ERROR
    if YOLO is None and ULTRALYTICS_IMPORT_ERROR is None:
        try:
            YOLO = importlib.import_module("ultralytics").YOLO
        except Exception as exc:  # pragma: no cover
            ULTRALYTICS_IMPORT_ERROR = exc
    if YOLO is None:
        raise RuntimeError(f"Cannot initialize ultralytics/YOLO: {ULTRALYTICS_IMPORT_ERROR}")
    return YOLO


def _resolve_auto_label_model_path() -> Path:
    candidates = [
        resolve_trained_model_path(required=False, fallback="yolo11s.pt"),
        resolve_model_source("yolo11n.pt"),
    ]
    for candidate in candidates:
        resolved = resolve_model_source(candidate)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "No model found for auto labeling. Check models/trained/best.pt or models/pretrained/yolo11s.pt."
    )


def _iter_raw_images() -> list[Path]:
    if not RAW_IMAGES_DIR.exists():
        return []
    return sorted(path for path in RAW_IMAGES_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _to_yolo_line(class_id: int, bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> str:
    image_width, image_height = image_size
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(x1, image_width - 1))
    y1 = max(0.0, min(y1, image_height - 1))
    x2 = max(0.0, min(x2, image_width - 1))
    y2 = max(0.0, min(y2, image_height - 1))
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    x_center = x1 + (box_width / 2.0)
    y_center = y1 + (box_height / 2.0)
    return (
        f"{class_id} "
        f"{x_center / image_width:.6f} "
        f"{y_center / image_height:.6f} "
        f"{box_width / image_width:.6f} "
        f"{box_height / image_height:.6f}"
    )


def auto_label_raw_images(*, overwrite: bool = False, conf: float = 0.25, device: str = "cpu", batch_size: int = 16) -> dict[str, object]:
    ensure_project_directories()
    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = _iter_raw_images()
    if not image_paths:
        return {
            "images": 0,
            "generated": 0,
            "skipped_existing": 0,
            "no_detection": [],
            "model_path": None,
        }

    model_path = _resolve_auto_label_model_path()
    yolo_cls = _require_yolo()
    model = yolo_cls(str(model_path))
    generated = 0
    skipped_existing = 0
    no_detection: list[str] = []

    pending_paths: list[Path] = []
    for image_path in image_paths:
        label_path = RAW_LABELS_DIR / f"{image_path.stem}.txt"
        if label_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        pending_paths.append(image_path)

    if not pending_paths:
        return {
            "images": len(image_paths),
            "generated": 0,
            "skipped_existing": skipped_existing,
            "no_detection": [],
            "model_path": model_path,
        }

    for i in range(0, len(pending_paths), batch_size):
        batch_paths = pending_paths[i:i + batch_size]
        image_sizes = {}
        for p in batch_paths:
            with Image.open(p) as img:
                image_sizes[p] = img.size

        results = model.predict(source=[str(p) for p in batch_paths], conf=conf, device=device, verbose=False)
        for result, image_path in zip(results, batch_paths):
            lines: list[str] = []
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                lines.append(_to_yolo_line(class_id, (x1, y1, x2, y2), image_sizes[image_path]))
            if not lines:
                no_detection.append(image_path.name)
            else:
                (RAW_LABELS_DIR / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                generated += 1
                logger.info("Auto label created for %s", image_path.name)

    return {
        "images": len(image_paths),
        "generated": generated,
        "skipped_existing": skipped_existing,
        "no_detection": no_detection,
        "model_path": model_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto generate YOLO labels for dataset/object_detection/raw/images using existing model")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing label .txt files")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for parallel inference")
    args = parser.parse_args()

    for item in header("YOLO AUTO LABEL :: GENERATE COORDINATES FOR RAW IMAGES"):
        print(item)
    try:
        report = auto_label_raw_images(overwrite=args.overwrite, conf=args.conf, device=args.device, batch_size=args.batch_size)
    except Exception as exc:
        print(section("ERROR", RED))
        print(row("Reason cannot run", str(exc), RED, bounded=False))
        print(line(rule("-"), CYAN))
        print(section("TRY COMMANDS", CYAN))
        print(command_row(1, r".\.venv\Scripts\python training\download_models.py"))
        print(command_row(2, r".\.venv\Scripts\python training\auto_label_raw.py"))
        print(line(rule("="), CYAN))
        raise SystemExit(1)

    if report["images"] == 0:
        print(section("RESULT", YELLOW))
        print(row("Reason cannot run", "No images in dataset/object_detection/raw/images", YELLOW, bounded=False))
        print(line(rule("-"), CYAN))
        print(section("NEXT COMMANDS", CYAN))
        print(command_row(1, r".\.venv\Scripts\python training\prepare_dataset.py"))
        print(command_row(2, r".\.venv\Scripts\python training\auto_label_raw.py"))
        print(line(rule("="), CYAN))
        raise SystemExit(1)

    no_detection = report["no_detection"]
    print(section("RESULT", GREEN if report["generated"] else YELLOW))
    print(row("Total raw images", str(report["images"]), GREEN))
    print(row("Labels generated", str(report["generated"]), GREEN if report["generated"] else YELLOW))
    print(row("Skipped existing", str(report["skipped_existing"]), YELLOW if report["skipped_existing"] else GREEN))
    print(row("No detection", str(len(no_detection)), YELLOW if no_detection else GREEN))
    print(row("Model used", str(report["model_path"]) if report["model_path"] else "Unknown", CYAN, bounded=False))
    if no_detection:
        print(line(rule("-"), CYAN))
        print(section("NEEDS REVIEW", YELLOW))
        for name in no_detection[:8]:
            print(row("Image without coordinates", name, YELLOW, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("NEXT COMMANDS", CYAN))
    print(command_row(1, r".\.venv\Scripts\python training\validate_dataset.py"))
    print(command_row(2, r".\.venv\Scripts\python training\split_dataset.py"))
    print(command_row(3, r".\.venv\Scripts\python run_train.py"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
