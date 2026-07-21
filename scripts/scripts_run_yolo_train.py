"""Train YOLO detection nhe (yolo11s) cho dataset YOLO da split.
Supports resuming from the last checkpoint if available.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

from ultralytics import YOLO

PROCESSED = Path("dataset/processed")
CLASSES = [
    "Ung thư gan", "Ung thư phổi", "Ung thư vú", "Ung thư dạ dày",
    "Ung thư đại trực tràng", "Ung thư tuyến tiền liệt", "Ung thư cổ tử cung",
]
YAML_PATH = PROCESSED / "data.yaml"
BASE_MODEL = "models/pretrained/yolo11s.pt"
RUN_DIR = Path("runs/detect/medical_yolo")
OUT_MODEL = "models/trained/medical_yolo_detect.pt"

yaml_text = (
    f"path: {PROCESSED.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    f"nc: {len(CLASSES)}\n"
    "names: " + str(CLASSES) + "\n"
)
YAML_PATH.write_text(yaml_text, encoding="utf-8")
print("data.yaml ->", YAML_PATH)

def main() -> None:
    yaml_text = (
        f"path: {PROCESSED.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        "names: " + str(CLASSES) + "\n"
    )
    YAML_PATH.write_text(yaml_text, encoding="utf-8")
    print("data.yaml ->", YAML_PATH)

    last_ckpt = RUN_DIR / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"[resume] Tim thay checkpoint cu: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume = True
    else:
        print(f"[start] Train tu pretrained: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
        resume = False

    model.train(
        data=str(YAML_PATH),
        epochs=10,
        imgsz=320,
        batch=4,
        name="medical_yolo",
        patience=5,
        device=0,
        resume=resume,
    )
    model.save(str(OUT_MODEL))
    print("YOLO DONE ->", OUT_MODEL)


if __name__ == "__main__":
    main()
