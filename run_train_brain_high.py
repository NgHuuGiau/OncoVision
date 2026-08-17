from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from medical.training import (
    _compute_class_weights,
    medical_training_paths,
    prepare_medical_training_dataset,
)
from medical.cnn_classifier import train_cnn_classifier
from utils.entrypoint_common import run_entrypoint
from utils.terminal_encoding import ensure_utf8_console

_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_TRAIN_LOG_PATH = Path("output/medical/train_log.txt")
_CKPT_DIR = Path("output/medical/checkpoints")

MODEL_NAME = "brain"
# Cau hinh MAX cho GPU 4GB VRAM (RTX 3050 Ti Laptop).
# convnext_tiny (ImageNet pretrained, co san 109MB trong torch cache) @ 512x512 bs=4 = ~1.5GB peak.
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_ACCUM_STEPS = 4

# 4 sub-labels cua ung thu nao (khop brain model goc + ten file trong dataset).
BRAIN_SUBLABELS = ("glioma", "meningioma", "pituitary", "no_tumor")


def _class_names() -> tuple[str, ...]:
    return BRAIN_SUBLABELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ung thu nao (1 lop) - cau hinh cao cho GPU 4GB VRAM."
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f"Kich thuoc anh train (mac dinh: {DEFAULT_IMAGE_SIZE}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch GPU (mac dinh: {DEFAULT_BATCH_SIZE}, giam xuong 1 neu OOM).")
    parser.add_argument("--accum-steps", type=int, default=DEFAULT_ACCUM_STEPS,
                        help=f"Gradient accumulation (mac dinh: {DEFAULT_ACCUM_STEPS}).")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--detached", action="store_true")
    return parser


def _samples_for(root: Path, label: str, split: str) -> list[tuple[Path, int]]:
    from medical.classifier import iter_medical_image_paths

    samples: list[tuple[Path, int]] = []
    for img in iter_medical_image_paths(root / label / "processed" / "images" / split):
        name = img.name.lower()
        if "glioma" in name:
            idx = 0
        elif "meningioma" in name:
            idx = 1
        elif "pituitary" in name:
            idx = 2
        else:
            idx = 3  # no_tumor / notumor / cac ten khac
        samples.append((img, idx))
    return samples


def _print_brain_distribution(samples: list[tuple[Path, int]]) -> None:
    counts = {label: 0 for label in BRAIN_SUBLABELS}
    for _, idx in samples:
        counts[BRAIN_SUBLABELS[idx]] += 1
    for label, count in counts.items():
        print(f"  {label}: {count}", flush=True)


def run_brain_training(args) -> int:
    paths = medical_training_paths()
    class_names = _class_names()
    root = paths.dataset_root
    label = "Ung thư não"

    print("=" * 60, flush=True)
    print("TRAIN UNG THU NAO (4 sub-labels) - PRODUCTION CONFIG", flush=True)
    print("=" * 60, flush=True)
    print("[1/3] Quet dataset brain (4 sub-labels)...", flush=True)
    prepare_medical_training_dataset(paths)

    train_samples = _samples_for(root, label, "train")
    val_samples = _samples_for(root, label, "val")
    if args.max_train_samples:
        train_samples = train_samples[: args.max_train_samples]
    if not train_samples:
        raise FileNotFoundError("Khong co du lieu train ung thu nao.")

    print(f"  Train ({len(train_samples)}):", flush=True)
    _print_brain_distribution(train_samples)
    print(f"  Val ({len(val_samples)}):", flush=True)
    _print_brain_distribution(val_samples)

    class_weights = _compute_class_weights(train_samples, class_names)

    ckpt_dir = _CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "brain_classifier.pt"

    print(
        f"[2/3] Train CNN 4 lop: backbone=convnext_tiny(pretrained) image={args.image_size}x{args.image_size} "
        f"batch={args.batch_size} accum={args.accum_steps} (effective={args.batch_size * args.accum_steps}) "
        f"epochs={args.epochs} lr={args.learning_rate} loss=focal_loss gamma=2.0 "
        f"train={len(train_samples)} anh val={len(val_samples)} anh",
        flush=True,
    )

    wrapper, history = train_cnn_classifier(
        train_samples,
        class_labels=class_names,
        image_size=args.image_size,
        backbone="convnext_tiny",
        pretrained=True,
        dropout=0.25,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        val_samples=val_samples or None,
        early_stopping_patience=15,
        label_smoothing=0.1,
        mixed_precision=True,
        warmup_epochs=3,
        class_weights=class_weights,
        gradient_accumulation_steps=args.accum_steps,
        enable_ema=True,
        ema_decay=0.999,
        enable_checkpoint_averaging=True,
        checkpoint_averaging_window=5,
        scheduler_type="cosine_warmup_restart",
        optimizer_type="adamw",
        gradient_clip_norm=1.0,
        loss_function="focal_loss",
        focal_gamma=2.0,
        checkpoint_path=ckpt_path,
        progress_tag="train",
    )

    target_path = Path("models/pretrained/brain_classifier.pt")
    wrapper.save(target_path)
    if ckpt_path.exists():
        ckpt_path.unlink()  # xoa checkpoint tam khi da xong
    print(f"[3/3] Da luu model: {target_path.resolve()}", flush=True)
    best_val_acc = max(history.get("val_acc", [0.0]))
    best_val_f1 = max(history.get("val_f1", [0.0]))
    print(f"Best val_acc={best_val_acc:.4f} | Best val_f1={best_val_f1:.4f}", flush=True)
    return 0


def launch_detached(args) -> int:
    _TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "run_train_brain_high.py"]
    env = dict(os.environ)
    env["ONCOVISION_ALLOW_WEIGHT_DOWNLOAD"] = "1"
    with open(_TRAIN_LOG_PATH, "w", encoding="utf-8") as log_file:
        subprocess.Popen(
            cmd,
            creationflags=_DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parent,
            env=env,
        )
    print("Da khoi dong train brain o che do detached.")
    print(f"Log: {_TRAIN_LOG_PATH}")
    return 0


def main() -> int:
    ensure_utf8_console()
    os.environ.setdefault("ONCOVISION_ALLOW_WEIGHT_DOWNLOAD", "1")
    args = build_parser().parse_args()
    if getattr(args, "detached", False):
        return launch_detached(args)
    start = time.perf_counter()
    try:
        code = run_brain_training(args)
    except Exception as exc:
        print(f"Loi training: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return 1
    print(f"Tong thoi gian: {time.perf_counter() - start:.2f}s", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))