from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from medical.training import (
    _compute_class_weights,
    _load_medical_settings,
    medical_training_paths,
    prepare_medical_training_dataset,
    _samples_for_split,
)
from medical.cnn_classifier import train_cnn_classifier
from utils.entrypoint_common import run_entrypoint
from utils.terminal_encoding import ensure_utf8_console

_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_TRAIN_LOG_PATH = Path("output/medical/train_log.txt")

# Cau hinh MAX cho GPU 4GB VRAM (RTX 3050 Ti Laptop).
# convnext_tiny (ImageNet pretrained, co san 109MB trong torch cache) @ 512x512 bs=4 = ~1.5GB peak.
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_ACCUM_STEPS = 4
EFFECTIVE_BATCH = DEFAULT_BATCH_SIZE * DEFAULT_ACCUM_STEPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 7 ung thu voi cau hinh cao (512x512) dap ung GPU 4GB VRAM."
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f"Kich thuoc anh train (mac dinh: {DEFAULT_IMAGE_SIZE}, giam xuong 448/384/320 neu OOM).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size GPU (mac dinh: {DEFAULT_BATCH_SIZE}, giam xuong 1 neu OOM).")
    parser.add_argument("--accum-steps", type=int, default=DEFAULT_ACCUM_STEPS,
                        help=f"Gradient accumulation steps, effective batch = batch*accum (mac dinh: {DEFAULT_ACCUM_STEPS}).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="So epochs (mac dinh: tu cau hinh trong config/medical_settings.yaml).")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (mac dinh: tu cau hinh config/medical_settings.yaml).")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Gioi han so anh train (dung de test nhanh pipeline).")
    parser.add_argument("--detached", action="store_true",
                        help="Chay detached, ghi log ra output/medical/train_log.txt.")
    return parser


def build_high_override(args) -> dict:
    settings = _load_medical_settings()
    override = {
        "cnn_image_size": args.image_size,
        "cnn_batch_size": args.batch_size,
        "cnn_gradient_accumulation_steps": args.accum_steps,
        "cnn_mixed_precision": True,
        "cnn_pretrained": False,
        "cnn_backbone": str(settings.get("cnn_backbone", "resnet18")),
        "cnn_dropout": float(settings.get("cnn_dropout", 0.25)),
        "cnn_label_smoothing": float(settings.get("cnn_label_smoothing", 0.08)),
        "cnn_warmup_epochs": int(settings.get("cnn_warmup_epochs", 3)),
        "cnn_class_weighting": bool(settings.get("cnn_class_weighting", True)),
        "loss_function": str(settings.get("loss_function", "focal_loss")),
        "focal_loss_gamma": float(settings.get("focal_loss_gamma", 2.0)),
    }
    if args.epochs is not None:
        override["cnn_num_epochs"] = args.epochs
    else:
        override["cnn_num_epochs"] = int(settings.get("cnn_num_epochs", 15))
    if args.learning_rate is not None:
        override["cnn_learning_rate"] = args.learning_rate
    else:
        override["cnn_learning_rate"] = float(settings.get("cnn_learning_rate", 0.00005))
    override["cnn_early_stopping_patience"] = int(settings.get("cnn_early_stopping_patience", 10))
    return override


def run_high_training(args) -> int:
    paths = medical_training_paths()
    settings = _load_medical_settings()
    override = build_high_override(args)
    config = {**settings, **override}

    print("=" * 60, flush=True)
    print("TRAIN 7 UNG THU - PRODUCTION CONFIG (resnet50 pretrained)", flush=True)
    print("=" * 60, flush=True)
    print(f"[1/3] Chuan bi dataset...", flush=True)
    prepare_medical_training_dataset(paths)

    train_samples = _samples_for_split(paths, "train")
    val_samples = _samples_for_split(paths, "val")
    if args.max_train_samples:
        train_samples = train_samples[: args.max_train_samples]
    if not train_samples:
        raise FileNotFoundError("Khong co du lieu train medical.")

    class_weights = None
    if bool(config.get("cnn_class_weighting", True)):
        class_weights = _compute_class_weights(train_samples, paths.class_names)

    ckpt_dir = Path("output/medical/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "medical_7_cancers_cnn.pt"

    effective_batch = args.batch_size * args.accum_steps
    print(
        f"[2/3] Train CNN: backbone=convnext_tiny(pretrained) image={config['cnn_image_size']}x{config['cnn_image_size']} "
        f"batch={args.batch_size} accum={args.accum_steps} (effective={effective_batch}) "
        f"epochs={config['cnn_num_epochs']} lr={config['cnn_learning_rate']} loss=focal_loss gamma=2.0 "
        f"train={len(train_samples)} anh val={len(val_samples) if val_samples else 0} anh",
        flush=True,
    )

    wrapper, history = train_cnn_classifier(
        train_samples,
        class_labels=paths.class_names,
        image_size=int(config["cnn_image_size"]),
        backbone="convnext_tiny",
        pretrained=True,
        dropout=float(config["cnn_dropout"]),
        batch_size=args.batch_size,
        num_epochs=int(config["cnn_num_epochs"]),
        learning_rate=float(config["cnn_learning_rate"]),
        val_samples=val_samples or None,
        early_stopping_patience=int(config["cnn_early_stopping_patience"]),
        label_smoothing=0.1,
        mixed_precision=True,
        warmup_epochs=int(config["cnn_warmup_epochs"]),
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
        focal_gamma=float(config.get("focal_loss_gamma", 2.0)),
        checkpoint_path=ckpt_path,
        progress_tag="train",
        verbose=bool(getattr(args, "verbose", False)),
    )

    target_path = paths.cnn_model_path
    wrapper.save(target_path)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"[3/3] Da luu model: {target_path.resolve()}", flush=True)
    best_val_acc = max(history.get("val_acc", [0.0]))
    best_val_f1 = max(history.get("val_f1", [0.0]))
    print(f"Best val_acc={best_val_acc:.4f} | Best val_f1={best_val_f1:.4f}", flush=True)
    return 0


def launch_detached(args) -> int:
    _TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "run_train_7cancers_high.py"]
    cmd += _cli_args_to_list(args)
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
    print("Da khoi dong training cao-cap o che do detached.")
    print(f"Log: {_TRAIN_LOG_PATH}")
    print(f"Theo doi: Get-Content -Tail 20 -Wait {_TRAIN_LOG_PATH}")
    return 0


def _cli_args_to_list(args) -> list[str]:
    result: list[str] = []
    if getattr(args, "image_size", None):
        result += ["--image-size", str(args.image_size)]
    if getattr(args, "batch_size", None):
        result += ["--batch-size", str(args.batch_size)]
    if getattr(args, "accum_steps", None):
        result += ["--accum-steps", str(args.accum_steps)]
    if getattr(args, "epochs", None):
        result += ["--epochs", str(args.epochs)]
    if getattr(args, "learning_rate", None):
        result += ["--learning-rate", str(args.learning_rate)]
    if getattr(args, "max_train_samples", None):
        result += ["--max-train-samples", str(args.max_train_samples)]
    return result


def main() -> int:
    ensure_utf8_console()
    os.environ.setdefault("ONCOVISION_ALLOW_WEIGHT_DOWNLOAD", "1")
    args = build_parser().parse_args()
    if getattr(args, "detached", False):
        return launch_detached(args)
    start = time.perf_counter()
    try:
        code = run_high_training(args)
    except Exception as exc:
        print(f"Loi training: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return 1
    print(f"Tong thoi gian: {time.perf_counter() - start:.2f}s", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))