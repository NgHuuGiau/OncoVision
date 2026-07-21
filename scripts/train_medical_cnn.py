"""Train CNN classifier THAT tren dataset medical 7 ung thu, roi danh gia test set.

Day la script chay CHU DONG (can GPU + nhieu gio) de thay the centroid classifier
bang mo hinh CNN hoc sau that su. Vi train la hanh dong chu dong, script tu dong
cho phep tai trong so ImageNet pretrained (dat ONCOVISION_ALLOW_WEIGHT_DOWNLOAD=1)
tru khi truyen --no-pretrained.

Sau khi train xong, script tu dong danh gia tren TEST split va cong bo metric
per-class (accuracy, sensitivity, specificity, precision, recall, F1, ROC-AUC,
PR-AUC), ghi bao cao JSON + Markdown vao output/medical/reports/.

Vi du:

    python scripts/train_medical_cnn.py --backbone convnextv2_tiny --epochs 30
    python scripts/train_medical_cnn.py --no-pretrained --epochs 5   # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CNN classifier medical + danh gia test set.")
    parser.add_argument("--backbone", default=None, help="Backbone (mac dinh: lay tu config cnn_backbone).")
    parser.add_argument("--epochs", type=int, default=None, help="So epoch (mac dinh: config cnn_num_epochs).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (mac dinh: uoc luong theo GPU).")
    parser.add_argument("--no-pretrained", action="store_true", help="Khong dung ImageNet pretrained (train tu dau).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--eval-split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--resume-path", default=None, help="Checkpoint path de tiep tuc train (luu full state moi epoch).")
    parser.add_argument("--fresh", action="store_true", help="Bo qua checkpoint cu va train tu dau.")
    args = parser.parse_args()

    # Train la hanh dong chu dong: cho phep tai ImageNet pretrained tru khi tat.
    if not args.no_pretrained:
        os.environ.setdefault("ONCOVISION_ALLOW_WEIGHT_DOWNLOAD", "1")

    from medical.evaluation import evaluate_on_test_set, write_evaluation_report
    from medical.training import (
        medical_training_paths,
        prepare_medical_training_dataset,
        train_cnn_medical_model,
    )
    from utils.file_utils import load_yaml

    settings_path = Path("config/medical_settings.yaml")
    settings = load_yaml(settings_path).get("medical", {})
    if str(settings.get("classifier_backend", "")).lower() != "cnn":
        print("[warn] classifier_backend trong config không phải 'cnn'. "
              "Van train CNN nhung nen dat classifier_backend: cnn de runtime dung CNN.")

    paths = medical_training_paths()
    print("=" * 60)
    print("TRAIN CNN MEDICAL 7 UNG THU (that)")
    print("=" * 60)

    print("[1/3] Chuan bi dataset (split train/val/test theo benh nhan)...")
    summary = prepare_medical_training_dataset(paths)
    print(f"      train={summary.train_count} val={summary.val_count} test={summary.test_count}")

    # Override in-memory (KHONG ghi de config/medical_settings.yaml tren dia).
    override: dict = {"classifier_backend": "cnn", "cnn_pretrained": not args.no_pretrained}
    if args.backbone:
        override["cnn_backbone"] = args.backbone
    if args.epochs is not None:
        override["cnn_num_epochs"] = args.epochs
    if args.batch_size is not None:
        override["cnn_batch_size"] = args.batch_size

    default_ckpt = Path("output/medical/cnn_checkpoint.pt")
    default_ckpt.parent.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume_path
    if not args.fresh and resume_path is None:
        if default_ckpt.exists():
            resume_path = str(default_ckpt)
            print(f"[resume] Tim thay checkpoint mac dinh: {resume_path}")
        else:
            print("[start] Khong tim thay checkpoint cu, train tu dau.")
    elif args.fresh and resume_path is None:
        print("[fresh] Bo qua checkpoint cu, train tu dau.")

    print(f"[2/3] Train CNN backbone={override.get('cnn_backbone', settings.get('cnn_backbone'))} "
          f"epochs={override.get('cnn_num_epochs', settings.get('cnn_num_epochs'))} "
          f"pretrained={not args.no_pretrained}...")
    model_path = train_cnn_medical_model(
        paths, prepare_dataset=False, verbose=args.verbose, settings_override=override,
        checkpoint_path=resume_path or str(default_ckpt),
    )
    print(f"      Model da luu: {model_path}")

    print(f"[3/3] Danh gia tren split '{args.eval_split}' (giu rieng)...")
    report = evaluate_on_test_set(paths, model_path=model_path, split=args.eval_split)
    json_path, md_path = write_evaluation_report(report)
    print("-" * 60)
    print(f"Accuracy:      {report['accuracy']:.4f}")
    print(f"Macro F1:      {report['macro_f1']:.4f}")
    print(f"Macro ROC-AUC: {report['macro_roc_auc']:.4f}")
    print("Per-class:")
    for entry in report["per_class"]:
        print(f"  {entry['label']}: sens={entry['sensitivity']:.3f} "
              f"spec={entry['specificity']:.3f} f1={entry['f1_score']:.3f} "
              f"auc={entry['roc_auc']:.3f} (n={entry['support']})")
    print(f"Bao cao: {json_path}")
    print(f"         {md_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
