from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from medical.case_payloads import build_case_export_payload, build_detection_metadata
from medical.cancer_dataset_registry import common_cancer_dataset_source_dicts
from medical.cancer_overview import build_cancer_overview
from medical.cli_helpers import print_medical_readiness, print_medical_status_block
from medical.compliance import MEDICAL_DISCLAIMER
from medical.dataset import (
    count_medical_class_split_images,
    create_default_medical_dataset_config,
    ensure_medical_dataset_structure,
)
from medical.metrics import compute_medical_metrics
from medical.output_management import _medical_output_directories
from medical.modality_calibration import apply_calibrated_modality_tuning, calibrate_modality_tuning
from medical.pipeline import MedicalImageAnalyzer
from medical.reporting import export_case_bundle, update_case_report_case_id
from medical.active_learning import suggest_active_learning_samples
from medical.modality_training import train_modality_classifier
from medical.storage import MedicalCaseDatabase
from medical.validator import validate_image
from medical.system_status import get_medical_system_status, recommended_medical_commands
from medical.training import (
    _load_medical_settings,
    audit_medical_raw_dataset,
    medical_training_paths,
    prepare_medical_training_dataset,
    run_full_medical_training_pipeline,
    train_medical_model,
    validate_medical_model,
)
from utils.cleanup_utils import cleanup_directories
from utils.entrypoint_common import run_entrypoint


@dataclass
class _SimpleModelConfig:
    model_path: Path
    allow_fallback_model: bool = False
    fallback_model_path: Path | None = None




def _dataset_split_counts(dataset_root: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        counts[split] = sum(count_medical_class_split_images(dataset_root, split).values())
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medical CLI: quan ly 7 ung thu da co san trong dataset/medical.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-dataset", help="Kiểm tra dataset medical và tạo data.yaml nếu thiếu.")
    init_parser.add_argument("--dataset-root", default="dataset/medical")

    cancer_init_parser = subparsers.add_parser("init-cancer-dataset", help="Alias cho init-dataset.")
    cancer_init_parser.add_argument("--dataset-root", default="dataset/medical")

    analyze_parser = subparsers.add_parser("analyze", help="Phân tích ảnh medical.")
    analyze_parser.add_argument("--image", required=True)
    analyze_parser.add_argument("--patient-code", required=True)

    validate_image_parser = subparsers.add_parser("validate-image", help="Kiểm tra ảnh y khoa hợp lệ và phân loại modality/vùng cơ thể.")
    validate_image_parser.add_argument("--image", required=True)
    validate_image_parser.add_argument("--min-confidence", type=float, default=0.70, help="Ngưỡng confidence tối thiểu (0-1). Mặc định: 0.70")

    subparsers.add_parser("audit-dataset", help="Kiểm tra ảnh train/val/test trong 7 thư mục ung thư.")
    subparsers.add_parser("split-dataset", help="Đồng bộ và xác thực lại cấu trúc dataset medical.")
    subparsers.add_parser("status", help="Xem trạng thái model, dataset và output medical.")
    subparsers.add_parser("dataset-counts", help="Xem số ảnh theo split của dataset medical.")
    subparsers.add_parser("report", help="Báo cáo gọn cho dataset medical.")
    subparsers.add_parser("sources", help="Xem 7 nhóm ung thư đã có sẵn.")
    subparsers.add_parser("cancer", help="Tổng quan 7 ung thư và số ảnh local.")
    subparsers.add_parser("ready", help="Kiểm tra sẵn sàng train medical.")
    train_parser = subparsers.add_parser("train", help="Train medical classifier.")
    train_parser.add_argument("--verbose", action="store_true", help="In chi tiết từng batch kể cả khi chạy qua pipe (menu).")
    train_parser.add_argument("--resume", default=None, help="Đường dẫn checkpoint .pt để tiếp tục train.")
    subparsers.add_parser("validate", help="Validate medical classifier.")
    evaluate_parser = subparsers.add_parser("evaluate", help="Danh gia model tren TEST set, cong bo metric per-class (AUC/sensitivity/specificity).")
    evaluate_parser.add_argument("--model", default=None, help="Duong dan model .pt (mac dinh: model da train).")
    evaluate_parser.add_argument("--split", default="test", choices=["test", "val", "train"], help="Split de danh gia (mac dinh: test).")
    train_all_parser = subparsers.add_parser("train-all", help="Split/validate/train theo dataset medical hiện có.")
    train_all_parser.add_argument("--verbose", action="store_true", help="In chi tiết từng batch kể cả khi chạy qua pipe (menu).")
    train_all_parser.add_argument("--resume", default=None, help="Đường dẫn checkpoint .pt để tiếp tục train.")
    calibrate_parser = subparsers.add_parser("calibrate-modality-tuning", help="Thống kê dataset và đề xuất ngưỡng tuning theo modality.")
    calibrate_parser.add_argument("--dataset-root", default="dataset/medical")
    calibrate_parser.add_argument("--settings-path", default="config/medical_settings.yaml")
    calibrate_parser.add_argument("--report-path", default="output/medical/reports/modality_tuning_report.json")
    calibrate_parser.add_argument("--apply", action="store_true", help="Cập nhật modality_tuning trong file config.")

    active_learning_parser = subparsers.add_parser("active-learning", help="Gợi ý ảnh cần dán nhãn thêm từ thư mục ảnh chưa dán nhãn.")
    active_learning_parser.add_argument("--image-dir", default="dataset/medical/unlabeled", help="Thư mục ảnh chưa dán nhãn.")
    active_learning_parser.add_argument("--uncertainty-threshold", type=float, default=0.15)
    active_learning_parser.add_argument("--max-samples", type=int, default=10)

    modality_parser = subparsers.add_parser("train-modality", help="Train classifier phân loại modality (ct/mri/xray/...).")
    modality_parser.add_argument("--dataset-root", default="dataset/medical_modality")
    modality_parser.add_argument("--output-path", default="models/pretrained/modality_classifier.pt")
    modality_parser.add_argument("--epochs", type=int, default=10)
    modality_parser.add_argument("--batch-size", type=int, default=16)
    modality_parser.add_argument("--verbose", action="store_true", help="In chi tiết từng batch kể cả khi chạy qua pipe (menu).")

    history_parser = subparsers.add_parser("history", help="Xem lịch sử phân tích.")
    history_parser.add_argument("--limit", type=int, default=10)

    detail_parser = subparsers.add_parser("show-case", help="Xem chi tiết một ca bệnh.")
    detail_parser.add_argument("--case-id", type=int, required=True)

    export_parser = subparsers.add_parser("export-case", help="Đóng gói report và ảnh của một ca bệnh.")
    export_parser.add_argument("--case-id", type=int, required=True)
    export_parser.add_argument("--output-dir", default="output/medical/exports")
    export_parser.add_argument("--pdf", action="store_true", help="Tạo thêm file báo cáo PDF.")

    delete_parser = subparsers.add_parser("delete-case", help="Xóa một bản ghi ca bệnh khỏi lịch sử.")
    delete_parser.add_argument("--case-id", type=int, required=True)
    delete_parser.add_argument("--delete-files", action="store_true", help="Xóa cả file vật lý liên quan.")

    metrics_parser = subparsers.add_parser("metrics", help="Tính metric medical từ file JSON.")
    metrics_parser.add_argument("--truths", required=True, help="Mảng JSON giá trị bool.")
    metrics_parser.add_argument("--predictions", required=True, help="Mảng JSON giá trị bool.")

    cleanup_parser = subparsers.add_parser("cleanup-output", help="Dọn file output medical cũ.")
    cleanup_parser.add_argument("--older-than-days", type=int, default=None)

    parser.epilog = (
        "Medical: init-dataset, init-cancer-dataset, status, sources, cancer, analyze\n"
        "Medical: ready, report, dataset-counts, calibrate-modality-tuning\n"
        "Medical: audit-dataset, split-dataset, train, validate, train-all\n"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    def handle_init_dataset() -> int:
        config = create_default_medical_dataset_config(args.dataset_root)
        ensure_medical_dataset_structure(config)
        print(f"Dataset root: {config.dataset_root}")
        print(f"Data yaml: {config.data_yaml_path}")
        print(f"Metadata: {config.metadata_dir}")
        print(f"Reports: {config.reports_dir}")
        print("Da tao cau truc medical cho 7 ung thu da co san.")
        print(MEDICAL_DISCLAIMER)
        return 0

    def handle_init_cancer_dataset() -> int:
        return handle_init_dataset()

    def handle_analyze() -> int:
        analyzer = MedicalImageAnalyzer()
        db = MedicalCaseDatabase()
        try:
            result = analyzer.analyze_image(args.image, patient_code=args.patient_code)
        except ValueError as exc:
            message = str(exc)
            if ": " in message:
                error_code, error_message = message.split(": ", 1)
            else:
                error_code = "UNKNOWN_ERROR"
                error_message = message
            print(f"Loi: [{error_code}] {error_message}")
            print("Vui long tai len dung loai anh y khoa duoc ho tro.")
            return 1
        case_id = db.save_case(
            patient_code=result.patient_code,
            image_path=str(result.source_image),
            processed_image_path=str(result.processed_image),
            report_json_path=str(result.report_json_path),
            report_md_path=str(result.report_md_path),
            suspected_malignant=result.suspected_malignant,
            risk_level=result.risk_level,
            recommendation=result.recommendation,
            metadata=build_detection_metadata(result),
        )
        update_case_report_case_id(result.report_json_path, result.report_md_path, case_id=case_id)
        print(f"Ma ca benh: {case_id}")
        print(f"Muc do sang loc nguy co: {result.risk_level}")
        if result.risk_level == "uncertain":
            print("LOI: Ket qua khong du tin tuong de dua ra chan doan.")
        if result.quality_warnings:
            print("Canh bao chat luong anh:")
            for warning in result.quality_warnings:
                print(f"- {warning}")
        print(f"Anh da xu ly: {result.processed_image}")
        print(f"Report JSON: {result.report_json_path}")
        print(f"Report MD: {result.report_md_path}")
        print(MEDICAL_DISCLAIMER)
        return 0

    def handle_validate_image() -> int:
        result = validate_image(args.image, min_confidence=args.min_confidence)
        if result.status == "success":
            print(f"Trang thai: {result.status}")
            print(f"Modality: {result.modality}")
            print(f"Body region: {result.body_region}")
            print(f"Modality confidence: {result.modality_confidence:.2f}")
            print(f"Body region confidence: {result.body_region_confidence:.2f}")
        else:
            print(f"Trang thai: {result.status}")
            print(f"Error code: {result.error_code}")
            print(f"Message: {result.message}")
            return 1
        return 0

    def handle_history() -> int:
        db = MedicalCaseDatabase()
        for item in db.list_cases()[: args.limit]:
            print(f"#{item.case_id} | {item.patient_code} | {item.risk_level} | suspicious={item.suspected_malignant} | {item.created_at}")
            print(f"  image={item.image_path}")
            print(f"  report={item.report_md_path}")
        return 0

    def handle_show_case() -> int:
        db = MedicalCaseDatabase()
        item = db.get_case(args.case_id)
        if item is None:
            print(f"Khong tim thay ca benh #{args.case_id}.")
            return 1
        print(f"Ca benh #{item.case_id}")
        print(f"Ma benh nhan: {item.patient_code}")
        print(f"Thoi gian: {item.created_at}")
        print(f"Nguy co: {item.risk_level}")
        print(f"Anh goc: {item.image_path}")
        print(f"Anh xu ly: {item.processed_image_path}")
        print(f"Report JSON: {item.report_json_path}")
        print(f"Report MD: {item.report_md_path}")
        print(f"Metadata: {json.dumps(item.metadata, ensure_ascii=False, indent=2)}")
        return 0

    def handle_status() -> int:
        status = get_medical_system_status()
        print_medical_status_block(status, status.dataset_root)
        print("Recommended commands:")
        for command in recommended_medical_commands(status):
            print(f"- {command}")
        return 0

    def handle_dataset_counts() -> int:
        paths = medical_training_paths()
        counts = _dataset_split_counts(paths.dataset_root)
        print("Medical dataset counts:")
        print(f"- train images: {counts['train']}")
        print(f"- val images: {counts['val']}")
        print(f"- test images: {counts['test']}")
        return 0

    def handle_report() -> int:
        status = get_medical_system_status()
        counts = _dataset_split_counts(status.dataset_root)
        print("Bao cao gon medical")
        print(f"- train {counts['train']} | val {counts['val']} | test {counts['test']}")
        print(f"- model ready: {status.model_ready} | dataset initialized: {status.dataset_initialized}")
        print(f"- ready for train: {status.dataset_initialized and status.raw_dataset_ready and status.processed_dataset_ready and status.model_ready}")
        return 0

    def handle_sources() -> int:
        print("Nguon 7 ung thu da co san:")
        for source in common_cancer_dataset_source_dicts():
            print(f"- {source['source_name']} | {source['cancer_type']} | {source['status']}")
            print(f"  {source['official_url']}")
            print(f"  {source['notes']}")
        return 0

    def handle_cancer() -> int:
        overview = build_cancer_overview()
        summary = cast(dict[str, Any], overview["summary"])
        cancers = cast(list[dict[str, Any]], overview["cancers"])
        print("Tong quan ung thu")
        print(f"Tong anh ung thu local: {summary['total_cancer_images']}")
        print("Theo tung nhom ung thu:")
        for item in cancers:
            print(
                f"- {item['label']}: {item['local_image_count']} anh | "
                f"local_status={item['local_status']} | model_ready={item['model_ready']}"
            )
            for source in item["local_sources"]:
                print(
                    f"  {source['collection_name']}: {source['image_count']} anh | "
                    f"dir={source['collection_root']}"
                )
        return 0

    def handle_ready() -> int:
        status = get_medical_system_status()
        paths = medical_training_paths()
        counts = _dataset_split_counts(paths.dataset_root)
        print("Medical:")
        print(f"- dataset root: {paths.dataset_root}")
        print(f"- train images: {counts['train']}")
        print(f"- val images: {counts['val']}")
        print(f"- test images: {counts['test']}")
        print(f"- dataset initialized: {status.dataset_initialized}")
        print(f"- model ready: {status.model_ready}")
        print("Toan he thong:")
        print(f"- medical reports: {status.report_files}")
        print(f"- normalized: {status.normalized_files}")
        print(f"- overlay: {status.overlay_files}")
        print(f"- exports: {status.export_files}")
        print(f"- case db: {status.case_count}")
        print_medical_readiness(status)
        return 0

    def handle_export_case() -> int:
        db = MedicalCaseDatabase()
        item = db.get_case(args.case_id)
        if item is None:
            print(f"Khong tim thay ca benh #{args.case_id}.")
            return 1
        payload = build_case_export_payload(item)
        bundle_path = export_case_bundle(
            payload,
            args.output_dir,
            include_files=[item.image_path, item.processed_image_path, item.report_json_path, item.report_md_path],
            include_pdf=args.pdf,
        )
        print(f"Da export ca benh #{item.case_id} ra: {bundle_path}")
        if args.pdf:
            pdf_path = Path(args.output_dir) / f"case_{item.case_id}.pdf"
            print(f"Bao cao PDF: {pdf_path if pdf_path.exists() else '(chua tao duoc)'}")
        return 0

    def handle_active_learning() -> int:
        from medical.active_learning import ActiveLearningConfig
        from medical.model_policy import resolve_medical_runtime_model_path

        config = ActiveLearningConfig(
            uncertainty_threshold=args.uncertainty_threshold,
            max_samples_per_batch=args.max_samples,
        )
        settings = _load_medical_settings()
        model_config = _SimpleModelConfig(
            model_path=Path(settings.get("model", "medical_7_cancers.pt")),
            allow_fallback_model=bool(settings.get("allow_fallback_model", False)),
            fallback_model_path=Path(settings["fallback_model"]) if settings.get("fallback_model") else None,
        )
        try:
            model_path = resolve_medical_runtime_model_path(model_config)
        except FileNotFoundError as exc:
            print(f"Khong the chay active-learning: {exc}")
            return 1
        candidates = suggest_active_learning_samples(args.image_dir, config=config, model_path=model_path)
        if not candidates:
            print("Khong co anh nao can dan nhan them (hoac chua co model / thu muc anh).")
            return 0
        print(f"Goi y {len(candidates)} anh can dan nhan them tu {args.image_dir}:")
        for path, confidence, uncertainty in candidates:
            print(f"- {path} (conf={confidence:.3f}, uncertainty={uncertainty:.3f})")
        print(f"Danh sach chi tiet: {config.output_dir / 'active_learning_candidates.csv'}")
        return 0

    def handle_train_modality() -> int:
        path = train_modality_classifier(
            args.dataset_root,
            args.output_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        print(f"Da train modality classifier: {path}")
        return 0

    def handle_delete_case() -> int:
        db = MedicalCaseDatabase()
        if args.delete_files:
            deleted, deleted_paths = db.delete_case_with_files(args.case_id)
        else:
            deleted = db.delete_case(args.case_id)
            deleted_paths = []
        if not deleted:
            print(f"Khong tim thay ca benh #{args.case_id}.")
            return 1
        print(f"Da xoa ban ghi ca benh #{args.case_id}.")
        if deleted_paths:
            print("Da xoa cac file lien quan:")
            for path in deleted_paths:
                print(f"- {path}")
        return 0

    def handle_metrics() -> int:
        truths = json.loads(args.truths)
        predictions = json.loads(args.predictions)
        metrics = compute_medical_metrics(truths, predictions)
        print(json.dumps(metrics.__dict__, indent=2, ensure_ascii=False))
        print(MEDICAL_DISCLAIMER)
        return 0

    def handle_cleanup_output() -> int:
        summary = cleanup_directories(_medical_output_directories(), older_than_days=args.older_than_days)
        print(f"Da xoa file: {summary.removed_files}")
        print(f"Da xoa thu muc rong: {summary.removed_dirs}")
        print(f"Dung luong giai phong: {summary.freed_bytes} bytes")
        return 0

    def handle_audit_dataset() -> int:
        audit = audit_medical_raw_dataset()
        print(f"Train classes: {len(audit['train_images'])}")
        print(f"Val classes: {len(audit['val_images'])}")
        print(f"Test classes: {len(audit['test_images'])}")
        print(f"Missing classes: {len(audit['missing_classes'])}")
        for class_name, count in audit["class_counts"].items():
            print(f"- {class_name}: {count}")
        return 0

    def handle_split_dataset() -> int:
        summary = prepare_medical_training_dataset()
        print(f"Train: {summary.train_count}")
        print(f"Val: {summary.val_count}")
        print(f"Test: {summary.test_count}")
        return 0

    def handle_train() -> int:
        model_path = train_medical_model(verbose=args.verbose, resume_path=args.resume)
        print(f"Da luu model medical: {model_path}")
        return 0

    def handle_validate() -> int:
        metrics = validate_medical_model()
        print(metrics)
        print(MEDICAL_DISCLAIMER)
        return 0

    def handle_evaluate() -> int:
        from medical.evaluation import evaluate_on_test_set, write_evaluation_report

        report = evaluate_on_test_set(model_path=args.model, split=args.split)
        print(f"Model: {report['model_path']}")
        print(f"Split: {report['split']} | So mau: {report['num_samples']}")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1: {report['macro_f1']:.4f} | Macro ROC-AUC: {report['macro_roc_auc']:.4f}")
        print("Per-class (label | sensitivity | specificity | f1 | roc_auc):")
        for entry in report["per_class"]:
            print(
                f"  {entry['label']}: sens={entry['sensitivity']:.3f} "
                f"spec={entry['specificity']:.3f} f1={entry['f1_score']:.3f} "
                f"auc={entry['roc_auc']:.3f} (n={entry['support']})"
            )
        json_path, md_path = write_evaluation_report(report)
        print(f"Bao cao JSON: {json_path}")
        print(f"Bao cao MD:   {md_path}")
        print(MEDICAL_DISCLAIMER)
        return 0

    def handle_train_all() -> int:
        report = run_full_medical_training_pipeline(verbose=args.verbose, resume_path=args.resume)
        print(f"Train: {report['train_count']}")
        print(f"Val: {report['val_count']}")
        print(f"Test: {report['test_count']}")
        print(f"Model: {report['trained_model_path']}")
        prepare_seconds = report.get("prepare_seconds")
        if prepare_seconds is not None:
            print(f"Prepare: {prepare_seconds:.2f}s")
        train_seconds = report.get("train_seconds")
        if train_seconds is not None:
            print(f"Train: {train_seconds:.2f}s")
        validate_seconds = report.get("validate_seconds")
        if validate_seconds is not None:
            print(f"Validate: {validate_seconds:.2f}s")
        total_seconds = report.get("total_seconds")
        if total_seconds is not None:
            print(f"Total: {total_seconds:.2f}s")
        print(report["validation_metrics"])
        print(MEDICAL_DISCLAIMER)
        return 0

    def handle_calibrate_modality_tuning() -> int:
        if args.apply:
            report = apply_calibrated_modality_tuning(args.dataset_root, settings_path=args.settings_path)
            action = "Da cap nhat"
        else:
            report = calibrate_modality_tuning(args.dataset_root, settings_path=args.settings_path)
            action = "Da de xuat"
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{action} modality_tuning cho {report['sample_count']} anh.")
        print(f"Report: {report_path}")
        for modality, tuning in report["modality_tuning"].items():
            if modality == "default":
                continue
            print(
                f"- {modality}: certainty={tuning['certainty_threshold']:.2f}, "
                f"medium={tuning['medium_threshold']:.2f}, quality={tuning['quality_threshold']:.2f}, "
                f"contrast={tuning['contrast_boost']:.2f}"
            )
        print(MEDICAL_DISCLAIMER)
        return 0

    handlers = {
        "init-dataset": handle_init_dataset,
        "init-cancer-dataset": handle_init_cancer_dataset,
        "analyze": handle_analyze,
        "validate-image": handle_validate_image,
        "history": handle_history,
        "show-case": handle_show_case,
        "status": handle_status,
        "dataset-counts": handle_dataset_counts,
        "report": handle_report,
        "sources": handle_sources,
        "cancer": handle_cancer,
        "ready": handle_ready,
        "export-case": handle_export_case,
        "delete-case": handle_delete_case,
        "metrics": handle_metrics,
        "cleanup-output": handle_cleanup_output,
        "audit-dataset": handle_audit_dataset,
        "split-dataset": handle_split_dataset,
        "train": handle_train,
        "validate": handle_validate,
        "evaluate": handle_evaluate,
        "train-all": handle_train_all,
        "calibrate-modality-tuning": handle_calibrate_modality_tuning,
        "active-learning": handle_active_learning,
        "train-modality": handle_train_modality,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.error("Lenh chua duoc ho tro")
        return 1
    return handler()


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
