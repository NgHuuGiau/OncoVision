from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical.case_payloads import build_case_export_payload, build_detection_metadata
from medical.compliance import build_medical_disclaimer
from medical.dataset import create_default_skin_cancer_dataset_config, ensure_medical_dataset_structure
from medical.metrics import compute_medical_metrics
from medical.pipeline import MedicalImageAnalyzer
from medical.reporting import export_case_bundle
from medical.storage import MedicalCaseDatabase
from medical.training import (
    audit_medical_raw_dataset,
    prepare_medical_training_dataset,
    run_full_medical_training_pipeline,
    train_medical_model,
    validate_medical_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medical imaging workflow cho sàng lọc ung thư da từ ảnh upload.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-dataset", help="Tạo cấu trúc dataset y khoa mặc định.")
    init_parser.add_argument("--dataset-root", default="dataset/medical_skin_lesion")

    analyze_parser = subparsers.add_parser("analyze", help="Phân tích ảnh y khoa.")
    analyze_parser.add_argument("--image", required=True)
    analyze_parser.add_argument("--patient-code", required=True)

    subparsers.add_parser("audit-dataset", help="Kiểm tra dataset raw cho medical training.")
    subparsers.add_parser("split-dataset", help="Chia raw dataset medical sang train/val/test.")
    subparsers.add_parser("train", help="Train medical model và cập nhật config model.")
    subparsers.add_parser("validate", help="Validate medical model hiện tại.")
    subparsers.add_parser("train-all", help="Chạy split + train + validate cho medical pipeline.")

    history_parser = subparsers.add_parser("history", help="Xem lịch sử phân tích.")
    history_parser.add_argument("--limit", type=int, default=10)

    detail_parser = subparsers.add_parser("show-case", help="Xem chi tiết một ca bệnh.")
    detail_parser.add_argument("--case-id", type=int, required=True)

    export_parser = subparsers.add_parser("export-case", help="Đóng gói report và ảnh của một ca bệnh.")
    export_parser.add_argument("--case-id", type=int, required=True)
    export_parser.add_argument("--output-dir", default="output/medical/exports")

    delete_parser = subparsers.add_parser("delete-case", help="Xóa một bản ghi ca bệnh khỏi lịch sử.")
    delete_parser.add_argument("--case-id", type=int, required=True)
    delete_parser.add_argument("--delete-files", action="store_true", help="Xóa cả ảnh và report vật lý liên quan.")

    metrics_parser = subparsers.add_parser("metrics", help="Tính metric y khoa từ file JSON.")
    metrics_parser.add_argument("--truths", required=True, help="JSON array bool.")
    metrics_parser.add_argument("--predictions", required=True, help="JSON array bool.")
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-dataset":
        config = create_default_skin_cancer_dataset_config(args.dataset_root)
        summary = ensure_medical_dataset_structure(config)
        print(f"Đã tạo dataset tại: {summary.dataset_root}")
        print(f"Data config: {summary.data_yaml_path}")
        print(build_medical_disclaimer())
        return 0

    if args.command == "analyze":
        analyzer = MedicalImageAnalyzer()
        db = MedicalCaseDatabase()
        result = analyzer.analyze_image(args.image, patient_code=args.patient_code)
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
        print(f"Mã ca bệnh: {case_id}")
        print(f"Mức độ sàng lọc nguy cơ: {result.risk_level}")
        if result.quality_warnings:
            print("Cảnh báo chất lượng ảnh:")
            for warning in result.quality_warnings:
                print(f"- {warning}")
        print(f"Report JSON: {result.report_json_path}")
        print(f"Report MD: {result.report_md_path}")
        print(result.disclaimer)
        return 0

    if args.command == "history":
        db = MedicalCaseDatabase()
        for item in db.list_cases()[: args.limit]:
            print(f"#{item.case_id} | {item.patient_code} | {item.risk_level} | suspicious={item.suspected_malignant} | {item.created_at}")
            print(f"  image={item.image_path}")
            print(f"  report={item.report_md_path}")
        return 0

    if args.command == "show-case":
        db = MedicalCaseDatabase()
        item = db.get_case(args.case_id)
        if item is None:
            print(f"Không tìm thấy ca bệnh #{args.case_id}.")
            return 1
        print(f"Ca bệnh #{item.case_id}")
        print(f"Mã bệnh nhân: {item.patient_code}")
        print(f"Thời gian: {item.created_at}")
        print(f"Nguy cơ: {item.risk_level}")
        print(f"Ảnh gốc: {item.image_path}")
        print(f"Ảnh xử lý: {item.processed_image_path}")
        print(f"Report JSON: {item.report_json_path}")
        print(f"Report MD: {item.report_md_path}")
        print(f"Metadata: {json.dumps(item.metadata, ensure_ascii=False, indent=2)}")
        return 0

    if args.command == "export-case":
        db = MedicalCaseDatabase()
        item = db.get_case(args.case_id)
        if item is None:
            print(f"Không tìm thấy ca bệnh #{args.case_id}.")
            return 1
        payload = build_case_export_payload(item)
        bundle_path = export_case_bundle(
            payload,
            args.output_dir,
            include_files=[item.image_path, item.processed_image_path, item.report_json_path, item.report_md_path],
        )
        print(f"Đã export ca bệnh #{item.case_id} ra: {bundle_path}")
        return 0

    if args.command == "delete-case":
        db = MedicalCaseDatabase()
        if args.delete_files:
            deleted, deleted_paths = db.delete_case_with_files(args.case_id)
        else:
            deleted = db.delete_case(args.case_id)
            deleted_paths = []
        if not deleted:
            print(f"Không tìm thấy ca bệnh #{args.case_id}.")
            return 1
        print(f"Đã xóa bản ghi ca bệnh #{args.case_id}.")
        if deleted_paths:
            print("Đã xóa các file liên quan:")
            for path in deleted_paths:
                print(f"- {path}")
        return 0

    if args.command == "metrics":
        truths = json.loads(args.truths)
        predictions = json.loads(args.predictions)
        metrics = compute_medical_metrics(truths, predictions)
        print(json.dumps(metrics.__dict__, indent=2, ensure_ascii=False))
        print(build_medical_disclaimer())
        return 0

    if args.command == "audit-dataset":
        audit = audit_medical_raw_dataset()
        print(f"Ảnh raw: {len(audit['raw_images'])}")
        print(f"Nhãn raw: {len(audit['raw_labels'])}")
        print(f"Cặp hợp lệ: {len(audit['eligible'])}")
        print(f"Thiếu nhãn: {len(audit['missing_labels'])}")
        print(f"Nhãn lỗi: {len(audit['invalid_labels'])}")
        return 0

    if args.command == "split-dataset":
        summary = prepare_medical_training_dataset()
        print(f"Train: {summary.train_count}")
        print(f"Val: {summary.val_count}")
        print(f"Test: {summary.test_count}")
        return 0

    if args.command == "train":
        model_path = train_medical_model()
        print(f"Đã lưu model medical: {model_path}")
        return 0

    if args.command == "validate":
        metrics = validate_medical_model()
        print(metrics)
        print(build_medical_disclaimer())
        return 0

    if args.command == "train-all":
        report = run_full_medical_training_pipeline()
        print(f"Train: {report['train_count']}")
        print(f"Val: {report['val_count']}")
        print(f"Test: {report['test_count']}")
        print(f"Model: {report['trained_model_path']}")
        print(report["validation_metrics"])
        print(build_medical_disclaimer())
        return 0

    parser.error("Lệnh chưa được hỗ trợ")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
