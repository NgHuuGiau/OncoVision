from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical.case_payloads import build_case_export_payload, build_detection_metadata
from medical.compliance import build_medical_disclaimer
from medical.cancer_overview import build_cancer_overview
from medical.cancer_dataset_registry import common_cancer_dataset_source_dicts
from medical.cli_helpers import print_dataset_counts, print_output_counts, print_skin_readiness, print_skin_status_block
from medical.dataset import create_default_medical_cancer_dataset_config, create_default_skin_cancer_dataset_config
from medical.metrics import compute_medical_metrics
from medical.output_management import cleanup_medical_outputs
from medical.pipeline import MedicalImageAnalyzer
from medical.reporting import export_case_bundle, update_case_report_case_id
from medical.status_helpers import skin_dataset_counts
from medical.storage import MedicalCaseDatabase
from medical.system_status import get_medical_system_status, recommended_medical_commands
from medical.training import (
    audit_medical_raw_dataset,
    medical_training_paths,
    prepare_medical_training_dataset,
    run_full_medical_training_pipeline,
    train_medical_model,
    validate_medical_model,
)
from utils.entrypoint_common import run_entrypoint


SKIN_LESION_ROOT = create_default_skin_cancer_dataset_config().dataset_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Luồng Medical: y dược, ung thư và nhận diện vật thể trong một bộ công cụ."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-dataset", help="Y dược: kiểm tra dataset mặc định, không tự tạo.")
    init_parser.add_argument("--dataset-root", default="dataset/medical/skin_lesion")

    cancer_init_parser = subparsers.add_parser("init-cancer-dataset", help="Y dược: kiểm tra dataset ung thư tổng hợp, không tự tạo.")
    cancer_init_parser.add_argument("--dataset-root", default="dataset/medical")

    analyze_parser = subparsers.add_parser("analyze", help="Vật thể / y dược: phân tích ảnh.")
    analyze_parser.add_argument("--image", required=True)
    analyze_parser.add_argument("--patient-code", required=True)

    subparsers.add_parser("audit-dataset", help="Vật thể: xem số lượng ảnh raw và nhãn.")
    subparsers.add_parser("split-dataset", help="Vật thể: chia raw dataset sang train/val/test.")
    subparsers.add_parser("status", help="Y dược: xem trạng thái model, dataset, output và số ảnh.")
    subparsers.add_parser("dataset-counts", help="Y dược: xem riêng số ảnh skin lesion và dataset y khoa.")
    subparsers.add_parser("report", help="Y dược: màn hình tổng hợp ngắn gọn cho dataset và sẵn sàng train.")
    subparsers.add_parser("sources", help="Y dược: xem danh sách nguồn ung thư và dataset liên quan.")
    subparsers.add_parser("cancer", help="Y dược: tổng quan gói gọn cho toàn bộ dữ liệu ung thư.")
    subparsers.add_parser("ready", help="Kiểm tra sẵn sàng train cho hệ y và toàn hệ thống.")
    subparsers.add_parser("train", help="Vật thể: train model hiện tại.")
    subparsers.add_parser("validate", help="Vật thể: validate model hiện tại.")
    subparsers.add_parser("train-all", help="Vật thể: split + train + validate.")

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
    metrics_parser.add_argument("--truths", required=True, help="Mảng JSON giá trị bool.")
    metrics_parser.add_argument("--predictions", required=True, help="Mảng JSON giá trị bool.")

    cleanup_parser = subparsers.add_parser("cleanup-output", help="Dọn file output medical cũ.")
    cleanup_parser.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        help="Chỉ xóa file cũ hơn số ngày này. Bỏ qua để xóa toàn bộ file trong thư mục output medical.",
    )
    parser.epilog = (
        "Y dược: init-dataset, init-cancer-dataset, status, sources, cancer, analyze\n"
        "Y dược: ready, report, dataset-counts\n"
        "Vật thể: audit-dataset, split-dataset, train, validate, train-all\n"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    def handle_init_dataset() -> int:
        config = create_default_skin_cancer_dataset_config(args.dataset_root)
        print(f"Dataset root: {config.dataset_root}")
        print(f"Raw images: {config.raw_images_dir}")
        print(f"Raw labels: {config.raw_labels_dir}")
        print(f"Processed images: {config.processed_images_dir}")
        print(f"Processed labels: {config.processed_labels_dir}")
        print(f"Metadata: {config.metadata_dir}")
        print(f"Reports: {config.reports_dir}")
        print("Da tao dataset. Nguoi dung can chuan bi du lieu thu cong hoac chay luong import/download rieng neu muon.")
        print(build_medical_disclaimer())
        return 0

    def handle_init_cancer_dataset() -> int:
        config = create_default_medical_cancer_dataset_config(args.dataset_root)
        print(f"Dataset root: {config.dataset_root}")
        print(f"Raw images: {config.raw_images_dir}")
        print(f"Raw labels: {config.raw_labels_dir}")
        print(f"Processed images: {config.processed_images_dir}")
        print(f"Processed labels: {config.processed_labels_dir}")
        print(f"Metadata: {config.metadata_dir}")
        print(f"Reports: {config.reports_dir}")
        print("Da tao dataset. Chi kiem tra layout mong doi de nguoi dung tu chuan bi du lieu.")
        print(build_medical_disclaimer())
        return 0

    def handle_analyze() -> int:
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
        update_case_report_case_id(result.report_json_path, result.report_md_path, case_id=case_id)
        print(f"Ma ca benh: {case_id}")
        print(f"Muc do sang loc nguy co: {result.risk_level}")
        if result.quality_warnings:
            print("Canh bao chat luong anh:")
            for warning in result.quality_warnings:
                print(f"- {warning}")
        print(f"Anh da xu ly: {result.processed_image}")
        print(f"Report JSON: {result.report_json_path}")
        print(f"Report MD: {result.report_md_path}")
        print(result.disclaimer)
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

    def handle_status() -> int:
        status = get_medical_system_status()
        skin_counts = skin_dataset_counts(SKIN_LESION_ROOT)
        print_skin_status_block(status, skin_counts, SKIN_LESION_ROOT)
        print("Recommended commands:")
        for command in recommended_medical_commands(status):
            print(f"- {command}")
        return 0

    def handle_dataset_counts() -> int:
        skin_root = create_default_skin_cancer_dataset_config().dataset_root
        skin_counts = skin_dataset_counts(skin_root)
        # Shared dataset counters keep every medical status command in sync.
        print("Y dược dataset counts:")
        print(f"- skin lesion raw images: {skin_counts.raw_images}")
        print(f"- skin lesion raw labels: {skin_counts.raw_labels}")
        print(f"- skin lesion train images: {skin_counts.train_images}")
        print(f"- skin lesion val images: {skin_counts.val_images}")
        return 0

    def handle_report() -> int:
        status = get_medical_system_status()
        skin_counts = skin_dataset_counts(SKIN_LESION_ROOT)
        print("Báo cáo ngắn y dược")
        print(f"- Skin lesion: raw {skin_counts.raw_images}/{skin_counts.raw_labels} | train {skin_counts.train_images} | val {skin_counts.val_images}")
        print(f"- Skin lesion images dir: {SKIN_LESION_ROOT / 'raw' / 'images'}")
        print(f"- Model ready: {status.model_ready} | dataset initialized: {status.dataset_initialized}")
        print(f"- Ready for train skin: {status.dataset_initialized and status.raw_dataset_ready and status.processed_dataset_ready and status.model_ready}")
        return 0

    def handle_sources() -> int:
        print("Nguồn ung thư và mức độ sẵn sàng:")
        for source in common_cancer_dataset_source_dicts():
            print(f"- {source['source_name']} | {source['cancer_type']} | {source['status']}")
            print(f"  {source['official_url']}")
            print(f"  {source['notes']}")
        print("Gợi ý: nếu muốn xem tổng quan gọn hơn, dùng `run_medical.py cancer`.")
        return 0

    def handle_cancer() -> int:
        overview = build_cancer_overview()
        summary = overview["summary"]
        print("Tong quan ung thu")
        print(f"Tong anh ung thu local: {summary['total_cancer_images']}")
        print(f"- skin local: {summary['skin_raw_images']}")
        print(f"Skin lesion dir: {summary['skin_image_dir']}")
        print("Theo tung nhom ung thu:")
        for item in overview["cancers"]:
            print(
                f"- {item['label']}: {item['local_image_count']} anh | "
                f"local_status={item['local_status']} | model_ready={item['model_ready']}"
            )
            if item["local_sources"]:
                for source in item["local_sources"]:
                    print(
                        f"  {source['collection_name']}: {source['image_count']} anh | "
                        f"dir={source['collection_root']}"
                    )
            else:
                print("  chua co anh local.")
            if item["sources"]:
                print("  nguon lien quan:")
                for source in item["sources"]:
                    print(f"    - {source['source_name']} | {source['status']}")
            print(f"  ghi chu model: {item['model_notes']}")
        return 0

    def handle_ready() -> int:
        status = get_medical_system_status()
        paths = medical_training_paths()
        medical_root = create_default_skin_cancer_dataset_config().dataset_root
        medical_counts = skin_dataset_counts(medical_root)
        print("Y dược:")
        print(f"- dataset root: {paths.dataset_root}")
        print(f"- raw images: {status.raw_images}")
        print(f"- raw labels: {status.raw_labels}")
        print(f"- processed train/val/test: {status.train_images}/{status.val_images}/{status.test_images}")
        print(f"- skin lesion raw images/labels: {medical_counts.raw_images}/{medical_counts.raw_labels}")
        print(f"- dataset initialized: {status.dataset_initialized}")
        print(f"- model ready: {status.model_ready}")
        print("Toàn hệ thống:")
        print(f"- medical reports: {status.report_files}")
        print(f"- normalized: {status.normalized_files}")
        print(f"- overlay: {status.overlay_files}")
        print(f"- exports: {status.export_files}")
        print(f"- case db: {status.case_count}")
        print_skin_readiness(status)
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
        )
        print(f"Da export ca benh #{item.case_id} ra: {bundle_path}")
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
        print(build_medical_disclaimer())
        return 0

    def handle_cleanup_output() -> int:
        summary = cleanup_medical_outputs(older_than_days=args.older_than_days)
        print(f"Da xoa file: {summary.removed_files}")
        print(f"Da xoa thu muc rong: {summary.removed_dirs}")
        print(f"Dung luong giai phong: {summary.freed_bytes} bytes")
        return 0

    def handle_audit_dataset() -> int:
        audit = audit_medical_raw_dataset()
        print(f"Anh raw: {len(audit['raw_images'])}")
        print(f"Nhan raw: {len(audit['raw_labels'])}")
        print(f"Cap hop le: {len(audit['eligible'])}")
        print(f"Thieu nhan: {len(audit['missing_labels'])}")
        print(f"Nhan loi: {len(audit['invalid_labels'])}")
        return 0

    def handle_split_dataset() -> int:
        summary = prepare_medical_training_dataset()
        print(f"Train: {summary.train_count}")
        print(f"Val: {summary.val_count}")
        print(f"Test: {summary.test_count}")
        return 0

    def handle_train() -> int:
        model_path = train_medical_model()
        print(f"Da luu model medical: {model_path}")
        return 0

    def handle_validate() -> int:
        metrics = validate_medical_model()
        print(metrics)
        print(build_medical_disclaimer())
        return 0

    def handle_train_all() -> int:
        report = run_full_medical_training_pipeline()
        print(f"Train: {report['train_count']}")
        print(f"Val: {report['val_count']}")
        print(f"Test: {report['test_count']}")
        print(f"Model: {report['trained_model_path']}")
        print(report["validation_metrics"])
        print(build_medical_disclaimer())
        return 0

    handlers = {
        "init-dataset": handle_init_dataset,
        "init-cancer-dataset": handle_init_cancer_dataset,
        "analyze": handle_analyze,
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
        "train-all": handle_train_all,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.error("Lenh chua duoc ho tro")
        return 1
    return handler()


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))

