from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical.case_payloads import build_case_export_payload, build_detection_metadata
from medical.cancer_catalog import supported_cancer_labels
from medical.cancer_dataset_registry import common_cancer_dataset_source_dicts
from medical.cancer_overview import build_cancer_overview
from medical.cli_helpers import print_medical_readiness, print_medical_status_block
from medical.compliance import MEDICAL_DISCLAIMER
from medical.dataset import create_default_medical_dataset_config, ensure_medical_dataset_structure
from medical.metrics import compute_medical_metrics
from medical.output_management import _medical_output_directories
from medical.pipeline import MedicalImageAnalyzer
from medical.reporting import export_case_bundle, update_case_report_case_id
from medical.status_helpers import count_files
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
from utils.cleanup_utils import cleanup_directories
from utils.entrypoint_common import run_entrypoint


def _dataset_split_counts(dataset_root: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        counts[split] = sum(
            count_files(dataset_root / class_name / "processed" / "images" / split)
            for class_name in supported_cancer_labels()
        )
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medical CLI: quan ly 7 ung thu da co san trong dataset/medical.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-dataset", help="Kiem tra dataset medical va tao data.yaml neu thieu.")
    init_parser.add_argument("--dataset-root", default="dataset/medical")

    cancer_init_parser = subparsers.add_parser("init-cancer-dataset", help="Alias cho init-dataset.")
    cancer_init_parser.add_argument("--dataset-root", default="dataset/medical")

    analyze_parser = subparsers.add_parser("analyze", help="Phan tich anh medical.")
    analyze_parser.add_argument("--image", required=True)
    analyze_parser.add_argument("--patient-code", required=True)

    subparsers.add_parser("audit-dataset", help="Kiem tra anh train/val/test trong 7 thu muc ung thu.")
    subparsers.add_parser("split-dataset", help="Dong bo va xac thuc lai cau truc dataset medical.")
    subparsers.add_parser("status", help="Xem trang thai model, dataset va output medical.")
    subparsers.add_parser("dataset-counts", help="Xem so anh theo split cua dataset medical.")
    subparsers.add_parser("report", help="Bao cao gon cho dataset medical.")
    subparsers.add_parser("sources", help="Xem 7 nhom ung thu da co san.")
    subparsers.add_parser("cancer", help="Tong quan 7 ung thu va so anh local.")
    subparsers.add_parser("ready", help="Kiem tra san sang train medical.")
    subparsers.add_parser("train", help="Train medical classifier.")
    subparsers.add_parser("validate", help="Validate medical classifier.")
    subparsers.add_parser("train-all", help="Split/validate/train theo dataset medical hien co.")

    history_parser = subparsers.add_parser("history", help="Xem lich su phan tich.")
    history_parser.add_argument("--limit", type=int, default=10)

    detail_parser = subparsers.add_parser("show-case", help="Xem chi tiet mot ca benh.")
    detail_parser.add_argument("--case-id", type=int, required=True)

    export_parser = subparsers.add_parser("export-case", help="Dong goi report va anh cua mot ca benh.")
    export_parser.add_argument("--case-id", type=int, required=True)
    export_parser.add_argument("--output-dir", default="output/medical/exports")

    delete_parser = subparsers.add_parser("delete-case", help="Xoa mot ban ghi ca benh khoi lich su.")
    delete_parser.add_argument("--case-id", type=int, required=True)
    delete_parser.add_argument("--delete-files", action="store_true", help="Xoa ca file vat ly lien quan.")

    metrics_parser = subparsers.add_parser("metrics", help="Tinh metric medical tu file JSON.")
    metrics_parser.add_argument("--truths", required=True, help="Mang JSON gia tri bool.")
    metrics_parser.add_argument("--predictions", required=True, help="Mang JSON gia tri bool.")

    cleanup_parser = subparsers.add_parser("cleanup-output", help="Don file output medical cu.")
    cleanup_parser.add_argument("--older-than-days", type=int, default=None)

    parser.epilog = (
        "Medical: init-dataset, init-cancer-dataset, status, sources, cancer, analyze\n"
        "Medical: ready, report, dataset-counts\n"
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
        print(MEDICAL_DISCLAIMER)
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
        summary = overview["summary"]
        print("Tong quan ung thu")
        print(f"Tong anh ung thu local: {summary['total_cancer_images']}")
        print("Theo tung nhom ung thu:")
        for item in overview["cancers"]:
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
        model_path = train_medical_model()
        print(f"Da luu model medical: {model_path}")
        return 0

    def handle_validate() -> int:
        metrics = validate_medical_model()
        print(metrics)
        print(MEDICAL_DISCLAIMER)
        return 0

    def handle_train_all() -> int:
        report = run_full_medical_training_pipeline()
        print(f"Train: {report['train_count']}")
        print(f"Val: {report['val_count']}")
        print(f"Test: {report['test_count']}")
        print(f"Model: {report['trained_model_path']}")
        print(f"Prepare: {report['prepare_seconds']:.2f}s")
        print(f"Train: {report['train_seconds']:.2f}s")
        print(f"Validate: {report['validate_seconds']:.2f}s")
        print(f"Total: {report['total_seconds']:.2f}s")
        print(report["validation_metrics"])
        print(MEDICAL_DISCLAIMER)
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
