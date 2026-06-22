from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical.case_payloads import build_case_export_payload, build_detection_metadata
from medical.compliance import build_medical_disclaimer
from medical.dataset import create_default_skin_cancer_dataset_config, ensure_medical_dataset_structure
from medical.metrics import compute_medical_metrics
from medical.output_management import cleanup_medical_outputs
from medical.pipeline import MedicalImageAnalyzer
from medical.reporting import export_case_bundle, update_case_report_case_id
from medical.storage import MedicalCaseDatabase
from medical.system_status import get_medical_system_status, recommended_medical_commands
from medical.training import (
    audit_medical_raw_dataset,
    prepare_medical_training_dataset,
    run_full_medical_training_pipeline,
    train_medical_model,
    validate_medical_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medical imaging workflow cho sang loc ung thu da tu anh upload.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-dataset", help="Tao cau truc dataset y khoa mac dinh.")
    init_parser.add_argument("--dataset-root", default="dataset/medical_skin_lesion")

    analyze_parser = subparsers.add_parser("analyze", help="Phan tich anh y khoa.")
    analyze_parser.add_argument("--image", required=True)
    analyze_parser.add_argument("--patient-code", required=True)

    subparsers.add_parser("audit-dataset", help="Kiem tra dataset raw cho medical training.")
    subparsers.add_parser("split-dataset", help="Chia raw dataset medical sang train/val/test.")
    subparsers.add_parser("status", help="Xem nhanh trang thai model, dataset va output medical.")
    subparsers.add_parser("train", help="Train medical model va cap nhat config model.")
    subparsers.add_parser("validate", help="Validate medical model hien tai.")
    subparsers.add_parser("train-all", help="Chay split + train + validate cho medical pipeline.")

    history_parser = subparsers.add_parser("history", help="Xem lich su phan tich.")
    history_parser.add_argument("--limit", type=int, default=10)

    detail_parser = subparsers.add_parser("show-case", help="Xem chi tiet mot ca benh.")
    detail_parser.add_argument("--case-id", type=int, required=True)

    export_parser = subparsers.add_parser("export-case", help="Dong goi report va anh cua mot ca benh.")
    export_parser.add_argument("--case-id", type=int, required=True)
    export_parser.add_argument("--output-dir", default="output/medical/exports")

    delete_parser = subparsers.add_parser("delete-case", help="Xoa mot ban ghi ca benh khoi lich su.")
    delete_parser.add_argument("--case-id", type=int, required=True)
    delete_parser.add_argument("--delete-files", action="store_true", help="Xoa ca anh va report vat ly lien quan.")

    metrics_parser = subparsers.add_parser("metrics", help="Tinh metric y khoa tu file JSON.")
    metrics_parser.add_argument("--truths", required=True, help="JSON array bool.")
    metrics_parser.add_argument("--predictions", required=True, help="JSON array bool.")

    cleanup_parser = subparsers.add_parser("cleanup-output", help="Don file output medical cu.")
    cleanup_parser.add_argument(
        "--older-than-days",
        type=int,
        default=None,
        help="Chi xoa file cu hon so ngay nay. Bo qua de xoa tat ca file trong thu muc output medical.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-dataset":
        config = create_default_skin_cancer_dataset_config(args.dataset_root)
        summary = ensure_medical_dataset_structure(config)
        print(f"Da tao dataset tai: {summary.dataset_root}")
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
        update_case_report_case_id(
            result.report_json_path,
            result.report_md_path,
            case_id=case_id,
        )
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

    if args.command == "status":
        status = get_medical_system_status()
        print("Medical system status")
        print(f"Model config: {status.configured_model_path}")
        if status.resolved_model_path is not None:
            print(f"Model runtime: {status.resolved_model_path}")
        print(f"Fallback allowed: {status.allow_fallback_model}")
        print(f"Model ready: {status.model_ready}")
        print(f"Model detail: {status.model_message}")
        print(f"Dataset root: {status.dataset_root}")
        print(f"Data yaml: {status.data_yaml_path} | exists={status.dataset_initialized}")
        print(
            "Dataset counts: "
            f"raw_images={status.raw_images}, raw_labels={status.raw_labels}, "
            f"train={status.train_images}, val={status.val_images}, test={status.test_images}"
        )
        print(
            "Outputs: "
            f"cases={status.case_count}, reports={status.report_files}, "
            f"normalized={status.normalized_files}, overlay={status.overlay_files}, exports={status.export_files}"
        )
        print("Recommended commands:")
        for command in recommended_medical_commands(status):
            print(f"- {command}")
        return 0

    if args.command == "export-case":
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

    if args.command == "delete-case":
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

    if args.command == "metrics":
        truths = json.loads(args.truths)
        predictions = json.loads(args.predictions)
        metrics = compute_medical_metrics(truths, predictions)
        print(json.dumps(metrics.__dict__, indent=2, ensure_ascii=False))
        print(build_medical_disclaimer())
        return 0

    if args.command == "cleanup-output":
        summary = cleanup_medical_outputs(older_than_days=args.older_than_days)
        print(f"Da xoa file: {summary.removed_files}")
        print(f"Da xoa thu muc rong: {summary.removed_dirs}")
        print(f"Dung luong giai phong: {summary.freed_bytes} bytes")
        return 0

    if args.command == "audit-dataset":
        audit = audit_medical_raw_dataset()
        print(f"Anh raw: {len(audit['raw_images'])}")
        print(f"Nhan raw: {len(audit['raw_labels'])}")
        print(f"Cap hop le: {len(audit['eligible'])}")
        print(f"Thieu nhan: {len(audit['missing_labels'])}")
        print(f"Nhan loi: {len(audit['invalid_labels'])}")
        return 0

    if args.command == "split-dataset":
        summary = prepare_medical_training_dataset()
        print(f"Train: {summary.train_count}")
        print(f"Val: {summary.val_count}")
        print(f"Test: {summary.test_count}")
        return 0

    if args.command == "train":
        model_path = train_medical_model()
        print(f"Da luu model medical: {model_path}")
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

    parser.error("Lenh chua duoc ho tro")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
