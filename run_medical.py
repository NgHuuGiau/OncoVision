from __future__ import annotations

import argparse
import json

from medical.case_payloads import build_case_export_payload, build_detection_metadata
from medical.cli_helpers import print_medical_status_block
from medical.compliance import MEDICAL_DISCLAIMER
from medical.output_management import _medical_output_directories
from medical.pipeline import MedicalImageAnalyzer
from medical.reporting import export_case_bundle, update_case_report_case_id
from medical.storage import MedicalCaseDatabase
from medical.system_status import get_medical_system_status
from medical.validator import validate_image
from utils.cleanup_utils import cleanup_directories
from utils.entrypoint_common import run_entrypoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OncoVision: phân tích ảnh y khoa và quản lý ca bệnh.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Phân tích một ảnh y khoa bằng model có sẵn.")
    analyze.add_argument("--image", required=True)
    analyze.add_argument("--patient-code", required=True)

    validate = subparsers.add_parser("validate-image", help="Kiểm tra ảnh y khoa và nhận diện modality/vùng cơ thể.")
    validate.add_argument("--image", required=True)
    validate.add_argument("--min-confidence", type=float, default=0.70)

    subparsers.add_parser("status", help="Xem trạng thái model suy luận và ca bệnh.")
    subparsers.add_parser("report", help="Xem số ca và tệp kết quả đã lưu.")

    history = subparsers.add_parser("history", help="Xem lịch sử phân tích.")
    history.add_argument("--limit", type=int, default=10)

    detail = subparsers.add_parser("show-case", help="Xem chi tiết một ca bệnh.")
    detail.add_argument("--case-id", type=int, required=True)

    export = subparsers.add_parser("export-case", help="Đóng gói báo cáo và ảnh của một ca bệnh.")
    export.add_argument("--case-id", type=int, required=True)
    export.add_argument("--output-dir", default="output/medical/exports")
    export.add_argument("--pdf", action="store_true")

    delete = subparsers.add_parser("delete-case", help="Xóa một ca khỏi lịch sử.")
    delete.add_argument("--case-id", type=int, required=True)
    delete.add_argument("--delete-files", action="store_true", help="Xóa cả các tệp vật lý liên quan.")

    cleanup = subparsers.add_parser("cleanup-output", help="Dọn tệp kết quả y khoa cũ.")
    cleanup.add_argument("--older-than-days", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "analyze":
        analyzer = MedicalImageAnalyzer()
        db = MedicalCaseDatabase()
        try:
            result = analyzer.analyze_image(args.image, patient_code=args.patient_code)
        except ValueError as exc:
            message = str(exc)
            error_code, _, error_message = message.partition(": ")
            print(f"Lỗi: [{error_code or 'UNKNOWN_ERROR'}] {error_message or message}")
            print("Vui lòng chọn đúng loại ảnh y khoa được hỗ trợ.")
            return 1
        except Exception as exc:
            print(f"Lỗi phân tích ảnh: {exc}")
            return 1

        case_id = db.save_case(
            patient_code=result.patient_code,
            image_path=str(result.source_image) if result.source_image else "",
            processed_image_path=str(result.processed_image) if result.processed_image else "",
            report_json_path=str(result.report_json_path) if result.report_json_path else "",
            report_md_path=str(result.report_md_path) if result.report_md_path else "",
            suspected_malignant=result.suspected_malignant,
            risk_level=result.risk_level,
            recommendation=result.recommendation,
            metadata=build_detection_metadata(result),
        )
        if result.report_json_path and result.report_md_path:
            update_case_report_case_id(result.report_json_path, result.report_md_path, case_id=case_id)
        print(f"Mã ca bệnh: {case_id}")
        print(f"Mức độ sàng lọc nguy cơ: {result.risk_level}")
        if result.risk_level == "uncertain":
            print("Kết quả chưa đủ tin cậy để đưa ra nhận định.")
        for warning in result.quality_warnings:
            print(f"Cảnh báo chất lượng ảnh: {warning}")
        print(f"Ảnh đã xử lý: {result.processed_image}")
        print(f"Báo cáo JSON: {result.report_json_path}")
        print(f"Báo cáo MD: {result.report_md_path}")
        print(MEDICAL_DISCLAIMER)
        return 0

    if args.command == "validate-image":
        result = validate_image(args.image, min_confidence=args.min_confidence)
        print(f"Trạng thái: {result.status}")
        if result.status != "success":
            print(f"Lỗi: [{result.error_code}] {result.message}")
            return 1
        print(f"Modality: {result.modality} ({result.modality_confidence:.2f})")
        print(f"Vùng cơ thể: {result.body_region} ({result.body_region_confidence:.2f})")
        return 0

    if args.command == "status":
        print_medical_status_block(get_medical_system_status())
        return 0

    if args.command == "report":
        status = get_medical_system_status()
        print(f"Số ca bệnh: {status.case_count}")
        print(f"Báo cáo: {status.report_files} | ảnh chuẩn hóa: {status.normalized_files} | heatmap: {status.overlay_files}")
        return 0

    if args.command == "history":
        for item in MedicalCaseDatabase().list_cases()[: args.limit]:
            print(f"#{item.case_id} | {item.patient_code} | {item.risk_level} | nghi ngờ={item.suspected_malignant} | {item.created_at}")
            print(f"  Ảnh: {item.image_path}")
            print(f"  Báo cáo: {item.report_md_path}")
        return 0

    if args.command == "show-case":
        item = MedicalCaseDatabase().get_case(args.case_id)
        if item is None:
            print(f"Không tìm thấy ca bệnh #{args.case_id}.")
            return 1
        print(f"Ca bệnh #{item.case_id} | Mã bệnh nhân: {item.patient_code} | Thời gian: {item.created_at}")
        print(f"Nguy cơ: {item.risk_level}")
        print(f"Ảnh gốc: {item.image_path}")
        print(f"Ảnh xử lý: {item.processed_image_path}")
        print(f"Báo cáo JSON: {item.report_json_path}")
        print(f"Báo cáo MD: {item.report_md_path}")
        print(f"Metadata: {json.dumps(item.metadata, ensure_ascii=False, indent=2)}")
        return 0

    if args.command == "export-case":
        item = MedicalCaseDatabase().get_case(args.case_id)
        if item is None:
            print(f"Không tìm thấy ca bệnh #{args.case_id}.")
            return 1
        bundle = export_case_bundle(
            build_case_export_payload(item),
            args.output_dir,
            include_files=[item.image_path, item.processed_image_path, item.report_json_path, item.report_md_path],
            include_pdf=args.pdf,
        )
        print(f"Đã xuất ca bệnh #{item.case_id}: {bundle}")
        return 0

    if args.command == "delete-case":
        db = MedicalCaseDatabase()
        if args.delete_files:
            deleted, deleted_paths = db.delete_case_with_files(args.case_id)
        else:
            deleted, deleted_paths = db.delete_case(args.case_id), []
        if not deleted:
            print(f"Không tìm thấy ca bệnh #{args.case_id}.")
            return 1
        print(f"Đã xóa ca bệnh #{args.case_id}.")
        for path in deleted_paths:
            print(f"Đã xóa tệp: {path}")
        return 0

    if args.command == "cleanup-output":
        summary = cleanup_directories(_medical_output_directories(), older_than_days=args.older_than_days)
        print(f"Đã xóa {summary.removed_files} tệp, {summary.removed_dirs} thư mục rỗng; giải phóng {summary.freed_bytes} byte.")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
