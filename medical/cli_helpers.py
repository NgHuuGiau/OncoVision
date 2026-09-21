from __future__ import annotations

from pathlib import Path

from medical.system_status import MedicalSystemStatus


def print_medical_status_block(status: MedicalSystemStatus) -> None:
    print("Trạng thái phân tích y khoa")
    cfg = Path(status.configured_model_path)
    suffix = "" if cfg.exists() else " (chưa có file — chỉ dùng brain model)"
    print(f"Model cấu hình: {status.configured_model_path}{suffix}")
    if status.resolved_model_path is not None:
        print(f"Model đang dùng: {status.resolved_model_path}")
    print(f"Sẵn sàng phân tích: {status.model_ready}")
    print(f"Chi tiết: {status.model_message}")
    print("Nhóm bệnh có trong catalog:")
    for name, ready in status.screening_targets:
        print(f"- {name}: {'có model' if ready else 'chờ model suy luận'}")
    print(
        f"Ca đã lưu: {status.case_count} | báo cáo: {status.report_files} | "
        f"ảnh chuẩn hóa: {status.normalized_files} | heatmap: {status.overlay_files} | "
        f"bản xuất: {status.export_files}"
    )
