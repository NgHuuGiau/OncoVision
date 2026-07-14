from __future__ import annotations

from medical.system_status import MedicalSystemStatus
from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, line, row, rule, section


def medical_status_color(status: MedicalSystemStatus) -> str:
    if not status.model_ready:
        return RED
    if status.using_fallback_model or status.allow_fallback_model:
        return YELLOW
    return GREEN


def print_medical_status(status: MedicalSystemStatus) -> None:
    color = medical_status_color(status)
    print(line(rule("-"), CYAN))
    print(section("MEDICAL", color))
    print(row("Model config", str(status.configured_model_path), CYAN, bounded=False))
    if status.resolved_model_path is not None:
        print(row("Model runtime", str(status.resolved_model_path), color, bounded=False))
    print(row("Fallback", "Bật" if status.allow_fallback_model else "Tắt", YELLOW if status.allow_fallback_model else GREEN, bounded=False))
    print(row("Trạng thái", status.model_message, color, bounded=False))
    print(row("Dataset root", str(status.dataset_root), CYAN, bounded=False))
    print(
        row(
            "Medical data",
            f"raw {status.raw_images}/{status.raw_labels} | train {status.train_images} | val {status.val_images} | test {status.test_images}",
            GREEN if status.processed_dataset_ready else YELLOW,
            bounded=False,
        )
    )
    print(
        row(
            "Cases / output",
            f"{status.case_count} ca | reports {status.report_files} | normalized {status.normalized_files} | overlay {status.overlay_files} | exports {status.export_files}",
            CYAN,
            bounded=False,
        )
    )
    if status.screening_targets:
        ready_targets = ", ".join(label for label, ready in status.screening_targets if ready) or "không có"
        missing_targets = ", ".join(label for label, ready in status.screening_targets if not ready) or "không có"
        print(row("Target đã sẵn sàng", ready_targets, GREEN, bounded=False))
        print(row("Target cần mở rộng", missing_targets, YELLOW, bounded=False))


def print_recommended_commands(
    *,
    missing_models: list[str],
    icon_count: int,
    dataset_ok: bool,
    split_ok: bool,
    medical_commands: list[str],
    icon_warning_threshold: int,
) -> None:
    print(line(rule("-"), CYAN))
    print(section("LỆNH NÊN CHẠY", CYAN))
    command_index = 1
    commands: list[str] = []
    if missing_models:
        commands.append("python training/download_models.py")
    if icon_count < icon_warning_threshold:
        commands.append("python run_doctor.py --fix")
    if not dataset_ok:
        commands.append("python training/prepare_dataset.py")
    elif not split_ok:
        commands.append("python training/validate_dataset.py")
        commands.append("python training/split_dataset.py")
    else:
        commands.append("python run_chat.py")
        commands.append("python run_train.py")
    commands.extend(medical_commands)
    for command in dict.fromkeys(commands):
        print(command_row(command_index, command))
        command_index += 1
