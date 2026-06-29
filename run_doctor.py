from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.chat_ui.paths import get_chat_capture_dir
from core.hardware_info import detect_hardware
from core.model_catalog import YOLO11_MODELS_ASC
from core.runtime_advisor import select_runtime_config_optimized
from medical.system_status import MedicalSystemStatus, get_medical_system_status, recommended_medical_commands
from medical.dataset import create_default_skin_cancer_dataset_config
from utils.entrypoint_checks import medical_config_issues, runtime_config_issues
from training.terminal_ui import CYAN, GREEN, RED, YELLOW, command_row, header, line, row, rule, section
from utils.entrypoint_common import run_entrypoint
from utils.camera_probe import probe_camera
from utils.camera_utils import open_camera_capture
from utils.file_utils import ensure_project_directories
from utils.path_counts import count_files


YOLO11_MODELS = YOLO11_MODELS_ASC
PRETRAINED_DIR = Path("models/pretrained")
RAW_IMAGES_DIR = Path("dataset/object_detection/raw/images")
RAW_LABELS_DIR = Path("dataset/object_detection/raw/labels")
PROCESSED_TRAIN_DIR = Path("dataset/object_detection/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/object_detection/processed/images/val")
MEDICAL_SKIN_ROOT = create_default_skin_cancer_dataset_config().dataset_root
MEDICAL_SKIN_RAW_IMAGES_DIR = MEDICAL_SKIN_ROOT / "raw" / "images"
MEDICAL_SKIN_RAW_LABELS_DIR = MEDICAL_SKIN_ROOT / "raw" / "labels"
MEDICAL_SKIN_PROCESSED_TRAIN_DIR = MEDICAL_SKIN_ROOT / "processed" / "images" / "train"
MEDICAL_SKIN_PROCESSED_VAL_DIR = MEDICAL_SKIN_ROOT / "processed" / "images" / "val"
ICONS_DIR = Path("assets/icons")
ICON_WARNING_THRESHOLD = 10
ICON_AUTOFIX_THRESHOLD = 5
RUNTIME_RECOMMENDATION_MODES = (
    ("Cao nhat", "high"),
    ("Trung binh", "medium"),
    ("Yeu", "low"),
)


@dataclass
class CameraProbeResult:
    level: str
    summary: str
    detail: str

    @property
    def color(self) -> str:
        return {"PASS": GREEN, "WARN": YELLOW, "ERROR": RED}.get(self.level, CYAN)


def _count_files(path: Path) -> int:
    return count_files(path)


def _count_project_files(*paths: Path) -> tuple[int, ...]:
    return tuple(_count_files(path) for path in paths)


def _medical_skin_counts() -> tuple[int, int, int, int]:
    return _count_project_files(
        MEDICAL_SKIN_RAW_IMAGES_DIR,
        MEDICAL_SKIN_RAW_LABELS_DIR,
        MEDICAL_SKIN_PROCESSED_TRAIN_DIR,
        MEDICAL_SKIN_PROCESSED_VAL_DIR,
    )


def _present_and_missing_models(model_dir: Path = PRETRAINED_DIR) -> tuple[list[str], list[str]]:
    present = [name for name in YOLO11_MODELS if (model_dir / name).exists()]
    missing = [name for name in YOLO11_MODELS if name not in present]
    return present, missing


def _runtime_recommendations(hardware) -> dict[str, object]:
    return {
        label: select_runtime_config_optimized(mode, hardware)
        for label, mode in RUNTIME_RECOMMENDATION_MODES
    }


_open_camera_capture = open_camera_capture


def _probe_camera(index: int = 0) -> CameraProbeResult:
    result = probe_camera(
        index=index,
        attempts=3,
        unavailable_detail="Ly do khong chay   Webcam khong san sang, dang bi app khac chiem hoac chua cam.",
        open_camera_capture_fn=_open_camera_capture,
    )
    return CameraProbeResult(level=result.level, summary=result.summary, detail=result.detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiem tra suc khoe toan he thong cho du an OncoVision.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index de kiem tra webcam that.")
    parser.add_argument(
        "--skip-camera-check",
        action="store_true",
        help="Bo qua buoc kiem tra camera that.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Tu dong sua cac loi co ban nhu thieu icon.",
    )
    return parser.parse_args()


def _run_autofix() -> None:
    print(line(rule("-"), CYAN))
    print(section("AUTO-FIX", YELLOW))
    if not ICONS_DIR.exists() or sum(1 for _ in ICONS_DIR.iterdir()) < ICON_AUTOFIX_THRESHOLD:
        print(row("Icons", "Dang tao bo icon mac dinh...", YELLOW))
        from utils.icons import create_default_icons

        create_default_icons()
    print(row("Trang thai", "Da chay xong Auto-fix!", GREEN))


def _print_camera_probe(camera_probe: CameraProbeResult) -> None:
    print(line(rule("-"), CYAN))
    print(section("CAMERA THAT", camera_probe.color))
    print(row("Trang thai", camera_probe.summary.split("|", 1)[-1].strip(), camera_probe.color, bounded=False))
    print(
        row(
            "Chi tiet",
            camera_probe.detail.replace("Chi tiet          ", "").replace("Ly do khong chay   ", ""),
            camera_probe.color,
            bounded=False,
        )
    )


def _print_recommendations(recommendations: dict[str, object]) -> None:
    print(line(rule("-"), CYAN))
    print(section("GOI Y CHAY THEO MAY", GREEN))
    for label, runtime in recommendations.items():
        value = f"{runtime.primary_model_name} / {runtime.resolved_device} / imgsz {runtime.imgsz}"
        color = GREEN if runtime.primary_model_name != "yolo11n.pt" else YELLOW
        print(row(label, value, color, bounded=False))


def _medical_status_color(status: MedicalSystemStatus) -> str:
    if not status.model_ready:
        return RED
    if status.using_fallback_model or status.allow_fallback_model:
        return YELLOW
    return GREEN


def _print_medical_status(status: MedicalSystemStatus) -> None:
    color = _medical_status_color(status)
    print(line(rule("-"), CYAN))
    print(section("MEDICAL", color))
    print(row("Model config", str(status.configured_model_path), CYAN, bounded=False))
    if status.resolved_model_path is not None:
        print(row("Model runtime", str(status.resolved_model_path), color, bounded=False))
    print(row("Fallback", "Bat" if status.allow_fallback_model else "Tat", YELLOW if status.allow_fallback_model else GREEN, bounded=False))
    print(row("Trang thai", status.model_message, color, bounded=False))
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
        ready_targets = ", ".join(label for label, ready in status.screening_targets if ready) or "khong co"
        missing_targets = ", ".join(label for label, ready in status.screening_targets if not ready) or "khong co"
        print(row("Target da san sang", ready_targets, GREEN, bounded=False))
        print(row("Target can mo rong", missing_targets, YELLOW, bounded=False))


def _print_config_health() -> None:
    issues = runtime_config_issues() + medical_config_issues()
    print(line(rule("-"), CYAN))
    print(section("CAU HINH", GREEN if not issues else YELLOW))
    if not issues:
        print(row("Runtime config", "Hop le", GREEN, bounded=False))
    else:
        print(row("Runtime config", "Can kiem tra", YELLOW, bounded=False))
        for issue in issues:
            print(row("Van de", issue, RED, bounded=False))


def _print_recommended_commands(
    *,
    missing_models: list[str],
    icon_count: int,
    dataset_ok: bool,
    split_ok: bool,
    medical_commands: list[str],
) -> None:
    print(line(rule("-"), CYAN))
    print(section("LENH NEN CHAY", CYAN))
    command_index = 1
    commands: list[str] = []
    if missing_models:
        commands.append("python training\\download_models.py")
    if icon_count < ICON_WARNING_THRESHOLD:
        commands.append("python run_doctor.py --fix")
    if not dataset_ok:
        commands.append("python training\\prepare_dataset.py")
    elif not split_ok:
        commands.append("python training\\validate_dataset.py")
        commands.append("python training\\split_dataset.py")
    else:
        commands.append("python run_chat.py")
        commands.append("python run_train.py")
    commands.extend(medical_commands)
    for command in dict.fromkeys(commands):
        print(command_row(command_index, command))
        command_index += 1


def main() -> None:
    args = parse_args()
    ensure_project_directories()

    if args.fix:
        _run_autofix()

    hardware = detect_hardware()
    present_models, missing_models = _present_and_missing_models()
    chat_capture_dir = get_chat_capture_dir(ensure_exists=False)
    raw_images, raw_labels, train_images, val_images, icon_count, chat_capture_count = _count_project_files(
        RAW_IMAGES_DIR,
        RAW_LABELS_DIR,
        PROCESSED_TRAIN_DIR,
        PROCESSED_VAL_DIR,
        ICONS_DIR,
        chat_capture_dir,
    )
    medical_status = get_medical_system_status()
    med_raw_images, med_raw_labels, med_train_images, med_val_images = _medical_skin_counts()
    camera_probe = None if args.skip_camera_check else _probe_camera(args.camera_index)
    recommendations = _runtime_recommendations(hardware)

    for item in header("OncoVision DOCTOR :: KIEM TRA TOAN HE THONG"):
        print(item)

    print(section("PHAN CUNG", GREEN if hardware.cuda_available else YELLOW))
    print(row("CPU", hardware.cpu_name, GREEN))
    print(row("RAM / OS", f"{hardware.ram_gb:.1f} GB / {hardware.os_name}", GREEN))
    print(row("GPU", hardware.gpu_name, GREEN if hardware.gpu_hardware_available else YELLOW))
    print(row("VRAM / GPU", f"{hardware.vram_gb:.1f} GB / {hardware.gpu_count}", GREEN if hardware.gpu_hardware_available else YELLOW))
    print(row("PyTorch", hardware.torch_version, GREEN if hardware.torch_version != "Khong co PyTorch" else RED, bounded=False))
    print(row("CUDA", hardware.cuda_runtime_reason, GREEN if hardware.cuda_available else YELLOW, bounded=False))

    if camera_probe is not None:
        _print_camera_probe(camera_probe)

    print(line(rule("-"), CYAN))
    icons_ok = icon_count >= ICON_WARNING_THRESHOLD
    print(section("GIAO DIEN & ICONS", GREEN if icons_ok else YELLOW))
    print(row("Icons (.svg)", f"{icon_count} file trong assets/icons", GREEN if icons_ok else RED, bounded=False))
    if not icons_ok:
        print(row("Canh bao", "Thieu icon se lam giao dien bi den trang.", RED, bounded=False))

    print(line(rule("-"), CYAN))
    print(section("MODEL YOLO11", GREEN if not missing_models else YELLOW))
    print(row("Da co", ", ".join(present_models) if present_models else "Chua co model nao", GREEN if present_models else RED, bounded=False))
    if missing_models:
        print(row("Thieu", ", ".join(missing_models), RED, bounded=False))
    else:
        print(row("Trang thai", "Da co du 5 model YOLO11.", GREEN, bounded=False))

    _print_recommendations(recommendations)

    print(line(rule("-"), CYAN))
    dataset_ok = raw_images > 0 and raw_labels > 0
    split_ok = train_images > 0 and val_images > 0
    print(section("DU LIEU", GREEN if dataset_ok or med_raw_images > 0 else YELLOW))
    print(row("Vat the raw images", f"{RAW_IMAGES_DIR} ({raw_images} file)", GREEN if raw_images else RED, bounded=False))
    print(row("Vat the raw labels", f"{RAW_LABELS_DIR} ({raw_labels} file)", GREEN if raw_labels else RED, bounded=False))
    print(row("Vat the train split", f"{PROCESSED_TRAIN_DIR} ({train_images} file)", GREEN if train_images else YELLOW, bounded=False))
    print(row("Vat the val split", f"{PROCESSED_VAL_DIR} ({val_images} file)", GREEN if val_images else YELLOW, bounded=False))
    print(row("Y duoc raw images", f"{MEDICAL_SKIN_RAW_IMAGES_DIR} ({med_raw_images} file)", GREEN if med_raw_images else RED, bounded=False))
    print(row("Y duoc raw labels", f"{MEDICAL_SKIN_RAW_LABELS_DIR} ({med_raw_labels} file)", GREEN if med_raw_labels else RED, bounded=False))
    print(row("Y duoc train split", f"{MEDICAL_SKIN_PROCESSED_TRAIN_DIR} ({med_train_images} file)", GREEN if med_train_images else YELLOW, bounded=False))
    print(row("Y duoc val split", f"{MEDICAL_SKIN_PROCESSED_VAL_DIR} ({med_val_images} file)", GREEN if med_val_images else YELLOW, bounded=False))

    _print_medical_status(medical_status)
    _print_config_health()

    print(line(rule("-"), CYAN))
    print(section("OUTPUT", GREEN))
    print(row("Chat captures", f"{chat_capture_dir} ({chat_capture_count} file)", CYAN, bounded=False))

    print(line(rule("-"), CYAN))
    ready = bool(present_models) and dataset_ok and medical_status.model_ready
    print(section("KET LUAN", GREEN if ready else YELLOW))
    if not present_models:
        print(row("Ly do", "Chua co model local trong models/pretrained.", RED, bounded=False))
    elif missing_models:
        print(row("Ly do", "May van chay duoc, nhung chua co du 5 model de chon het moi muc.", YELLOW, bounded=False))
    else:
        print(row("Model", "Da san sang de chay du cac muc YOLO11.", GREEN, bounded=False))

    if camera_probe is not None and camera_probe.level != "PASS":
        print(row("Camera", camera_probe.detail.replace("Ly do khong chay   ", ""), YELLOW, bounded=False))

    if not dataset_ok:
        print(row("Dataset", "Chua co du lieu raw cho luong vat the.", YELLOW, bounded=False))
    elif not split_ok:
        print(row("Dataset", "Da co raw vat the nhung chua split train/val.", YELLOW, bounded=False))
    else:
        print(row("Dataset", "Du lieu train/val vat the da san sang.", GREEN, bounded=False))
    print(row("Medical", f"{medical_status.model_message} | raw={med_raw_images}/{med_raw_labels}", _medical_status_color(medical_status), bounded=False))

    _print_recommended_commands(
        missing_models=missing_models,
        icon_count=icon_count,
        dataset_ok=dataset_ok,
        split_ok=split_ok,
        medical_commands=recommended_medical_commands(medical_status),
    )
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
