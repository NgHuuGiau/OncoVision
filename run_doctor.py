from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.chat_ui.paths import get_chat_capture_dir
from core.hardware_info import detect_hardware
from core.model_catalog import YOLO11_MODELS_ASC
from core.runtime_advisor import optimized_runtime
from medical.dataset import create_default_medical_dataset_config
from medical.status_helpers import count_files
from medical.system_status import get_medical_system_status, recommended_medical_commands
from training.terminal_ui import CYAN, GREEN, RED, YELLOW, header, line, row, rule, section
from utils.doctor_helpers import medical_status_color, print_medical_status, print_recommended_commands
from utils.camera_probe import probe_camera
from utils.camera_utils import open_camera_capture
from utils.entrypoint_checks import medical_config_issues, runtime_config_issues
from utils.entrypoint_common import run_entrypoint
from utils.file_utils import ensure_project_directories


YOLO11_MODELS = YOLO11_MODELS_ASC
PRETRAINED_DIR = Path("models/pretrained")
RAW_IMAGES_DIR = Path("dataset/object_detection/raw/images")
RAW_LABELS_DIR = Path("dataset/object_detection/raw/labels")
PROCESSED_TRAIN_DIR = Path("dataset/object_detection/processed/images/train")
PROCESSED_VAL_DIR = Path("dataset/object_detection/processed/images/val")
MEDICAL_SKIN_ROOT = create_default_medical_dataset_config().dataset_root
MEDICAL_SKIN_RAW_IMAGES_DIR = MEDICAL_SKIN_ROOT / "raw" / "images"
MEDICAL_SKIN_RAW_LABELS_DIR = MEDICAL_SKIN_ROOT / "raw" / "labels"
MEDICAL_SKIN_PROCESSED_TRAIN_DIR = MEDICAL_SKIN_ROOT / "processed" / "images" / "train"
MEDICAL_SKIN_PROCESSED_VAL_DIR = MEDICAL_SKIN_ROOT / "processed" / "images" / "val"
ICONS_DIR = Path("assets/icons")
ICON_WARNING_THRESHOLD = 10
ICON_AUTOFIX_THRESHOLD = 5
RUNTIME_RECOMMENDATION_MODES = (
    ("Cao nhất", "high"),
    ("Trung bình", "medium"),
    ("Yếu", "low"),
)


@dataclass
class CameraProbeResult:
    level: str
    summary: str
    detail: str

    @property
    def color(self) -> str:
        return {"PASS": GREEN, "WARN": YELLOW, "ERROR": RED}.get(self.level, CYAN)
def _present_and_missing_models(model_dir: Path = PRETRAINED_DIR) -> tuple[list[str], list[str]]:
    present = [name for name in YOLO11_MODELS if (model_dir / name).exists()]
    missing = [name for name in YOLO11_MODELS if name not in present]
    return present, missing


def _runtime_recommendations(hardware) -> dict[str, object]:
    return {
        label: optimized_runtime(mode, hardware)
        for label, mode in RUNTIME_RECOMMENDATION_MODES
    }


_open_camera_capture = open_camera_capture


def _probe_camera(index: int = 0) -> CameraProbeResult:
    result = probe_camera(
        index=index,
        attempts=3,
        unavailable_detail="Lý do không chạy  Webcam không sẵn sàng, đang bị app khác chiếm hoặc chưa cắm.",
        open_camera_capture_fn=_open_camera_capture,
    )
    return CameraProbeResult(level=result.level, summary=result.summary, detail=result.detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra sức khỏe toàn hệ thống cho dự án OncoVision.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index để kiểm tra webcam thật.")
    parser.add_argument(
        "--skip-camera-check",
        action="store_true",
        help="Bỏ qua bước kiểm tra camera thật.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Tự động sửa các lỗi cơ bản như thiếu icon.",
    )
    return parser.parse_args()


def _run_autofix() -> None:
    print(line(rule("-"), CYAN))
    print(section("AUTO-FIX", YELLOW))
    if not ICONS_DIR.exists() or sum(1 for _ in ICONS_DIR.iterdir()) < ICON_AUTOFIX_THRESHOLD:
        print(row("Icons", "Đang tạo bộ icon mặc định...", YELLOW))
        from utils.icons import create_default_icons

        create_default_icons()
    print(row("Trạng thái", "Đã chạy xong Auto-fix!", GREEN))


def _print_camera_probe(camera_probe: CameraProbeResult) -> None:
    print(line(rule("-"), CYAN))
    print(section("CAMERA THẬT", camera_probe.color))
    print(row("Trạng thái", camera_probe.summary.split("|", 1)[-1].strip(), camera_probe.color, bounded=False))
    print(
        row(
            "Chi tiết",
            camera_probe.detail.replace("Chi tiết          ", "").replace("Lý do không chạy  ", ""),
            camera_probe.color,
            bounded=False,
        )
    )


def _print_recommendations(recommendations: dict[str, object]) -> None:
    print(line(rule("-"), CYAN))
    print(section("GỢI Ý CHẠY THEO MÁY", GREEN))
    for label, runtime in recommendations.items():
        value = f"{runtime.primary_model_name} / {runtime.resolved_device} / imgsz {runtime.imgsz}"
        color = GREEN if runtime.primary_model_name != "yolo11n.pt" else YELLOW
        print(row(label, value, color, bounded=False))


def _print_config_health() -> None:
    issues = runtime_config_issues() + medical_config_issues()
    print(line(rule("-"), CYAN))
    print(section("CẤU HÌNH", GREEN if not issues else YELLOW))
    if not issues:
        print(row("Runtime config", "Hợp lệ", GREEN, bounded=False))
    else:
        print(row("Runtime config", "Cần kiểm tra", YELLOW, bounded=False))
        for issue in issues:
            print(row("Vấn đề", issue, RED, bounded=False))


def main() -> None:
    args = parse_args()
    ensure_project_directories()

    if args.fix:
        _run_autofix()

    hardware = detect_hardware()
    present_models, missing_models = _present_and_missing_models()
    chat_capture_dir = get_chat_capture_dir(ensure_exists=False)
    raw_images = count_files(RAW_IMAGES_DIR)
    raw_labels = count_files(RAW_LABELS_DIR)
    train_images = count_files(PROCESSED_TRAIN_DIR)
    val_images = count_files(PROCESSED_VAL_DIR)
    icon_count = count_files(ICONS_DIR)
    chat_capture_count = count_files(chat_capture_dir)
    medical_status = get_medical_system_status()
    med_raw_images = count_files(MEDICAL_SKIN_RAW_IMAGES_DIR)
    med_raw_labels = count_files(MEDICAL_SKIN_RAW_LABELS_DIR)
    med_train_images = count_files(MEDICAL_SKIN_PROCESSED_TRAIN_DIR)
    med_val_images = count_files(MEDICAL_SKIN_PROCESSED_VAL_DIR)
    camera_probe = None if args.skip_camera_check else _probe_camera(args.camera_index)
    recommendations = _runtime_recommendations(hardware)

    for item in header("OncoVision DOCTOR :: KIỂM TRA TOÀN HỆ THỐNG"):
        print(item)

    print(section("PHẦN CỨNG", GREEN if hardware.cuda_available else YELLOW))
    print(row("CPU", hardware.cpu_name, GREEN))
    print(row("RAM / OS", f"{hardware.ram_gb:.1f} GB / {hardware.os_name}", GREEN))
    print(row("GPU", hardware.gpu_name, GREEN if hardware.gpu_hardware_available else YELLOW))
    print(row("VRAM / GPU", f"{hardware.vram_gb:.1f} GB / {hardware.gpu_count}", GREEN if hardware.gpu_hardware_available else YELLOW))
    print(row("PyTorch", hardware.torch_version, GREEN if hardware.torch_version != "Không có PyTorch" else RED, bounded=False))
    print(row("CUDA", hardware.cuda_runtime_reason, GREEN if hardware.cuda_available else YELLOW, bounded=False))

    if camera_probe is not None:
        _print_camera_probe(camera_probe)

    print(line(rule("-"), CYAN))
    icons_ok = icon_count >= ICON_WARNING_THRESHOLD
    print(section("GIAO DIỆN & ICONS", GREEN if icons_ok else YELLOW))
    print(row("Icons (.svg)", f"{icon_count} file trong assets/icons", GREEN if icons_ok else RED, bounded=False))
    if not icons_ok:
        print(row("Cảnh báo", "Thiếu icon sẽ làm giao diện bị đen trắng.", RED, bounded=False))

    print(line(rule("-"), CYAN))
    print(section("MODEL YOLO11", GREEN if not missing_models else YELLOW))
    print(row("Đã có", ", ".join(present_models) if present_models else "Chưa có model nào", GREEN if present_models else RED, bounded=False))
    if missing_models:
        print(row("Thiếu", ", ".join(missing_models), RED, bounded=False))
    else:
        print(row("Trạng thái", "Đã có đủ 5 model YOLO11.", GREEN, bounded=False))

    _print_recommendations(recommendations)

    print(line(rule("-"), CYAN))
    dataset_ok = raw_images > 0 and raw_labels > 0
    split_ok = train_images > 0 and val_images > 0
    print(section("DỮ LIỆU", GREEN if dataset_ok or med_raw_images > 0 else YELLOW))
    print(row("Vật thể raw images", f"{RAW_IMAGES_DIR} ({raw_images} file)", GREEN if raw_images else RED, bounded=False))
    print(row("Vật thể raw labels", f"{RAW_LABELS_DIR} ({raw_labels} file)", GREEN if raw_labels else RED, bounded=False))
    print(row("Vật thể train split", f"{PROCESSED_TRAIN_DIR} ({train_images} file)", GREEN if train_images else YELLOW, bounded=False))
    print(row("Vật thể val split", f"{PROCESSED_VAL_DIR} ({val_images} file)", GREEN if val_images else YELLOW, bounded=False))
    print(row("Y dược raw images", f"{MEDICAL_SKIN_RAW_IMAGES_DIR} ({med_raw_images} file)", GREEN if med_raw_images else RED, bounded=False))
    print(row("Y dược raw labels", f"{MEDICAL_SKIN_RAW_LABELS_DIR} ({med_raw_labels} file)", GREEN if med_raw_labels else RED, bounded=False))
    print(row("Y dược train split", f"{MEDICAL_SKIN_PROCESSED_TRAIN_DIR} ({med_train_images} file)", GREEN if med_train_images else YELLOW, bounded=False))
    print(row("Y dược val split", f"{MEDICAL_SKIN_PROCESSED_VAL_DIR} ({med_val_images} file)", GREEN if med_val_images else YELLOW, bounded=False))

    print_medical_status(medical_status)
    _print_config_health()

    print(line(rule("-"), CYAN))
    print(section("OUTPUT", GREEN))
    print(row("Chat captures", f"{chat_capture_dir} ({chat_capture_count} file)", CYAN, bounded=False))

    print(line(rule("-"), CYAN))
    ready = bool(present_models) and dataset_ok and medical_status.model_ready
    print(section("KẾT LUẬN", GREEN if ready else YELLOW))
    if not present_models:
        print(row("Lý do", "Chưa có model local trong models/pretrained.", RED, bounded=False))
    elif missing_models:
        print(row("Lý do", "Máy vẫn chạy được, nhưng chưa có đủ 5 model để chọn hết mọi mức.", YELLOW, bounded=False))
    else:
        print(row("Model", "Đã sẵn sàng để chạy đủ các mức YOLO11.", GREEN, bounded=False))

    if camera_probe is not None and camera_probe.level != "PASS":
        print(row("Camera", camera_probe.detail.replace("Lý do không chạy  ", ""), YELLOW, bounded=False))

    if not dataset_ok:
        print(row("Dataset", "Chưa có dữ liệu raw cho luồng vật thể.", YELLOW, bounded=False))
    elif not split_ok:
        print(row("Dataset", "Đã có raw vật thể nhưng chưa split train/val.", YELLOW, bounded=False))
    else:
        print(row("Dataset", "Dữ liệu train/val vật thể đã sẵn sàng.", GREEN, bounded=False))
    print(row("Medical", f"{medical_status.model_message} | raw={med_raw_images}/{med_raw_labels}", medical_status_color(medical_status), bounded=False))

    print_recommended_commands(
        missing_models=missing_models,
        icon_count=icon_count,
        dataset_ok=dataset_ok,
        split_ok=split_ok,
        medical_commands=recommended_medical_commands(medical_status),
        icon_warning_threshold=ICON_WARNING_THRESHOLD,
    )
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
