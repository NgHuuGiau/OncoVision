from __future__ import annotations

from pathlib import Path

from core.hardware_info import detect_hardware
from core.runtime_advisor import (
    MODE_ORDER,
    YOLO11_VARIANTS,
    build_recommendations,
    ceiling_mode_for_hardware,
    gpu_tier,
    load_level,
    mode_label,
    mode_title,
    quality_score,
    stability_score,
)
from core.model_selector import load_settings
from utils.console_ui import (
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    YELLOW,
    header,
    line,
    row,
    rule,
    section,
)
from utils.file_utils import load_yaml_cached


MODEL_CONFIG_PATH = "config/model_config.yaml"
MODE_PROMPT_CHOICES = {"0": "exit", "1": "high", "2": "medium", "3": "low"}


def _usage_text(value) -> str:
    if value is None:
        return "không rõ"
    return f"{float(value):.1f}%"


def _usage_color(value) -> str:
    if value is None:
        return YELLOW
    if float(value) < 60:
        return GREEN
    if float(value) < 85:
        return YELLOW
    return RED


def _available_models() -> tuple[list[str], list[str]]:
    available = []
    missing = []
    for name in YOLO11_VARIANTS:
        if Path("models/pretrained", name).exists() or Path(name).exists():
            available.append(name)
        else:
            missing.append(name)
    if Path("models/trained/best.pt").exists():
        available.insert(0, "models/trained/best.pt")
    return available, missing


def _load_level_label(hardware) -> str:
    return {
        "cool": "nhẹ",
        "warm": "trung bình",
        "busy": "cao",
        "very_busy": "rất cao",
    }[load_level(hardware)]


def _mode_color(mode: str) -> str:
    return {"high": GREEN, "medium": YELLOW, "low": MAGENTA}.get(mode, CYAN)


def _runtime_brief(runtime) -> str:
    return f"{runtime.primary_model_name} | {runtime.resolved_device} | imgsz {runtime.imgsz} | max_det {runtime.max_det}"


def _cuda_status_text(hardware, runtime) -> str:
    if getattr(hardware, "cuda_available", False):
        return f"Có, sẽ chạy bằng {runtime.resolved_device}"
    return f"Không, sẽ chạy bằng {runtime.resolved_device}"


def _mode_when_to_choose(mode: str, hardware) -> str:
    load_label = _load_level_label(hardware)
    if mode == "high":
        return f"Chọn khi ưu tiên chất lượng nhận diện cao nhất và máy đang đủ tải. Tải hiện tại: {load_label}."
    if mode == "medium":
        return f"Chọn khi cần cân bằng giữa độ mượt và độ chính xác. Tải hiện tại: {load_label}."
    return f"Chọn khi cần mở nhanh, mượt, ít rủi ro khựng hình. Tải hiện tại: {load_label}."


def _mode_risk_text(mode: str, hardware) -> str:
    load = load_level(hardware)
    if mode == "high":
        if load in {"busy", "very_busy"}:
            return "Không nên chọn nếu đang mở nhiều app hoặc muốn FPS ổn định."
        return "Phù hợp khi bạn chấp nhận nặng hơn để đổi lấy chất lượng cao hơn."
    if mode == "medium":
        if load == "very_busy":
            return "Nếu máy đang quá tải thì vẫn có thể giật, lúc đó nên xuống low."
        return "Đây thường là mức dễ chọn đúng nhất cho chạy camera hằng ngày."
    if load in {"cool", "warm"}:
        return "Nếu thấy nhận diện còn thiếu chi tiết, có thể tăng lên medium."
    return "Đây là mức an toàn nhất khi máy đang bận hoặc bạn cần ưu tiên độ mượt."


def _best_solution_text(auto_runtime, hardware) -> str:
    return (
        f"Chọn {mode_label(auto_runtime.mode)} ngay lúc này vì tải máy đang {_load_level_label(hardware)}. "
        f"Thực tế sẽ chạy: {_runtime_brief(auto_runtime)}."
    )


def _recommended_models_for_mode(mode: str, recommendations: dict[str, object] | None) -> list[str]:
    if not recommendations:
        return []
    runtime = recommendations.get(mode)
    if runtime is None:
        return []
    primary = getattr(runtime, "primary_model_name", "")
    candidates = list(getattr(runtime, "candidate_models", []) or [])
    return [item for item in dict.fromkeys([primary, *candidates]) if item]


def _model_local_text(models: list[str]) -> str:
    return ", ".join(models) if models else "không có model local nào"


def _project_model_text(settings: dict, mode: str) -> str:
    profile = settings["models"][mode]
    device = "gpu" if mode in {"high", "medium"} else "auto"
    return f"{profile['model']} / {device} / imgsz {profile['imgsz']}"


def _mode_decision_lines(mode: str, runtime, hardware) -> list[tuple[str, str, str]]:
    return [
        ("  Nên chọn khi", _mode_when_to_choose(mode, hardware), DIM),
        ("  Model / Imgsz", f"{runtime.primary_model_name} / {runtime.imgsz}", CYAN),
        ("  Thiết bị chạy", _cuda_status_text(hardware, runtime), CYAN),
        ("  Max det", str(runtime.max_det), CYAN),
        (
            "  Điểm số",
            f"chất lượng {quality_score(runtime)}/100 | ổn định {stability_score(mode, hardware)}/100",
            DIM,
        ),
        ("  Lưu ý", _mode_risk_text(mode, hardware), DIM),
    ]


def _print_lines(lines: list[str], *, print_fn=print) -> None:
    for item in lines:
        print_fn(item)


def _runtime_overview_lines(
    *,
    hardware,
    settings: dict,
    model_config: dict,
    recommendations: dict[str, object],
) -> list[str]:
    auto_runtime = recommendations["auto"]
    available_models, missing_models = _available_models()
    preferred = model_config.get("preferred_models", {})
    priority_order = model_config.get("priority_order", [])
    ceiling_mode = ceiling_mode_for_hardware(hardware)
    ceiling_runtime = recommendations[ceiling_mode]

    lines: list[str] = []
    lines.extend(header("BỘ TƯ VẤN RUNTIME YOLO :: THĂM DÒ MÁY VÀ CHỌN ĐÚNG MỨC CHẠY"))
    lines.extend(
        [
            section("TỔNG QUAN MÁY", GREEN),
            row("CPU", hardware.cpu_name, GREEN, bounded=False),
            row("RAM / OS", f"{hardware.ram_gb:.1f} GB / {hardware.os_name}", GREEN, bounded=False),
            row("GPU", hardware.gpu_name, GREEN if hardware.cuda_available else YELLOW, bounded=False),
            row("VRAM", f"{hardware.vram_gb:.1f} GB", GREEN if hardware.vram_gb else YELLOW, bounded=False),
            row("CUDA", "có" if hardware.cuda_available else "không", GREEN if hardware.cuda_available else RED, bounded=False),
            row("PyTorch", hardware.torch_version, CYAN, bounded=False),
            row("CUDA build", hardware.torch_cuda_version, CYAN, bounded=False),
            row("Phân hạng GPU", gpu_tier(hardware), YELLOW, bounded=False),
            row("Tải CPU", _usage_text(hardware.cpu_usage_percent), _usage_color(hardware.cpu_usage_percent), bounded=False),
            row("Tải GPU", _usage_text(hardware.gpu_usage_percent), _usage_color(hardware.gpu_usage_percent), bounded=False),
            row("Tải VRAM", _usage_text(hardware.vram_usage_percent), _usage_color(hardware.vram_usage_percent), bounded=False),
            row("Tải tổng thể", _load_level_label(hardware), MAGENTA, bounded=False),
            line(rule("-"), CYAN),
            section("YOLO11 VÀ MODEL LOCAL", MAGENTA),
            row("5 phiên bản", ", ".join(YOLO11_VARIANTS), CYAN, bounded=False),
            row("Model sẵn sàng", _model_local_text(available_models), GREEN, bounded=False),
            row(
                "Model còn thiếu",
                _model_local_text(missing_models) if missing_models else "đã có đủ các model chính",
                YELLOW,
                bounded=False,
            ),
            line(rule("-"), CYAN),
            section("HIỂU NHANH 3 MỨC", YELLOW),
            row("Mạnh nhất", "ưu tiên chất lượng cao nhất, nặng nhất", GREEN, bounded=False),
            row("Trung bình", "cân bằng tốt nhất để chạy thường xuyên", YELLOW, bounded=False),
            row("Yếu nhất", "ưu tiên mượt và an toàn, nhẹ nhất", MAGENTA, bounded=False),
            line(rule("-"), CYAN),
            section("3 MỨC THỰC TẾ TRÊN MÁY NÀY", CYAN),
        ]
    )

    for mode in MODE_ORDER:
        runtime = recommendations[mode]
        lines.append(row(mode_title(mode), "", _mode_color(mode), bounded=False))
        for label, value, style in _mode_decision_lines(mode, runtime, hardware):
            lines.append(row(label, value, style, bounded=False))
        lines.append(line(rule("."), DIM))

    lines.extend(
        [
            line(rule("-"), CYAN),
            section("CẤU HÌNH DỰ ÁN", GREEN),
            row("Thiết lập high", _project_model_text(settings, "high"), GREEN, bounded=False),
            row("Thiết lập medium", _project_model_text(settings, "medium"), YELLOW, bounded=False),
            row("Thiết lập low", _project_model_text(settings, "low"), MAGENTA, bounded=False),
            row(
                "Model ưu tiên",
                (
                    f"primary {preferred.get('primary_gpu', 'không rõ')} | "
                    f"backup GPU {preferred.get('stable_backup_gpu', 'không rõ')} | "
                    f"backup CPU {preferred.get('stable_backup_cpu', 'không rõ')}"
                ),
                CYAN,
                bounded=False,
            ),
        ]
    )
    if priority_order:
        lines.append(row("Thứ tự load", ", ".join(priority_order), DIM, bounded=False))

    lines.extend(
        [
            line(rule("-"), CYAN),
            section("CHỐT CÁCH CHỌN", MAGENTA),
            row("Trần tối đa", _runtime_brief(ceiling_runtime), GREEN, bounded=False),
            row("Nên chọn hằng ngày", _runtime_brief(recommendations["medium"]), YELLOW, bounded=False),
            row("Nên chọn ngay", _best_solution_text(auto_runtime, hardware), GREEN, bounded=False),
        ]
    )
    return lines


def _runtime_mode_choice_lines(hardware, recommendations: dict[str, object]) -> list[str]:
    return [
        line(rule("-"), CYAN),
        section("CHỌN CHẾ ĐỘ SẼ CHẠY", CYAN),
        row("Nên chọn ngay", _best_solution_text(recommendations["auto"], hardware), YELLOW, bounded=False),
        row("1 | MẠNH NHẤT", _runtime_brief(recommendations["high"]), GREEN, bounded=False),
        row("2 | TRUNG BÌNH", _runtime_brief(recommendations["medium"]), YELLOW, bounded=False),
        row("3 | YẾU NHẤT", _runtime_brief(recommendations["low"]), MAGENTA, bounded=False),
        line(rule("."), DIM),
        row("0 | THOÁT", "Đóng chương trình ngay tại đây.", RED, bounded=False),
        line(rule("="), CYAN),
    ]


def prompt_runtime_mode(hardware=None, recommendations=None, input_fn=input, print_fn=print) -> str:
    hardware = hardware or detect_hardware()
    recommendations = recommendations or build_recommendations(hardware)
    settings = load_settings()
    model_config = load_yaml_cached(MODEL_CONFIG_PATH)

    while True:
        _print_lines(
            _runtime_overview_lines(
                hardware=hardware,
                settings=settings,
                model_config=model_config,
                recommendations=recommendations,
            ),
            print_fn=print_fn,
        )
        _print_lines(_runtime_mode_choice_lines(hardware, recommendations), print_fn=print_fn)

        raw_value = input_fn(line("Nhập lựa chọn của bạn (0/1/2/3): ", GREEN)).strip()
        selected_mode = MODE_PROMPT_CHOICES.get(raw_value)
        if selected_mode == "exit":
            raise SystemExit(0)
        if selected_mode:
            print_fn("")
            print_fn(line(f"Đã chọn chế độ: {mode_label(selected_mode)}", GREEN))
            return selected_mode
        print_fn(line("Lựa chọn không hợp lệ. Vui lòng nhập 0, 1, 2 hoặc 3.", RED))


def prompt_runtime_model(
    *,
    selected_mode: str,
    recommendations: dict[str, object] | None = None,
    input_fn=input,
    print_fn=print,
) -> str:
    available_models, missing_models = _available_models()
    recommended = _recommended_models_for_mode(selected_mode, recommendations)
    options = list(dict.fromkeys([*recommended, *available_models]))
    if not options:
        raise RuntimeError("Khong co model local nao de chon. Hay chay training/download_models.py truoc.")
    if len(options) == 1:
        chosen = options[0]
        print_fn("")
        print_fn(line(f"Tự động chọn model duy nhất: {chosen}", GREEN))
        return chosen

    while True:
        print_fn(line(rule("="), CYAN))
        print_fn(section("CHỌN MODEL SẼ CHẠY", CYAN))
        print_fn(row("Chế độ đã chọn", mode_title(selected_mode), _mode_color(selected_mode), bounded=False))
        if recommended:
            print_fn(row("Model nên dùng", ", ".join(recommended), GREEN, bounded=False))
        if missing_models:
            print_fn(row("Model còn thiếu", ", ".join(missing_models), YELLOW, bounded=False))
        print_fn(line(rule("-"), CYAN))
        for index, model_name in enumerate(options, start=1):
            hint = "khuyến nghị" if model_name in recommended else "có sẵn"
            color = GREEN if model_name in recommended else CYAN
            print_fn(row(f"{index} | {model_name}", hint, color))
        print_fn(line(rule("."), DIM))
        print_fn(row("0 | THOÁT", "Đóng chương trình ngay tại đây.", RED))
        print_fn(line(rule("-"), CYAN))

        raw_value = input_fn(line(f"Nhập lựa chọn của bạn (0-{len(options)}): ")).strip()
        if raw_value == "0":
            raise SystemExit(0)
        if raw_value.isdigit():
            selected_index = int(raw_value) - 1
            if 0 <= selected_index < len(options):
                selected_model = options[selected_index]
                print_fn("")
                print_fn(line(f"Đã chọn model: {selected_model}", GREEN))
                return selected_model
        print_fn(line("Lựa chọn không hợp lệ. Vui lòng nhập số trong danh sách.", RED))
        input_fn(line("Nhấn Enter để chọn lại...", DIM))


def main() -> None:
    hardware = detect_hardware()
    settings = load_settings()
    model_config = load_yaml_cached(MODEL_CONFIG_PATH)
    recommendations = build_recommendations(hardware)
    _print_lines(
        _runtime_overview_lines(
            hardware=hardware,
            settings=settings,
            model_config=model_config,
            recommendations=recommendations,
        )
    )
    print(line(rule("="), CYAN))
