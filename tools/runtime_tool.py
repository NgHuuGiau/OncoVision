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
        return "khong ro"
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


def _summary_line(mode: str, runtime) -> str:
    return (
        f"{mode_title(mode)} ({mode}) -> "
        f"model={runtime.primary_model_name}, "
        f"thiet bi={runtime.resolved_device}, "
        f"imgsz={runtime.imgsz}, "
        f"max_det={runtime.max_det}, "
        f"fallback={', '.join(runtime.candidate_models)}"
    )


def _mode_reason(mode: str, runtime, hardware) -> str:
    quality = quality_score(runtime)
    stability = stability_score(mode, hardware)
    if mode == "high":
        return f"Tran cao nhat may con ganh duoc. chat luong={quality}/100, on dinh={stability}/100."
    if mode == "medium":
        return f"Muc can bang dep nhat de dung thuong xuyen. chat luong={quality}/100, on dinh={stability}/100."
    return f"Muc an toan nhat khi uu tien do muot. chat luong={quality}/100, on dinh={stability}/100."


def _best_solution_text(auto_runtime, hardware) -> str:
    load_text = {
        "cool": "tai hien tai nhe",
        "warm": "tai hien tai trung binh",
        "busy": "tai hien tai kha cao",
        "very_busy": "tai hien tai rat cao",
    }[load_level(hardware)]
    return (
        f"De xuat nen chay ngay la {mode_label(auto_runtime.mode)} vi {load_text}, "
        f"voi {auto_runtime.primary_model_name} / {auto_runtime.resolved_device} / "
        f"imgsz {auto_runtime.imgsz} / max_det {auto_runtime.max_det}."
    )


def _wow_conclusion(hardware, recommendations, auto_runtime) -> list[str]:
    ceiling_mode = ceiling_mode_for_hardware(hardware)
    ceiling_runtime = recommendations[ceiling_mode]
    return [
        (
            f"tran toi da may dang ganh duoc: {mode_label(ceiling_mode)} ({ceiling_mode}) / "
            f"{ceiling_runtime.primary_model_name} / {ceiling_runtime.resolved_device} / "
            f"imgsz {ceiling_runtime.imgsz} / max_det {ceiling_runtime.max_det}"
        ),
        (
            f"muc dep nhat de chay thuong xuyen: trung binh / "
            f"{recommendations['medium'].primary_model_name} / {recommendations['medium'].resolved_device} / "
            f"imgsz {recommendations['medium'].imgsz} / max_det {recommendations['medium'].max_det}"
        ),
        (
            f"muc an toan nhat khi muon muot: yeu nhat / "
            f"{recommendations['low'].primary_model_name} / {recommendations['low'].resolved_device} / "
            f"imgsz {recommendations['low'].imgsz} / max_det {recommendations['low'].max_det}"
        ),
        (
            f"de xuat chay ngay luc nay: {mode_label(auto_runtime.mode)} ({auto_runtime.mode}) / "
            f"{auto_runtime.primary_model_name} / {auto_runtime.resolved_device} / "
            f"imgsz {auto_runtime.imgsz} / max_det {auto_runtime.max_det}"
        ),
    ]


def _print_lines(lines: list[str], *, print_fn=print) -> None:
    for item in lines:
        print_fn(item)


def _model_local_text(models: list[str]) -> str:
    return ", ".join(models) if models else "khong co model local nao"


def _recommended_models_for_mode(mode: str, recommendations: dict[str, object] | None) -> list[str]:
    if not recommendations:
        return []
    runtime = recommendations.get(mode)
    if runtime is None:
        return []
    primary = getattr(runtime, "primary_model_name", "")
    candidates = list(getattr(runtime, "candidate_models", []) or [])
    return [item for item in dict.fromkeys([primary, *candidates]) if item]


def _mode_color(mode: str) -> str:
    return {"high": GREEN, "medium": YELLOW, "low": MAGENTA}.get(mode, CYAN)


def _project_model_text(settings: dict, mode: str) -> str:
    profile = settings["models"][mode]
    device = "gpu" if mode in {"high", "medium"} else "auto"
    return f"{profile['model']} / {device} / imgsz {profile['imgsz']}"


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

    lines: list[str] = []
    lines.extend(header("BO TU VAN RUNTIME YOLO :: THAM DO MAY VA DE XUAT 3 MUC TOI UU"))
    lines.extend(
        [
            section("TONG QUAN MAY", GREEN),
            row("CPU", hardware.cpu_name, GREEN, bounded=False),
            row("RAM / OS", f"{hardware.ram_gb:.1f} GB / {hardware.os_name}", GREEN, bounded=False),
            row("GPU", hardware.gpu_name, GREEN if hardware.cuda_available else YELLOW, bounded=False),
            row("VRAM", f"{hardware.vram_gb:.1f} GB", GREEN if hardware.vram_gb else YELLOW, bounded=False),
            row("CUDA", "co" if hardware.cuda_available else "khong", GREEN if hardware.cuda_available else RED, bounded=False),
            row("PyTorch", hardware.torch_version, CYAN, bounded=False),
            row("CUDA build", hardware.torch_cuda_version, CYAN, bounded=False),
            row("Phan hang GPU", gpu_tier(hardware), YELLOW, bounded=False),
            row("Tai CPU", _usage_text(hardware.cpu_usage_percent), _usage_color(hardware.cpu_usage_percent), bounded=False),
            row("Tai GPU", _usage_text(hardware.gpu_usage_percent), _usage_color(hardware.gpu_usage_percent), bounded=False),
            row("Tai VRAM", _usage_text(hardware.vram_usage_percent), _usage_color(hardware.vram_usage_percent), bounded=False),
            row("Trang thai tai", load_level(hardware), MAGENTA, bounded=False),
            line(rule("-"), CYAN),
            section("YOLO11 VA MODEL LOCAL", MAGENTA),
            row("5 phien ban", ", ".join(YOLO11_VARIANTS), CYAN, bounded=False),
            row("Model san sang", _model_local_text(available_models), GREEN, bounded=False),
            row(
                "Model con thieu",
                _model_local_text(missing_models) if missing_models else "da co du cac model chinh",
                YELLOW,
                bounded=False,
            ),
            line(rule("-"), CYAN),
            section("DINH NGHIA 3 MUC", YELLOW),
            row("Manh nhat", "muc cao nhat may con ganh duoc", GREEN, bounded=False),
            row("Trung binh", "muc can bang dep nhat de chay thuong xuyen", YELLOW, bounded=False),
            row("Yeu nhat", "muc nhe nhat de uu tien do muot va an toan", MAGENTA, bounded=False),
            line(rule("-"), CYAN),
            section("3 MUC TOI UU TREN MAY NAY", CYAN),
        ]
    )

    for mode in MODE_ORDER:
        runtime = recommendations[mode]
        color = _mode_color(mode)
        lines.extend(
            [
                row(mode_title(mode), _summary_line(mode, runtime), color, bounded=False),
                row(
                    "  Danh gia",
                    f"chat luong {quality_score(runtime)}/100 | on dinh {stability_score(mode, hardware)}/100",
                    DIM,
                    bounded=False,
                ),
                row("  Giai thich", _mode_reason(mode, runtime, hardware), DIM, bounded=False),
                line(rule("."), DIM),
            ]
        )

    lines.extend(
        [
            line(rule("-"), CYAN),
            section("CAU HINH DU AN", GREEN),
            row("Thiet lap high", _project_model_text(settings, "high"), GREEN, bounded=False),
            row("Thiet lap medium", _project_model_text(settings, "medium"), YELLOW, bounded=False),
            row("Thiet lap low", _project_model_text(settings, "low"), MAGENTA, bounded=False),
            row(
                "Model uu tien",
                (
                    f"primary {preferred.get('primary_gpu', 'khong ro')} | "
                    f"backup GPU {preferred.get('stable_backup_gpu', 'khong ro')} | "
                    f"backup CPU {preferred.get('stable_backup_cpu', 'khong ro')}"
                ),
                CYAN,
                bounded=False,
            ),
        ]
    )
    if priority_order:
        lines.append(row("Thu tu load", ", ".join(priority_order), DIM, bounded=False))

    lines.extend(
        [
            line(rule("-"), CYAN),
            section("KET LUAN WOW", MAGENTA),
        ]
    )
    for item in _wow_conclusion(hardware, recommendations, auto_runtime):
        lines.append(row("Giai phap", item, GREEN, bounded=False))
    lines.append(row("De xuat tot nhat", _best_solution_text(auto_runtime, hardware), YELLOW, bounded=False))
    return lines


def _runtime_mode_choice_lines(hardware, recommendations: dict[str, object]) -> list[str]:
    return [
        line(rule("-"), CYAN),
        section("CHON CHE DO SE CHAY", CYAN),
        row("De xuat", _best_solution_text(recommendations["auto"], hardware), YELLOW, bounded=False),
        row("1 | MANH NHAT", _summary_line("high", recommendations["high"]), GREEN, bounded=False),
        row("2 | TRUNG BINH", _summary_line("medium", recommendations["medium"]), YELLOW, bounded=False),
        row("3 | YEU NHAT", _summary_line("low", recommendations["low"]), MAGENTA, bounded=False),
        line(rule("."), DIM),
        row("0 | THOAT", "Dong chuong trinh ngay tai day.", RED, bounded=False),
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

        raw_value = input_fn(line("Nhap lua chon cua ban (0/1/2/3): ", GREEN)).strip()
        selected_mode = MODE_PROMPT_CHOICES.get(raw_value)
        if selected_mode == "exit":
            raise SystemExit(0)
        if selected_mode:
            print_fn("")
            print_fn(line(f"Da chon che do: {mode_label(selected_mode)}", GREEN))
            return selected_mode
        print_fn(line("Lua chon khong hop le. Vui long nhap 0, 1, 2 hoac 3.", RED))


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
        raise RuntimeError("Khong co model local nao de chon. Hay chay training\\download_models.py truoc.")
    if len(options) == 1:
        chosen = options[0]
        print_fn("")
        print_fn(line(f"Tu dong chon model duy nhat: {chosen}", GREEN))
        return chosen

    while True:
        print_fn(line(rule("="), CYAN))
        print_fn(section("CHON MODEL SE CHAY", CYAN))
        print_fn(row("Che do da chon", mode_title(selected_mode), _mode_color(selected_mode), bounded=False))
        if recommended:
            print_fn(row("Model nen dung", ", ".join(recommended), GREEN, bounded=False))
        if missing_models:
            print_fn(row("Model con thieu", ", ".join(missing_models), YELLOW, bounded=False))
        print_fn(line(rule("-"), CYAN))
        for index, model_name in enumerate(options, start=1):
            hint = "khuyen nghi" if model_name in recommended else "co san"
            color = GREEN if model_name in recommended else CYAN
            print_fn(row(f"{index} | {model_name}", hint, color))
        print_fn(line(rule("."), DIM))
        print_fn(row("0 | THOAT", "Dong chuong trinh ngay tai day.", RED))
        print_fn(line(rule("-"), CYAN))

        raw_value = input_fn(line(f"Nhap lua chon cua ban (0-{len(options)}): ")).strip()
        if raw_value == "0":
            raise SystemExit(0)
        if raw_value.isdigit():
            selected_index = int(raw_value) - 1
            if 0 <= selected_index < len(options):
                selected_model = options[selected_index]
                print_fn("")
                print_fn(line(f"Da chon model: {selected_model}", GREEN))
                return selected_model
        print_fn(line("Lua chon khong hop le. Vui long nhap so trong danh sach.", RED))
        input_fn(line("Nhan Enter de chon lai...", DIM))


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
