from __future__ import annotations

from core.hardware_info import detect_hardware
from core.model_catalog import MODEL_QUALITY_SCORES, YOLO11_MODELS_DESC
from core.model_selector import RuntimeConfig, select_runtime_config

MODE_ORDER = ("high", "medium", "low")
YOLO11_VARIANTS = YOLO11_MODELS_DESC
MODE_META = {
    "high": {"label": "manh nhat", "title": "MANH NHAT", "meaning": "muc cao nhat co the uu tien"},
    "medium": {"label": "trung binh", "title": "TRUNG BINH", "meaning": "muc can bang de dung nhat"},
    "low": {"label": "yếu nhất", "title": "YẾU NHẤT", "meaning": "mức nhẹ nhất / dễ chạy nhất"},
}


def mode_label(mode: str) -> str:
    return MODE_META.get(mode, MODE_META["low"])["label"]


def mode_title(mode: str) -> str:
    return MODE_META.get(mode, MODE_META["low"])["title"]


def load_level(hardware) -> str:
    cpu = float(getattr(hardware, "cpu_usage_percent", 0.0) or 0.0)
    ram = float(getattr(hardware, "ram_usage_percent", 0.0) or 0.0)
    gpu = float(getattr(hardware, "gpu_usage_percent", 0.0) or 0.0)
    vram = float(getattr(hardware, "vram_usage_percent", 0.0) or 0.0)
    peak = max(cpu, ram, gpu, vram)
    if peak >= 85:
        return "very_busy"
    if peak >= 70:
        return "busy"
    if peak >= 50:
        return "warm"
    return "cool"


def gpu_tier(hardware) -> str:
    if not getattr(hardware, "cuda_available", False):
        return "cpu_only"

    name = str(getattr(hardware, "gpu_name", "")).lower()
    vram = float(getattr(hardware, "vram_gb", 0.0) or 0.0)
    if "4090" in name or "4080" in name or vram >= 16:
        return "enthusiast"
    if "3090" in name or "3080" in name or "3070" in name or vram >= 8:
        return "strong"
    if "3060" in name or "3050" in name or "1660" in name or vram >= 4:
        return "entry"
    if vram >= 2:
        return "weak"
    return "cpu_only"


def profile_specs_for_hardware(hardware) -> dict[str, dict]:
    runtimes = {mode: select_runtime_config(mode, hardware) for mode in MODE_ORDER}
    return {
        mode: {
            "model": runtime.primary_model_name,
            "device": runtime.resolved_device,
            "imgsz": runtime.imgsz,
            "max_det": runtime.max_det,
        }
        for mode, runtime in runtimes.items()
    }


def default_mode_for_hardware(hardware) -> str:
    tier = gpu_tier(hardware)
    load = load_level(hardware)
    if load == "very_busy":
        return "low"
    if load in {"busy", "warm"}:
        return "medium" if tier in {"enthusiast", "strong", "entry"} else "low"
    if tier in {"enthusiast", "strong"}:
        return "high"
    if tier == "entry":
        return "medium"
    return "low"


def ceiling_mode_for_hardware(hardware) -> str:
    tier = gpu_tier(hardware)
    if tier in {"enthusiast", "strong", "entry"}:
        return "high"
    if tier == "weak":
        return "medium"
    return "low"


def quality_score(runtime: RuntimeConfig) -> int:
    model_score = MODEL_QUALITY_SCORES.get(runtime.primary_model_name, 50)
    imgsz_bonus = min(18, max(0, (int(runtime.imgsz) - 320) // 32))
    det_bonus = min(10, max(0, (int(runtime.max_det) - 60) // 15))
    return min(100, model_score + imgsz_bonus + det_bonus)


def stability_score(mode: str, hardware) -> int:
    base = {"high": 68, "medium": 88, "low": 96}[mode]
    penalty = {"cool": 0, "warm": 6, "busy": 14, "very_busy": 24}[load_level(hardware)]
    return max(40, min(99, base - penalty))


def optimized_runtime(mode: str, hardware) -> RuntimeConfig:
    resolved_mode = default_mode_for_hardware(hardware) if mode == "auto" else mode
    return select_runtime_config(resolved_mode, hardware)


def select_runtime_config_optimized(mode: str, hardware):
    return optimized_runtime(mode, hardware)


def build_recommendations(hardware=None) -> dict[str, RuntimeConfig]:
    hardware = hardware or detect_hardware()
    recommendations = {mode: select_runtime_config_optimized(mode, hardware) for mode in MODE_ORDER}
    recommendations["auto"] = select_runtime_config_optimized("auto", hardware)
    return recommendations
