from __future__ import annotations

import platform
from dataclasses import dataclass

import psutil

try:
    import GPUtil
except ImportError:  # pragma: no cover
    GPUtil = None

torch = None
TORCH_IMPORT_ERROR = None


@dataclass
class HardwareInfo:
    cpu_name: str
    ram_gb: float
    gpu_name: str
    vram_gb: float
    cuda_available: bool
    os_name: str
    gpu_count: int
    mps_available: bool = False
    xpu_available: bool = False
    rocm_available: bool = False
    torch_version: str = "Không có PyTorch"
    torch_cuda_version: str = "CPU-only"
    cuda_runtime_status: str = "Không"
    cuda_runtime_reason: str = "Chưa kiểm tra"
    gpu_hardware_available: bool = False
    cpu_usage_percent: float | None = None
    ram_usage_percent: float | None = None
    gpu_usage_percent: float | None = None
    vram_usage_percent: float | None = None

    def pretty_report(self) -> str:
        lines = [
            "===== KIỂM TRA CẤU HÌNH =====",
            f"CPU: {self.cpu_name}",
            f"RAM: {self.ram_gb:.1f} GB",
            f"GPU: {self.gpu_name}",
            f"VRAM: {self.vram_gb:.1f} GB",
            f"CPU dùng: {self.cpu_usage_percent:.1f}%" if self.cpu_usage_percent is not None else "CPU dùng: Không rõ",
            f"RAM dùng: {self.ram_usage_percent:.1f}%" if self.ram_usage_percent is not None else "RAM dùng: Không rõ",
            f"GPU dùng: {self.gpu_usage_percent:.1f}%" if self.gpu_usage_percent is not None else "GPU dùng: Không rõ",
            f"VRAM dùng: {self.vram_usage_percent:.1f}%" if self.vram_usage_percent is not None else "VRAM dùng: Không rõ",
            f"CUDA: {'Có' if self.cuda_available else 'Không'}",
            f"MPS (Apple Silicon): {'Có' if self.mps_available else 'Không'}",
            f"XPU (Intel): {'Có' if self.xpu_available else 'Không'}",
            f"ROCM (AMD): {'Có' if self.rocm_available else 'Không'}",
            f"PyTorch: {self.torch_version}",
            f"CUDA build: {self.torch_cuda_version}",
            f"OS: {self.os_name}",
        ]
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "cpu_name": self.cpu_name,
            "ram_gb": round(self.ram_gb, 2),
            "gpu_name": self.gpu_name,
            "vram_gb": round(self.vram_gb, 2),
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
            "xpu_available": self.xpu_available,
            "rocm_available": self.rocm_available,
            "os_name": self.os_name,
            "gpu_count": self.gpu_count,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_runtime_status": self.cuda_runtime_status,
            "cuda_runtime_reason": self.cuda_runtime_reason,
            "gpu_hardware_available": self.gpu_hardware_available,
            "cpu_usage_percent": round(self.cpu_usage_percent, 2) if self.cpu_usage_percent is not None else None,
            "ram_usage_percent": round(self.ram_usage_percent, 2) if self.ram_usage_percent is not None else None,
            "gpu_usage_percent": round(self.gpu_usage_percent, 2) if self.gpu_usage_percent is not None else None,
            "vram_usage_percent": round(self.vram_usage_percent, 2) if self.vram_usage_percent is not None else None,
        }


def _detect_cpu_name() -> str:
    cpu_name = platform.processor().strip()
    if cpu_name:
        return cpu_name
    uname = platform.uname()
    if uname.processor:
        return uname.processor
    if platform.system() == "Darwin":
        # On macOS, try to get more detailed CPU info
        try:
            import subprocess
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2, check=False)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    return platform.uname().processor or "Không rõ CPU"


def _get_gpu_info_linux() -> tuple[str, float, int]:
    """Get GPU info on Linux using nvidia-smi, rocm-smi, or intel_gpu_top."""
    gpu_name = "Không phát hiện GPU"
    gpu_count = 0

    # Try NVIDIA (nvidia-smi)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            gpu_count = len(lines)
            first = lines[0].split(", ")
            gpu_name = first[0].strip()
            vram_mb = int(first[1].strip()) if len(first) > 1 else 0
            return gpu_name, round(vram_mb / 1024, 2), gpu_count
    except Exception:
        pass

    # Try AMD ROCm (rocm-smi)
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse rocm-smi output
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "GPU" in line and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        gpu_name = parts[1].strip()
                        break
            return gpu_name, 0.0, 1  # VRAM detection more complex for ROCm
    except Exception:
        pass

    # Try Intel GPU (intel_gpu_top)
    try:
        import subprocess
        result = subprocess.run(
            ["intel_gpu_top", "-J"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            return "Intel GPU (XPU)", 0.0, 1
    except Exception:
        pass

    return "Không phát hiện GPU", 0.0, 0


def _get_gpu_info_macos() -> tuple[str, float, int]:
    """Get GPU info on macOS (Apple Silicon MPS or Intel)."""
    import subprocess
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            for display in displays:
                gpu_name = display.get("sppci_model", "Apple GPU")
                vram_mb = display.get("sppci_vram", 0)
                if isinstance(vram_mb, str):
                    vram_mb = int(vram_mb.replace(" MB", "").replace(",", ""))
                vram_gb = round(vram_mb / 1024, 2)
                return gpu_name, vram_gb, 1
    except Exception:
        pass

    # Fallback: check chip type
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2, check=False)
        if result.returncode == 0:
            cpu_brand = result.stdout.strip()
            if "Apple" in cpu_brand:
                return "Apple Silicon (MPS)", 0.0, 1
    except Exception:
        pass

    return "Apple GPU (MPS)", 0.0, 1


def _get_gpu_info_windows() -> tuple[str, float, int, float | None, float | None]:
    """Get GPU info on Windows using GPUtil or wmic."""
    if GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                primary = gpus[0]
                return primary.name, round(primary.memoryTotal / 1024, 2), len(gpus), primary.load * 100 if primary.load is not None else None, primary.memoryUtil * 100 if primary.memoryUtil is not None else None
        except Exception:
            pass

    # Fallback: wmic
    try:
        import subprocess
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name,adapterram", "/format:csv"],
            capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split(",")
                    if len(parts) >= 3 and parts[1].strip():
                        name = parts[1].strip()
                        vram = int(parts[2].strip()) if parts[2].strip().isdigit() else 0
                        return name, round(vram / (1024**3), 2), 1, None, None
    except Exception:
        pass
    return "Không phát hiện GPU", 0.0, 0, None, None


def _detect_gpu() -> tuple[str, float, int, bool, bool, bool, bool, float | None, float | None]:
    """
    Returns: (gpu_name, vram_gb, gpu_count, cuda_available, mps_available, xpu_available,
    rocm_available, gpu_usage_percent, vram_usage_percent).

    shortcut: Linux/macOS không trả load/VRAM usage (luôn None); thêm query
    nvidia-smi utilization.gpu/memory.used nếu cần hiển thị live usage trên Linux.
    """
    system = platform.system()

    if system == "Linux":
        gpu_name, vram_gb, gpu_count = _get_gpu_info_linux()
        # Check CUDA
        cuda_available = False
        rocm_available = False
        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
            # Check ROCm
            if hasattr(torch, 'hip') and hasattr(torch.hip, 'is_available'):
                if torch.hip.is_available():
                    rocm_available = True
        except Exception:
            pass
        return gpu_name, vram_gb, gpu_count, cuda_available, False, False, rocm_available, None, None

    elif platform.system() == "Darwin":  # macOS
        gpu_name, vram_gb, gpu_count = _get_gpu_info_macos()
        mps_available = False
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                mps_available = True
        except Exception:
            pass
        return gpu_name, vram_gb, gpu_count, False, mps_available, False, False, None, None

    elif system == "Windows":
        gpu_name, vram_gb, gpu_count, gpu_usage_percent, vram_usage_percent = _get_gpu_info_windows()
        cuda_available = False
        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
        except Exception:
            pass
        return gpu_name, vram_gb, gpu_count, cuda_available, False, False, False, gpu_usage_percent, vram_usage_percent

    return "Không phát hiện GPU", 0.0, 0, False, False, False, False, None, None


def _load_torch():
    global torch, TORCH_IMPORT_ERROR
    if torch is not None or TORCH_IMPORT_ERROR is not None:
        return torch
    try:
        import torch as imported_torch
        torch = imported_torch
    except Exception as exc:  # pragma: no cover
        TORCH_IMPORT_ERROR = exc
        torch = None
    return torch


def detect_hardware() -> HardwareInfo:
    torch_module = torch if torch is not None else _load_torch()
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    ram_usage_percent = float(getattr(memory, "percent", 0.0))
    cpu_usage_percent = float(psutil.cpu_percent(interval=0.15))

    # Detect GPU and acceleration backends
    gpu_name, vram_gb, gpu_count, cuda_available, mps_available, xpu_available, rocm_available, gpu_usage_percent, vram_usage_percent = _detect_gpu()

    gpu_hardware_available = gpu_name != "Không phát hiện GPU"

    torch_module = _load_torch()
    torch_version = "Không có PyTorch"
    torch_cuda_version = "CPU-only"
    cuda_runtime_status = "Không"
    cuda_runtime_reason = "Chưa kiểm tra"

    if torch_module is not None:
        torch_version = getattr(torch_module, "__version__", "Không có PyTorch")
        torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        torch_cuda_label = torch_cuda_version or "CPU-only"

        # CUDA
        cuda_available = bool(torch_module.cuda.is_available()) if hasattr(torch_module, "cuda") else False
        if cuda_available:
            torch_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
            torch_cuda_label = torch_cuda_version or "CPU-only"

        # MPS (Apple Silicon)
        mps_available = False
        if hasattr(torch_module, "backends") and hasattr(torch_module.backends, "mps"):
            mps_available = torch_module.backends.mps.is_available()

        # XPU (Intel)
        xpu_available = False
        if hasattr(torch_module, "xpu") and hasattr(torch_module.xpu, "is_available"):
            xpu_available = torch_module.xpu.is_available()

        # ROCM (AMD)
        rocm_available = False
        if hasattr(torch_module, "hip") and hasattr(torch_module.hip, "is_available"):
            rocm_available = torch_module.hip.is_available()

        # Determine overall acceleration availability
        any_accel = cuda_available or mps_available or xpu_available

        if torch_module is None:
            cuda_runtime_reason = (
                f"Không khởi tạo được PyTorch: {TORCH_IMPORT_ERROR}"
                if TORCH_IMPORT_ERROR is not None
                else "Chưa cài PyTorch trong môi trường hiện tại"
            )
        elif not gpu_hardware_available:
            cuda_runtime_reason = "Không phát hiện GPU tương thích để chạy CUDA/MPS/XPU"
        elif not any_accel:
            cuda_runtime_reason = "PyTorch không hỗ trợ tăng tốc phần cứng trên thiết bị này"
        else:
            cuda_runtime_reason = "PyTorch runtime sẵn sàng với tăng tốc phần cứng"

        # Get detailed GPU info from PyTorch if CUDA available
        gpu_name_from_torch = "Không phát hiện GPU"
        vram_gb_from_torch = 0.0
        gpu_count_from_torch = 0

        if cuda_available and torch_module is not None:
            try:
                index = torch_module.cuda.current_device()
                gpu_name_from_torch = torch_module.cuda.get_device_name(index)
                properties = torch_module.cuda.get_device_properties(index)
                vram_gb_from_torch = properties.total_memory / (1024**3)
                gpu_count_from_torch = torch_module.cuda.device_count()
            except Exception:
                pass

        # Prefer PyTorch info over GPUtil for NVIDIA GPUs
        if cuda_available and gpu_name_from_torch != "Không phát hiện GPU":
            gpu_name = gpu_name_from_torch
            vram_gb = vram_gb_from_torch
            gpu_count = gpu_count_from_torch
            gpu_hardware_available = True
        else:
            gpu_hardware_available = gpu_name != "Không phát hiện GPU"

    else:
        torch_version = "Không có PyTorch"
        torch_cuda_label = "CPU-only"
        cuda_runtime_status = "Không"
        cuda_runtime_reason = "Chưa cài PyTorch"
        gpu_hardware_available = False

    # Determine overall acceleration
    any_accel = cuda_available or mps_available or xpu_available or rocm_available
    cuda_runtime_status = "Có" if any_accel else "Không"

    # Set runtime reason based on what's available
    if not gpu_hardware_available:
        cuda_runtime_reason = "Không phát hiện GPU tương thích"
    elif not any_accel:
        cuda_runtime_reason = "PyTorch không hỗ trợ tăng tốc phần cứng trên thiết bị này"
    else:
        reasons = []
        if cuda_available:
            reasons.append("CUDA")
        if mps_available:
            reasons.append("MPS (Apple Silicon)")
        if xpu_available:
            reasons.append("XPU (Intel)")
        if rocm_available:
            reasons.append("ROCM (AMD)")
        cuda_runtime_reason = f"PyTorch runtime sẵn sàng với tăng tốc: {', '.join(reasons)}"

    # Get live usage
    memory = psutil.virtual_memory()
    ram_usage_percent = float(getattr(memory, "percent", 0.0))
    cpu_usage_percent = float(psutil.cpu_percent(interval=0.15))

    return HardwareInfo(
        cpu_name=_detect_cpu_name(),
        ram_gb=round(memory.total / (1024**3), 2),
        gpu_name=gpu_name if 'gpu_name' in locals() else "Không phát hiện GPU",
        vram_gb=vram_gb if 'vram_gb' in locals() else 0.0,
        cuda_available=cuda_available if 'cuda_available' in locals() else False,
        mps_available=mps_available if 'mps_available' in locals() else False,
        xpu_available=xpu_available if 'xpu_available' in locals() else False,
        rocm_available=rocm_available if 'rocm_available' in locals() else False,
        os_name=f"{platform.system()} {platform.release()}",
        gpu_count=gpu_count if gpu_name != "Không phát hiện GPU" else 0,
        torch_version=torch_version if 'torch_version' in locals() else "Không có PyTorch",
        torch_cuda_version=torch_cuda_label if 'torch_cuda_label' in locals() else "CPU-only",
        cuda_runtime_status=cuda_runtime_status if 'cuda_runtime_status' in locals() else "Không",
        cuda_runtime_reason=cuda_runtime_reason if 'cuda_runtime_reason' in locals() else "Chưa kiểm tra",
        gpu_hardware_available=gpu_hardware_available,
        cpu_usage_percent=cpu_usage_percent,
        ram_usage_percent=ram_usage_percent,
        gpu_usage_percent=gpu_usage_percent,
        vram_usage_percent=vram_usage_percent,
    )


def get_live_usage_snapshot() -> dict[str, float | None]:
    try:
        memory = psutil.virtual_memory()
        ram_usage_percent = float(getattr(memory, "percent", 0.0))
    except Exception:
        ram_usage_percent = None

    try:
        cpu_usage_percent = float(psutil.cpu_percent(interval=None))
    except Exception:
        cpu_usage_percent = None

    # If basic system metrics fail, don't trust GPU metrics either
    if cpu_usage_percent is None or ram_usage_percent is None:
        return {
            "cpu_usage_percent": cpu_usage_percent,
            "ram_usage_percent": ram_usage_percent,
            "gpu_usage_percent": None,
            "vram_usage_percent": None,
        }

    _gpu_name, _vram_gb, _gpu_count, _cuda_available, _mps_available, _xpu_available, _rocm_available, gpu_usage_percent, vram_usage_percent = _detect_gpu()
    return {
        "cpu_usage_percent": cpu_usage_percent,
        "ram_usage_percent": ram_usage_percent,
        "gpu_usage_percent": gpu_usage_percent,
        "vram_usage_percent": vram_usage_percent,
    }
