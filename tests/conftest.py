import subprocess
import sys
from pathlib import Path

import pytest

collect_ignore = ["tests/manual"]

# Check PyTorch availability in a subprocess to avoid crashing the main process
_torch_available = False
try:
    result = subprocess.run(
        [sys.executable, "-c", "from torchvision.models import resnet18; resnet18(weights=None)"],
        check=False, capture_output=True, timeout=30
    )
    _torch_available = (result.returncode == 0)
except Exception:
    _torch_available = False

# Skip collection of torch-dependent test files when PyTorch is broken
if not _torch_available:
    _torch_test_files = [
        "tests/integration/test_web_auth.py",
        "tests/integration/test_web_full_flow.py",
        "tests/integration/test_web_cases.py",
        "tests/integration/test_medical_chat_service.py",
        "tests/integration/test_medical_advanced_integration.py",
        "tests/integration/test_medical_controller.py",
        "tests/integration/test_medical_worker.py",
        "tests/integration/test_camera_detector.py",
        "tests/integration/test_camera_outputs.py",
        "tests/integration/test_run_medical.py",
        "tests/integration/test_model_evaluation.py",
        "tests/integration/test_detector_evaluation.py",
        "tests/integration/test_yolo_loader.py",
        "tests/integration/test_clinical_validation.py",
        "tests/unit/test_cnn_classifier.py",
        "tests/unit/test_volume_inference.py",
        "tests/unit/test_medical_pipeline.py",
        "tests/unit/test_medical_modules.py",
        "tests/unit/test_new_modules.py",
    ]
    # Resolve relative paths to absolute
    _project_root = Path(__file__).resolve().parent.parent
    collect_ignore.extend([str(_project_root / f) for f in _torch_test_files])


def pytest_configure(config):
    """Add torch_required marker."""
    config.addinivalue_line(
        "markers", "torch_required: tests that require a working PyTorch installation"
    )
    if not _torch_available:
        config.option.markexpr = "not torch_required"


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that import torchvision modules when PyTorch is broken."""
    if _torch_available:
        return  # PyTorch works fine, no skipping needed
    skip_torch = pytest.mark.skip(reason="PyTorch/torchvision native code crash detected")
    for item in items:
        fspath = str(getattr(item, "fspath", ""))
        skip_names = {
            "test_web_auth.py", "test_web_full_flow.py", "test_web_cases.py",
            "test_medical_chat_service.py", "test_medical_advanced_integration.py",
            "test_medical_controller.py", "test_medical_worker.py",
            "test_cnn_classifier.py", "test_volume_inference.py",
            "test_medical_pipeline.py", "test_medical_modules.py",
            "test_camera_detector.py", "test_camera_outputs.py",
            "test_run_medical.py", "test_model_evaluation.py",
            "test_detector_evaluation.py", "test_yolo_loader.py",
            "test_clinical_validation.py", "test_new_modules.py",
        }
        if any(name in fspath for name in skip_names):
            item.add_marker(skip_torch)
