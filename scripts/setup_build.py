# -*- mode: python ; coding: utf-8 -*-
"""
Setup script for building OncoVision Desktop Chat EXE using cx_Freeze 8.x
Replaces the old OncoVision_chat.spec file.

Usage:
    python setup_build.py build_exe
"""

import sys
import os
from pathlib import Path
sys.setrecursionlimit(5000)  # Increase recursion limit for PyTorch

# Project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add project root to sys.path so cx_Freeze can find app/, medical/, core/, utils/
sys.path.insert(0, str(PROJECT_ROOT))

from cx_Freeze import setup, Executable

# Build options
build_exe_options = {
    "includes": [
        "entrypoints.run_chat",
        "app.chat_ui.window",
        "app.chat_ui.desktop_auth",
        "app.chat_ui.models",
        "app.chat_ui.paths",
        "app.chat_ui.storage",
        "app.chat_ui.medical_controller",
        "app.web_auth",
        "app.email_service",
        "medical.pipeline",
        "medical.cnn_classifier",
        "medical.cancer_catalog",
        "medical.cancer_model_registry",
        "medical.model_policy",
        "medical.system_status",
        "medical.validator",
        "medical.dataset",
        "medical.reporting",
        "medical.storage",
        "medical.case_payloads",
        "medical.chat_service",
        "medical.compliance",
        "medical.clinical_validation",
        "medical.volume_inference",
        "medical.status_helpers",
        "utils.logger",
        "utils.sqlite_utils",
        "utils.cleanup_utils",
        "utils.entrypoint_checks",
        "utils.entrypoint_common",
        "utils.icons",
        "timm",
        "torch",
        "torch.cuda",
        "torch.distributed",
        "torch.fx",
        "torch._dynamo.polyfills.fx",  # import động trong torch._dynamo.polyfills.loader
        "torch.multiprocessing",
        "torchvision",
        "cv2",
        "numpy",
        "ultralytics",
        "pydicom",
        "sqlite3",
        "json",
        "pathlib",
        "PIL",
        "PIL.Image",
        "PIL.ImageTk",
        "email",
        "email.mime",
        "email.mime.text",
        "email.mime.multipart",
        "email.mime.base",
        "email.encoders",
        "smtplib",
        "ssl",
        "multiprocessing",
        "multiprocessing.pool",
        "multiprocessing.context",
        "multiprocessing.sharedctypes",
        "multiprocessing.queues",
        "multiprocessing.synchronize",
        "unittest",
        "unittest.mock",
        "concurrent",
        "concurrent.futures",
    ],
    "excludes": [
        "matplotlib",
        "fastapi",
        "uvicorn",
        "starlette",
        "pydantic",
        "jinja2",
        "reportlab",
        "pytest",
        "aiofiles",
        "doctest",
        # Exclude heavy/unnecessary torch modules to avoid recursion
        # (khong exclude torch.onnx: torchvision import torch.onnx ngay khi nap)
        "torch.elastic",
        "torch.testing",
        "torch.utils.tensorboard",
    ],
    "packages": [],
    "include_files": [
        (str(PROJECT_ROOT / "app" / "chat_ui"), "app/chat_ui"),
        (str(PROJECT_ROOT / "entrypoints"), "entrypoints"),
        (str(PROJECT_ROOT / "medical"), "medical"),
        (str(PROJECT_ROOT / "core"), "core"),
        (str(PROJECT_ROOT / "utils"), "utils"),
        (str(PROJECT_ROOT / "config"), "config"),
        (str(PROJECT_ROOT / "assets"), "assets"),
        (str(PROJECT_ROOT / "models"), "models"),
        (str(PROJECT_ROOT / "templates"), "templates"),
    ],
    "optimize": 1,
    "zip_include_packages": ["PySide6", "shiboken6"],
}

# Executable
exe = Executable(
    script="entrypoints/run_chat.py",
    base="gui",  # No console window (cx_Freeze 8.x uses "gui" instead of "Win32GUI")
    target_name="OncoVision.exe",
    copyright="Copyright © 2025 OncoVision",
)

setup(
    name="OncoVision",
    version="1.0.0",
    description="OncoVision Desktop Chat - AI Y Khoa",
    options={"build_exe": build_exe_options},
    executables=[exe],
)
