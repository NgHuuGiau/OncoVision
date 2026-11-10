import sys
import os
from cx_Freeze import setup, Executable

# Very minimal - exclude torch/cv2 from packages, let them be auto-detected if needed
build_exe_options = {
    "packages": [
        "app", "medical", "core", "utils", "config", "assets",
        "numpy", "ultralytics", "PIL", "pydicom", "pydantic", "yaml",
        "PySide6", "fastapi", "uvicorn", "starlette", "jinja2",
    ],
    "excludes": [
        "matplotlib", "tkinter", "test", "unittest", "doctest",
        "pytest", "sphinx", "notebook", "jupyter", "ipython",
        "sqlalchemy", "setuptools", "pip", "wheel",
        "torch", "torchvision", "cv2",
    ],
    "include_files": [
        ("config", "config"),
        ("static", "static"),
        ("templates", "templates"),
        ("assets", "assets"),
        ("models", "models"),
    ],
    "include_msvcr": True,
    "optimize": 1,
    "silent": False,
    "zip_include_packages": ["*"],
    "zip_exclude_packages": [],
}

base = "gui" if sys.platform == "win32" else None

executables = [
    Executable(
        "run_chat.py",
        base=base,
        target_name="OncoVisionChat.exe",
    ),
]

if __name__ == "__main__":
    setup(
        name="OncoVision",
        version="1.0.0",
        description="OncoVision Medical AI System",
        options={"build_exe": build_exe_options},
        executables=executables,
    )