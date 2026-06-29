from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "DEFAULT_CAMERA_MODE": ("app.camera_runtime.bootstrap", "DEFAULT_CAMERA_MODE"),
    "DEFAULT_UI_MODE": ("app.camera_runtime.bootstrap", "DEFAULT_UI_MODE"),
    "StartOptions": ("app.camera_runtime.bootstrap", "StartOptions"),
    "build_targeted_parser": ("app.camera_runtime.entrypoint", "build_targeted_parser"),
    "resolve_start_bundle": ("app.camera_runtime.bootstrap", "resolve_start_bundle"),
    "resolve_start_options": ("app.camera_runtime.bootstrap", "resolve_start_options"),
    "run_camera_entrypoint": ("app.camera_runtime.entrypoint", "run_camera_entrypoint"),
    "run_targeted_entrypoint": ("app.camera_runtime.entrypoint", "run_targeted_entrypoint"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
