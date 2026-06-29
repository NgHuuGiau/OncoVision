from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CHAT_CAPTURES_DIR": ("app.chat_ui.paths", "CHAT_CAPTURES_DIR"),
    "CHAT_HISTORY_DB_PATH": ("app.chat_ui.paths", "CHAT_HISTORY_DB_PATH"),
    "ChatDatabase": ("app.chat_ui.storage", "ChatDatabase"),
    "ChatMessage": ("app.chat_ui.models", "ChatMessage"),
    "Conversation": ("app.chat_ui.models", "Conversation"),
    "OUTPUT_DIR": ("app.chat_ui.paths", "OUTPUT_DIR"),
    "PROJECT_ROOT": ("app.chat_ui.paths", "PROJECT_ROOT"),
    "build_chat_arg_parser": ("app.chat_ui.cli", "build_chat_arg_parser"),
    "build_chat_capture_path": ("app.chat_ui.paths", "build_chat_capture_path"),
    "get_chat_capture_dir": ("app.chat_ui.paths", "get_chat_capture_dir"),
    "launch_chat_app": ("app.chat_ui.window", "launch_chat_app"),
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
