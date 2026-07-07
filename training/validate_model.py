from __future__ import annotations

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.training import validate_medical_model
from utils.terminal_encoding import ensure_utf8_console


def main():
    ensure_utf8_console()
    try:
        metrics = validate_medical_model()
    except FileNotFoundError as exc:
        print(f"Medical model not ready: {exc}")
        print("Chạy `python run_train.py` trước để tạo medical_7_cancers.pt.")
        return None
    print(metrics)
    return metrics


if __name__ == "__main__":
    raise SystemExit(0 if main() is not None else 1)
