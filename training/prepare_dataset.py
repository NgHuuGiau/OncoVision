from __future__ import annotations

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.training import prepare_medical_training_dataset
from utils.terminal_encoding import ensure_utf8_console


def main():
    ensure_utf8_console()
    summary = prepare_medical_training_dataset()
    print("Medical dataset ready")
    print(f"- Classes: {summary.class_count}")
    print(f"- Train/val/test: {summary.train_count}/{summary.val_count}/{summary.test_count}")
    print(f"- Total: {summary.total_count}")
    return summary


if __name__ == "__main__":
    main()
