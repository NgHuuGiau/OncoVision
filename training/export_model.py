from __future__ import annotations

import shutil

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.training import medical_training_paths
from utils.file_utils import ensure_project_directories
from utils.terminal_encoding import ensure_utf8_console


def main() -> None:
    ensure_utf8_console()
    ensure_project_directories()
    paths = medical_training_paths()
    source_path = paths.trained_model_path
    export_path = source_path.with_name(source_path.stem + "_exported.pt")
    if not source_path.exists():
        print(f"Medical model not ready: {source_path}")
        print("Chạy `python run_train.py` trước rồi hãy export.")
        raise SystemExit(1)
    shutil.copy2(source_path, export_path)
    print(f"Exported medical model: {export_path}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
