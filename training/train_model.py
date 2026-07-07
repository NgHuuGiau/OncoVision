from __future__ import annotations

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.training import train_medical_model


def main():
    return train_medical_model()


if __name__ == "__main__":
    main()
