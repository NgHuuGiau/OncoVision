from __future__ import annotations

import json
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from training.tcia_downloader import verify_downloads


def main() -> None:
    report = verify_downloads(Path("training/tcia_collections_5.json"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
