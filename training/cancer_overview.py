from __future__ import annotations

import json
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.cancer_dataset_registry import common_cancer_dataset_source_dicts
from training.cancer_download_plan import build_download_plan
from training.terminal_ui import CYAN, GREEN, command_row, header, line, row, rule, section
from utils.file_utils import ensure_project_directories
from utils.terminal_encoding import ensure_utf8_console


OUTPUT_PATH = Path("dataset/cancer_overview.json")


def main() -> None:
    ensure_utf8_console()
    ensure_project_directories()
    overview = {
        "sources": common_cancer_dataset_source_dicts(),
        "download_plan": build_download_plan(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in header("YOLO DATASET :: CANCER OVERVIEW"):
        print(item)
    print(section("NGUON", GREEN))
    for item in overview["sources"]:
        print(row(item["source_name"], item["cancer_type"], GREEN, bounded=False))
        print(row("Trang thai", item["status"], CYAN, bounded=False))
        print(row("Ghi chu", item["notes"], CYAN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("KE HOACH TAI", GREEN))
    for item in overview["download_plan"]["planned_downloads"]:
        print(row(item["cancer_name"], item["primary_source"], GREEN if item["status"] == "download_planned" else CYAN, bounded=False))
        print(row("Trang thai", item["status"], CYAN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("KET QUA", GREEN))
    print(row("Overview", str(OUTPUT_PATH), GREEN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("BUOC TIEP", CYAN))
    print(command_row(1, r".\.venv\Scripts\python run_medical.py cancer"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
