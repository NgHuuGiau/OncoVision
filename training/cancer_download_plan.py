from __future__ import annotations

import json
from pathlib import Path

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.cancer_dataset_registry import common_cancer_dataset_source_dicts
from training.terminal_ui import CYAN, GREEN, command_row, header, line, row, rule, section
from utils.file_utils import ensure_project_directories
from utils.terminal_encoding import ensure_utf8_console


OUTPUT_PATH = Path("dataset/cancer_download_plan.json")


def build_download_plan() -> dict[str, object]:
    sources = common_cancer_dataset_source_dicts()
    ready_sources = [item for item in sources if item["status"] == "ready"]
    planned_sources = [item for item in sources if item["source_name"].startswith("TCIA")]
    return {
        "priority_downloads": [
            {
                "cancer_name": item["cancer_type"],
                "primary_source": item["source_name"],
                "status": item["status"],
                "note": item["notes"],
            }
            for item in ready_sources
        ],
        "planned_downloads": [
            {
                "cancer_name": item["cancer_type"],
                "primary_source": item["source_name"],
                "status": item["status"],
                "note": item["notes"],
            }
            for item in planned_sources
        ],
        "ready_for_train": [],
    }


def main() -> None:
    ensure_utf8_console()
    ensure_project_directories()
    plan = build_download_plan()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in header("YOLO DATASET :: CANCER DOWNLOAD PLAN"):
        print(item)
    print(section("Uu tien tai ngay", GREEN))
    for item in plan["priority_downloads"]:
        print(row(item["cancer_name"], item["primary_source"], GREEN, bounded=False))
        print(row("Trạng thái", item["status"], CYAN, bounded=False))
        print(row("Ghi chu", item["note"], CYAN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("CO THE TAI TIEP", GREEN))
    for item in plan["planned_downloads"]:
        print(row(item["cancer_name"], item["primary_source"], GREEN if item["status"] == "download_planned" else CYAN, bounded=False))
        print(row("Trạng thái", item["status"], CYAN, bounded=False))
        print(row("Ghi chu", item["note"], CYAN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("KET QUA", GREEN))
    print(row("Plan", str(OUTPUT_PATH), GREEN, bounded=False))
    print(line(rule("-"), CYAN))
    print(section("BUOC TIEP", CYAN))
    print(command_row(1, r".\.venv\Scripts\python run_medical.py tcia-download"))
    print(command_row(2, r".\.venv\Scripts\python run_medical.py verify-tcia"))
    print(line(rule("="), CYAN))


if __name__ == "__main__":
    main()
