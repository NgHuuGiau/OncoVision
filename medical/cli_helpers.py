from __future__ import annotations

from pathlib import Path


def print_dataset_counts(prefix: str, *, raw_images: int, raw_labels: int, train: int, val: int, test: int) -> None:
    print(f"{prefix}: raw_images={raw_images}, raw_labels={raw_labels}, train={train}, val={val}, test={test}")


def print_output_counts(*, case_count: int, report_files: int, normalized_files: int, overlay_files: int, export_files: int) -> None:
    print(
        f"Outputs: cases={case_count}, reports={report_files}, "
        f"normalized={normalized_files}, overlay={overlay_files}, exports={export_files}"
    )


def print_skin_readiness(status) -> None:
    print(
        "ready_for_train_skin: "
        f"{status.dataset_initialized and status.raw_dataset_ready and status.processed_dataset_ready and status.model_ready}"
    )


def print_skin_status_block(status, skin_counts, skin_root: Path) -> None:
    print("Trạng thái hệ thống y")
    print(f"Model config: {status.configured_model_path}")
    if status.resolved_model_path is not None:
        print(f"Model runtime: {status.resolved_model_path}")
    print(f"Fallback allowed: {status.allow_fallback_model}")
    print(f"Model ready: {status.model_ready}")
    print(f"Model detail: {status.model_message}")
    print("Hệ thống đang phân tích các ung thư:")
    for name in status.analyzed_cancers:
        print(f"- {name}")
    print(f"Dataset root: {status.dataset_root}")
    print(f"Data yaml: {status.data_yaml_path} | exists={status.dataset_initialized}")
    print_dataset_counts(
        "Dataset counts",
        raw_images=status.raw_images,
        raw_labels=status.raw_labels,
        train=status.train_images,
        val=status.val_images,
        test=status.test_images,
    )
    print_output_counts(
        case_count=status.case_count,
        report_files=status.report_files,
        normalized_files=status.normalized_files,
        overlay_files=status.overlay_files,
        export_files=status.export_files,
    )
    print(
        "Medical skin lesion counts: "
        f"raw={skin_counts.raw_images}/{skin_counts.raw_labels}, train={skin_counts.train_images}, val={skin_counts.val_images}"
    )
    print(f"Skin lesion images live at: {skin_root / 'raw' / 'images'}")
