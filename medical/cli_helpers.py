from __future__ import annotations

from pathlib import Path


def print_dataset_counts(prefix: str, *, train: int, val: int, test: int, total: int) -> None:
    print(f"{prefix}: train={train}, val={val}, test={test}, total={total}")


def print_output_counts(*, case_count: int, report_files: int, normalized_files: int, overlay_files: int, export_files: int) -> None:
    print(
        f"Outputs: cases={case_count}, reports={report_files}, "
        f"normalized={normalized_files}, overlay={overlay_files}, exports={export_files}"
    )


def print_medical_readiness(status) -> None:
    print(
        "ready_for_train_medical: "
        f"{status.dataset_initialized and status.raw_dataset_ready and status.processed_dataset_ready and status.model_ready}"
    )


def print_medical_status_block(status, dataset_root: Path) -> None:
    print("Trang thai he thong medical")
    print(f"Model config: {status.configured_model_path}")
    if status.resolved_model_path is not None:
        print(f"Model runtime: {status.resolved_model_path}")
    print(f"Fallback allowed: {status.allow_fallback_model}")
    print(f"Model ready: {status.model_ready}")
    print(f"Model detail: {status.model_message}")
    print("He thong dang phan tich cac ung thu:")
    for name in status.analyzed_cancers:
        print(f"- {name}")
    print(f"Dataset root: {status.dataset_root}")
    print(f"Data yaml: {status.data_yaml_path} | exists={status.dataset_initialized}")
    print_dataset_counts(
        "Dataset counts",
        train=status.train_images,
        val=status.val_images,
        test=status.test_images,
        total=status.total_images,
    )
    print_output_counts(
        case_count=status.case_count,
        report_files=status.report_files,
        normalized_files=status.normalized_files,
        overlay_files=status.overlay_files,
        export_files=status.export_files,
    )
    print(f"Medical dataset root: {dataset_root}")
