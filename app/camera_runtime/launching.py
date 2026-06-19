from __future__ import annotations

from utils.console_ui import BootProgress


CAMERA_BOOT_STEPS = (
    (16, "Dang nhan cau hinh khoi dong"),
    (42, "Dang kiem tra CPU / GPU / CUDA"),
    (68, "Dang chon model va runtime phu hop"),
    (88, "Dang chuan bi mo camera"),
)
CAMERA_BOOT_FINISH_MESSAGE = "San sang mo camera"


def run_camera_launch_flow(
    *,
    dashboard_title: str,
    runtime,
    hardware,
    camera_index: int,
    launch_target: str,
    print_runtime_dashboard_fn,
    run_camera_session_fn,
    boot_progress_cls=BootProgress,
    boot_steps=CAMERA_BOOT_STEPS,
    finish_message: str = CAMERA_BOOT_FINISH_MESSAGE,
) -> int:
    progress = boot_progress_cls(dashboard_title)
    for percent, message in boot_steps:
        progress.advance_to(percent, message)
    progress.finish(finish_message)

    print_runtime_dashboard_fn(
        title=dashboard_title,
        runtime=runtime,
        hardware=hardware,
        camera_index=camera_index,
        launch_target=launch_target,
    )
    run_camera_session_fn(runtime=runtime, camera_index=camera_index)
    return 0
