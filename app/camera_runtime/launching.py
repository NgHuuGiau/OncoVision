from __future__ import annotations

from utils.console_ui import BootProgress


CAMERA_BOOT_STEPS = (
    (16, "Đang nhận cấu hình khởi động"),
    (42, "Đang kiểm tra CPU / GPU / CUDA"),
    (68, "Đang chọn model và runtime phù hợp"),
    (88, "Đang chuẩn bị mở camera"),
)
CAMERA_BOOT_FINISH_MESSAGE = "Sẵn sàng mở camera"


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
