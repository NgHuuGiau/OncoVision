from __future__ import annotations

import argparse
from functools import lru_cache
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from utils.entrypoint_common import run_entrypoint


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SmokeCheck:
    key: str
    title: str
    description: str
    command: tuple[str, ...]
    warning_exit_codes: tuple[int, ...] = ()
    ci_safe: bool = True


BASE_SMOKE_CHECKS: tuple[SmokeCheck, ...] = (
    SmokeCheck(
        key="runtime-advisor",
        title="Runtime advisor",
        description="Kiểm tra entrypoint tư vấn runtime mà không mở camera.",
        command=("run_app.py", "--advisor-only"),
    ),
    SmokeCheck(
        key="doctor",
        title="Doctor scan",
        description="Rà soát tổng thể phần cứng, model và dữ liệu mà không cần webcam thật.",
        command=("run_doctor.py", "--skip-camera-check"),
        ci_safe=False,
    ),
    SmokeCheck(
        key="chat-preflight",
        title="Chat preflight",
        description="Kiểm tra dependency bắt buộc, icon và độ sẵn sàng của luồng chat/medical.",
        command=("run_chat.py", "--check-only", "--auto-fix-icons"),
        warning_exit_codes=(2,),
        ci_safe=False,
    ),
    SmokeCheck(
        key="training-preflight",
        title="Training preflight",
        description="Kiểm tra config, model và dataset cho entrypoint train tổng quát.",
        command=("run_train.py", "--check-only"),
    ),
    SmokeCheck(
        key="medical-status",
        title="Medical status",
        description="Kiểm tra nhanh model, dataset và output của nhánh medical.",
        command=("run_medical.py", "status"),
        ci_safe=False,
    ),
)

TEST_SUITE_CHECK = SmokeCheck(
    key="test-suite",
    title="Test suite",
    description="Chạy unit test và hồi quy mà không yêu cầu camera thật.",
    command=("run_tests.py", "--skip-camera-check"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy smoke-check an toàn cho các entrypoint chính của repo.")
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Chạy thêm run_tests.py --skip-camera-check ở cuối chuỗi smoke-check.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Dừng ngay khi gặp một check fail thay vì đi hết cả danh sách.",
    )
    parser.add_argument(
        "--ci-safe",
        action="store_true",
        help="Chỉ chạy những smoke-check nhẹ, phù hợp với môi trường CI không có camera/dataset đầy đủ.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in danh sách check và lệnh sẽ chạy, không thực thi gì cả.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


@lru_cache(maxsize=8)
def select_checks(*, include_tests: bool = False, ci_safe: bool = False) -> tuple[SmokeCheck, ...]:
    checks = [check for check in BASE_SMOKE_CHECKS if not ci_safe or check.ci_safe]
    if include_tests:
        checks.append(TEST_SUITE_CHECK)
    return tuple(checks)


def _display_command(check: SmokeCheck) -> str:
    return f"python {' '.join(check.command)}"


def _run_command(check: SmokeCheck) -> int:
    return subprocess.call([sys.executable, *check.command], cwd=PROJECT_ROOT)


def execute_checks(
    checks: tuple[SmokeCheck, ...],
    *,
    stop_on_fail: bool = False,
    dry_run: bool = False,
    run_command_fn=_run_command,
    print_fn=print,
) -> int:
    pass_count = 0
    warn_count = 0
    fail_count = 0
    failed_checks: list[str] = []

    for index, check in enumerate(checks, start=1):
        print_fn("=" * 78)
        print_fn(f"[{index}/{len(checks)}] {check.title}")
        print_fn(f"Command : {_display_command(check)}")
        print_fn(f"Purpose : {check.description}")
        if dry_run:
            print_fn("Result  : DRY-RUN (skipped)")
            pass_count += 1
            continue
        exit_code = int(run_command_fn(check))

        if exit_code == 0:
            pass_count += 1
            print_fn(f"Result  : PASS (exit={exit_code})")
        elif exit_code in check.warning_exit_codes:
            warn_count += 1
            print_fn(f"Result  : WARN (exit={exit_code})")
        else:
            fail_count += 1
            failed_checks.append(check.key)
            print_fn(f"Result  : FAIL (exit={exit_code})")
            if stop_on_fail:
                break

    print_fn("=" * 78)
    print_fn("Smoke summary")
    print_fn(f"- PASS: {pass_count}")
    print_fn(f"- WARN: {warn_count}")
    print_fn(f"- FAIL: {fail_count}")
    if failed_checks:
        print_fn("- Failed checks: " + ", ".join(failed_checks))

    return 0 if fail_count == 0 else 1


def main() -> int:
    args = parse_args()
    checks = select_checks(include_tests=args.include_tests, ci_safe=args.ci_safe)
    return execute_checks(checks, stop_on_fail=args.stop_on_fail, dry_run=getattr(args, "dry_run", False))


if __name__ == "__main__":
    raise SystemExit(run_entrypoint(main))
