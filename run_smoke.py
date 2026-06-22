from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SmokeCheck:
    key: str
    title: str
    description: str
    command: tuple[str, ...]
    warning_exit_codes: tuple[int, ...] = ()


BASE_SMOKE_CHECKS: tuple[SmokeCheck, ...] = (
    SmokeCheck(
        key="runtime-advisor",
        title="Runtime advisor",
        description="Kiem tra entrypoint tu van runtime ma khong mo camera.",
        command=("run_app.py", "--advisor-only"),
    ),
    SmokeCheck(
        key="doctor",
        title="Doctor scan",
        description="Ra soat tong the phan cung, model va du lieu ma khong can webcam that.",
        command=("run_doctor.py", "--skip-camera-check"),
    ),
    SmokeCheck(
        key="chat-preflight",
        title="Chat preflight",
        description="Kiem tra dep bat buoc, icon va do san sang cua luong chat/medical.",
        command=("run_chat.py", "--check-only"),
        warning_exit_codes=(2,),
    ),
    SmokeCheck(
        key="training-preflight",
        title="Training preflight",
        description="Kiem tra config, model va dataset cho entrypoint train tong quat.",
        command=("run_train.py", "--check-only"),
    ),
    SmokeCheck(
        key="medical-status",
        title="Medical status",
        description="Kiem tra nhanh model, dataset va output cua nhanh medical.",
        command=("run_medical.py", "status"),
    ),
)

TEST_SUITE_CHECK = SmokeCheck(
    key="test-suite",
    title="Test suite",
    description="Chay unit test va hoi quy ma khong yeu cau camera that.",
    command=("run_tests.py", "--skip-camera-check"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chay smoke-check an toan cho cac entrypoint chinh cua repo.")
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Chay them run_tests.py --skip-camera-check o cuoi chuoi smoke-check.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Dung ngay khi gap mot check fail thay vi di het ca danh sach.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def select_checks(*, include_tests: bool = False) -> tuple[SmokeCheck, ...]:
    checks = list(BASE_SMOKE_CHECKS)
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
    checks = select_checks(include_tests=args.include_tests)
    return execute_checks(checks, stop_on_fail=args.stop_on_fail)


if __name__ == "__main__":
    raise SystemExit(main())
