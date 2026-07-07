from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    run_files = sorted(
        path.name
        for path in ROOT.glob("run_*.py")
        if path.is_file() and path.name != "run_menu.py"
    )
    failures: list[str] = []
    for script in run_files:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            failures.append(script)
            print(f"[FAIL] {script}")
            print(result.stdout)
            print(result.stderr)
        else:
            print(f"[OK] {script}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
