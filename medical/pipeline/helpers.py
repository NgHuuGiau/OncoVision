"""Section 1: Helper functions.

Các hàm utility dùng trong pipeline: sanitize filename, validate path, canonical body region.
"""
from __future__ import annotations

import re
from pathlib import Path


def _sanitize_filename_part(text: str, max_len: int = 50) -> str:
    """Sanitize text for safe use in filenames: chỉ A-Z a-z 0-9 . _ -"""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    return safe[:max_len]


def _validate_output_path(base_dir: Path, target_path: Path) -> Path:
    """Validate that target_path is within base_dir to prevent path traversal."""
    try:
        base_resolved = base_dir.resolve()
        target_resolved = target_path.resolve()
        if not target_resolved.is_relative_to(base_resolved):
            raise ValueError(f"Path traversal detected: {target_path} outside {base_dir}")
    except (ValueError, OSError):
        raise ValueError(f"Invalid path: {target_path}")
    return target_resolved


def _canonical_body_region_key(target_key: str) -> str | None:
    """Chuyen target_key sang body_region (hiện chỉ 'brain')."""
    from medical.validator import _canonical_body_region
    return _canonical_body_region(target_key)
