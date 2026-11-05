"""Chỉ dùng checkpoint local ngoài script tải YOLO11 tường minh."""

from __future__ import annotations

import os
import sys

_ENV_REQUIRE = "ONCOVISION_REQUIRE_PRETRAINED"


def weight_download_allowed() -> bool:
    """Pretrained backbone weights must be provided locally."""
    return False


def _require_pretrained() -> bool:
    value = os.environ.get(_ENV_REQUIRE, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def resolve_pretrained(requested: bool, *, context: str = "backbone") -> bool:
    """Áp dụng chính sách offline cho cờ pretrained.

    Nếu gọi yêu cầu pretrained=True nhưng chính sách offline đang bật, hạ về
    False và in cảnh báo một lần để người dùng biết model khởi tạo với trọng số
    ngẫu nhiên (không phù hợp cho suy luận; cần cung cấp checkpoint local).

    Nếu đặt ONCOVISION_REQUIRE_PRETRAINED=1 (dành cho serving/production), thay
    vì hạ thầy lang sẽ raise RuntimeError để fail-loud, tránh degrade thầy lang.
    """
    if not requested:
        return False
    if _require_pretrained():
        raise RuntimeError(
            f"Yêu cầu pretrained cho '{context}' nhưng chỉ cho phép dùng checkpoint local."
        )
    _warn_once(context)
    return False


_warned_contexts: set[str] = set()


def _warn_once(context: str) -> None:
    if context in _warned_contexts:
        return
    _warned_contexts.add(context)
    _safe_print(
        f"[NetworkPolicy] Không tải trọng số pretrained cho '{context}'; cần checkpoint local."
    )


def _safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoded = message.encode(sys.stdout.encoding or "ascii", errors="replace").decode(
            sys.stdout.encoding or "ascii"
        )
        print(encoded, flush=True)


__all__ = ["resolve_pretrained", "weight_download_allowed"]
