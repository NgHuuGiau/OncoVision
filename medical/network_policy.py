"""Chính sách tải trọng số từ mạng cho OncoVision.

Quy tắc: KHI CHẠY HỆ THỐNG KHÔNG TẢI GÌ TRỪ YOLO.

- Model detection YOLO (Ultralytics) vẫn được phép tải bình thường.
- Mọi trọng số khác (ImageNet pretrained cho backbone CNN, DINOv2, SAM, ...)
  bị chặn mặc định để tránh tải mạng không mong muốn lúc runtime.

Người dùng có thể mở lại bằng biến môi trường:

    ONCOVISION_ALLOW_WEIGHT_DOWNLOAD=1

Khi đó pretrained=True sẽ tải trọng số ImageNet như bình thường.
"""

from __future__ import annotations

import os

_ENV_ALLOW = "ONCOVISION_ALLOW_WEIGHT_DOWNLOAD"
_ENV_REQUIRE = "ONCOVISION_REQUIRE_PRETRAINED"


def weight_download_allowed() -> bool:
    """Tra ve True neu cho phép tải trọng số pretrained (ngoai YOLO)."""
    value = os.environ.get(_ENV_ALLOW, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _require_pretrained() -> bool:
    value = os.environ.get(_ENV_REQUIRE, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def resolve_pretrained(requested: bool, *, context: str = "backbone") -> bool:
    """Áp dụng chính sách offline cho cờ pretrained.

    Nếu gọi yêu cầu pretrained=True nhưng chính sách offline đang bật, hạ về
    False và in cảnh báo một lần để người dùng biết model khởi tạo với trọng số
    ngẫu nhiên (cần train từ đầu hoặc cung cấp checkpoint local).

    Nếu đặt ONCOVISION_REQUIRE_PRETRAINED=1 (dành cho serving/production), thay
    vì hạ thầy lang sẽ raise RuntimeError để fail-loud, tránh degrade thầy lang.
    """
    if not requested:
        return False
    if weight_download_allowed():
        return True
    if _require_pretrained():
        raise RuntimeError(
            f"Yêu cầu pretrained cho '{context}' nhưng tải trọng số bị chặn. "
            f"Đặt {_ENV_ALLOW}=1 để cho phép tải, hoặc cung cấp checkpoint local."
        )
    _warn_once(context)
    return False


_warned_contexts: set[str] = set()


def _warn_once(context: str) -> None:
    if context in _warned_contexts:
        return
    _warned_contexts.add(context)
    print(
        f"[NetworkPolicy] Chặn tải trọng số pretrained cho '{context}' (offline). "
        f"Đặt {_ENV_ALLOW}=1 để cho phép tải. YOLO không bị ảnh hưởng.",
        flush=True,
    )


__all__ = ["weight_download_allowed", "resolve_pretrained"]
