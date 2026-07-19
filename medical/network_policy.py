"""Chinh sach tai trong so tu mang cho OncoVision.

Quy tac: KHI CHAY HE THONG KHONG TAI GI TRU YOLO.

- Model detection YOLO (Ultralytics) van duoc phep tai binh thuong.
- Moi trong so khac (ImageNet pretrained cho backbone CNN, DINOv2, SAM, ...)
  bi chan mac dinh de tranh tai mang khong mong muon luc runtime.

Nguoi dung co the mo lai bang bien moi truong:

    ONCOVISION_ALLOW_WEIGHT_DOWNLOAD=1

Khi do pretrained=True se tai trong so ImageNet nhu binh thuong.
"""

from __future__ import annotations

import os

_ENV_ALLOW = "ONCOVISION_ALLOW_WEIGHT_DOWNLOAD"


def weight_download_allowed() -> bool:
    """Tra ve True neu cho phep tai trong so pretrained (ngoai YOLO)."""
    value = os.environ.get(_ENV_ALLOW, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def resolve_pretrained(requested: bool, *, context: str = "backbone") -> bool:
    """Ap dung chinh sach offline cho co pretrained.

    Neu goi yeu cau pretrained=True nhung chinh sach offline dang bat, ha ve
    False va in canh bao mot lan de nguoi dung biet model khoi tao voi trong so
    ngau nhien (can train tu dau hoac cung cap checkpoint local).
    """
    if not requested:
        return False
    if weight_download_allowed():
        return True
    _warn_once(context)
    return False


_warned_contexts: set[str] = set()


def _warn_once(context: str) -> None:
    if context in _warned_contexts:
        return
    _warned_contexts.add(context)
    print(
        f"[NetworkPolicy] Chan tai trong so pretrained cho '{context}' (offline). "
        f"Dat {_ENV_ALLOW}=1 de cho phep tai. YOLO khong bi anh huong.",
        flush=True,
    )


__all__ = ["weight_download_allowed", "resolve_pretrained"]
