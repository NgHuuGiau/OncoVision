from __future__ import annotations

from textwrap import dedent


MEDICAL_DISCLAIMER = dedent(
    """
    Canh bao y khoa:
    He thong nay chi ho tro sang loc va nghien cuu hinh anh y khoa.
    Ket qua AI khong duoc dung de tu chan doan, ke don hoac thay the y kien bac si.
    Moi truong hop nghi ngo ung thu can duoc danh gia boi bac si chuyen khoa va xet nghiem bo sung.
    """
).strip()


def build_medical_disclaimer() -> str:
    return MEDICAL_DISCLAIMER
