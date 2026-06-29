from __future__ import annotations

from textwrap import dedent


MEDICAL_DISCLAIMER = dedent(
    """
    Canh bao y khoa:
    He thong nay chi ho tro sang loc va nghien cuu hinh anh y khoa.
    Kết quả phân tích tự động không được dùng để tự chẩn đoán, kê đơn hoặc thay thế ý kiến bác sĩ.
    Moi truong hop nghi ngo ung thu can duoc danh gia boi bac si chuyen khoa va xet nghiem bo sung.
    """
).strip()


def build_medical_disclaimer() -> str:
    return MEDICAL_DISCLAIMER
