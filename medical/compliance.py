from __future__ import annotations

from textwrap import dedent


MEDICAL_DISCLAIMER = dedent(
    """
    Cảnh báo y khoa:
    Hệ thống này chỉ hỗ trợ sàng lọc và nghiên cứu hình ảnh y khoa.
    Kết quả phân tích tự động không được dùng để tự chẩn đoán, kê đơn hoặc thay thế ý kiến bác sĩ.
    Mọi trường hợp nghi ngờ ung thư cần được đánh giá bởi bác sĩ chuyên khoa và xét nghiệm bổ sung.
    """
).strip()


def build_medical_disclaimer() -> str:
    return MEDICAL_DISCLAIMER
