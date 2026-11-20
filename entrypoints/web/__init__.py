"""Web routers package — Kế hoạch refactor web_app.py (1,441 dòng).

Tài liệu này mô tả kế hoạch refactor web_app.py thành các router modules nhỏ hơn.
Hiện tại web_app.py vẫn là file duy nhất; package này sẽ được lấp đầy dần dần.

Cấu trúc mục tiêu sau khi refactor hoàn tất:
    entrypoints/web/
    ├── __init__.py        # Re-export app instance
    ├── auth.py            # Login, logout, password recovery, change credentials (10 routes)
    ├── admin.py           # User management (8 routes)
    ├── medical.py         # Analysis, upload, system status (8 routes)
    ├── chat.py            # Chat messages (4 routes)
    ├── cases.py           # Case management (10 routes)
    ├── anatomy.py         # 3D anatomy viewer (1 route)
    └── health.py          # Health check, favicon (2 routes)

Cách refactor từng bước:
    1. Tách auth.py: login, logout, forgot-password, change-credentials
    2. Tách admin.py: /admin/users/* routes
    3. Tách medical.py: /api/analyze, /api/upload, /api/medical/*
    4. Tách chat.py: /api/chat/* routes
    5. Tách cases.py: /api/cases/* routes
    6. Tách anatomy.py: /api/anatomy/* routes
    7. Cập nhật web_app.py để include các router
    8. Chạy test sau mỗi bước

Lưu ý: Các router sẽ dùng APIRouter của FastAPI và được include vào app
chính trong web_app.py. Các helper function (require_authenticated, CSRF,
role hierarchy) sẽ được chuyển vào entrypoints/web/deps.py.
"""
from __future__ import annotations

__all__ = []
