# Contributing to OncoVision

Cảm ơn bạn đã quan tâm đến OncoVision! Hướng dẫn này giúp bạn đóng góp vào dự án.

## Code of Conduct

- Tôn trọng người dùng và cộng đồng
- Kết quả chỉ dùng để hỗ trợ sàng lọc, không thay thế chẩn đoán y khoa
- Báo cáo lỗi và đề xuất cải tiến một cách xây dựng

## Development Setup

### 1. Clone repository
```bash
git clone https://github.com/your-org/OncoVision.git
cd OncoVision
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
# Production only
pip install -r requirements/requirements.txt

# Or with development tools (recommended for contributors)
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
```

### 4. Setup database
```bash
# Initialize database (automatic on first run)
python -c "from medical.storage import MedicalCaseDatabase; MedicalCaseDatabase()"
```

### 5. Run tests
```bash
pytest
# Or with coverage
pytest --cov=medical --cov=app --cov-report=term-missing
```

## Code Style

- **Formatter**: Ruff format (config in `config/ruff.toml`)
- **Linter**: Ruff (config in `config/ruff.toml`)
- **Type checking**: mypy
- **Line length**: 120 characters

Run quality checks:
```bash
# Format
ruff format .

# Lint
ruff check .

# Type check
mypy medical/ app/ core/
```

## Commit Messages

Viết commit message bằng tiếng Việt, theo format:
```
<loại>: <tiêu đề ngắn gọn>

<Mô tả chi tiết nếu cần>

Loại:
- feat: Tính năng mới
- fix: Sửa lỗi
- refactor: Refactor code
- docs: Tài liệu
- test: Test
- chore: Bảo trì
```

Ví dụ:
```
feat: thêm preprocessor cho colonoscopy

Thêm tiền xử lý riêng cho nội soi đại tràng:
- Loại bỏ bọt khí
- Tăng contrast cho polyp
- Normalization đặc biệt
```

## Pull Request Process

1. Fork repository
2. Tạo branch mới từ `main`
3. Thực hiện thay đổi
4. Chạy test và quality checks
5. Push và tạo Pull Request
6. Đợi review từ maintainer

## Project Structure

```
OncoVision/
├── app/              # Desktop UI (PySide6) & web auth
├── core/             # Camera & model runtime
├── medical/          # Medical pipeline (core logic)
├── utils/            # Common utilities
├── scripts/          # Automation scripts
├── tests/            # Test suite
├── config/           # Configuration files
├── models/           # Pretrained models
├── docs/             # Documentation
├── alembic/          # Database migrations
└── web_app.py        # Web app entry point
```

## Adding a New Cancer Target

1. Thêm vào `medical/cancer_catalog.py`:
```python
CancerScreeningTarget(
    key="new_cancer",
    label="Tên ung thư",
    description="Mô tả",
    modalities=("CT", "MRI"),
    model_ready=False,
)
```

2. Đặt model vào `models/pretrained/cancers/new_cancer/`

3. Cập nhật `models/model_catalog.json`

4. Thêm test vào `tests/`

## Running the Application

### Desktop Chat
```bash
python entrypoints/run_chat.py
```

### Web App
```bash
python entrypoints/web_app.py
# Access at http://localhost:8000
```

### CLI Medical Analysis
```bash
python entrypoints/run_medical.py analyze --image path/to/image.jpg --target brain
```

### Menu
```bash
python entrypoints/run_menu.py
```

## Reporting Issues

Khi báo cáo lỗi, hãy cung cấp:
- Mô tả lỗi
- Các bước để tái hiện
- Expected vs actual behavior
- Environment (OS, Python version, GPU)
- Logs nếu có

## License

Bằng phép MIT. Xem `LICENSE` để biết chi tiết.
