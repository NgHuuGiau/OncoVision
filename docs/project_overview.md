# Tổng quan dự án

## 1. Điểm vào chính

| File | Chức năng | Mô tả chi tiết |
|------|-----------|----------------|
| `run_menu.py` | Menu chính | Điểm vào để truy cập các chức năng khác |
| `run_app.py` | Camera realtime | Chạy camera thời gian thực với detection |
| `run_chat.py` | Chat + medical analysis | Giao diện chat tích hợp phân tích y khoa |
| `run_doctor.py` | System health check | Kiểm tra sức khỏe hệ thống |
| `run_tests.py` | Unit tests | Chạy các bài test đơn vị |
| `run_train.py` | Training pipeline | Pipeline huấn luyện mô hình |
| `run_medical.py` | Medical analysis CLI | CLI phân tích y khoa (init-dataset, analyze) |
| `run_smoke.py` | Smoke tests | Kiểm tra nhanh các chức năng cơ bản |

## 2. Cấu trúc thư mục dự án

### `app/`
- **Files**: `chat_ui/*.py`
- **Chức năng**: Giao diện chat UI, bootstrap camera
- **Xử lý**: UI người dùng, khởi động camera pipeline

### `core/`
- **Files**: `runtime_advisor.py`, `model_selector.py`, `model_loader.py`, `camera_runner.py`
- **Chức năng**: Runtime engine, load model, camera pipeline
- **Xử lý**: Đề xuất cấu hình runtime, chọn model, tải mô hình, chạy camera

### `docs/`
- **Files**: `*.md`
- **Chức năng**: Tài liệu dự án

### `medical/`
- **Files**: `chat_service.py`, `controller.py`, `system_status.py`, `storage.py`
- **Chức năng**: Medical imaging pipeline
- **Xử lý**: Dịch vụ chat y tế, điều khiển phân tích, trạng thái hệ thống, lưu trữ dữ liệu

### `tests/`
- **Files**: `test_*.py`
- **Chức năng**: Unit tests (30+ files)

### `training/`
- **Files**: `train_model.py`, `validate_dataset.py`, `split_dataset.py`, `export_model.py`
- **Chức năng**: Training pipeline
- **Xử lý**: Huấn luyện mô hình, xác thực dataset, chia dữ liệu, xuất mô hình

### `utils/`
- **Files**: `icons.py`, `draw_utils.py`, `file_utils.py`
- **Chức năng**: Helper utilities
- **Xử lý**: Quản lý icon, vẽ bounding box, xử lý file

### `config/`
- **Files**: `settings.yaml`, `model_config.yaml`
- **Chức năng**: Cấu hình inference, model

### `assets/`
- **Files**: `icons/*.svg`
- **Chức năng**: Icons SVG cho UI

### `models/`
- **Files**: `pretrained/*.pt`, `trained/*.pt`
- **Chức năng**: Model storage (gitignore)

### `dataset/`
- **Files**: `raw/images`, `raw/labels`, `processed/`
- **Chức năng**: Dữ liệu training (gitignore)

### `output/`
- **Files**: `chat_captures/`, `medical/`
- **Chức năng**: Kết quả xuất (gitignore)

## 3. Kiến trúc runtime camera

```text
run_app.py
└─> app/camera_runtime/bootstrap
    ├─> core/runtime_advisor (recommendation)
    ├─> core/model_selector (config)
    ├─> core/model_loader (load model)
    ├─> core/camera_runner (stream + detect)
    └─> utils/draw_utils (vẽ box)
```

## 4. Training Flow

```text
dataset/raw/images
  → training/validate_dataset.py (kiểm tra ảnh/label)
  → training/split_dataset.py (chia train/val/test)
  → run_train.py (huấn luyện)
  → training/validate_model.py (đánh giá)
  → training/export_model.py (xuất model)
```

## 5. Medical Flow

```text
run_medical.py init-dataset
  → dataset/medical_skin_lesion/ (tạo cấu trúc)

run_medical.py analyze --image <file>
  → medical/controller
    → normalize image
    → detect lesion
    → classify risk (low/med/high)
    → save report + case to SQLite

run_chat.py
  → upload image
  → medical pipeline (xử lý trên)
```

## 6. Mode inference

| Mode | Mục đích |
|------|----------|
| `high` | Chất lượng cao nhất máy gánh được |
| `medium` | Cân bằng performance/chất lượng |
| `low` | Tối ưu FPS, ít tài nguyên |
| `auto` | Hệ thống tự chọn dựa phần cứng |