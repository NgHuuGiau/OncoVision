# Tổng Quan Dự Án

## Mục đích

OncoVision là bộ ứng dụng dùng để:

- chạy camera realtime với YOLO
- phân tích ảnh y khoa qua chat UI
- huấn luyện model custom
- kiểm tra sức khỏe hệ thống
- chạy smoke test an toàn cho CI

## Entry points chính

| File | Mục đích |
|---|---|
| `run_menu.py` | Menu trung tâm để mở các chức năng chính |
| `run_app.py` | Camera realtime |
| `run_chat.py` | Chat UI và phân tích ảnh y khoa |
| `run_doctor.py` | Kiểm tra sức khỏe hệ thống |
| `run_tests.py` | Chạy test hồi quy |
| `run_train.py` | Huấn luyện model |
| `run_medical.py` | CLI cho luồng medical |
| `run_smoke.py` | Kiểm tra an toàn entrypoint |

## Cấu trúc thư mục

### `app/`

Giao diện chat, bootstrap camera runtime và logic khởi chạy ứng dụng.

### `core/`

Runtime advisor, model selector, model loader, camera runner và logic nền.

### `medical/`

Pipeline phân tích ảnh y khoa, lưu lịch sử ca, export report và trạng thái hệ thống.

### `training/`

Chuẩn bị dữ liệu, chia dataset, train, validate và export model.

### `utils/`

Helper dùng chung: icon, terminal, file, draw, camera probe.

### `config/`

Cấu hình runtime, model và medical settings.

### `docs/`

Tài liệu cài đặt, vận hành, training và tổng quan.

### `tests/`

Unit tests và smoke/regression checks.

## Luồng chạy chính

```text
run_menu.py
├── run_app.py
├── run_chat.py
├── run_doctor.py
├── run_train.py
└── run_medical.py
```

## Luồng camera realtime

```text
run_app.py
→ runtime advisor
→ model selector
→ model loader
→ camera runner
→ draw utils
```

## Luồng training

```text
dataset/raw
→ training/validate_dataset.py
→ training/split_dataset.py
→ run_train.py
→ training/validate_model.py
→ training/export_model.py
```

## Luồng medical

```text
run_medical.py init-dataset
→ tạo cấu trúc dataset medical

run_medical.py analyze
→ normalize ảnh
→ detect
→ phân loại nguy cơ
→ lưu report + history
```

## Ghi chú

- `dataset/`, `models/`, `output/` thường là dữ liệu sinh ra lúc chạy và nên quản lý riêng.
- Khi đổi branding hoặc title UI, nên cập nhật entrypoint và test contract cùng lúc.
