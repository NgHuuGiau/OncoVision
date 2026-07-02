# Tổng Quan Dự Án

[![Overview](https://img.shields.io/badge/Docs-Overview-0F766E?logo=readthedocs&logoColor=white)](project_overview.md)

Tài liệu này mô tả bố cục repo OncoVision ở mức kiến trúc: entrypoint nào dùng cho tác vụ nào, thư mục nào chịu trách nhiệm phần gì, và luồng dữ liệu đi qua hệ thống ra sao.

> Nếu bạn chỉ đọc một file trong `docs/`, hãy đọc file này trước.

## Tóm Tắt Nhanh

| Điểm cần nhớ | Ý nghĩa |
|---|---|
| `run_*.py` | Lớp entrypoint mỏng, mỗi file giữ một vai trò rõ ràng |
| `medical/` | Luồng y dược, status, report, dataset và training liên quan |
| `training/` | Luồng object detection và pipeline train / validate / export |
| `app/` | Phần UI và runtime đi sát trải nghiệm người dùng |
| `utils/` | Helper dùng chung, ưu tiên gọn và ít indirection |

## 1. Toàn Cảnh

OncoVision là một monorepo gồm:

- bộ entrypoint `run_*.py`,
- module `core/` cho camera realtime,
- module `medical/` cho nhánh y dược,
- module `training/` cho object detection,
- `app/` cho runtime camera và chat UI,
- `utils/` cho helper dùng chung,
- `tests/` cho unit test và hồi quy.

Sau các đợt dọn gần đây, package root đã chuyển sang import tĩnh thay vì lazy export để giảm indirection ở cấp package.

## 2. Cây Thư Mục

```text
OncoVision/
|-- app/
|-- assets/
|-- config/
|-- core/
|-- dataset/
|-- docs/
|-- medical/
|-- models/
|-- output/
|-- runs/
|-- scripts/
|-- tests/
|-- training/
|-- utils/
|-- run_app.py
|-- run_chat.py
|-- run_doctor.py
|-- run_medical.py
|-- run_menu.py
|-- run_smoke.py
`-- run_train.py
```

## 3. Thư Mục Chính

### `app/`

Chứa lớp gần với UI và runtime:

- `camera_runtime/`: parser, bootstrap và launch flow cho camera
- `chat_ui/`: chat window, storage, medical controller, theme, widgets

### `core/`

Xử lý camera realtime và object detection runtime:

- `camera_runner.py`: vòng lặp camera, detect, overlay, record, capture
- `model_loader.py`: nạp YOLO model và fallback
- `hardware_info.py`: đọc CPU/GPU/CUDA/PyTorch
- `frame_processing.py`: tiền xử lý frame
- `tracking/`: gán track, smooth và filter detection

### `dataset/`

Chứa dữ liệu vận hành của dự án, tách theo 2 nhánh:

- `dataset/medical/`
- `dataset/object_detection/`

### `medical/`

Package nghiệp vụ cho nhánh y dược:

- quản lý catalog ung thư,
- quản lý dataset structure medical,
- quản lý model medical,
- tổng hợp system status,
- lưu case DB,
- sinh report và phục vụ chat UI.

Các helper liên quan đã được gom rõ hơn:

- `medical/training.py` gom audit / split / train / validate cho luồng medical
- `medical/cli_helpers.py` chứa các helper in trạng thái y dược dùng chung cho CLI

### `training/`

Package cho object detection training:

- chuẩn bị dataset,
- validate dataset,
- split train/val/test,
- train model,
- validate/export model.

### `utils/`

Helper dùng chung:

- `console_ui.py`
- `entrypoint_checks.py`
- `file_utils.py`
- `logger.py`
- `camera_utils.py`
- `cleanup_utils.py`
- `sqlite_utils.py`

## 4. Entrypoint Gốc

| File | Trách nhiệm |
|---|---|
| `run_menu.py` | Cửa vào tổng hợp cho người vận hành |
| `run_app.py` | Runtime advisor và camera realtime |
| `run_chat.py` | Chat UI và cleanup output |
| `run_doctor.py` | Doctor scan tổng quát |
| `run_medical.py` | CLI quản lý nhánh y dược |
| `run_train.py` | Entrypoint training object detection |
| `run_smoke.py` | Smoke check entrypoint |
| `run_tests.py` | Dashboard unit test |

Các file `run_*.py` đang được giữ theo hướng entrypoint mỏng, còn logic chính nằm trong module helper tương ứng.

## 5. Luồng Dữ Liệu

### Camera realtime

```text
run_app.py
-> app/camera_runtime/*
-> core/hardware_info.py
-> core/model_loader.py
-> core/camera_runner.py
-> output/captures | output/recordings
```

### Object detection training

```text
dataset/object_detection/raw
-> training/prepare_dataset.py
-> training/validate_dataset.py
-> training/split_dataset.py
-> run_train.py / training/train_model.py
-> models/trained/best.pt
-> run_app.py --model models/trained/best.pt
```

### Medical

```text
dataset/medical/*
-> medical/dataset.py
-> medical/system_status.py
-> run_medical.py
-> output/medical/*
-> run_chat.py --check-only / launch chat
```

## 6. Nên Mở Đầu Tiên Khi Debug

| Vấn đề | Mở đầu tiên |
|---|---|
| Camera không chạy | `run_app.py`, `core/camera_runner.py`, `utils/camera_utils.py` |
| Runtime gợi ý sai | `core/hardware_info.py`, `core/runtime_advisor.py`, `app/camera_runtime/bootstrap.py` |
| Chat UI không sẵn sàng | `run_chat.py`, `utils/entrypoint_checks.py`, `app/chat_ui/` |
| Medical status sai | `medical/system_status.py`, `medical/model_policy.py`, `medical/storage.py` |
| Train fail | `run_train.py`, `training/train_model.py`, `training/validate_dataset.py` |
| CI fail | `.github/workflows/test.yml`, `run_smoke.py`, `requirements.txt` |
