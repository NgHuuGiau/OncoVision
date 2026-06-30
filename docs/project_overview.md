# Tổng Quan Dự Án

Tài liệu này mô tả bố cục repo OncoVision ở mức kiến trúc: entrypoint nào dùng cho tác vụ nào, thư mục nào phụ trách phần nào, và luồng dữ liệu đi qua hệ thống ra sao.

## 1. Toàn Cảnh

OncoVision là một monorepo kỹ thuật gồm:

- bộ entrypoint `run_*.py` để vận hành,
- module `core/` cho camera realtime,
- module `medical/` cho nhánh y dược,
- module `training/` cho object detection và downloader,
- `app/` cho runtime camera và chat UI,
- `utils/` cho helper dùng chung,
- `tests/` cho unit test và hồi quy.

## 2. Cây Thư Mục Dự Án

```text
OncoVision/
|-- .github/
|   `-- workflows/
|       `-- test.yml
|-- app/
|   |-- camera_runtime/
|   `-- chat_ui/
|-- assets/
|-- config/
|-- core/
|-- dataset/
|   |-- medical/
|   `-- object_detection/
|-- docs/
|-- medical/
|-- models/
|   |-- pretrained/
|   `-- trained/
|-- output/
|   |-- captures/
|   |-- chat_captures/
|   |-- logs/
|   `-- medical/
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

## 3. Mô Tả Chi Tiết Từng Thư Mục

### `.github/`

- chứa workflow CI cho repo,
- hiện tại workflow chính là `workflows/test.yml`,
- có log artifact và smoke mode an toàn cho CI.

### `app/`

Nơi chứa các lớp ở mức ứng dụng, gần với entrypoint và UI hơn là với thuật toán nền.

#### `app/camera_runtime/`

Dùng để:

- build argument parser cho `run_app.py`,
- chọn runtime mode,
- bootstrap luồng khởi động camera,
- giữ logic điều phối giữa phần cứng, runtime và luồng launch.

File đáng chú ý:

- `cli.py`: parser cho mode, camera-index, model
- `bootstrap.py`: tạo `StartOptions`, resolve runtime
- `launching.py`: boot progress và flow chạy camera

#### `app/chat_ui/`

Dùng để:

- cung cấp giao diện chat,
- lưu hội thoại,
- quản lý icon, paths, output,
- kết nối chat UI với logic medical.

File đáng chú ý:

- `cli.py`: parser cho `run_chat.py`
- `window.py`: launch giao diện chat
- `storage.py`: lưu hội thoại sqlite
- `medical_controller.py`: state giữa UI và medical service
- `voice_worker.py`: thu âm và chuyển giọng nói thành text

### `assets/`

- nơi để tài nguyên tĩnh,
- có thể chứa icon, ảnh mẫu, hoặc static asset phục vụ UI / demo.

### `config/`

- chứa file YAML cho settings runtime và medical,
- là nơi cần xem đầu tiên nếu muốn đổi default path, recording, confidence, iou, output.

### `core/`

Đây là lớp xử lý camera realtime và object detection runtime.

Thành phần chính:

- `camera_runner.py`: vòng lặp camera, đọc frame, detect, overlay, record, capture
- `model_loader.py`: nạp YOLO model và fallback
- `hardware_info.py`: đọc CPU/GPU/CUDA/PyTorch
- `frame_processing.py`: tiền xử lý frame, low-light, motion
- `tracking/`: logic gán track, smooth, filter detection
- `recorder.py`, `frame_capture.py`: quay video và chụp frame

Nội dung `core/` rất quan trọng với:

- `run_app.py`
- `run_doctor.py`
- `run_tests.py`

### `dataset/`

Chứa dữ liệu vận hành của dự án, tách làm 2 nhánh rõ ràng.

#### `dataset/medical/`

- skin lesion dataset,
- TCIA dataset,
- và những artifact dữ liệu medical liên quan.

#### `dataset/object_detection/`

- raw images / labels cho object detection,
- processed images sau split train/val/test,
- là đầu vào cho pipeline training YOLO.

### `docs/`

- bộ tài liệu vận hành và hướng dẫn cho thành viên nhóm,
- được tổ chức theo chủ đề: install, runtime, training, medical, quick commands, project overview.

### `medical/`

Đây là package nghiệp vụ cho nhánh y dược.

Trách nhiệm chính:

- mô tả catalog ung thư,
- quản lý dataset structure medical,
- quản lý model medical,
- tổng hợp system status,
- lưu case DB,
- report / output / service phục vụ chat UI.

Những file quan trọng:

- `system_status.py`: gom status model + data + output + DB
- `dataset.py`: mô tả layout dataset, đường dẫn mặc định và helper kiểm tra
- `pipeline.py`: xử lý / phân tích ảnh medical
- `storage.py`: medical case database
- `chat_service.py`: logic phản hồi cho chat UI
- `model_policy.py`: resolve runtime medical model

### `models/`

Chứa weights model.

#### `models/pretrained/`

- model có sẵn dùng để khởi động nhanh,
- thường dùng cho object detection baseline.

#### `models/trained/`

- model custom sau khi train nội bộ,
- là nơi `best.pt` thường được đưa vào runtime hoặc validate.

### `output/`

Chứa kết quả sinh ra trong quá trình chạy.

Sơ đồ tổng quát:

```text
output/
|-- captures/          # snapshot camera realtime
|-- recordings/        # video recording nếu có bật
|-- chat_captures/     # capture / attachment từ chat
|-- logs/              # app.log và log runtime
`-- medical/           # report, normalized, overlay, exports, db
```

### `runs/`

- kết quả huấn luyện / validate sinh bởi YOLO và training scripts,
- thường chứa artifact theo từng lần train.

### `scripts/`

- script phụ để verify, maintenance, hoặc tooling nhỏ,
- ví dụ `verify_entrypoints_help.py` được workflow CI gọi để check `--help`.

### `tests/`

- unit test và regression test,
- bao gồm test cho runtime, medical, training, UI logic, status, smoke support.

Nhóm test quan trọng:

- `test_run_smoke.py`
- `test_medical_system_status.py`
- `test_camera_detector.py`
- `test_runtime_prompt.py`
- `test_training_pipeline.py`

### `training/`

Đây là package training object detection và downloader phụ trợ.

Vai trò chính:

- chuẩn bị dataset,
- validate dataset,
- split train/val/test,
- chạy train model,
- validate model,
- export model,
- quản lý TCIA collections và downloader.

File quan trọng:

- `prepare_dataset.py`
- `validate_dataset.py`
- `split_dataset.py`
- `train_model.py`
- `validate_model.py`
- `export_model.py`
- `tcia_downloader.py`
- `download_models.py`

### `utils/`

Chứa helper dùng chung cho toàn repo.

Nhóm chức năng:

- `console_ui.py`: in bảng, dashboard, terminal rendering
- `entrypoint_checks.py`: preflight checks cho chat / runtime / training
- `file_utils.py`: helper file / YAML / path
- `logger.py`: logger có fallback
- `camera_utils.py`: wrapper mở camera
- `cleanup_utils.py`: dọn dẹp output
- `sqlite_utils.py`: helper sqlite

## 4. Vai Trò Của Từng Entrypoint Gốc

| File | Trách nhiệm |
|---|---|
| `run_menu.py` | Cửa vào tổng hợp cho người vận hành |
| `run_app.py` | Runtime advisor và camera realtime |
| `run_chat.py` | Chat UI, preflight chat, cleanup output |
| `run_doctor.py` | Doctor scan tổng quát cho môi trường |
| `run_medical.py` | CLI quản lý nhánh y dược |
| `run_train.py` | Entrypoint object detection training |
| `run_smoke.py` | Smoke check entrypoint |
| `run_tests.py` | Dashboard unit test với camera check tùy chọn |

## 5. Luồng Dữ Liệu Tổng Quan

### Luồng camera realtime

```text
run_app.py
-> app/camera_runtime/*
-> core/hardware_info.py
-> core/model_loader.py
-> core/camera_runner.py
-> output/captures | output/recordings
```

### Luồng object detection training

```text
dataset/object_detection/raw
-> training/prepare_dataset.py
-> training/validate_dataset.py
-> training/split_dataset.py
-> run_train.py / training/train_model.py
-> models/trained/best.pt
-> run_app.py --model models/trained/best.pt
```

### Luồng medical

```text
dataset/medical/*
-> medical/dataset.py
-> medical/system_status.py
-> run_medical.py
-> output/medical/*
-> run_chat.py --check-only / launch chat
```

## 6. Thư Mục Nào Nên Mở Đầu Tiên Khi Debug

| Vấn đề | Thư mục / file nên mở đầu tiên |
|---|---|
| Camera không chạy | `run_app.py`, `core/camera_runner.py`, `utils/camera_utils.py` |
| Runtime gợi ý sai | `core/hardware_info.py`, `core/runtime_advisor.py`, `app/camera_runtime/bootstrap.py` |
| Chat UI không sẵn sàng | `run_chat.py`, `utils/entrypoint_checks.py`, `app/chat_ui/` |
| Medical status sai | `medical/system_status.py`, `medical/model_policy.py`, `medical/storage.py` |
| Train fail | `run_train.py`, `training/train_model.py`, `training/validate_dataset.py` |
| CI fail | `.github/workflows/test.yml`, `run_smoke.py`, `requirements.txt` |

## 7. Nguyên Tắc Kiến Trúc Đang Thể Hiện Trong Repo

- Entry point rõ ràng, module nghiệp vụ tách riêng.
- Package import đang được tối ưu để tránh side effect quá sớm.
- CI ưu tiên smoke mode an toàn, không cố gắng mở tất cả feature nặng.
- Medical và object detection tách layout dữ liệu để tránh lẫn nghiệp vụ.

## 8. Cách Dùng Tài Liệu Này

Nếu bạn mới vào repo:

1. Đọc file này trước.
2. Sau đó đọc `install_guide.md`.
3. Nếu phụ trách object detection, đọc tiếp `training_guide.md`.
4. Nếu phụ trách luồng y dược, đọc tiếp `medical_imaging_guide.md`.
