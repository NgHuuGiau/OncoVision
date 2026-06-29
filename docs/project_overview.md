# Tổng Quan Dự Án

OncoVision hiện gồm 2 luồng chính:

- `Y dược`
- `Vật thể`

## Entry point chính

| File | Mục đích |
|---|---|
| `run_menu.py` | Menu trung tâm |
| `run_app.py` | Camera realtime |
| `run_chat.py` | Chat UI và phân tích ảnh y khoa |
| `run_doctor.py` | Kiểm tra sức khỏe hệ thống |
| `run_train.py` | Huấn luyện YOLO vật thể |
| `run_medical.py` | CLI cho luồng y dược |
| `run_smoke.py` | Kiểm tra an toàn entrypoint |

## Cấu trúc thư mục

- `medical/`: pipeline y khoa, hồ sơ ca bệnh, trạng thái hệ thống
- `training/`: split, train, validate, downloader TCIA
- `utils/`: helper dùng chung
- `docs/`: tài liệu vận hành
- `tests/`: test và smoke check

## Ghi chú

- `dataset/medical/skin_lesion` là khung y dược chính để train skin lesion.
- `dataset/medical/tcia` là nơi đổ dữ liệu TCIA.
- `dataset/object_detection/` là luồng vật thể riêng.
