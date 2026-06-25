# 🧬 OncoVision

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111?style=for-the-badge&logo=github&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PySide6](https://img.shields.io/badge/PySide6-Desktop%20UI-41CD52?style=for-the-badge&logo=qt&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D4?style=for-the-badge&logo=windows&logoColor=white)

**Tiếng Việt** | [English](README_EN.md)

OncoVision là hệ thống phân tích hình ảnh y khoa thời gian thực dựa trên YOLO, kết hợp camera realtime, chat hỗ trợ phân tích ảnh, kiểm tra sức khỏe hệ thống, và luồng huấn luyện model nội bộ.

## Tổng quan

OncoVision được thiết kế cho các tình huống cần:

- Camera realtime để nhận diện nhanh
- Chat UI để phân tích ảnh y khoa
- Medical workflow cho dataset, train, validate, history và export
- Runtime advisor để chọn cấu hình phù hợp với máy
- Smoke test và doctor scan để kiểm tra hệ thống trước khi chạy thật

## Tính năng nổi bật

- Camera realtime với runtime advisor
- Chat UI tích hợp phân tích ảnh y khoa
- Medical CLI riêng cho dataset, train và history
- Bộ công cụ kiểm tra hệ thống an toàn
- GitHub Actions workflow lưu log `.txt` khi CI chạy

## Công nghệ sử dụng

| Nhóm | Công nghệ | Vai trò |
|---|---|---|
| Ngôn ngữ | Python 3.10+ | Ngôn ngữ chính của toàn bộ dự án |
| Deep learning | PyTorch, Ultralytics YOLO | Suy luận và huấn luyện model |
| Xử lý ảnh | OpenCV, Pillow, NumPy | Đọc, biến đổi và hiển thị ảnh |
| Tiện ích | PyYAML, tqdm, psutil, GPUtil | Cấu hình, progress, theo dõi tài nguyên |
| Giao diện | PySide6 | UI desktop và chat UI |
| Âm thanh | faster-whisper, pyaudio | Hỗ trợ luồng voice/audio nếu bật |
| Lưu trữ | SQLite | Lưu lịch sử ca y khoa và report |

## Nền tảng hỗ trợ

- Windows 10/11
- PowerShell / Windows Terminal
- Webcam nếu muốn chạy camera realtime
- GPU NVIDIA là tùy chọn, nhưng giúp inference và training nhanh hơn

## Bắt đầu nhanh

```powershell
python run_menu.py
```

Hoặc chạy trực tiếp:

```powershell
python run_app.py
python run_chat.py
python run_doctor.py
python run_train.py
python run_medical.py
python run_smoke.py
```

## Cấu trúc dự án

```text
OncoVision/
├── run_*.py          Entry points chính
├── app/              Camera runtime và chat UI
├── core/             Runtime, model, camera pipeline
├── medical/          Luồng phân tích ảnh y khoa
├── training/         Chuẩn bị dữ liệu, train, validate, export
├── utils/            Icon, file, terminal, helper
├── config/           Cấu hình model và runtime
├── docs/             Tài liệu hướng dẫn
└── tests/            Unit tests và regression checks
```

## Thành phần chính

- `run_menu.py` - menu trung tâm
- `run_app.py` - camera realtime
- `run_chat.py` - chat UI và phân tích ảnh
- `run_doctor.py` - kiểm tra sức khỏe hệ thống
- `run_train.py` - huấn luyện model
- `run_medical.py` - CLI cho workflow medical
- `run_smoke.py` - smoke test an toàn

## Tài liệu

- [Tổng quan dự án](docs/project_overview.md)
- [Hướng dẫn cài đặt](docs/install_guide.md)
- [Lệnh nhanh](docs/quick_commands.md)
- [Hướng dẫn runtime](docs/runtime_tool_guide.md)
- [Hướng dẫn training](docs/training_guide.md)
- [Hướng dẫn medical imaging](docs/medical_imaging_guide.md)

## Hỗ trợ chính

- Runtime advisor để chọn mode phù hợp với máy
- Doctor scan để kiểm tra hệ thống trước khi chạy thật
- Smoke check để test an toàn các entrypoint chính
- Unit tests để kiểm tra logic lõi
- Workflow GitHub Actions để lưu log chi tiết

## Thông số kỹ thuật

| Thông số | Giá trị |
|---|---|
| Hỗ trợ Python | 3.10+ |
| Loại inference | YOLO realtime |
| Giao diện | Desktop chat UI |
| Lưu trữ | SQLite |
| Kiểm thử | `unittest`, smoke checks |
| CI | GitHub Actions |

## Yêu cầu

- Windows 10/11
- Python 3.10+
- Webcam nếu muốn dùng camera realtime
- GPU NVIDIA là tùy chọn

## Giấy phép

MIT
