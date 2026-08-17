# OncoVision

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/Windows-11%2B-0078D6)](https://www.microsoft.com/windows)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111111)](https://www.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)](https://opencv.org/)
[![PySide6](https://img.shields.io/badge/PySide6-Desktop%20UI-41CD52)](https://www.qt.io/qt-for-python)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Chat-009688)](https://fastapi.tiangolo.com/)
[![Medical](https://img.shields.io/badge/Medical-Screening-00A6A6)](docs/medical_imaging_guide.md)
[![Training](https://img.shields.io/badge/Training-YOLO-FFB000)](docs/training_guide.md)

**OncoVision** là nền tảng hỗ trợ chẩn đoán hình ảnh y khoa tích hợp: từ quản lý dữ liệu y tế, huấn luyện mô hình YOLO/CNN đến giao diện chat AI cho bác sĩ và nhân viên y tế. Hệ thống chạy hoàn toàn trên máy local (Windows), hỗ trợ xử lý đa dạng định dạng ảnh y khoa (DICOM, NIfTI, JPG, PNG).

---

## Tính năng chính

| Nhóm | Mô tả |
|---|---|
| **Chat AI Y khoa** | Giao diện desktop (PySide6) và web (FastAPI) để đặt câu hỏi, tải ảnh y khoa và nhận phân tích tự động |
| **Phân tích ảnh y tế** | Hỗ trợ 7 nhóm ung thư (gan, phổi, vú, dạ dày, đại trực tràng, tuyến tiền liệt, cổ tử cung) với đa modality (CT, MRI, X-quang, siêu âm, nội soi, PET/CT, mammogram) |
| **Camera thông minh** | Chạy realtime object detection với nhiều chế độ (auto/high/medium/low), tự động gợi ý cấu hình runtime phù hợp với máy |
| **Huấn luyện mô hình** | Pipeline train YOLO detection và CNN classifier đầy đủ, hỗ trợ resume, augment dữ liệu, export model |
| **YOLO Detection** | Phát hiện vật thể trong ảnh y khoa |
| **CNN Classifier** | Phân loại modality, nhóm ung thư |

---

## Bắt đầu nhanh

### Yêu cầu hệ thống

- Windows 10/11
- Python 3.10+
- GPU NVIDIA khuyến nghị cho train và inference

### Cài đặt

```powershell
git clone <repo-url>
cd OncoVision
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Database

Hệ thống dùng **SQLite** tập trung tại `output/onco.db` — lưu hội thoại chat + case y tế trong 1 file. Không cần cài DB server.

### Kiểm tra môi trường

```powershell
python run_doctor.py --skip-camera-check
python run_smoke.py
```

### Chạy chat AI

```powershell
# Giao diện desktop
python run_chat.py

# Giao diện web (optional)
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
```

---

## Bản đồ entrypoint

| Entrypoint | Vai trò |
|---|---|---|
| `run_chat.py` | Giao diện chat AI desktop — kiểm tra trạng thái, mở chat, dọn dẹp output |
| `run_app.py` | Camera realtime — gợi ý cấu hình runtime, chạy object detection trực tiếp |
| `run_menu.py` | Menu tổng hợp, cửa vào cho người vận hành |
| `run_doctor.py` | Quét tổng thể hệ thống — dependency, model, dataset, output |
| `run_medical.py` | CLI quản lý luồng y dược — dataset, model, training, modality, phân tích |
| `run_smoke.py` | Kiểm tra nhanh chuỗi entrypoint (CI-friendly) |
| `run_tests.py` | Dashboard chạy unit test |
| `run_train_brain_high.py` | Train CNN não (4 sub-label) — cấu hình cao, checkpoint 15 phút |
| `run_train_7cancers_high.py` | Train CNN 7 ung thư — cấu hình cao, checkpoint 15 phút |

### Luồng dữ liệu cơ bản

```
Camera:  run_app.py → core/camera_runner.py → output/captures/
Chat:    run_chat.py → app/chat_ui/ → medical/phân tích → output/chat/
Medical: run_medical.py → medical/dataset.py → output/medical/
Train:   run_train.py → models/trained/*.pt
Web:     web_app.py → SQLite → output/chat_history.db
```

---

## Huấn luyện mô hình

```powershell
# YOLO detection
python run_train.py

# CNN modality (8 loại hình ảnh y tế) — resnet18 @320, epochs 20
python run_medical.py train-modality

# CNN 7 ung thư (cấu hình cao, convnext_tiny @512, epochs 30) — cần ~190k ảnh, GPU 4GB
python run_train_7cancers_high.py

# CNN não (4 sub-label, convnext_tiny @512, epochs 35)
python run_train_brain_high.py
```

> **Cấu hình train nâng cao:** các script `run_train_*_high.py` dùng convnext_tiny pretrained @512px, focal loss,
> mixup 0.2, EMA 0.999, checkpoint averaging, checkpoint tự lưu mỗi 15 phút (resume được), `num_workers=2`.
> Model chưa train xong sẽ được `run_doctor.py` báo là chưa sẵn sàng.

---

## Cấu trúc thư mục

```text
OncoVision/
├── app/                    # Giao diện và runtime
│   ├── camera_runtime/     # Bootstrap và launch camera
│   └── chat_ui/            # Chat desktop, theme, storage, widgets
├── core/                   # Xử lý camera, model loader, hardware info
├── medical/                # Luồng y dược — dataset, model, pipeline, chat, report
├── training/               # Pipeline huấn luyện object detection
├── utils/                  # Helper dùng chung
├── config/                 # Cấu hình YAML
├── dataset/                # Dữ liệu
│   ├── medical/            # Dataset y tế
│   ├── medical_modality/   # Dataset modality (8 loại)
│   └── object_detection/   # Dataset detection
├── models/                 # Mô hình
│   ├── pretrained/         # Model tiền huấn luyện
│   └── trained/            # Model đã train
├── output/                 # Kết quả đầu ra
├── docs/                   # Tài liệu
└── tests/                  # Unit test
```

---

## Medical Pipeline (OncoVision AI)

Hệ thống phân tích ảnh y khoa với CNN classifier (convnext_tiny pretrained @512px cho ung thư/não, resnet18 @320px cho modality), hỗ trợ 7 nhóm ung thư:

| Bước | Module | Mô tả |
|---|---|---|
| Validate | `medical/validator.py` | Kiểm tra ảnh đầu vào, modality, body region |
| DICOM parse | `medical/dataset.py` | Parse DICOM header, window/level rendering |
| CNN inference | `medical/cnn_classifier.py` | FP16 trên GPU, TTA, confidence calibration |
| Grad-CAM | `medical/explainability.py` | Heatmap vùng CNN tập trung |
| Chat UI | `app/chat_ui/` | Desktop (PySide6) + Web (FastAPI) |
| Báo cáo | `medical/reporting.py` | JSON/MD/HTML dashboard |

## Tài liệu tham khảo

| File | Nội dung |
|---|---|
| [docs/project_overview.md](docs/project_overview.md) | Tổng quan kiến trúc và cây thư mục |
| [docs/install_guide.md](docs/install_guide.md) | Hướng dẫn cài đặt chi tiết |
| [docs/medical_imaging_guide.md](docs/medical_imaging_guide.md) | Luồng y dược — dataset, model, training |
| [docs/training_guide.md](docs/training_guide.md) | Huấn luyện object detection YOLO |
| [docs/runtime_tool_guide.md](docs/runtime_tool_guide.md) | Runtime advisor và camera realtime |
| [docs/quick_commands.md](docs/quick_commands.md) | Lệnh nhanh hàng ngày |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Lỗi thường gặp và cách xử lý |
| [docs/ci_and_quality.md](docs/ci_and_quality.md) | CI pipeline và quality gate |
