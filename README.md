# OncoVision

[![Python](https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.13-3776AB)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6)](https://www.microsoft.com/windows)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111111)](https://www.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)](https://opencv.org/)
[![PySide6](https://img.shields.io/badge/PySide6-Desktop%20UI-41CD52)](https://www.qt.io/qt-for-python)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Chat-009688)](https://fastapi.tiangolo.com/)
[![Medical](https://img.shields.io/badge/Medical-Screening-00A6A6)](docs/medical_imaging_guide.md)

Ứng dụng suy luận ảnh y khoa: nhận ảnh → kiểm tra dữ liệu → chạy model có sẵn → trả kết quả sàng lọc và gợi ý bước tiếp theo. Hỗ trợ DICOM, NIfTI, JPG, PNG cùng giao diện desktop (PySide6/QtWebEngine) và web (FastAPI).

## Chạy dự án

### Yêu cầu
- Windows 10/11
- Python 3.10–3.13 (khuyến nghị 3.12)
- GPU NVIDIA khuyến nghị; CPU vẫn chạy được

### Cài đặt
```powershell
git clone <repo-url>
cd OncoVision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements/requirements.txt
```

Với GPU NVIDIA (CUDA 13.0):
```powershell
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Kiểm tra môi trường
```powershell
python entrypoints/run_doctor.py --skip-camera-check
python entrypoints/run_smoke.py
```

### Chạy giao diện Chat AI (Desktop)
```powershell
python entrypoints/run_chat.py
```

### Chạy giao diện Web (FastAPI)
```powershell
python entrypoints/web_app.py
# hoặc
python -m uvicorn entrypoints.web_app:app --host 127.0.0.1 --port 8000
```
Mở http://127.0.0.1:8000. Xem các endpoint đăng nhập và xác thực trong [docs/api.md](docs/api.md).

### Chạy Camera Realtime
```powershell
python entrypoints/run_app.py
```

### Menu tổng hợp
```powershell
python entrypoints/run_menu.py
```

### Chạy bằng Docker (Web)
```powershell
docker build -t onkovision:latest .
docker run -d --name onkovision -p 8000:8000 `
  -v ${PWD}/models:/app/models -v ${PWD}/output:/app/output `
  -v ${PWD}/data:/app/data -v ${PWD}/config:/app/config onkovision:latest
# GPU: thêm --gpus all
```

### Build file cài đặt Desktop (EXE)
```powershell
python scripts/setup_build.py build_exe
```
Kết quả ở `dist/` (~30 phút, ~12 GB). Giao diện y hệt web.

## Cấu trúc thư mục chính

```
OncoVision/
├── entrypoints/      # 9 điểm vào: run_chat.py, run_app.py, run_menu.py, run_doctor.py, run_medical.py, run_smoke.py, run_tests.py, web_app.py, launcher.py
├── app/              # Giao diện: camera_runtime/, chat_ui/ (PySide6 theme, storage, widgets)
├── core/             # Camera runner, model loader, hardware info
├── medical/          # Pipeline y khoa: validator, cnn_classifier, explainability (Grad-CAM), reporting, dataset (DICOM)
├── models/           # Model suy luận (gitignored; tự đặt)
│   ├── pretrained/   # brain_classifier.pt, modality_classifier.pt, brain_detector.pt, brain_volume_model.pt2 + .json
│   └── trained/      # Model bổ sung do người dùng cung cấp
├── cancers/          # 11 nhóm ung thư: classifier.pt, detector.pt, volume.pt2 + volume.json, clinical_validation.json
├── output/           # Kết quả đầu ra (gitignored): onco.db (SQLite), captures/, chat/, medical/
├── deploy/           # Dockerfile, docker-compose.yml
├── docs/             # Tài liệu chi tiết
├── scripts/          # fetch_model, import_foreign_model, setup_build.py, dev/
├── static/           # CSS/JS cho web
├── templates/        # Jinja2 templates
├── config/           # YAML, Alembic, ruff
├── tests/            # 440 unit and integration tests
└── utils/            # Helper: console, terminal UI, file, camera
```

## Model & Phân tích

Hệ thống **chỉ suy luận bằng model có sẵn**. Đặt model vào `models/pretrained/`:

```
models/pretrained/
├── brain_classifier.pt        # Não (4 sub-label) — có sẵn máy dev, KHÔNG đi kèm repo
├── modality_classifier.pt     # Modality 8 loại ảnh — có sẵn máy dev, 99.93%
├── brain_detector.pt          # YOLO detect/segment y khoa
├── brain_volume_model.pt2     # torch.export MRI/CT 3D (.pt cũ vẫn hỗ trợ)
├── brain_volume_model.json    # Manifest: task, nhãn, input shape, ngưỡng mask
└── cancers/<key>/             # 11 nhóm: brain, breast, cervical, colorectal, kidney, liver, lung, pancreas, prostate, stomach, thyroid
    ├── classifier.pt
    ├── detector.pt
    ├── volume.pt2 + volume.json
    └── clinical_validation.json
```

Khi biết modality, model nằm đúng thư mục con: `cancers/lung/ct/` hoặc `cancers/lung/xray/`.

### Kiểm tra model đã nhận đủ
```powershell
python entrypoints/run_doctor.py --skip-camera-check
```

### CLI suy luận ảnh y khoa
```powershell
# Phân tích 1 ảnh não
python entrypoints/run_medical.py analyze --image path/to/ảnh.jpg --patient-code BN001

# Validate & split dataset (không rò rỉ bệnh nhân)
python entrypoints/run_medical.py validate-dataset --csv dataset.csv
python entrypoints/run_medical.py split-dataset --csv dataset.csv --output dataset_split.csv

# Ước lượng cỡ mẫu kiểm định
python entrypoints/run_medical.py plan-validation --sensitivity 0.90 --specificity 0.85 --prevalence 0.20
```

## Trạng thái model (11/2026)

| Thành phần | Trạng thái |
|---|---|
| Modality Classifier | ✅ ResNet18, 6 loại MedNIST, **99.93% acc** |
| 11 Classifier ung thư | ✅ 14 classifier cho 11 nhóm |
| 16 Detector ung thư | ✅ 15/16 hoạt động (thyroid INVALID — tự bỏ qua) |
| Volume 3D | ⏳ Runtime sẵn sàng, fallback 3 cấp; **chưa có model 3D huấn luyện sẵn** |
| Catalog 11 nhóm | ✅ 11/11 runtime_ready |
| Model manifests | ✅ 31 manifests |
| Web UI (FastAPI) | ✅ `python entrypoints/web_app.py` |
| Desktop (QtWebEngine) | ✅ `python entrypoints/run_chat.py` |
| Desktop EXE | ✅ `python scripts/setup_build.py build_exe` |
| Test suite | ✅ **440 tests** |

> **Lưu ý**: 11/11 nhóm ung thư inference thành công. Volume 3D cần re-export với PyTorch tương thích. Kết quả chỉ hỗ trợ sàng lọc, chưa có kiểm định lâm sàng độc lập.

## Medical Pipeline

| Bước | Module | Mô tả |
|---|---|---|
| Validate | `medical/validator.py` | Kiểm tra ảnh đầu vào, modality, body region |
| DICOM parse | `medical/dataset.py` | Parse header, window/level rendering |
| CNN inference | `medical/cnn_classifier.py` | FP16 GPU, TTA, confidence calibration |
| Grad-CAM | `medical/explainability.py` | Heatmap vùng CNN tập trung |
| Chat UI | `app/chat_ui/` | Desktop (PySide6) + Web (FastAPI) chung giao diện |
| Báo cáo | `medical/reporting.py` | JSON/MD/HTML dashboard |

## Tài liệu tham khảo

| File | Nội dung |
|---|---|
| [docs/project_overview_2026-11-19.md](docs/project_overview_2026-11-19.md) | Kiến trúc và cây thư mục |
| [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) | Cài đặt và build desktop |
| [docs/model_inference_audit.md](docs/model_inference_audit.md) | Luồng suy luận và trạng thái model |
| [models/README.md](models/README.md) | Quy ước model và tình trạng checkpoint |
| [docs/clinical_validation_guide.md](docs/clinical_validation_guide.md) | Kiểm định lâm sàng |
| [docs/demo_guide.md](docs/demo_guide.md) | Chạy demo web và desktop |
| [docs/api.md](docs/api.md) | API, đăng nhập và xác thực |
| [scripts/README.md](scripts/README.md) | Danh sách scripts |
