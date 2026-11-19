# Giải Pháp Xử Lý 8 Vấn Đề Dự Án OncoVision

## Tổng Quan Sau Khi Khảo Sát

Sau khi khảo sát chi tiết codebase, phát hiện **3/8 vấn đề đã được xử lý một phần hoặc hoàn toàn**:
- ✅ **CI/CD**: Đã có `.github/workflows/test.yml` (4 Python versions, lint, type check, smoke test, unit tests) + `codeql.yml` (security analysis)
- ✅ **Clinical validation framework**: Đã có `medical/clinical_validation.py` với Wilson CI, sample size planning, SHA256 model verification
- ✅ **Model download**: Đã có `scripts/fetch_model.py` với catalog support, SHA256 verification, HuggingFace/GitHub integration

Dưới đây là giải pháp chi tiết cho từng vấn đề:

---

## 1. Model chưa đầy đủ (2/14 model)

### Hiện trạng
- Có sẵn: `modality_classifier.pt` (99.93%), `brain_classifier.pt` (4 loại u não)
- Thiếu: `brain_detector.pt`, `brain_volume_model.pt2`, 10 nhóm ung thư còn lại
- `scripts/fetch_model.py` đã hỗ trợ tải từ HuggingFace/GitHub với catalog

### Giải pháp

#### 1a. Tạo model catalog JSON mẫu
Tạo file `models/model_catalog.json` với danh sách model có sẵn trên HuggingFace:

```json
{
  "version": 1,
  "description": "Catalog model OncoVision — chỉ chứa model suy luận đã kiểm định",
  "entries": [
    {
      "repo": "owner/repo",
      "file": "model.pt",
      "target": "lung",
      "kind": "classifier",
      "modality": "ct",
      "sha256": "..."
    }
  ]
}
```

#### 1b. Cải thiện `fetch_model.py`
- Thêm progress bar khi tải
- Thêm retry logic cho kết nối mạng yếu
- Thêm `--dry-run` để xem trước trước khi tải

#### 1c. Tài liệu hướng dẫn tải model
Cập nhật `README.md` và `docs/medical_imaging_guide.md` với:
```bash
# Tải tất cả model có sẵn
python scripts/fetch_model.py --all

# Tải model cụ thể
python scripts/fetch_model.py owner/repo --file model.pt --target lung --kind classifier --modality ct
```

#### 1d. Model placeholder strategy
Với 10 nhóm ung thư chưa có model, tạo file placeholder `.placeholder` trong mỗi thư mục:
```
models/pretrained/cancers/lung/ct/classifier.pt.placeholder
```
Khi user chạy `fetch_model.py --all`, script sẽ tự động bỏ qua các mục không có repo.

---

## 2. Clinical validation độc lập

### Hiện trạng
- `medical/clinical_validation.py` đã có framework hoàn chỉnh
- CLI `run_medical.py` đã có `plan-validation`, `validate-dataset`, `split-dataset`
- Thiếu: Protocol template, hướng dẫn thực hiện, integration với pipeline

### Giải pháp

#### 2a. Tạo protocol template
Tạo `config/clinical_validation_template.json`:

```json
{
  "schema_version": 1,
  "cohort_id": "",
  "dataset_provenance": "",
  "ground_truth_method": "",
  "sample_size_justification": "",
  "site_count": 0,
  "inclusion_criteria": [],
  "exclusion_criteria": [],
  "approved_by": null,
  "approved_at": null
}
```

#### 2b. Thêm CLI command cho validation
Mở rộng `run_medical.py` với:
```bash
# Tạo báo cáo validation từ CSV kết quả
python run_medical.py validate-clinical \
  --csv results.csv \
  --model models/pretrained/brain_classifier.pt \
  --cohort-id "BRAIN-2026-001" \
  --output models/pretrained/cancers/brain/clinical_validation.json
```

#### 2c. Integration với pipeline
Trong `medical/pipeline.py`, khi có `clinical_validation.json` hợp lệ:
- Hiển thị badge "Đã kiểm định lâm sàng" trong báo cáo
- Bao gồm confidence intervals trong kết quả
- Cảnh báo nếu model hash không khớp với báo cáo validation

#### 2d. Hướng dẫn thực hiện
Tạo `docs/clinical_validation_guide.md` với:
- Bước 1: Ước lượng cỡ mẫu (`plan-validation`)
- Bước 2: Chuẩn bị cohort độc lập
- Bước 3: Chạy inference trên cohort
- Bước 4: Tạo CSV kết quả (patient_id, truth, prediction)
- Bước 5: Tạo báo cáo validation
- Bước 6: Gửi đơn vị lâm sàng phê duyệt

---

## 3. File lớn — Refactor pipeline.py (2242 dòng)

### Hiện trạng
- `MedicalImageAnalyzer` class có 53 methods trong 1 file
- Trộn lẫn: validation, preprocessing, inference, detection, volume, reporting, UI progress

### Giải pháp — Tách thành 6 module con

#### 3a. Cấu trúc mới
```
medical/pipeline/
├── __init__.py              # Re-export MedicalImageAnalyzer
├── analyzer.py              # Class chính — orchestrate các stage
├── config.py                # MedicalImageAnalyzerConfig + build_default
├── validation.py            # validate_input, DICOM parsing
├── preprocessing.py         # Normalize, resize, modality-specific prep
├── inference.py             # CNN inference, YOLO detection, ensemble
├── volume.py                # 3D volume inference (tách từ volume_inference)
├── detection.py             # Detection finding, bounding box mapping
├── reporting.py             # Report generation, dashboard writing
├── progress.py              # Progress callback handling
└── types.py                 # Dataclasses: DetectionFinding, PipelineStageResult, MedicalAnalysisResult
```

#### 3b. Chiến lược refactor
1. **Bước 1**: Tạo `medical/pipeline/` directory với `__init__.py` re-export tất cả symbols hiện tại
2. **Bước 2**: Tách các dataclasses vào `types.py`
3. **Bước 3**: Tách config vào `config.py`
4. **Bước 4**: Tách validation methods vào `validation.py`
5. **Bước 5**: Tách inference methods vào `inference.py`
6. **Bước 6**: Tách reporting methods vào `reporting.py`
7. **Bước 7**: Class `MedicalImageAnalyzer` chỉ còn là orchestrator (~200 dòng)
8. **Bước 8**: Giữ `medical/pipeline.py` làm compatibility shim (import từ `medical/pipeline/analyzer.py`)

#### 3c. Lợi ích
- Mỗi file ~100-300 dòng thay vì 2242 dòng
- Dễ test từng module riêng lẻ
- Dễ đọc và bảo trì
- Không thay đổi API — backward compatible

---

## 4. File lớn — Refactor web_app.py (1277 dòng)

### Hiện trạng
- FastAPI app với 1277 dòng trong 1 file
- Trộn lẫn: app setup, auth, upload, analysis, chat, admin, error handling

### Giải pháp — Tách thành module

#### 4a. Cấu trúc mới
```
app/web/
├── __init__.py              # Re-export app
├── app.py                   # FastAPI app setup, lifespan, middleware
├── auth.py                  # Login, logout, session, CSRF, password recovery
├── upload.py                # File upload, validation, analysis endpoints
├── chat.py                  # Chat endpoints, medical service integration
├── admin.py                 # Admin panel, user management
├── static.py                # Static files, templates
├── error.py                 # Error handlers
└── types.py                 # Shared types, constants
```

#### 4b. Chiến lược refactor
1. **Bước 1**: Tạo `app/web/` directory
2. **Bước 2**: Tách auth routes vào `auth.py`
3. **Bước 3**: Tách upload/analysis routes vào `upload.py`
4. **Bước 4**: Tách chat routes vào `chat.py`
5. **Bước 5**: Tách admin routes vào `admin.py`
6. **Bước 6**: File `web_app.py` chỉ còn ~50 dòng import và re-export

---

## 5. Scripts rời rạc (45 files)

### Hiện trạng
- 45 scripts trong `scripts/`
- Nhiều file là debug one-off: `check_models.py`, `check_models2.py`, `check_models_full.py`
- Training scripts: `train_modality_classifier.py`, `train_modality_fixed.py`, `train_modality_simple.py`
- Download scripts: `download_all_models.py`, `download_final_main.py`, `download_final_part1.py`, ...
- Probe scripts: `probe_liver.py`, `probe_liver2.py`

### Giải pháp — Phân loại và dọn dẹp

#### 5a. Phân loại scripts

| Nhóm | Files | Hành động |
|---|---|---|
| **Core** | `fetch_model.py`, `import_foreign_model.py`, `organize_models.py` | Giữ lại, cải thiện |
| **Download** | `download_models.py`, `download_missing_models.py`, `download_volume_models.py`, `download_all_models.py`, `download_final_*.py` | Gộp vào `fetch_model.py` |
| **Training** | `train_modality_*.py`, `generate_volume*.py` | Di chuyển vào `scripts/training/` |
| **Debug one-off** | `check_models*.py`, `probe_liver*.py`, `check_*.py` | Di chuyển vào `scripts/debug/` hoặc xóa |
| **Maintenance** | `maintenance/*.py` | Giữ lại |
| **Deploy** | `deploy/*.py` | Giữ lại |
| **Build** | `build_exe*.bat` | Di chuyển vào `scripts/build/` |
| **Summary** | `final_summary.py`, `summary_models.py` | Gộp vào `fetch_model.py --status` |

#### 5b. Cấu trúc mới
```
scripts/
├── fetch_model.py           # Core — tải và quản lý model
├── import_foreign_model.py  # Core — import model từ bên ngoài
├── organize_models.py       # Core — tổ chức thư mục model
├── build/                   # Build scripts
│   ├── build_exe.bat
│   ├── build_exe2.bat
│   └── build_exe3.bat
├── training/                # Training scripts (không dùng trong production)
│   ├── train_modality_classifier.py
│   ├── generate_volume_models.py
│   └── ...
├── debug/                   # Debug one-off scripts (có thể xóa sau)
│   ├── check_models.py
│   ├── probe_liver.py
│   └── ...
├── deploy/                  # Deploy scripts (giữ nguyên)
└── maintenance/             # Maintenance scripts (giữ nguyên)
```

#### 5c. Tạo `scripts/README.md`
```markdown
# Scripts OncoVision

## Core (sử dụng thường xuyên)
- `fetch_model.py` — Tải model từ HuggingFace/GitHub
- `import_foreign_model.py` — Import model từ bên ngoài
- `organize_models.py` — Tổ chức thư mục model

## Build
- `build/` — Build executable (cx_Freeze)

## Training (phát triển)
- `training/` — Scripts huấn luyện model

## Debug (tạm thời)
- `debug/` — Scripts debug một lần, có thể xóa
```

---

## 6. Windows-centric scripts

### Hiện trạng
- 3 file `.ps1` ở root: `cleanup_main.ps1`, `create_dist.ps1`, `deploy_production.ps1`
- 3 file `.bat` trong `scripts/`: `build_exe*.bat`
- 2 file `.ps1` trong `scripts/`: `create_recovery_branch.ps1`, `run_recovery.ps1`
- CI/CD dùng `shell: pwsh` — chỉ chạy trên Windows

### Giải pháp

#### 6a. Tạo script cross-platform
Thay thế các `.ps1` bằng Python scripts tương đương:

```python
# scripts/cleanup.py — thay thế cleanup_main.ps1
"""Dọn dẹp thư mục output và cache."""
import argparse
from pathlib import Path

def cleanup(keep_days: int = 7):
    output_dir = Path("output")
    # Logic cleanup...
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-days", type=int, default=7)
    args = parser.parse_args()
    cleanup(args.keep_days)
```

#### 6b. Cải thiện CI/CD
Thêm Ubuntu runner vào `.github/workflows/test.yml`:

```yaml
jobs:
  test:
    name: CI (${{ matrix.os }} / Python ${{ matrix.python-version }})
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
```

#### 6c. Cải thiện Dockerfile
- Dockerfile đã dùng Ubuntu 22.04 — tốt
- Thêm `requirements-web.txt` cho web-only deployment — đã có
- Cần thêm: multi-stage build để giảm kích thước image

#### 6d. Makefile
Tạo `Makefile` cross-platform:

```makefile
.PHONY: install install-dev test lint clean docker-build docker-run

install:
	pip install -r requirements/requirements.txt

install-dev:
	pip install -r requirements/requirements.txt
	pip install -r requirements/requirements-dev.txt

test:
	python -m pytest tests/ -v

lint:
	ruff check .

clean:
	python scripts/cleanup.py

docker-build:
	docker build -t onkovision:latest .

docker-run:
	docker compose up -d
```

---

## 7. PyAudio dependency

### Hiện trạng
- `requirements.txt`: `pyaudio>=0.2.14; platform_system == "Windows"` — đã conditional
- `requirements-web.txt`: Không bao gồm PyAudio — tốt
- Dùng trong: `app/chat_ui/voice_worker.py` và `app/chat_ui/window.py`
- PyAudio khó cài trên Linux/macOS (cần portaudio dev library)

### Giải pháp

#### 7a. Graceful fallback
Trong `voice_worker.py`, thêm try/except:

```python
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio không khả dụng — voice input sẽ bị vô hiệu hóa")
```

#### 7b. Alternative cho Linux/macOS
Thêm `sounddevice` + `soundfile` làm alternative:

```txt
# requirements.txt
pyaudio>=0.2.14; platform_system == "Windows"
sounddevice>=0.4.6; platform_system != "Windows"
soundfile>=0.12.1; platform_system != "Windows"
```

#### 7c. UI feedback
Trong `window.py`, khi PyAudio không khả dụng:
- Ẩn nút microphone
- Hiển thị thông báo: "Voice input không khả dụng trên hệ thống này"

#### 7d. Tài liệu
Cập nhật `docs/install_guide.md`:
```markdown
## Voice Input (tùy chọn)

### Windows
PyAudio được tự động cài đặt qua `requirements.txt`.

### Linux
Cài portaudio trước:
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev

# Fedora
sudo dnf install portaudio-devel
```

### macOS
```bash
brew install portaudio
```
```

---

## 8. Model gitignored — Automated download

### Hiện trạng
- `models/` được gitignore — đúng
- `scripts/fetch_model.py` đã hỗ trợ tải từ HuggingFace/GitHub
- Thiếu: Model catalog JSON, hướng dẫn rõ ràng, first-run experience

### Giải pháp

#### 8a. Tạo model catalog
Tạo `models/model_catalog.json` với danh sách model có sẵn:

```json
{
  "version": 1,
  "description": "Catalog model OncoVision",
  "entries": []
}
```

Khi có model mới trên HuggingFace, thêm vào catalog.

#### 8b. Cải thiện first-run experience
Trong `run_doctor.py` và `run_medical.py status`:
- Kiểm tra model catalog
- Nếu không có model nào, hiển thị hướng dẫn:
```
⚠️  Không tìm thấy model nào. Để tải model:

  python scripts/fetch_model.py --all

Hoặc tải thủ công theo hướng dẫn: docs/model_download.md
```

#### 8c. Tạo `docs/model_download.md`
Hướng dẫn chi tiết:
- Cách tải model từ HuggingFace
- Cách đặt model thủ công
- Cấu trúc thư mục model
- Cách kiểm tra model đã cài đúng

#### 8d. Model verification
Thêm command vào `run_medical.py`:
```bash
# Kiểm tra model đã cài
python run_medical.py verify-models

# Kết quả:
# ✅ brain_classifier.pt — SHA256 khớp
# ⚠️ brain_detector.pt — Thiếu
# 🔒 lung_classifier.pt — Chưa có trong catalog
```

---

## Tóm Tắt Hành Động

| # | Vấn đề | Trạng thái | Hành động | Ưu tiên |
|---|---|---|---|---|
| 1 | Model chưa đầy đủ | Partial | Tạo catalog, cải thiện fetch_model.py | Cao |
| 2 | Clinical validation | Framework có | Tạo template, CLI integration, hướng dẫn | Cao |
| 3a | pipeline.py lớn | Cần refactor | Tách thành 6 module con | Trung bình |
| 3b | web_app.py lớn | Cần refactor | Tách thành module app/web/ | Trung bình |
| 4 | Scripts rời rạc | Cần tổ chức | Phân loại, di chuyển, dọn dẹp | Thấp |
| 5 | Windows-centric | Partial | Tạo cross-platform alternatives | Trung bình |
| 6 | PyAudio | Partial | Graceful fallback, alternative | Thấp |
| 7 | CI/CD | ✅ Đã có | Cải thiện multi-OS runner | Thấp |
| 8 | Model download | Partial | Catalog, first-run experience | Cao |

---

## Lộ Trình Thực Hiện

### Giai đoạn 1 — Nhanh (1-2 ngày)
1. Tạo `models/model_catalog.json`
2. Cải thiện `fetch_model.py` với progress bar và retry
3. Tạo `docs/model_download.md`
4. Thêm graceful fallback cho PyAudio
5. Tạo `config/clinical_validation_template.json`

### Giai đoạn 2 — Trung hạn (1-2 tuần)
1. Refactor `pipeline.py` thành `medical/pipeline/`
2. Refactor `web_app.py` thành `app/web/`
3. Tổ chức lại `scripts/`
4. Tạo cross-platform alternatives cho .ps1/.bat

### Giai đoạn 3 — Dài hạn (1-2 tháng)
1. Bổ sung model cho các nhóm ung thư còn lại
2. Thực hiện clinical validation với cohort độc lập
3. Cải thiện CI/CD multi-OS
4. Multi-stage Docker build

---

## Lưu Ý Quan Trọng

1. **Không train model trong dự án**: Dự án chỉ suy luận (inference-only). Model phải được train bên ngoài và đặt vào `models/pretrained/`.
2. **Backward compatibility**: Mọi refactor phải giữ API cũ hoạt động — dùng re-export trong `__init__.py`.
3. **Test coverage**: Mỗi bước refactor phải có test tương ứng để đảm bảo không phá vỡ chức năng.
4. **Medical compliance**: Clinical validation phải có sự tham gia của đơn vị lâm sàng độc lập — không thể tự động hóa hoàn toàn.
