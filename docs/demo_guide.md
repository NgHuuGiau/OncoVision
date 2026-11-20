# Hướng dẫn chạy demo OncoVision

Hướng dẫn chi tiết cách chạy demo hệ thống OncoVision — AI y khoa phân tích ảnh y tế để sàng lọc 11 nhóm ung thư.

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt](#2-cài-đặt)
3. [Chạy demo Web](#3-chạy-demo-web)
4. [Chạy demo Desktop](#4-chạy-demo-desktop)
5. [Phân tích ảnh](#5-phân-tích-ảnh)
6. [Kiểm tra hệ thống](#6-kiểm-tra-hệ-thống)
7. [Xây dựng EXE](#7-xây-dựng-exe)
8. [Mục lục model](#8-mục-lục-model)
9. [Khắc phục sự cố](#9-khắc-phục-sự-cố)

---

## 1. Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| **OS** | Windows 10/11, Linux, macOS | Windows 11 |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | Không bắt buộc | NVIDIA GPU 4+ VRAM (CUDA) |
| **Python** | 3.10–3.13 | 3.12 |
| **Disk** | 10 GB trống | 20+ GB (khi có model) |

---

## 2. Cài đặt

### 2.1. Clone dự án

```bash
git clone <repository-url>
cd OncoVision
```

### 2.2. Tạo môi trường ảo

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2.3. Cài đặt dependencies

```bash
pip install -r requirements/requirements.txt
```

### 2.4. Chuẩn bị model

Hệ thống cần file model tại `models/pretrained/cancers/`. Các model **không được lưu trong git** (bị `.gitignore`).

**Cách 1:** Copy model từ máy dev vào thư mục `models/pretrained/cancers/`.

**Cách 2:** Nếu chưa có model, hệ thống vẫn chạy được nhưng sẽ báo `runtime_ready=False` cho các nhóm ung thư thiếu model.

Cấu trúc model:
```
models/pretrained/cancers/
├── modality_classifier.pt          # Phân loại 6 loại ảnh y tế
├── brain/
│   ├── classifier.pt
│   └── detector.pt
├── breast/
│   └── ultrasound/
│       ├── classifier.pt
│       └── detector.pt
├── ... (tương tự cho 11 nhóm ung thư)
```

### 2.5. Kiểm tra cài đặt

```bash
python entrypoints/run_smoke.py
```

**Kết quả mong đợi:** `PASS 4/4`

---

## 3. Chạy demo Web

### 3.1. Khởi động server

```bash
python entrypoints/web_app.py
```

Server chạy tại: **http://127.0.0.1:8000**

### 3.2. Truy cập web

1. Mở trình duyệt, truy cập `http://127.0.0.1:8000`
2. Đăng nhập bằng tài khoản admin (mặc định: `admin` / `admin`)
3. Lần đầu đăng nhập, hệ thống yêu cầu đổi tên đăng nhập và mật khẩu

### 3.3. Các trang chính

| Trang | URL | Mô tả |
|---|---|---|
| **Trang chủ** | `/` | Giao diện chat, upload ảnh |
| **Đăng nhập** | `/login` | Đăng nhập hệ thống |
| **Quản lý user** | `/admin/users` | Tạo/sửa tài khoản (admin) |
| **Danh sách ca** | `/api/cases` | Xem danh sách ca bệnh |
| **Kiểm tra hệ thống** | `/api/status` | Trạng thái model, DB |
| **Health check** | `/healthz` | Kiểm tra server hoạt động |

### 3.4. Quyền hạn

| Role | Mô tả | Quyền |
|---|---|---|
| **admin** | Quản trị viên | Toàn quyền |
| **clinician** | Nhân viên y tế | Phân tích, rà soát ca |
| **viewer** | Người dùng | Chỉ xem kết quả đã duyệt |

---

## 4. Chạy demo Desktop

### 4.1. Khởi động ứng dụng desktop

```bash
python entrypoints/run_chat.py
```

Ứng dụng desktop sử dụng PySide6 (Qt) với giao diện chat nhúng web UI.

### 4.2. Các tính năng desktop

- **Chat AI:** Trò chuyện với AI y khoa
- **Upload ảnh:** Kéo thả hoặc chọn file ảnh
- **Phân tích:** Tự động phân tích ảnh y tế
- **Lịch sử:** Lưu trữ lịch sử hội thoại

---

## 5. Phân tích ảnh

### 5.1. Qua giao diện Web

1. Truy cập trang chủ `/`
2. Chọn **nhóm ung thư** từ danh sách (bắt buộc)
3. Chọn **modality** (tùy chọn, hệ thống tự động phát hiện)
4. **Upload ảnh** — kéo thả hoặc chọn file
5. Nhấn **Phân tích**
6. Xem kết quả:
   - **Risk level:** `high_suspicion`, `indeterminate`, `uncertain`, `low`
   - **Confidence:** Mức độ tự tin của model (0–1)
   - **Detections:** Vùng phát hiện bất thường
   - **Recommendation:** Khuyến nghị bước tiếp theo

### 5.2. Qua CLI

```bash
python entrypoints/run_medical.py analyze --image path/to/image.jpg --target brain
```

### 5.3. Qua API

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "image_path=output/web_uploads/20261121_abc123_image.jpg" \
  -F "target_key=brain" \
  -F "patient_code=DEMO001"
```

### 5.4. 11 nhóm ung thư hỗ trợ

| # | Ung thư | Modality chính | File model |
|---|---|---|---|
| 1 | **U não** | MRI | `brain/classifier.pt`, `brain/detector.pt` |
| 2 | **U vú** | Siêu âm | `breast/ultrasound/classifier.pt`, `breast/ultrasound/detector.pt` |
| 3 | **U cổ tử cung** | Colposcopy | `cervical/colposcopy/classifier.pt`, `cervical/colposcopy/detector.pt` |
| 4 | **U đại trực tràng** | Colonoscopy | `colorectal/colonoscopy/classifier.pt`, `colorectal/colonoscopy/detector.pt` |
| 5 | **U thận** | CT | `kidney/ct/classifier.pt`, `kidney/ct/detector.pt` |
| 6 | **U gan** | CT | `liver/ct/classifier.pt`, `liver/ct/detector.pt` |
| 7 | **U phổi** | CT, X-ray | `lung/ct/classifier.pt`, `lung/ct/detector.pt` |
| 8 | **U tụy** | CT | `pancreas/ct/classifier.pt`, `pancreas/ct/detector.pt` |
| 9 | **U tuyến tiền liệt** | MRI | `prostate/mri/classifier.pt`, `prostate/mri/detector.pt` |
| 10 | **U dạ dày** | Nội soi | `stomach/endoscopy/classifier.pt`, `stomach/endoscopy/detector.pt` |
| 11 | **U tuyến giáp** | Siêu âm | `thyroid/ultrasound/classifier.pt`, `thyroid/ultrasound/detector.pt` |

### 5.5. Auto-detect cancer type

Khi không chọn `target_key`, hệ thống tự động xác định nhóm ung thư từ:
1. **Tên file** — tìm từ khóa liên quan (brain, lung, thyroid...)
2. **Phân tích pixel** — xác định modality (CT, MRI, siêu âm, nội soi...)
3. **CNN classifier** — so sánh score của các model khớp modality

**Lưu ý:** Auto-detect hoạt động tốt nhất với ảnh y khoa thực tế. Với ảnh không liên quan, hệ thống trả về `status="uncertain"`.

---

## 6. Kiểm tra hệ thống

### 6.1. Smoke test

```bash
python entrypoints/run_smoke.py
```

Kiểm tra 4 thành phần:
1. **Runtime advisor** — phát hiện phần cứng, gợi ý model phù hợp
2. **Doctor scan** — kiểm tra 11/11 nhóm ung thư `runtime_ready`
3. **Chat preflight** — kiểm tra chat UI sẵn sàng
4. **Medical status** — kiểm tra pipeline y tế

### 6.2. System health check

```bash
python entrypoints/run_doctor.py
```

Quét toàn bộ hệ thống: model, detector, database, network.

### 6.3. Kiểm tra web app

```bash
# Health endpoint
curl http://127.0.0.1:8000/healthz

# Status endpoint (cần auth)
curl http://127.0.0.1:8000/api/status
```

### 6.4. Chạy test

```bash
# Chạy toàn bộ test
pytest tests/ -v

# Chạy test integration
pytest tests/integration/ -v
```

---

## 7. Xây dựng EXE

### 7.1. Yêu cầu

- Windows 10/11
- Python 3.12
- ~30 phút build, ~12 GB disk

### 7.2. Build

```bash
python scripts/setup_build.py build_exe
```

File EXE sẽ được tạo tại `dist/OncoVision/`.

### 7.3. Chạy EXE

```bash
dist\OncoVision\OncoVision.exe
```

---

## 8. Mục lục model

Xem chi tiết tại:

- **`models/model_catalog.json`** — 35 entries, mô tả hệ thống mong đợi
- **`models/README.md`** — Hướng dẫn quản lý model
- **`models/backup/README.md`** — Danh sách file backup

### 8.1. Tổng quan model hiện tại

| Thành phần | Số lượng | Trạng thái |
|---|---|---|
| **Classifiers** | 14 file | ✅ Hoạt động |
| **Detectors** | 17 file | ✅ 16/17 hoạt động |
| **Modality classifier** | 1 file | ✅ 99.93% accuracy |
| **Volume 3D** | 0 file | ❌ Đã backup, cần re-export |
| **Clinical validation** | 0/11 | ⏳ Chưa có |

---

## 9. Khắc phục sự cố

### 9.1. Lỗi phổ biến

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `ModuleNotFoundError` | Thiếu package | `pip install -r requirements/requirements.txt` |
| `FileNotFoundError: classifier.pt` | Thiếu model | Copy model vào `models/pretrained/cancers/` |
| `CUDA out of memory` | GPU không đủ RAM | Chạy trên CPU: `export ONCOVISION_DEVICE=cpu` |
| `Port 8000 already in use` | Server đang chạy | Dừng server cũ hoặc đổi port: `export ONCOVISION_HOST=127.0.0.1` |
| `Backbone metadata mismatch` | Metadata backbone sai | Hệ thống tự động xử lý, không ảnh hưởng |

### 9.2. Warnings không ảnh hưởng

- **Backbone mismatch** (liver, thyroid, brain, cervical): Hệ thống tự động phát hiện và load backbone đúng
- **Volume 3D fail**: Hệ thống tự động fallback sang chế độ 2D
- **State_dict key mismatch**: Một số key không khớp nhưng model vẫn hoạt động

### 9.3. Log file

Log được lưu tại: `output/logs/`

```bash
# Xem log realtime
tail -f output/logs/oncovision.log
```

---

## Phụ lục: Kiến trúc web app (sau refactor)

```
entrypoints/
├── web_app.py              # Slim entry point (179 dòng)
└── web/                    # Blueprint package
    ├── __init__.py
    ├── shared.py           # Common helpers (384 dòng)
    ├── pages.py            # Pages: index, login, admin (335 dòng)
    ├── health.py           # Health & status (100 dòng)
    ├── upload_analyze.py   # Upload & analyze (178 dòng)
    ├── cases.py            # Case management (228 dòng)
    ├── conversations.py    # Admin conversations (110 dòng)
    ├── settings.py         # Settings (30 dòng)
    └── utils_routes.py     # Crop, output files (94 dòng)
```

Tổng: **~1,259 dòng** trong 8 file blueprint + **179 dòng** entry point = **~1,438 dòng** (so với 1,441 dòng trong 1 file trước đó).
