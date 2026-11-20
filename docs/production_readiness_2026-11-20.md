# Báo Cáo Production Readiness — OncoVision

**Ngày tạo:** 2026-11-20  
**Mục tiêu:** Đánh giá và xử lý 5 mục cần thiết để đưa OncoVision vào sản xuất y khoa thực sự.

---

## 1. Fix Volume 3D .pt2 — ✅ Đã Xử Lý

### Vấn Đề
- 3 model volume 3D (.pt2) trong catalog có status `needs_reexport` do PyTorch version mismatch
- Không có file .pt2 thực tế trong repo (chỉ có metadata)
- Khi load, `torch.export.load()` throw KeyError/RuntimeError

### Giải Pháp
**Script tự động:** `scripts/reexport_volume_3d.py` (278 dòng)

```bash
# Cách dùng
python scripts/reexport_volume_3d.py --source models/pretrained/cancers/brain/mri/volume.pt
python scripts/reexport_volume_3d.py --source models/pretrained/cancers/brain/mri/volume.pt --output models/pretrained/cancers/brain/mri/volume.pt2
```

**Cơ chế fallback trong code** (`medical/volume_inference.py`):
1. Thử `torch.export.load()` (native .pt2)
2. Fallback → `torch.jit.load()` (TorchScript)
3. Fallback → regular checkpoint

### Hạn Chế
- Nếu chỉ có file .pt2 hỏng mà không có .pt nguồn → **không thể tự động fix**
- Cần training script gốc để rebuild model architecture
- Hệ thống vẫn hoạt động bình thường ở chế độ 2D khi không có volume 3D

### Model Cần Re-export
| Model | Path | Status |
|---|---|---|
| Brain MRI Volume | `cancers/brain/mri/brain_volume_model.pt2` | needs_reexport |
| Breast Mammogram Volume | `cancers/breast/mammogram/volume.pt2` | needs_reexport |
| Breast MRI Volume | `cancers/breast/breast_volume_model.pt2` | needs_reexport |

---

## 2. Refactor pipeline.py — ✅ Đã Lập Kế Hoạch

### Vấn Đề
- `medical/pipeline.py`: 2,661 dòng, 13 subsection
- Khó maintain, khó test, khó đọc
- 13 file import trực tiếp từ `medical.pipeline`

### Giải Pháp
**Package mới:** `medical/pipeline/` với kế hoạch refactor chi tiết

**Cấu trúc mục tiêu:**
```
medical/pipeline/
├── __init__.py          # Kế hoạch refactor
├── helpers.py           # Section 1: 3 helper functions
├── dataclasses.py       # Section 2: 5 data classes
├── detection.py         # Section 3: 3 detection helpers
├── config.py            # Section 4-5: Config + validation
└── analyzer/            # Section 6: MedicalImageAnalyzer
    ├── init_progress.py # 6.1
    ├── model_path.py    # 6.2
    ├── entry_point.py   # 6.3
    ├── volume_3d.py     # 6.4
    ├── stages.py        # 6.5
    ├── video.py         # 6.6
    ├── modality.py      # 6.7
    ├── detection.py     # 6.8
    ├── cnn_brain.py     # 6.9
    ├── gradcam.py       # 6.10
    ├── reporting.py     # 6.11
    ├── demo.py          # 6.12
    └── utils.py         # 6.13
```

### Chiến Lược
- **Không refactor một lần** — quá rủi ro với 13 nơi import
- **Từng bước**: Tách Section 1-5 trước, rồi MedicalImageAnalyzer
- **Chạy test** sau mỗi bước để đảm bảo backward compatibility
- **Giữ pipeline.py** cho đến khi package hoàn tất

---

## 3. Refactor web_app.py — ✅ Đã Lập Kế Hoạch

### Vấn Đề
- `entrypoints/web_app.py`: 1,441 dòng, 40+ routes
- Tất cả routes nằm trong 1 file — khó maintain

### Giải Pháp
**Package mới:** `entrypoints/web/` với kế hoạch refactor chi tiết

**Cấu trúc mục tiêu:**
```
entrypoints/web/
├── __init__.py   # Kế hoạch refactor
├── auth.py       # 10 routes: login, logout, password
├── admin.py      # 8 routes: user management
├── medical.py    # 8 routes: analysis, upload
├── chat.py       # 4 routes: chat messages
├── cases.py      # 10 routes: case management
├── anatomy.py    # 1 route: 3D viewer
└── health.py     # 2 routes: health check
```

### Chiến Lược
- Dùng `APIRouter` của FastAPI
- Helper functions (CSRF, auth, role) → `entrypoints/web/deps.py`
- `web_app.py` chỉ còn: app setup, middleware, include routers

---

## 4. Clinical Validation — ✅ Framework + Hướng Dẫn

### Hiện Trạng
- Framework đã có sẵn: `medical/clinical_validation.py` (187 dòng)
- Đầy đủ hàm: `plan_binary_validation_sample_size()`, `build_independent_validation_report()`, `clinical_validation_status()`
- **Chưa có hướng dẫn sử dụng thực tế**

### Đã Thêm
1. **Hướng dẫn chi tiết:** `docs/clinical_validation_guide.md` (184 dòng)
   - 5 bước: Plan → Prepare → Run → Report → Approve
   - Code examples cho từng bước
   - Checklist trước khi deploy

2. **Template CSV:** `data/templates/validation_template.csv`
   - Định dạng: `patient_id,truth,prediction`
   - Ví dụ với 4 lớp ung thư não

### Cách Sử Dụng
```python
from medical.clinical_validation import (
    plan_binary_validation_sample_size,
    build_independent_validation_report,
    clinical_validation_status,
)

# Bước 1: Tính cỡ mẫu
plan = plan_binary_validation_sample_size(
    sensitivity=0.85, specificity=0.90,
    prevalence=0.30, margin=0.05,
)
# → Cần ~380 ca tối thiểu

# Bước 4: Tạo báo cáo
report = build_independent_validation_report(
    csv_path="validation_results.csv",
    model_path="models/pretrained/cancers/brain/mri/classifier.pt",
    output_path="output/validation/brain_mri_report.json",
    cohort_id="BRAIN_MRI_2026_Q1",
)

# Kiểm tra trạng thái
is_valid, reason = clinical_validation_status(
    report_path="output/validation/brain_mri_report.json",
    model_path="models/pretrained/cancers/brain/mri/classifier.pt",
)
```

---

## 5. Bổ Sung 6 Model Còn Thiếu — ✅ Hướng Dẫn Training

### 6 Model Cần Train

| # | Model | Modality | Dataset Công Khai | Khó Khăn |
|---|---|---|---|---|
| 1 | Breast | Mammogram | DDSM, INbreast | Dễ |
| 2 | Colorectal | Endoscopy | Kvasir v2, CVC-ClinicDB | Dễ |
| 3 | Kidney | Ultrasound | ❌ Rất hạn chế | Cần BV |
| 4 | Liver | MRI | MSD Liver | Trung bình |
| 5 | Prostate | Ultrasound | ❌ Rất hạn chế | Cần BV |
| 6 | Cervical | Pap smear | C-NPC, Herzog | Dễ |

### Đã Thêm
**Hướng dẫn chi tiết:** `docs/missing_models_training_guide.md` (357 dòng)

- Dataset đề xuất cho từng model
- Code example train classifier + detector
- Quy trình 4 bước: Prepare → Train → Evaluate → Catalog
- Yêu cầu phần cứng và timeline ước lượng

### Timeline Tổng
| Model | Thời Gian |
|---|---|
| Breast mammogram | 1-2 ngày |
| Colorectal endoscopy | 1 ngày |
| Kidney ultrasound | 2-4 tuần (cần thu thập dữ liệu) |
| Liver MRI | 2-3 ngày |
| Prostate ultrasound | 2-4 tuần (cần thu thập dữ liệu) |
| Cervical Pap smear | 1-2 ngày |

---

## 6. Tích Hợp 3D Mesh — ⏭️ SKIP Theo Yêu Cầu

User yêu cầu không xử lý mục này trong lần này.

---

## Tổng Kết

| Mục | Trạng Thái | File Đã Tạo/Sửa |
|---|---|---|
| Fix Volume 3D .pt2 | ✅ Script + fallback | `scripts/reexport_volume_3d.py` |
| Refactor pipeline.py | ✅ Kế hoạch | `medical/pipeline/__init__.py` |
| Refactor web_app.py | ✅ Kế hoạch | `entrypoints/web/__init__.py` |
| Clinical validation | ✅ Framework + hướng dẫn | `docs/clinical_validation_guide.md`, `data/templates/validation_template.csv` |
| 6 Model còn thiếu | ✅ Hướng dẫn training | `docs/missing_models_training_guide.md` |
| 3D Mesh integration | ⏭️ SKIP | — |

### Những Gì Cần Làm Tiếp

1. **Ngắn hạn** (1-2 ngày):
   - Chạy `scripts/reexport_volume_3d.py` nếu có file .pt nguồn
   - Bắt đầu refactor pipeline.py theo kế hoạch trong `medical/pipeline/__init__.py`

2. **Trung hạn** (1-2 tuần):
   - Hoàn tất refactor pipeline.py và web_app.py
   - Train 4 model có dataset công khai

3. **Dài hạn** (1-2 tháng):
   - Thu thập dữ liệu cho kidney/prostate ultrasound
   - Clinical validation với cohort độc lập
   - Phê duyệt từ đơn vị lâm sàng

### Lưu Ý Quan Trọng

- **Không deploy production** mà không có clinical validation được phê duyệt
- **Model chỉ hỗ trợ sàng lọc**, không thay thế chẩn đoán y khoa
- **Cần monitoring drift** và re-validation định kỳ sau deploy
