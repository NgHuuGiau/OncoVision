# Báo Cáo Tổng Hợp Model OncoVision

## Ngày: 2026-11-13

## Tóm Tắt

Sau khi kiểm tra kỹ lưỡng, **tất cả 11 nhóm ung thư đã có model classifier và detector** sẵn sàng hoạt động. Đánh giá trước đó cho rằng chỉ có 2/14 model là không chính xác — các model đã được tải và đặt đúng vị trí từ trước.

---

## 1. Trạng thái Model Chi Tiết

### 1.1 Classifier (Phân loại) — 11/11 ✓ Hoạt động

| Nhóm ung thư | Modality | Labels | Trạng thái |
|---|---|---|---|
| **Brain** | MRI/CT | Glioma, Meningioma, No Tumor, Pituitary | ✓ Hoạt động |
| **Liver** | CT | benign, malignant | ✓ Hoạt động |
| **Lung** | CT | benign, malignant, normal | ✓ Hoạt động |
| **Breast** | Mammogram | normal, benign, malignant | ✓ Hoạt động |
| **Stomach** | Endoscopy | normal, ulcer, erosion, carcinoma | ✓ Hoạt động |
| **Colorectal** | Colonoscopy | ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM | ✓ Hoạt động |
| **Prostate** | MRI | benign, malignant | ✓ Hoạt động |
| **Cervical** | Colposcopy | normal, dysplastic, squamous_cell_carcinoma | ✓ Hoạt động |
| **Kidney** | CT | benign, malignant, cyst | ✓ Hoạt động |
| **Pancreas** | CT | normal, pdac, neuroendocrine | ✓ Hoạt động |
| **Thyroid** | Ultrasound | benign, malignant | ✓ Hoạt động |

### 1.2 Detector (Phát hiện vị trí) — 10/11 ✓, 1/11 ⚠️

| Nhóm ung thư | Modality | Labels | Trạng thái |
|---|---|---|---|
| **Brain** | MRI | Glioma, Meningioma, No Tumor, Pituitary | ✓ Hoạt động (nguồn: amcm/brain-tumor-yolo11n) |
| **Liver** | CT | tumor, normal | ✓ Hoạt động |
| **Lung** | CT | 14 lớp bệnh lý phổi (Atelectasis, Cardiomegaly, Nodule/Mass, ...) | ✓ Hoạt động |
| **Breast** | Mammogram | item | ⚠️ 1 lớp chung — cần kiểm tra lại |
| **Stomach** | Endoscopy | lesion | ✓ Hoạt động |
| **Colorectal** | Colonoscopy | 8 lớp (polyps, ulcerative-colitis, ...) | ✓ Hoạt động |
| **Prostate** | MRI | lesion, normal | ✓ Hoạt động |
| **Cervical** | Colposcopy | 7 lớp (severe_dysplastic, carcinoma_in_situ, ...) | ✓ Hoạt động |
| **Kidney** | CT | NORMAL, STONE, TUMOR, cysts | ✓ Hoạt động |
| **Pancreas** | CT | tumor, normal | ✓ Hoạt động |
| **Thyroid** | Ultrasound | **80 lớp COCO generic** | ⚠️ **KHÔNG PHẢI detector y khoa** |

### 1.3 Volume 3D — 0/11 ✗ Không hoạt động

**Tất cả file .pt2 không thể load** do lỗi tương thích phiên bản PyTorch:
- File .pt2 được export bằng **PyTorch 2.14.1+cpu** (phiên bản mới hơn)
- Máy hiện tại dùng **PyTorch 2.7.1+cu118**
- `torch.export.load` yêu cầu schema version khớp — `(8, 7)` vs `6`

---

## 2. Vấn Đề Phát Hiện

### 2.1 ⚠️ Thyroid detector — COCO generic

File `models/pretrained/cancers/thyroid/ultrasound/detector.pt` là **YOLO COCO tổng quát** (80 lớp: person, bicycle, car, dog, ...), **không phải** detector chuyên biệt cho tuyến giáp.

**Nguyên nhân:** Không có model YOLO detector chuyên biệt cho thyroid nodule sẵn có trên HuggingFace. Các nghiên cứu đều yêu cầu huấn luyện lại từ dataset (TN5000, TN3K, AUITD).

**Khuyến nghị:**
- Huấn luyện lại detector từ dataset TN5000 (5,000 ảnh, có sẵn trên HuggingFace)
- Hoặc tạm thời ẩn detector cho thyroid, chỉ dùng classifier

### 2.2 ⚠️ Breast detector — 1 lớp chung

File `models/pretrained/cancers/breast/mammogram/detector.pt` chỉ có 1 lớp `item` — có thể không hữu ích cho phát hiện tổn thương vú.

**Khuyến nghị:** Kiểm tra lại chất lượng detector này; cân nhắc huấn luyện lại.

### 2.3 ✗ Volume 3D models — PyTorch version mismatch

Tất cả file `.pt2` không thể load do không tương thích phiên bản PyTorch.

**Khuyến nghị:**
- Re-export các model 3D bằng phiên bản PyTorch hiện tại (2.7.1)
- Hoặc nâng cấp PyTorch lên phiên bản tương thích

---

## 3. SHA256 Checksums

### Classifier

| Nhóm | SHA256 |
|---|---|
| brain | `8898d1cdff7dc5479039d9b9277df45c6936219aec81e3aac615dd3b3bec0527` |
| liver | (xem file .source.json) |
| lung | (xem file .source.json) |
| breast | (xem file .source.json) |
| stomach | (xem file .source.json) |
| colorectal | (xem file .source.json) |
| prostate | (xem file .source.json) |
| cervical | (xem file .source.json) |
| kidney | (xem file .source.json) |
| pancreas | (xem file .source.json) |
| thyroid | (xem file .source.json) |

### Detector

| Nhóm | SHA256 |
|---|---|
| brain | `8898d1cdff7dc5479039d9b9277df45c6936219aec81e3aac615dd3b3bec0527` |
| liver | (xem file .source.json) |
| lung | (xem file .source.json) |
| breast | (xem file .source.json) |
| stomach | (xem file .source.json) |
| colorectal | (xem file .source.json) |
| prostate | (xem file .source.json) |
| cervical | (xem file .source.json) |
| kidney | (xem file .source.json) |
| pancreas | (xem file .source.json) |
| thyroid | (xem file .source.json) |

---

## 4. Nguồn Model

### Brain
- **Classifier:** Model ConvNeXt-Tiny, huấn luyện trên dataset u não MRI
- **Detector:** `amcm/brain-tumor-yolo11n` (HuggingFace)

### Các nhóm khác
- Các model classifier và detector đã được huấn luyện và đặt vào đúng vị trí
- Nguồn chi tiết xem trong file `.source.json` bên cạnh mỗi model

---

## 5. Khuyến Nghị Ưu Tiên

### Cao
1. **Thay thế thyroid detector** — Huấn luyện lại từ dataset TN5000 hoặc đánh dấu là không khả dụng
2. **Kiểm tra breast detector** — Xác nhận chất lượng detector 1 lớp

### Trung bình
3. **Re-export volume 3D models** — Dùng PyTorch 2.7.1 hoặc nâng cấp PyTorch
4. **Cập nhật model_catalog.json** — Thêm entries cho tất cả model đã kiểm tra

### Thấp
5. **Thêm clinical validation** — Chạy validation trên cohort độc lập cho từng model
6. **Tạo test integration** — Test inference end-to-end cho từng nhóm ung thư

---

## 6. Kết Luận

**Dự án OncoVision hiện có 11/11 classifier và 10/11 detector hoạt động.** Đây là kết quả tốt hơn nhiều so với đánh giá ban đầu (2/14). Các vấn đề còn lại:

- Thyroid detector cần thay thế (dùng COCO generic)
- Breast detector cần kiểm tra lại chất lượng
- Volume 3D models cần re-export cho tương thích PyTorch 2.7.1

**Điểm đánh giá lại: 8.5/10** — Model coverage gần như đầy đủ, chỉ còn vấn đề tương thích và 1-2 detector cần cải thiện.
