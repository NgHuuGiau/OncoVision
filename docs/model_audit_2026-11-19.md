# Báo cáo kiểm tra model AI inference — 11 ung thư chính

**Ngày:** 2026-11-19  
**Mục đích:** Đánh giá đầy đủ model AI inference, tìm chỗ yếu cần bổ sung, dọn dẹp file không cần thiết.

---

## 1. Tổng quan

| Tiêu chí | Kết quả |
|---|---|
| **Tổng model files** | 95 files (~6 GB) |
| **11 ung thư chính** | 11/11 có runtime_ready ✅ |
| **Classifier** | 11/11 hoạt động ✅ |
| **Detector** | 11/11 hoạt động ✅ |
| **Volume 3D (.pt2)** | 0/11 load được ❌ (PyTorch version mismatch) |
| **File trùng SHA256** | 3 nhóm nhỏ (không nghiêm trọng) |
| **File rỗng** | 0 file model rỗng ✅ |

---

## 2. Bảng tổng hợp 11 ung thư chính

| Ung thư | Classifier | Detector | Volume 3D | Modality có model | Ghi chú |
|---|---|---|---|---|---|
| **🧠 Não** (brain) | ✅ | ✅ | ❌ | MRI | Backbone Swin mismatch |
| **🎀 Vú** (breast) | ✅ | ✅ | ❌ | Ultrasound, Mammogram(detector only) | Thiếu mammogram classifier |
| **🫁 Phổi** (lung) | ✅ | ✅ | ❌ | CT, X-ray | Đầy đủ |
| **🦋 Tuyến giáp** (thyroid) | ✅ | ✅ | ❌ | Ultrasound | Backbone ConvNeXt mismatch |
| **🫘 Thận** (kidney) | ✅ | ✅ | ❌ | CT | Thiếu ultrasound |
| **🫀 Gan** (liver) | ✅ | ✅ | ❌ | CT | Backbone Swin mismatch, thiếu MRI |
| **🍽️ Dạ dày** (stomach) | ✅ | ✅ | ❌ | Endoscopy | Đầy đủ |
| **🦴 Đại trực tràng** (colorectal) | ✅ | ✅ | ❌ | Colonoscopy, Endoscopy(detector only) | Thiếu endoscopy classifier |
| **👩 Cổ tử cung** (cervical) | ✅ | ✅ | ❌ | Colposcopy, Pap(classifier only) | Thiếu pap detector |
| **👨 Tiền liệt** (prostate) | ✅ | ✅ | ❌ | MRI | Thiếu ultrasound |
| **🟡 Tụy** (pancreas) | ✅ | ✅ | ❌ | CT | Đầy đủ |

---

## 3. Các chỗ yếu cần bổ sung

### Mức độ nghiêm trọng: Cao (ảnh hưởng demo)

| # | Ung thư | Modality | Thiếu | Mô tả |
|---|---|---|---|---|
| 1 | **Breast** | Mammogram | Classifier | Chỉ có detector, không có classifier cho mammogram |
| 2 | **Colorectal** | Endoscopy | Classifier | Chỉ có detector, không có classifier cho endoscopy |

### Mức độ nghiêm trọng: Trung bình (thiếu modality)

| # | Ung thư | Modality | Thiếu | Mô tả |
|---|---|---|---|---|
| 3 | **Kidney** | Ultrasound | Classifier + Detector | Thư mục ultrasound trống, chỉ có CT |
| 4 | **Liver** | MRI | Classifier + Detector | Thư mục MRI trống, chỉ có CT |
| 5 | **Prostate** | Ultrasound | Classifier + Detector | Thư mục ultrasound trống, chỉ có MRI |
| 6 | **Cervical** | Pap smear | Detector | Chỉ có classifier, không có detector cho pap smear |

---

## 3.1 Kết quả tìm kiếm model pretrained miễn phí

**Ngày tìm kiếm:** 2026-11-19  
**Nguồn:** HuggingFace, GitHub, Kaggle, PMC papers, Roboflow Universe

| # | Chỗ yếu | Kết quả tìm kiếm | Lý do không có |
|---|---|---|---|
| 1 | **Breast mammogram classifier** | ❌ Không có | Các model tìm được là TensorFlow (Mammo-FM, mammoscreen), Vision-Language models (MedGemma), hoặc cần training từ đầu. Không có model PyTorch 2D classification pretrained sẵn. |
| 2 | **Colorectal endoscopy classifier** | ❌ Không có | Model tìm được là pathology classification (SPIDER-colorectal — histopathology, không phải endoscopy), segmentation (Duck-Net polyp), hoặc foundation model (WOMA — cần fine-tune). Không có model 2D endoscopy classification pretrained. |
| 3 | **Kidney ultrasound classifier** | ❌ Không có | Model tìm được là 3D CT (RenalCLIP — Nature Communications), kidney disease chung (không phải cancer), hoặc dataset (RadImagenet). Không có model 2D ultrasound cancer classification. |
| 4 | **Liver MRI classifier** | ❌ Không có | Model tìm được là 3D MRI foundation (Triad, Decipher-MR), CT classification (seongun/liver-tumor-classification), hoặc segmentation (SWTR-Unet). Không có model 2D MRI cancer classification pretrained. |
| 5 | **Prostate ultrasound classifier** | ❌ Không có | Model tìm được là MRI (panorgan-oncology-prostate-mri-classifier), WSI pathology (self-supervised-prostate-cancer), hoặc 3D video (I3D ultrasound — cần training). Không có model 2D ultrasound cancer classification. |
| 6 | **Cervical pap smear detector** | ❌ Không có | Paper CerviCell-detector mô tả YOLOv5 cho pap smear nhưng không release weights. Chỉ có YOLO generic COCO model. Cần train từ dataset pap smear có bounding box. |

### Kết luận

**Không có model PyTorch pretrained miễn phí cho 6 chỗ yếu.** Đây là các task chuyên biệt y khoa mà cộng đồng open-source chưa release weights công khai. Các giải pháp:

1. **Giữ nguyên hiện tại** — 11/11 ung thư vẫn có classifier + detector hoạt động trên modality chính
2. **Train từ dataset** — cần GPU + dataset chuyên biệt cho từng modality
3. **Liên hệ tác giả papers** — xin weights từ các paper đã publish

---

## 4. Vấn đề đã biết (không ảnh hưởng demo 2D)

### 4.1. Backbone mismatch (3 ung thư)

| Ung thư | Metadata backbone | Thực tế backbone | Ảnh hưởng |
|---|---|---|---|
| **Brain** | ResNet50 | Swin Transformer | ⚠️ Warning log, vẫn hoạt động |
| **Thyroid** | ResNet50 | ConvNeXt-Tiny | ⚠️ Warning log, vẫn hoạt động |
| **Liver** | ResNet50 | Swin Tiny | ⚠️ Warning log, vẫn hoạt động |

**Nguyên nhân:** Model được train với backbone khác nhưng metadata ghi sai. Hệ thống tự động phát hiện và load backbone đúng.

### 4.2. Volume 3D (.pt2) — 100% fail

| Vấn đề | Chi tiết |
|---|---|
| **Số file** | 24 file `.pt2` |
| **Lỗi** | `torch.export.load` fail → `TorchScript` fail → fallback fail |
| **Nguyên nhân** | PyTorch version mismatch — model export với PyTorch cũ, không tương thích PyTorch 2.7.1 |
| **Giải pháp** | Cần re-export models với PyTorch phiên bản hiện tại |

### 4.3. File trùng SHA256 (3 nhóm)

| Nhóm | File trùng | Kích thước | Đánh giá |
|---|---|---|---|
| 1 | `lung/ct/detector.pt` = `lung/xray/detector.pt` | 5.3 MB | ⚠️ Cùng 1 file cho 2 modality khác nhau |
| 2 | `colorectal/endoscopy/detector_yolov11m.pt` = `stomach/endoscopy/detector.pt` | 38.6 MB | ⚠️ Cùng 1 detector cho 2 ung thư khác nhau |
| 3 | `lung/volume.pt` = `lung_nodule_ct_detection/models/model.pt` | 79.8 MB | ✅ OK — backup của cùng model |

---

## 5. Dọn dẹp đã thực hiện

| Thành phần | Số lượng | Dung lượng |
|---|---|---|
| Thư mục modality rỗng | 3 (liver/mri, prostate/ultrasound, kidney/ultrasound) | 0 MB |
| File `.gitkeep` | 4 files | 0 MB |
| Thư mục `.cache` rỗng | Đã tự dọn | 0 MB |

---

## 6. Đánh giá tổng thể

### ✅ Điểm mạnh

- **11/11 ung thư có classifier + detector hoạt động** — đủ để demo cơ bản
- **Không có file model rỗng** — tất cả model đều có nội dung
- **File trùng ít** — chỉ 3 nhóm nhỏ, không nghiêm trọng
- **Detector mới train** — thyroid (TN5000, mAP 87.4%) và lung (Kaggle, mAP 97.5%) đã train và deploy

### ⚠️ Điểm yếu

- **6 chỗ thiếu model** — 2 chỗ ảnh hưởng demo (breast mammogram classifier, colorectal endoscopy classifier)
- **Volume 3D không hoạt động** — 100% fail do PyTorch version mismatch
- **Backbone mismatch** — 3 ung thư có warning log khi load
- **Thiếu metric đánh giá** — không có AUC, sensitivity, specificity cho từng model

---

## 7. Đề xuất ưu tiên

### Ưu tiên cao (ảnh hưởng demo)

1. **Tìm hoặc train breast mammogram classifier** — mammogram là modality chính cho ung thư vú
2. **Tìm hoặc train colorectal endoscopy classifier** — endoscopy là modality chính cho đại trực tràng

### Ưu tiên trung bình (bổ sung modality)

3. **Tìm kidney ultrasound model** — siêu âm thận là modality phổ biến
4. **Tìm liver MRI model** — MRI gan cho chất lượng ảnh cao hơn
5. **Tìm prostate ultrasound model** — siêu âm tiền liệt là modality screening phổ biến
6. **Tìm cervical pap smear detector** — pap smear là screening tiêu chuẩn

### Ưu tiên thấp (cải thiện chất lượng)

7. **Re-export Volume 3D models** — cần PyTorch phiên bản tương thích
8. **Sửa metadata backbone** — cập nhật đúng backbone thực tế cho brain, thyroid, liver
9. **Thêm manifest + metric** — ghi rõ dataset, AUC, sensitivity cho từng model
10. **Xử lý file trùng** — train riêng detector cho lung CT vs X-ray, colorectal vs stomach

---

## 8. Model 3D

**Trạng thái:** Đã dọn dẹp hoàn toàn, sẵn sàng cho model 3D mới.

- Đã xóa 19 file GLB cũ (~735 MB)
- Đã xóa code 3D viewer, CSS, test pages
- Đã xóa `model_blender/` module
- Khi có model 3D mới, sẽ tích hợp lại từ đầu

---

## 9. Dung lượng dự án

| Thành phần | Dung lượng |
|---|---|
| Model AI inference | ~6.0 GB (95 files) |
| Dataset train | ~2.9 GB (TN5000 + lung nodule) |
| Code + templates + static | ~50 MB |
| Output (DB + logs) | ~20 MB |
| **Tổng** | **~9.0 GB** |
