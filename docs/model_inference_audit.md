# Báo cáo kiểm tra model AI inference — 11 ung thư chính

> Ngày kiểm tra: 18/11/2026  
> Phương pháp: SHA256 hash + runtime probe (cancer_model_registry.py)  
> Phạm vi: `models/pretrained/cancers/` — 82 file model (.pt, .pth, .safetensors, .pt2)

---

## 1. Tổng quan kết quả

| Ung thư | Classifier | Detector | Volume 3D | Files | Modalities | Đánh giá |
|---|---|---|---|---|---|---|
| 🧠 **Não** (brain) | ✅ | ✅ | ❌ | 8 | MRI | **TỐ** — Classifier + detector hoạt động |
| 🎀 **Vú** (breast) | ✅ | ✅ | ❌ | 14 | Mammogram, Ultrasound | **TỐ** — 2 modality, nhưng classifier trùng |
| 🫁 **Phổi** (lung) | ✅ | ❌ | ❌ | 7 | CT, X-ray | **TRUNG BÌNH** — Detector INVALID (COCO generic) |
| 🦋 **Tuyến giáp** (thyroid) | ✅ | ❌ | ❌ | 7 | Ultrasound | **TRUNG BÌNH** — Detector INVALID (COCO generic) |
| 🫘 **Thận** (kidney) | ✅ | ✅ | ❌ | 8 | CT, Ultrasound | **TỐ** — Nhưng có file rỗng + detector trùng |
| 🫀 **Gan** (liver) | ✅ | ✅ | ❌ | 11 | CT | **TỐ** — Backbone mismatch nhưng tự fix |
| 🍽️ **Dạ dày** (stomach) | ✅ | ✅ | ❌ | 6 | Endoscopy | **TỐ** — Detector dùng license AGPL-3.0 |
| 🦴 **Đại trực tràng** (colorectal) | ✅ | ✅ | ❌ | 7 | Colonoscopy, Endoscopy | **TỐ** — Detector trùng với stomach |
| 👩 **Cổ tử cung** (cervical) | ✅ | ✅ | ❌ | 13 | Colposcopy, Pap | **TỐ** — Nhưng 6 file safetensors trùng nhau |
| 👨 **Tiền liệt** (prostate) | ✅ | ✅ | ❌ | 6 | MRI | **TỐ** — Model hoạt động ổn |
| 🟡 **Tụy** (pancreas) | ✅ | ✅ | ❌ | 6 | CT | **TỐ** — Model hoạt động ổn |

---

## 2. Vấn đề nghiêm trọng cần xử lý

### 2.1. File rỗng

| File | Kích thước | Ung thư | Hành động |
|---|---|---|---|
| `kidney/ct/kidney_ann.pth` | **0 MB** | Thận | **Xóa ngay** — file rỗng, không dùng được |

### 2.2. Detector INVALID — COCO generic

2 detector dùng COCO generic (80 labels tổng quát), hệ thống tự động bỏ qua:

| File | Ung thư | Labels | Hành động |
|---|---|---|---|
| `thyroid/ultrasound/detector.pt` (5.4 MB) | Tuyến giáp | 80 labels COCO | **Thay thế** — cần detector chuyên biệt (TN3K, DDTI dataset) |
| `lung/ct/detector.pt` (21.6 MB) | Phổi CT | 14 labels không y khoa | **Thay thế** — cần detector lung nodule |
| `lung/xray/detector.pt` (21.6 MB) | Phổi X-ray | 14 labels không y khoa | **Thay thế** — cùng file với CT |

> ⚠️ **Lưu ý**: Lung CT và X-ray detector là cùng 1 file (SHA256 trùng).

### 2.3. Volume 3D (.pt2) — 100% fail

**Tất cả file .pt2 không load được** do PyTorch version mismatch:

```
torch.export.load thất bại (KeyError), thử TorchScript fallback cho .pt2
TorchScript fallback thất bại (RuntimeError), thử regular checkpoint
```

| Ung thư | File .pt2 | Tình trạng |
|---|---|---|
| Tất cả 11 ung thư | ~24 file .pt2 | ❌ Không load được |

**Nguyên nhân**: File .pt2 được export bằng phiên bản PyTorch khác, không tương thích với PyTorch hiện tại.

**Giải pháp**: Re-export models từ source code gốc với PyTorch phiên bản hiện tại.

### 2.4. Backbone mismatch — Metadata sai

Hệ thống tự động phát hiện và sửa, nhưng vẫn gây cảnh báo:

| Ung thư | Metadata nói | Thực tế | Kết quả |
|---|---|---|---|
| Gan (liver) | resnet50 | swin_tiny | ⚠️ Load backbone riêng, size mismatch classifier |
| Tuyến giáp (thyroid) | resnet50 | convnext_tiny | ⚠️ 180 keys missing, 320 keys unexpected |
| Não (brain) | — | swin (raw format) | ⚠️ 171 keys missing, 231 keys unexpected |
| Cervical/Pap | resnet50 | efficientnet-v2 | ⚠️ 311 keys missing, 362 keys unexpected |

> Hệ thống vẫn load được classifier sau khi tự động điều chỉnh, nhưng kết quả có thể không chính xác 100%.

---

## 3. File trùng nhau (SHA256)

### 3.1. Nhóm 1: Breast classifier (90 MB × 2 = 180 MB)

| File | Hash |
|---|---|
| `breast/mammogram/classifier.pt` | 09F4446D16291A79... |
| `breast/ultrasound/classifier.pt` | 09F4446D16291A79... |

**Vấn đề**: Classifier mammogram và ultrasound là cùng 1 file — không nên dùng chung model cho 2 modality khác nhau.

### 3.2. Nhóm 2: Cervical safetensors (101.7 MB × 6 = 610 MB)

| File | Hash |
|---|---|
| `cervical/colposcopy/cervical_alwaly.safetensors` | 9CEBF5B2B4B07292... |
| `cervical/colposcopy/model.safetensors` | 9CEBF5B2B4B07292... |
| `cervical/colposcopy/model_cervical.safetensors` | 9CEBF5B2B4B07292... |
| `cervical/endoscopy/cervical_alwaly.safetensors` | 9CEBF5B2B4B07292... |
| `cervical/endoscopy/model.safetensors` | 9CEBF5B2B4B07292... |
| `cervical/endoscopy/model_cervical.safetensors` | 9CEBF5B2B4B07292... |

**Vấn đề**: 6 file là bản sao của nhau — chỉ cần giữ 1 file.

### 3.3. Nhóm 3: Volume .pt2 (94.4 MB × nhiều)

Nhiều file `volume.pt2` có cùng kích thước 94.4 MB — có thể là cùng 1 model generic.

### 3.4. Nhóm 4: Detector trùng

| File | Hash |
|---|---|
| `colorectal/endoscopy/detector_yolov11m.pt` | 101AFA3026DF6A8C... |
| `stomach/endoscopy/detector.pt` | 101AFA3026DF6A8C... |

**Vấn đề**: Detector colorectal và stomach là cùng 1 file (REAL-Colon YOLO11m).

### 3.5. Nhóm 5: Kidney detector (6 MB × 2)

| File | Hash |
|---|---|
| `kidney/ct/detector.pt` | 9772844CFB846DEA... |
| `kidney/ultrasound/detector.pt` | 9772844CFB846DEA... |

**Vấn đề**: Detector CT và ultrasound là cùng 1 file.

### 3.6. Nhóm 6: Lung detector (21.6 MB × 2)

| File | Hash |
|---|---|
| `lung/ct/detector.pt` | 005C47688C5F806B... |
| `lung/xray/detector.pt` | 005C47688C5F806B... |

**Vấn đề**: Detector CT và X-ray là cùng 1 file — và cả 2 đều INVALID.

### 3.7. Nhóm 7: Cervical volume model (2.1 MB × 2)

| File | Hash |
|---|---|
| `cervical/colposcopy/cervical_volume_model.pt2` | D751055B6305F3A3... |
| `cervical/endoscopy/cervical_volume_model.pt2` | D751055B6305F3A3... |

---

## 4. Nguồn model (từ .source.json)

| Ung thư | Nguồn | License | Ghi chú |
|---|---|---|---|
| Não (brain) | amcm/brain-tumor-yolo11n (HF) | Không nêu | YOLO11n, 3 lớp u não |
| Vú (breast) | Local converted | — | Classifier chuyển đổi từ safetensors |
| Phổi (lung) | HuggingFace + local | — | Classifier từ HF, detector INVALID |
| Tuyến giáp (thyroid) | Local converted | — | Classifier chuyển đổi, detector COCO |
| Thận (kidney) | HuggingFace | — | Có file rỗng |
| Gan (liver) | HuggingFace | — | Swin Transformer backbone |
| Dạ dày (stomach) | Local converted | — | Detector AGPL-3.0 ⚠️ |
| Đại trực tràng | HuggingFace | — | Detector trùng stomach |
| Cổ tử cung | HuggingFace | — | 6 file safetensors trùng |
| Tiền liệt | HuggingFace | — | MRI model |
| Tụy (pancreas) | HuggingFace | — | CT model |

> ⚠️ **Stomach detector**: License AGPL-3.0 — cần lưu ý trước khi phát hành sản phẩm thương mại.

---

## 5. Đánh giá chất lượng tổng thể

### 5.1. Mức độ sẵn sàng cho demo

| Tiêu chí | Đánh giá | Chi tiết |
|---|---|---|
| **Classifier coverage** | ⭐⭐⭐⭐ | 11/11 ung thư có classifier hoạt động |
| **Detector coverage** | ⭐⭐⭐ | 9/11 có detector hợp lệ, 2 INVALID |
| **Volume 3D** | ⭐ | 0/11 load được — 100% fail |
| **File integrity** | ⭐⭐⭐ | 1 file rỗng, 7 nhóm file trùng |
| **Backbone accuracy** | ⭐⭐⭐ | 4/11 có metadata sai, nhưng tự fix |
| **Documentation** | ⭐⭐⭐⭐ | 24 file .source.json với thông tin nguồn |
| **License compliance** | ⭐⭐⭐ | 1 file AGPL-3.0, 1 file không nêu license |

### 5.2. Kết luận

**Đủ để demo cơ bản**: ✅

- 11/11 ung thư có classifier hoạt động → có thể phân loại ảnh
- 9/11 có detector hợp lệ → có thể khoanh vùng tổn thương
- Hệ thống tự động xử lý backbone mismatch → không crash

**Chưa đủ cho sản xuất**: ❌

- Volume 3D hoàn toàn không hoạt động
- 2 detector INVALID (lung, thyroid)
- Nhiều file trùng — số "82 files" bị thổi phồng
- Không có metric đánh giá (AUC, sensitivity, specificity)
- Không có clinical validation độc lập

---

## 6. Đề xuất ưu tiên

### Ưu tiên cao — Xử lý ngay

1. **Xóa `kidney/ct/kidney_ann.pth`** (0 MB) — file rỗng gây nhầm lẫn
2. **Thay thế detector thyroid** — tìm model từ dataset TN3K hoặc DDTI
3. **Thay thế detector lung** — tìm model lung nodule detection chuyên biệt
4. **Xóa 5 file cervical safetensors trùng** — tiết kiệm ~508 MB

### Ưu tiên trung bình — Nên làm

5. **Tách breast classifier** — mammogram và ultrasound cần model riêng
6. **Cập nhật metadata backbone** cho liver, thyroid, brain, cervical
7. **Re-export volume .pt2** với PyTorch phiên bản hiện tại
8. **Thêm manifest** cho từng model với metric (AUC, sensitivity, specificity)

### Ưu tiên thấp — Có thể làm sau

9. **Kiểm tra license AGPL-3.0** cho stomach detector
10. **Thêm segmentation** cho lung, kidney, pancreas, prostate
11. **Thêm clinical validation** với bộ test set độc lập

---

## 7. Dung lượng disk

| Thành phần | Số file | Dung lượng | Ghi chú |
|---|---|---|---|
| Classifier (.pt) | 15 | ~750 MB | 11/11 hoạt động |
| Detector (.pt) | 18 | ~500 MB | 2 INVALID, 2 trùng |
| Volume 3D (.pt2) | 24 | ~2.2 GB | 100% không load được |
| Safetensors | 23 | ~1.5 GB | 6 file trùng |
| Alternative (.pth) | 10 | ~1.2 GB | Không dùng trong pipeline |
| **Tổng** | **82** | **~6.1 GB** | |

**Có thể tiết kiệm**: ~1.5 GB (xóa file trùng + file rỗng + volume .pt2 không dùng được)

---

## 8. So sánh với đánh giá trước

| Vấn đề | Trạng thái trước | Trạng thái hiện tại |
|---|---|---|
| File rỗng kidney_ann.pth | ⚠️ Đã phát hiện | ⚠️ Vẫn còn — cần xóa |
| Detector thyroid INVALID | ⚠️ Đã biết | ⚠️ Vẫn INVALID |
| Volume 3D fail | ⚠️ Đã biết | ⚠️ Vẫn 100% fail |
| File trùng cervical | ⚠️ Đã phát hiện | ⚠️ Vẫn còn 6 file trùng |
| Backbone mismatch | ⚠️ Đã biết | ⚠️ Vẫn có 4/11 |
| Clinical validation | ❌ Chưa có | ❌ Vẫn chưa có |
| Metric đánh giá | ❌ Chưa có | ❌ Vẫn chưa có |

> **Nhận xét**: Các vấn đề đã biết từ lần đánh giá trước vẫn chưa được xử lý. Cần ưu tiên xử lý các vấn đề ưu tiên cao để cải thiện chất lượng demo.
