# Model Updates - 2026-11-20

Thư mục này chứa các model mới được tải từ HuggingFace để bổ sung cho dự án OncoVision demo.

## Model mới đã tải

### 1. Breast Mammogram (Ung thư vú - Mammogram)

| File | Nguồn | Kích thước | Mô tả |
|---|---|---|---|
| `breast/mammogram/classifier.pt` | PranayPalem/BYOL_Mammogram | 527.80 MB | BYOL ResNet50 self-supervised classifier cho mammogram |
| `breast/mammogram/detector_yolo.pt` | hab200/Breast-Cancer-Mammogram-Detection-Segmentation | 18.29 MB | YOLO11s detector cho mammogram lesion detection |

**Lưu ý:** Detector cũ (`detector.pt`) chỉ có 1 lớp 'item' với chất lượng thấp. Detector mới (`detector_yolo.pt`) được tải từ HuggingFace với chất lượng tốt hơn.

### 2. Colorectal Endoscopy (Ung thư đại trực tràng - Nội soi)

| File | Nguồn | Kích thước | Mô tả |
|---|---|---|---|
| `colorectal/endoscopy/classifier.pt` | umairinayat/medical-models | 188.79 MB | Classifier cho endoscopy colorectal |
| `colorectal/endoscopy/detector_rtdetr.pt` | sdoerrich97/true_colon_rtdetr_realcolon_s123 | 125.95 MB | RT-DETR detector cho polyp detection |

### 3. Cervical Pap Smear (Ung thư cổ tử cung - Pap smear)

| File | Nguồn | Kích thước | Mô tả |
|---|---|---|---|
| `cervical/pap/classifier.pt` | ethioel/cerviscan-b0 | 16.83 MB | CerviScan AI EfficientNet-B0 cho 5 lớp Pap smear cells |

**Classes:** dyskeratotic, koilocytotic, metaplastic, parabasal, superficial_intermediate

**Lưu ý:** Không tìm thấy detector model cho Pap smear trên HuggingFace.

### 4. Kidney MRI (Ung thư thận - MRI)

| File | Nguồn | Kích thước | Mô tả |
|---|---|---|---|
| `kidney/mri/classifier.pt` | theshoaibme/panorgan-nephrology-renal-ct-mri-classifier | 77.83 MB | EfficientNet-v2-s classifier cho renal CT/MRI |

**Classes:** ct, mri (modality classification)

### 5. Prostate Ultrasound (Ung thư tuyến tiền liệt - Siêu âm)

| File | Nguồn | Kích thước | Mô tả |
|---|---|---|---|
| `prostate/ultrasound/classifier.pt` | pimed/ProViCNet | 171.48 MB | ProViCNet TRUS classifier cho prostate ultrasound |

**Classes:** background, normal, indolent_cancer, significant_cancer

## Model không tìm thấy

- **Cervical Pap smear detector** - Không có model YOLO/detector cho Pap smear trên HuggingFace
- **Liver MRI models** - Chỉ có model cho CT, không có model pretrained cho MRI

## Tổng kết

| Thành phần | Số lượng | Tổng kích thước |
|---|---|---|
| Model mới tải | 7 | ~1.1 GB |
| Model cũ | 33 | ~1.9 GB |
| Tổng cộng | 40 | ~3.0 GB |

## Verification

Smoke test `entrypoints/run_smoke.py`: PASS 4/4

## Tài liệu tham khảo

- [PranayPalem/BYOL_Mammogram](https://huggingface.co/PranayPalem/BYOL_Mammogram)
- [hab200/Breast-Cancer-Mammogram-Detection-Segmentation](https://huggingface.co/hab200/Breast-Cancer-Mammogram-Detection-Segmentation)
- [umairinayat/medical-models](https://huggingface.co/umairinayat/medical-models)
- [sdoerrich97/true_colon_rtdetr_realcolon_s123](https://huggingface.co/sdoerrich97/true_colon_rtdetr_realcolon_s123)
- [ethioel/cerviscan-b0](https://huggingface.co/ethioel/cerviscan-b0)
- [theshoaibme/panorgan-nephrology-renal-ct-mri-classifier](https://huggingface.co/theshoaibme/panorgan-nephrology-renal-ct-mri-classifier)
- [pimed/ProViCNet](https://huggingface.co/pimed/ProViCNet)
