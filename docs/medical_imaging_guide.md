# Hướng Dẫn Luồng Y Dược

Tài liệu giải thích luồng y tế của OncoVision: dataset, entrypoint, lệnh CLI, cấu hình training và debug.

---

## 1. Tổng quan

Nhánh y dược phục vụ bốn việc chính:

1. Quản lý dataset y tế (tổ chức, kiểm tra, báo cáo)
2. Theo dõi trạng thái model và output medical
3. Phân tích ảnh y khoa và lưu trữ ca bệnh
4. Hỗ trợ chat AI kiểm tra trạng thái y dược

---

## 2. Entrypoint liên quan

| File | Vai trò |
|---|---|
| `run_medical.py` | CLI chính — dataset, model, training, modality |
| `run_chat.py` | Chat UI — kiểm tra preflight, tích hợp medical pipeline |
| `run_doctor.py` | Quét tổng thể model, dataset, output |

---

## 3. Lệnh CLI

```powershell
python run_medical.py status          # Trạng thái tổng quan
python run_medical.py ready           # Kiểm tra đủ điều kiện train
python run_medical.py sources         # Liệt kê nguồn ảnh
python run_medical.py cancer          # Danh sách nhóm ung thư
python run_medical.py init-dataset    # Khởi tạo layout dataset
python run_medical.py train-modality  # Train classifier modality
python run_medical.py train-cancer    # Train CNN classification
```

Lưu ý: `init-dataset` chỉ in layout mong đợi, không tự tạo dữ liệu.

---

## 4. Nhóm bệnh và modality hỗ trợ

| Nhóm | Ảnh / volume thường dùng |
|---|---|
| Gan | Siêu âm, CT, MRI, PET/CT |
| Phổi | X-quang ngực, CT ngực, PET/CT |
| Vú | Mammogram, siêu âm vú, MRI vú |
| Dạ dày | Nội soi, CT, MRI, PET, EUS |
| Đại trực tràng | Nội soi đại tràng, CT bụng-chậu, MRI trực tràng, PET |
| Tuyến tiền liệt | MRI tuyến tiền liệt, siêu âm, PET/CT |
| Cổ tử cung | MRI, CT, PET/CT |

### Định dạng ảnh hỗ trợ

- **JPG / PNG**: ảnh thông thường
- **DICOM**: file `.dcm` và series DICOM
- **NIfTI**: volume `.nii` / `.nii.gz`
- **Pap/HPV, soi cổ tử cung, sinh thiết**: đầu vào lâm sàng, không hỗ trợ upload trực tiếp

Chat UI cho phép chọn nhóm bệnh và modality để lọc file picker phù hợp.

---

## 5. Dataset modality

Dataset `dataset/medical_modality/` dùng để train classifier phân loại modality ảnh y khoa:

| Modality | Số ảnh | Nguồn |
|---|---|---|
| CT | 200 | OrganMNIST3D |
| MRI | 200 | OrganMNIST3D |
| X-quang | 200 | ChestMNIST |
| Mammogram | 200 | BreastMNIST |
| Nội soi | 200 | PathMNIST |
| Siêu âm | 200 | BloodMNIST |
| PET/CT | 200 | OrganMNIST3D + augment |
| EUS | 200 | PathMNIST + augment |

- Tổng: 1.600 ảnh (200 × 8 modality), chuẩn hóa 224×224 RGB
- Nguồn: [MedMNIST](https://medmnist.com/) (BSD license)
- Sinh lại: `python scripts/build_modality_dataset.py`
- Train: `python run_medical.py train-modality --epochs 12`

---

## 6. Module chính trong `medical/`

| Module | Trách nhiệm |
|---|---|
| `dataset.py` | Tạo và kiểm tra layout dataset |
| `system_status.py` | Tổng hợp trạng thái medical |
| `training.py` | Audit, split, train, validate |
| `output_management.py` | Quản lý output medical |
| `storage.py` | Lưu và truy vấn case DB |
| `validator.py` | Kiểm tra ảnh đầu vào |
| `cli_helpers.py` | Helper in trạng thái CLI |

---

## 7. Cấu hình training

### CNN Classification (ung thư)

| Tham số | Giá trị | Mục đích |
|---|---|---|
| Backbone | efficientnet_b2 | Nhẹ, hiệu năng cao |
| Image size | 288px | Chi tiết tốt, vừa VRAM 4GB |
| Batch size | 6 | An toàn với RTX 3050 Ti |
| Epochs | 30 | Đủ hội tụ |
| Loss | Focal Loss | Tập trung hard examples |
| Learning rate | 0.0001 | Ổn định |
| Warmup | 4 epochs | Tránh shock đầu train |
| Gradient accumulation | 4 | Effective batch = 24 |
| Class weights | Auto | Cân bằng mất cân bằng lớp |
| EMA | Có | Stabilize training |
| TTA | Có | Test time augmentation |

### Ngưỡng quyết định

| Ngưỡng | Giá trị | Mục đích |
|---|---|---|
| High risk | 0.35 | Ưu tiên recall, tránh bỏ sót |
| Medium risk | 0.25 | Cảnh báo sớm |
| Certainty | 0.30 | Threshold phân loại chung |

### YOLO Detection

| Tham số | Giá trị | Mục đích |
|---|---|---|
| Model | yolo11s | Nhẹ, phù hợp 4GB VRAM |
| Epochs | 50 | Train sâu |
| Imgsz | 512 | Chi tiết cao |
| Batch | 4 | An toàn VRAM |
| Optimizer | AdamW | Ổn định |
| Mosaic | 0.3 | Giảm nhiễu giải phẫu |
| Mixup | 0.1 | Tăng diversity |
| Copy-paste | 0.1 | Tăng diversity |
| Lật lên-xuống | 0.0 | Giữ hướng giải phẫu |

---

## 8. Debug theo triệu chứng

| Triệu chứng | Mở đầu tiên |
|---|---|
| `run_chat.py --check-only` fail | `utils/entrypoint_checks.py`, `medical/system_status.py` |
| Medical status sai | `medical/system_status.py`, `medical/model_policy.py` |
| Count train/val không đúng | `medical/training.py`, `medical/status_helpers.py` |
| Train medical fail | `medical/training.py`, `run_medical.py train` |
| CNN ảnh đầu vào lỗi | `medical/validator.py` |
