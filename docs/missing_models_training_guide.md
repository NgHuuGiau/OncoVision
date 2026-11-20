# Hướng Dẫn Training 6 Model Còn Thiếu — OncoVision

## Tổng Quan

OncoVision có 11/11 loại ung thư với classifier + detector hoạt động, nhưng **6 model còn thiếu pretrained weights chất lượng cao** cho các modality chuyên biệt. Tài liệu này hướng dẫn cách train và tích hợp các model còn thiếu.

## 6 Model Cần Bổ Sung

### 1. Ung Thư Vú — Mammogram Classifier

**Hiện trạng:** Có classifier ResNet50 nhưng chưa có pretrained weights chuyên biệt cho mammogram.

**Dataset đề xuất:**
- **DDSM** (Digital Database for Screening Mammography): ~3,300 trường hợp, MIT Open Access
- **INbreast**: 400 ảnh mammogram, UFRJ Brazil
- **BUSI** (Breast Ultrasound Images Dataset): 780 ảnh siêu âm vú

**Hướng dẫn train:**

```python
# scripts/train_breast_mammogram_classifier.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medical.cnn_classifier import MedicalCNNClassifierWrapper

# Dataset
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = MammogramDataset(root="data/dssm", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = MedicalCNNClassifierWrapper(
    backbone="resnet50",
    num_classes=3,  # normal, benign, malignant
    head="mlp",
)

# Train
trainer = MedicalTrainer(model, loader, epochs=50, device="cuda")
trainer.fit()

# Save
torch.save({
    "model_state_dict": model.state_dict(),
    "class_labels": ["normal", "benign", "malignant"],
    "num_classes": 3,
    "backbone": "resnet50",
    "head": "mlp",
    "image_size": 320,
    "temperature": 1.0,
}, "models/pretrained/cancers/breast/mammogram/classifier.pt")
```

**Đường dẫn đích:** `models/pretrained/cancers/breast/mammogram/classifier.pt`

---

### 2. Ung Thư Đại Trực Tràng — Endoscopy Classifier

**Hiện trạng:** Có classifier 9 lớp nhưng chất lượng chưa ổn định với endoscopy thực tế.

**Dataset đề xuất:**
- **Kvasir v2**: 1,000 ảnh nội soi, 8 lớp bệnh lý
- **CVC-ClinicDB**: 1,000 ảnh polyp đại tràng
- **ETIS-Larib Polyp Database**: 196 ảnh nội soi

**Hướng dẫn train:**

```python
# scripts/train_colorectal_endoscopy_classifier.py
from medical.cnn_classifier import MedicalCNNClassifierWrapper

model = MedicalCNNClassifierWrapper(
    backbone="convnext_tiny",
    num_classes=9,  # normal, epithelium, dysplasia, neoplasia, polyp, hyperplastic_polyp, sessile_serrated, adenoma, other
    head="mlp",
)

# Train với Kvasir v2 + CVC-ClinicDB combined
# ...

torch.save({
    "model_state_dict": model.state_dict(),
    "class_labels": ["normal", "epithelium", "dysplasia", "neoplasia", "polyp", "hyperplastic_polyp", "sessile_serrated", "adenoma", "other"],
    "num_classes": 9,
    "backbone": "convnext_tiny",
    "head": "mlp",
    "image_size": 320,
}, "models/pretrained/cancers/colorectal/endoscopy/classifier.pt")
```

**Đường dẫn đích:** `models/pretrained/cancers/colorectal/endoscopy/classifier.pt`

---

### 3. Ung Thư Thận — Ultrasound Classifier

**Hiện trạng:** Có classifier CT nhưng chưa có cho siêu âm thận.

**Dataset đề xuất:**
- **BreakHis** (không phải thận, nhưng có thể transfer learning)
- **Dữ liệu bệnh viện**: Cần thu thập từ cơ sở y tế

**Lưu ý:** Dataset siêu âm thận công khai rất hạn chế. Cần hợp tác với bệnh viện để có dữ liệu.

**Hướng dẫn train (khi có dữ liệu):**

```python
# scripts/train_kidney_ultrasound_classifier.py
from medical.cnn_classifier import MedicalCNNClassifierWrapper

model = MedicalCNNClassifierWrapper(
    backbone="resnet18",
    num_classes=3,  # benign, malignant, cyst
    head="mlp",
)

# Train với dữ liệu siêu âm thận
# ...

torch.save({
    "model_state_dict": model.state_dict(),
    "class_labels": ["benign", "malignant", "cyst"],
    "num_classes": 3,
    "backbone": "resnet18",
    "head": "mlp",
    "image_size": 320,
}, "models/pretrained/cancers/kidney/ultrasound/classifier.pt")
```

**Đường dẫn đích:** `models/pretrained/cancers/kidney/ultrasound/classifier.pt`

---

### 4. Ung Thư Gan — MRI Classifier

**Hiện trạng:** Có classifier CT nhưng chưa có cho MRI gan.

**Dataset đề xuất:**
- **LiTS** (Liver Tumor Segmentation Challenge): 131 ca CT, có thể transfer learning sang MRI
- **MSD Liver**: 50 ca MRI gan từ Medical Segmentation Decathlon
- **Dữ liệu bệnh viện**: Cần hợp tác để có MRI gan có nhãn

**Hướng dẫn train:**

```python
# scripts/train_liver_mri_classifier.py
from medical.cnn_classifier import MedicalCNNClassifierWrapper

model = MedicalCNNClassifierWrapper(
    backbone="resnet50",
    num_classes=2,  # benign, malignant
    head="mlp",
)

# Train với MSD Liver MRI
# ...

torch.save({
    "model_state_dict": model.state_dict(),
    "class_labels": ["benign", "malignant"],
    "num_classes": 2,
    "backbone": "resnet50",
    "head": "mlp",
    "image_size": 320,
}, "models/pretrained/cancers/liver/mri/classifier.pt")
```

**Đường dẫn đích:** `models/pretrained/cancers/liver/mri/classifier.pt`

---

### 5. Ung Thư Tiền Liệt — Ultrasound Classifier

**Hiện trạng:** Có classifier MRI nhưng chưa có cho siêu âm tuyến tiền liệt.

**Dataset đề xuất:**
- **Prostate MRI Dataset** (transfer learning sang ultrasound)
- **Dữ liệu bệnh viện**: Cần hợp tác để có siêu âm tuyến tiền liệt có nhãn

**Hướng dẫn train:**

```python
# scripts/train_prostate_ultrasound_classifier.py
from medical.cnn_classifier import MedicalCNNClassifierWrapper

model = MedicalCNNClassifierWrapper(
    backbone="resnet18",
    num_classes=2,  # benign, malignant
    head="mlp",
)

# Train với dữ liệu siêu âm tuyến tiền liệt
# ...

torch.save({
    "model_state_dict": model.state_dict(),
    "class_labels": ["benign", "malignant"],
    "num_classes": 2,
    "backbone": "resnet18",
    "head": "mlp",
    "image_size": 320,
}, "models/pretrained/cancers/prostate/ultrasound/classifier.pt")
```

**Đường dẫn đích:** `models/pretrained/cancers/prostate/ultrasound/classifier.pt`

---

### 6. Ung Thư Cổ Tử Cung — Pap Smear Detector

**Hiện trạng:** Có detector colposcopy nhưng chưa có cho Pap smear (phết cổ tử cung).

**Dataset đề xuất:**
- **C-NPC**: Cervical Neoplasia Prediction Challenge, ~10,000 ảnh tế bào
- **Herzog Dataset**: 917 ảnh tế bào cổ tử cung, 3 lớp
- **Pap-Pak Dataset**: 17,209 tế bào, 4 lớp

**Hướng dẫn train:**

```python
# scripts/train_cervical_pap_detector.py
from ultralytics import YOLO

# Khởi tạo model YOLO
model = YOLO("yolo11n.pt")

# Train với C-NPC dataset
model.train(
    data="data/cnpc/data.yaml",
    epochs=100,
    imgsz=320,
    batch=32,
    device="cuda",
)

# Export
model.export(format="torchscript")
```

**Đường dẫn đích:** `models/pretrained/cancers/cervical/pap_smear/detector.pt`

---

## Quy Trình Train Chung

### Bước 1: Chuẩn Bị Dữ Liệu

```bash
# Tạo thư mục dataset
mkdir -p data/dataset_name/train/{normal,benign,malignant}
mkdir -p data/dataset_name/val/{normal,benign,malignant}
mkdir -p data/dataset_name/test/{normal,benign,malignant}

# Chia dữ liệu theo bệnh nhân (không theo ảnh!)
python scripts/split_dataset.py \
    --source data/raw \
    --output data/dataset_name \
    --patient-id-col patient_id \
    --test-size 0.2 \
    --val-size 0.1
```

### Bước 2: Train Model

```bash
# Train classifier
python scripts/train_classifier.py \
    --dataset data/dataset_name \
    --backbone resnet50 \
    --num-classes 3 \
    --epochs 50 \
    --batch-size 32 \
    --output models/pretrained/cancers/<key>/<modality>/classifier.pt

# Train detector
python scripts/train_detector.py \
    --data data/dataset_name/data.yaml \
    --model yolo11n \
    --epochs 100 \
    --imgsz 320 \
    --output models/pretrained/cancers/<key>/<modality>/detector.pt
```

### Bước 3: Đánh Giá Chất Lượng

```python
from medical.clinical_validation import build_independent_validation_report

report = build_independent_validation_report(
    csv_path="data/dataset_name/test_results.csv",
    model_path="models/pretrained/cancers/<key>/<modality>/classifier.pt",
    output_path="output/validation/<key>_<modality>_report.json",
    cohort_id="<KEY>_<MODALITY>_2026",
)
```

### Bước 4: Cập Nhật Catalog

```python
# models/model_catalog.json — thêm entry mới
{
    "key": "breast",
    "type": "classifier",
    "modality": "mammogram",
    "path": "cancers/breast/mammogram/classifier.pt",
    "size_bytes": 94371473,
    "framework": "torch-state",
    "labels": ["normal", "benign", "malignant"],
    "input_size": 320,
    "backbone": "resnet50",
    "status": "ready",
    "notes": "Train từ DDSM + INbreast, accuracy 92.3%"
}
```

## Yêu Cầu Phần Cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| **GPU** | RTX 3060 12GB | RTX 4090 24GB |
| **RAM** | 16GB | 32GB |
| **Disk** | 100GB trống | 500GB trống |
| **PyTorch** | 2.1+ CUDA | 2.7+ CUDA |

## Timeline Ước Lượng

| Model | Dữ liệu | Train time | Tổng cộng |
|---|---|---|---|
| Breast mammogram | DDSM (công khai) | 2-4 giờ | 1-2 ngày |
| Colorectal endoscopy | Kvasir (công khai) | 1-2 giờ | 1 ngày |
| Kidney ultrasound | Cần thu thập | — | 2-4 tuần |
| Liver MRI | MSD (công khai) | 3-6 giờ | 2-3 ngày |
| Prostate ultrasound | Cần thu thập | — | 2-4 tuần |
| Cervical Pap smear | C-NPC (công khai) | 2-4 giờ | 1-2 ngày |

## Lưu Ý Quan Trọng

1. **Dataset công khai** không thay thế được dữ liệu lâm sàng thực tế
2. **Cần phê duyệt đạo đức** trước khi sử dụng dữ liệu bệnh nhân
3. **Clinical validation** là bắt buộc sau khi train xong
4. **License dataset** phải cho phép sử dụng thương mại nếu deploy sản phẩm
5. **Transfer learning** từ model có sẵn giúp giảm thời gian train

## Tham Khảo

- `models/README.md`: Quy ước checkpoint và hợp đồng runtime
- `medical/cnn_classifier.py`: Architecture classifier
- `docs/clinical_validation_guide.md`: Hướng dẫn clinical validation
- `scripts/import_foreign_model.py`: Import model từ nguồn ngoài
