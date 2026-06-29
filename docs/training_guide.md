# Huong Dan Training Object Detection

Tai lieu nay mo ta day du luong train YOLO object detection trong OncoVision, tu du lieu raw den model `best.pt` dua vao runtime camera.

## 1. Muc Tieu

Nhanh training duoc dung de:

- chuan bi dataset object detection,
- validate du lieu va label,
- split train/val/test,
- train model YOLO custom,
- validate va export model,
- dua model vao `run_app.py`.

## 2. Thu Muc Va Tep Lien Quan

```text
dataset/object_detection/raw/
  images/
  labels/

dataset/object_detection/processed/
  images/
    train/
    val/
    test/

training/
  data.yaml
  train_config.yaml
  prepare_dataset.py
  validate_dataset.py
  split_dataset.py
  train_model.py
  validate_model.py
  export_model.py
```

## 3. Dau Vao Chuan

### Anh goc

- dat trong `dataset/object_detection/raw/images/`

### Label YOLO

- dat trong `dataset/object_detection/raw/labels/`
- ten file phai khop ten anh

Vi du:

```text
images/
  sample_001.jpg
labels/
  sample_001.txt
```

## 4. Dinh Dang Label YOLO

Moi dong label:

```text
class_id x_center y_center width height
```

Trong do:

- `class_id`: chi so lop
- `x_center`, `y_center`, `width`, `height`: gia tri chuan hoa trong khoang `0..1`

## 5. File Cau Hinh Quan Trong

### `training/data.yaml`

Dung de:

- khai bao `train`, `val`, `test`,
- map class names,
- dua cho YOLO biet dataset dang dung class nao.

Neu doi class map, phai doi dong thoi:

- raw labels,
- dataset split,
- data.yaml,
- logic train / validate lien quan.

### `training/train_config.yaml`

Dung de:

- chon model mac dinh,
- fallback model,
- epoch, batch, image size, output setup neu du an dang su dung.

## 6. Cac Script Trong `training/`

| Script | Vai tro |
|---|---|
| `prepare_dataset.py` | Tao / dam bao khung dataset |
| `validate_dataset.py` | Soat loi anh, label, class id |
| `split_dataset.py` | Chia train / val / test |
| `train_model.py` | Logic train noi bo |
| `validate_model.py` | Danh gia model sau train |
| `export_model.py` | Dong bo / xuat model sau train |
| `download_models.py` | Tai pretrained models neu can |

## 7. Luong Training Khuyen Nghi

```powershell
python run_train.py --check-only
python training\prepare_dataset.py
python training\validate_dataset.py
python training\split_dataset.py
python run_train.py
python training\validate_model.py
python training\export_model.py
```

## 8. Y Nghia Tung Buoc

### Buoc 1. `run_train.py --check-only`

Dung de tra loi:

- dependency `ultralytics` / `torch` co san khong,
- pretrained model co ton tai khong,
- raw data da co chua,
- processed train/val da san sang chua.

### Buoc 2. `prepare_dataset.py`

Dung de:

- tao khung thu muc dataset can thiet,
- dong bo layout local.

### Buoc 3. `validate_dataset.py`

Dung de:

- phat hien anh thieu label,
- phat hien label sai format,
- phat hien `class_id` khong hop le,
- phat hien du lieu xau truoc khi train.

### Buoc 4. `split_dataset.py`

Dung de:

- chia du lieu sang `train`, `val`, `test`,
- dua du lieu vao layout ma YOLO co the dung.

### Buoc 5. `run_train.py`

Dung de:

- kich hoat training pipeline chinh,
- sinh artifact train trong `runs/`,
- tao model custom moi.

### Buoc 6. `validate_model.py`

Dung de:

- danh gia model sau train,
- xac minh model moi co dung duoc cho deployment / runtime hay khong.

### Buoc 7. `export_model.py`

Dung de:

- xuat model duoc chon,
- dong bo ve `models/trained/best.pt` neu quy trinh du an can.

## 9. Model Nao Nen Dung

### `models/pretrained/*.pt`

Dung khi:

- can baseline nhanh,
- chua co dataset custom du tot,
- dang debug pipeline runtime.

### `models/trained/best.pt`

Dung khi:

- da train xong bo du lieu noi bo,
- can tang do chinh xac cho class rieng,
- muon demo bang model cua du an thay vi pretrained.

## 10. Cach Dua Model Vao Runtime

```powershell
python run_app.py --model models/trained/best.pt
```

Co the ket hop voi mode:

```powershell
python run_app.py --model models/trained/best.pt --mode medium
```

## 11. Dau Hieu Dataset Chua Tot

Thuong gap:

- label thieu hoac sai format,
- class map khong dong nhat,
- qua it anh,
- anh train khac xa moi truong webcam that,
- object nho nhung `imgsz` thap,
- goc chup qua it bien the.

## 12. Cach Lam Model On Dinh Hon

- chup nhieu goc khac nhau,
- co anh sang yeu va anh sang manh,
- co background sach va background phuc tap,
- gan nhan nhat quan,
- khong doi class order tuy tien giua cac lan train.

## 13. Kiem Tra Sau Training

Sau khi train xong, nen chay:

```powershell
python training\validate_model.py
python run_doctor.py --skip-camera-check
python run_app.py --model models/trained/best.pt
```

Muc dich:

- xac minh model ton tai,
- kiem tra runtime van mo duoc,
- test trong boi canh webcam that.

## 14. Cac Cau Hoi Nen Tu Tra Loi Sau Moi Lan Train

1. Model co nhan dung class chinh khong?
2. Co bo sot object nho khong?
3. Co false positive qua nhieu khong?
4. Khi chay webcam that, FPS con chap nhan duoc khong?
5. Model custom co thuc su tot hon pretrained khong?

## 15. Kiem Loi Nhanh Theo Trieu Chung

| Trieu chung | Noi nen debug |
|---|---|
| `run_train.py --check-only` fail | `training/train_config.yaml`, `training/model_paths.py`, dependency local |
| Dataset split xong nhung count sai | `training/split_dataset.py`, `training/dataset_ops.py` |
| Train chay nhung model kem | dataset raw, class map, dieu kien chup, `data.yaml` |
| Runtime camera nhan dien khac training ky vong | `run_app.py`, `config/settings.yaml`, model da nap, image size |

## 16. Lien Quan Toi Nhanh Medical

Nhanh object detection va nhanh medical tach biet ve du lieu:

- object detection: `dataset/object_detection/`
- medical: `dataset/medical/`

Khong nen tron 2 layout nay vao nhau. Neu can huan luyen medical rieng, hay di theo huong dan o `medical_imaging_guide.md`.
