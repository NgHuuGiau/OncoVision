# Medical Imaging Guide

Tai lieu nay mo ta nhanh workflow moi de sang loc ung thu da tu anh upload.

## Muc tieu

- Nhan anh y khoa/da lieu tu file upload.
- Chuan hoa anh truoc khi suy luan.
- Phat hien vung nghi ngo bang model YOLO.
- Gan muc nguy co `low`, `medium`, `high`.
- Tao report JSON/Markdown.
- Luu lich su ca phan tich vao SQLite.
- Nhac lai canh bao phap ly va y khoa trong moi ket qua.

## Dataset duoc chuan hoa san

Lenh:

```powershell
.\.venv\Scripts\python run_medical.py init-dataset
```

Cau truc mac dinh:

```text
dataset/medical_skin_lesion/
  raw/images
  raw/labels
  processed/images/train
  processed/images/val
  processed/images/test
  processed/labels/train
  processed/labels/val
  processed/labels/test
  metadata
  reports
  data.yaml
```

## Quy trinh train medical day du

1. Bo anh va label YOLO vao:

```text
dataset/medical_skin_lesion/raw/images
dataset/medical_skin_lesion/raw/labels
```

2. Kiem tra dataset:

```powershell
.\.venv\Scripts\python run_medical.py audit-dataset
```

3. Chia train/val/test:

```powershell
.\.venv\Scripts\python run_medical.py split-dataset
```

4. Train model:

```powershell
.\.venv\Scripts\python run_medical.py train
```

5. Validate model:

```powershell
.\.venv\Scripts\python run_medical.py validate
```

6. Hoac chay tron goi:

```powershell
.\.venv\Scripts\python run_medical.py train-all
```

Sau khi train xong, `config/medical_settings.yaml` se tu dong cap nhat duong dan model moi de giao dien chat va CLI su dung cung mot model.

## Phan tich anh

```powershell
.\.venv\Scripts\python run_medical.py analyze --image path\to\image.jpg --patient-code BN001
```

Dau ra:

- `output/medical/normalized_images/`
- `output/medical/processed_images/`
- `output/medical/reports/`
- `output/medical/medical_cases.db`

## Phan tich ngay trong giao dien chat

Sau khi mo:

```powershell
.\.venv\Scripts\python run_chat.py
```

ban co the:

- bam `Chon anh`
- tai anh da lieu/y khoa len khung chat
- gui anh

Khi do giao dien chat se:

- goi medical pipeline
- tao anh overlay danh dau vung nghi ngo
- luu report JSON/Markdown
- luu ho so ca vao SQLite
- tra ket qua ngay trong cuoc tro chuyen

## Xem lich su

```powershell
.\.venv\Scripts\python run_medical.py history --limit 10
```

## Tinh metric y khoa

```powershell
.\.venv\Scripts\python run_medical.py metrics --truths "[true,false,true]" --predictions "[true,false,false]"
```

Metric hien co:

- sensitivity
- specificity
- precision
- recall
- accuracy
- f1_score

## Luu y quan trong

- Day la he thong ho tro sang loc, khong thay the bac si.
- Neu dung anh benh nhan that, can an danh du lieu va quan ly quyen truy cap.
- Neu doi bai toan sang MRI/CT/X-ray, can them buoc tien xu ly chuyen biet.
