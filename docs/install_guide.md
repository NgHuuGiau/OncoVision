# Huong Dan Cai Dat

Tai lieu nay huong dan cai dat OncoVision tren Windows, khoi tao thu muc du an, va kiem tra moi truong truoc khi dua vao su dung.

## 1. Yeu Cau He Thong

### Bat buoc

- Windows 10 hoac Windows 11
- Python 3.10 tro len
- Quyen tao virtual environment
- Quyen ghi trong thu muc du an

### Khuyen nghi

- GPU NVIDIA neu muon toi uu train va inference
- Webcam neu muon chay `run_app.py`
- Windows Terminal hoac PowerShell 7 de hien thi Unicode tot hon

## 2. Tao Moi Truong Ao

```powershell
cd D:\OncoVision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu bi chan script trong PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Khoi Tao Thu Muc Du An

Lan chay dau tien nen dung:

```powershell
python run_menu.py
```

Lenh nay giup tao va dong bo cac nhom thu muc nhu:

```text
dataset/
models/
output/
runs/
```

Sau do cac luong `medical`, `training`, `chat`, `camera` se tu bo sung nhung thu muc con can thiet.

## 4. Kiem Tra Sau Cai Dat

### Kiem tra tong quat an toan

```powershell
python run_doctor.py --skip-camera-check
```

Muc dich:

- kiem tra dependency quan trong,
- kiem tra model,
- kiem tra dataset,
- kiem tra output directories,
- khong yeu cau webcam that.

### Kiem tra chuoi entrypoint

```powershell
python run_smoke.py
```

Neu dang chay trong CI hoac muon check nhe:

```powershell
python run_smoke.py --ci-safe --stop-on-fail
```

### Kiem tra unit test

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## 5. Chay Thu Tung Chuc Nang

### Menu tong

```powershell
python run_menu.py
```

### Runtime advisor

```powershell
python run_app.py --advisor-only
```

### Camera realtime

```powershell
python run_app.py
python run_app.py --mode medium --camera-index 0
python run_app.py --model models/trained/best.pt
```

### Chat UI

```powershell
python run_chat.py --check-only
python run_chat.py
```

### Training

```powershell
python run_train.py --check-only
python run_train.py
```

### Medical CLI

```powershell
python run_medical.py status
python run_medical.py ready
python run_medical.py sources
```

## 6. Cau Truc Dependency

Repo hien co ba nhom dependency de de van hanh:

| File | Muc dich |
|---|---|
| `requirements.txt` | Bo dependency day du cho runtime chinh |
| `requirements-ci.txt` | Dependency toi thieu de chay workflow CI |
| `requirements-dev.txt` | Runtime day du + cong cu phat trien nhu `ruff`, `mypy` |

## 7. Xu Ly Loi Thuong Gap

### Khong mo duoc camera

Thu:

```powershell
python run_app.py --mode low --camera-index 1
```

Neu van loi:

- kiem tra app khac co dang giu webcam khong,
- chay `run_doctor.py --skip-camera-check` de xem canh bao he thong,
- doi `camera-index` sang `0`, `1`, `2`.

### Thieu model

Kiem tra:

```text
models/pretrained/
models/trained/
```

Neu can object detection pretrained model, xem cac script trong `training/download_models.py`.

### Loi CUDA hoac torch

Kiem tra:

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
```

Hai lenh nay se cho biet:

- GPU co duoc nhan khong,
- CUDA co san sang khong,
- torch dang build theo CPU hay CUDA.

### Giao dien chat chua san sang

Dung:

```powershell
python run_chat.py --check-only --auto-fix-icons
```

Lenh nay giup:

- check icon,
- check module bat buoc,
- check medical model status,
- tu tao icon neu dang thieu.

## 8. Checklist Sau Khi Cai Dat Xong

Moi truong co the xem la san sang khi cac muc sau deu on:

1. `python run_doctor.py --skip-camera-check` chay xong khong co loi nghiem trong.
2. `python run_smoke.py` hoac `python run_smoke.py --ci-safe` pass.
3. `python run_app.py --advisor-only` in duoc khuyen nghi runtime.
4. `python run_chat.py --check-only` bao trang thai san sang.
5. `python run_train.py --check-only` khong bi fail do thieu dependency.

## 9. Khuyen Nghi Cho Team

- Khong chay thang training hoac camera tren may moi ma chua chay doctor/smoke.
- Neu chi muon review code tren CI, uu tien `requirements-ci.txt`.
- Neu debug local feature day du, dung `requirements.txt` hoac `requirements-dev.txt`.
