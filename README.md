# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

## 1. Giới thiệu dự án

Đây là dự án nhận diện vật thể realtime bằng YOLO chạy trên desktop Python + OpenCV.

Dự án có các chức năng chính:

- mở webcam và detect vật thể theo thời gian thực
- tự dò cấu hình máy trước khi chạy
- tự chọn mức phù hợp theo `CPU`, `RAM`, `GPU`, `VRAM`, `CUDA`
- hỗ trợ chụp mẫu train trực tiếp từ camera
- có pipeline huấn luyện, validate và export model
- có dashboard terminal tiếng Việt để báo trạng thái và lý do lỗi

Dự án hiện dùng họ model `YOLO26`:

- `yolo26n.pt`
- `yolo26s.pt`
- `yolo26m.pt`
- `yolo26l.pt`
- `yolo26x.pt`

Menu người dùng hiện thấy 3 mức:

- `Cao nhất`
- `Trung bình`
- `Yếu`

Nhưng bên trong hệ thống sẽ tự map sang model thật theo cấu hình máy.

Ví dụ:

- máy rất mạnh -> có thể lên `yolo26x.pt`
- máy mạnh -> có thể lên `yolo26l.pt`
- máy tầm trung như RTX 3050 Ti 4GB -> thường hợp `yolo26s.pt`
- máy yếu hoặc CPU-only -> thường về `yolo26n.pt`

Lưu ý:

- repo này chỉ chứa code
- repo này không kèm model `.pt`
- repo này không kèm dataset

## 2. Cách cài

### 2.1. Yêu cầu

- Windows + PowerShell
- Python `3.10+`
- webcam hoạt động bình thường
- nếu muốn chạy GPU NVIDIA thì cần driver và PyTorch CUDA phù hợp

### 2.2. Clone dự án

```powershell
git clone <repo-url>
cd D:\YOLO
```

### 2.3. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2.4. Kích hoạt môi trường

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

Khi thành công, terminal sẽ có dạng:

```powershell
(.venv) PS D:\YOLO>
```

### 2.5. Cài thư viện Python

```powershell
pip install -r requirements.txt
```

### 2.6. Cài PyTorch

#### Chạy CPU

```powershell
.\\.venv\\Scripts\\python -m pip uninstall -y torch torchvision torchaudio
.\\.venv\\Scripts\\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Chạy GPU NVIDIA CUDA

```powershell
.\\.venv\\Scripts\\python -m pip uninstall -y torch torchvision torchaudio
.\\.venv\\Scripts\\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 2.7. Kiểm tra PyTorch và CUDA

```powershell
.\\.venv\\Scripts\\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Nếu `torch.cuda.is_available()` là `False` thì hiện tại máy đang chạy CPU hoặc môi trường CUDA chưa đúng.

### 2.8. Tạo thư mục hệ thống

```powershell
.\\.venv\\Scripts\\python training/prepare_dataset.py
```

Lệnh này sẽ tạo sẵn:

- `dataset/raw/images`
- `dataset/raw/labels`
- `dataset/processed/...`
- `dataset/sample/...`
- `models/pretrained`
- `models/trained`
- `models/exported`

### 2.9. Cài model

Repo không kèm model. Bạn phải tự đặt model vào:

```text
models/pretrained/
```

Nên chuẩn bị đủ 5 file:

- `models/pretrained/yolo26n.pt`
- `models/pretrained/yolo26s.pt`
- `models/pretrained/yolo26m.pt`
- `models/pretrained/yolo26l.pt`
- `models/pretrained/yolo26x.pt`

Nếu chưa đủ 5 file, hệ thống vẫn có thể chạy, nhưng sẽ không tận dụng được hết các mức.

### 2.10. Chạy test kiểm tra hệ thống

```powershell
.\\.venv\\Scripts\\python run_tests.py
```

## 3. Cách chạy

### 3.1. Chạy app camera chính

```powershell
.\\.venv\\Scripts\\python run_app.py
```

### 3.2. Chạy detect camera

```powershell
.\\.venv\\Scripts\\python run_detect.py
```

### 3.3. Chạy train

```powershell
.\\.venv\\Scripts\\python run_train.py
```

### 3.4. Chạy test

```powershell
.\\.venv\\Scripts\\python run_tests.py
```

### 3.5. Chạy với mode cố định

```powershell
.\\.venv\\Scripts\\python run_app.py --mode high
.\\.venv\\Scripts\\python run_app.py --mode medium
.\\.venv\\Scripts\\python run_app.py --mode low
```

Giải thích:

- `high`: yêu cầu mức cao
- `medium`: yêu cầu mức cân bằng
- `low`: yêu cầu mức nhẹ

Nhưng hệ thống vẫn tự hạ nếu phần cứng không chịu nổi.

### 3.6. Đổi camera index

```powershell
.\\.venv\\Scripts\\python run_app.py --camera-index 1
.\\.venv\\Scripts\\python run_detect.py --camera-index 1
```

### 3.7. Chụp mẫu train từ camera

Khi camera đang chạy:

- bấm `T` để vào chế độ chụp mẫu
- hệ thống kiểm tra độ ổn định trong `5` giây
- nếu rung/lắc thì đếm lại
- nếu đủ ổn định thì hiện bảng nhập tên mẫu

Phím dùng:

- `Enter`: lưu
- `Backspace`: xóa
- `Esc`: hủy

Mẫu sẽ lưu vào:

- `dataset/sample/images/`
- `dataset/sample/labels/`

### 3.8. Ý nghĩa màu terminal

- xanh lá: chạy được, trạng thái tốt
- vàng: cảnh báo hoặc trạng thái trung gian
- đỏ: lỗi hoặc không đủ điều kiện chạy

Nếu lệnh không chạy được, terminal sẽ hiện:

- `LÝ DO`
- `Lý do không chạy`
- `GỢI Ý`
- `LỆNH THỬ` hoặc `LỆNH NHANH`

## 4. Huấn luyện

Toàn bộ lệnh huấn luyện chạy từ thư mục gốc dự án:

```powershell
PS D:\YOLO>
```

### 4.1. Chuẩn bị dữ liệu

Bỏ dữ liệu vào:

- ảnh: `dataset/raw/images/`
- label YOLO: `dataset/raw/labels/`

Ví dụ:

```text
dataset/raw/images/frame_001.jpg
dataset/raw/labels/frame_001.txt
```

Mỗi file label YOLO có dạng:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ:

```text
0 0.512 0.438 0.220 0.310
```

### 4.2. Kiểm tra dataset raw

```powershell
.\\.venv\\Scripts\\python training/validate_dataset.py
```

Lệnh này sẽ báo:

- tổng số ảnh raw
- ảnh hợp lệ
- ảnh thiếu label
- label rỗng
- label lỗi
- label mồ côi

### 4.3. Chia train / val / test

```powershell
.\\.venv\\Scripts\\python training/split_dataset.py
```

Sau bước này dữ liệu train thật sẽ nằm ở:

- `dataset/processed/images/train`
- `dataset/processed/images/val`
- `dataset/processed/images/test`
- `dataset/processed/labels/train`
- `dataset/processed/labels/val`
- `dataset/processed/labels/test`

### 4.4. Kiểm tra `training/data.yaml`

File này map `class_id` sang tên class thật.

Ví dụ:

```yaml
path: ../dataset/processed
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: helmet
```

### 4.5. Kiểm tra `training/train_config.yaml`

Các mục quan trọng:

- `model`
- `fallback_model`
- `epochs`
- `imgsz`
- `batch`
- `device`
- `project`
- `name`

Nếu máy yếu hoặc thiếu VRAM, nên giảm:

- `imgsz`
- `batch`

### 4.6. Chạy train

```powershell
.\\.venv\\Scripts\\python run_train.py
```

Lệnh này sẽ:

1. đọc `training/train_config.yaml`
2. kiểm tra dataset đã split
3. load model train chính
4. fallback sang model nhẹ hơn nếu cần
5. copy `best.pt` về `models/trained/best.pt`

### 4.7. Validate model

```powershell
.\\.venv\\Scripts\\python training/validate_model.py
```

### 4.8. Export model

```powershell
.\\.venv\\Scripts\\python training/export_model.py
```

### 4.9. Chuỗi lệnh đầy đủ

```powershell
.\\.venv\\Scripts\\python training/prepare_dataset.py
.\\.venv\\Scripts\\python training/validate_dataset.py
.\\.venv\\Scripts\\python training/split_dataset.py
.\\.venv\\Scripts\\python run_train.py
.\\.venv\\Scripts\\python training/validate_model.py
.\\.venv\\Scripts\\python training/export_model.py
```

### 4.10. Khi nào lệnh train không chạy được

| Lệnh | Lý do thường gặp |
|---|---|
| `training/prepare_dataset.py` | hiếm khi lỗi, chủ yếu chỉ tạo thư mục |
| `training/validate_dataset.py` | chưa có ảnh trong `dataset/raw/images` |
| `training/split_dataset.py` | dataset raw rỗng hoặc raw không hợp lệ |
| `run_train.py` | chưa có `dataset/processed/images/train` hoặc `val` |
| `training/validate_model.py` | chưa có `dataset/processed/images/val` |
| `training/export_model.py` | chưa có `models/trained/best.pt` |

## 5. Giải thích cấu trúc thư mục

```text
YOLO/
|-- app/
|-- config/
|-- core/
|-- dataset/
|-- docs/
|-- models/
|-- output/
|-- runs/
|-- tests/
|-- training/
|-- utils/
|-- README.md
|-- requirements.txt
|-- run_app.py
|-- run_detect.py
|-- run_train.py
`-- run_tests.py
```

### `app/`

- helper cho entrypoint camera

### `config/`

- chứa YAML cấu hình hệ thống

### `core/`

- chứa lõi xử lý detect
- chọn runtime theo cấu hình máy
- load model
- chạy camera realtime

### `dataset/`

- `raw/`: dữ liệu gốc để train
- `processed/`: dữ liệu đã chia train/val/test
- `sample/`: mẫu chụp nhanh từ camera

### `docs/`

- tài liệu phụ của dự án

### `models/`

- `pretrained/`: model local bạn tự đặt vào
- `trained/`: model train xong, quan trọng nhất là `best.pt`
- `exported/`: model export ra để deploy

### `output/`

- ảnh, log, video sinh ra trong lúc chạy

### `runs/`

- artifact do Ultralytics sinh ra khi train / val / detect

### `tests/`

- test tự động của dự án

### `training/`

- pipeline huấn luyện
- chia dataset
- validate dataset
- validate model
- export model

### `utils/`

- helper dùng chung như terminal UI, file utils, draw utils

### Các file chạy chính

- `run_app.py`: chạy camera app chính
- `run_detect.py`: chạy detect camera
- `run_train.py`: chạy train
- `run_tests.py`: chạy toàn bộ test

## 6. Trạng thái hiện tại

- terminal tiếng Việt có dấu
- menu chọn mode đã gợi ý theo cấu hình máy
- hỗ trợ đủ logic cho `YOLO26 n/s/m/l/x`
- `run_tests.py` hiện pass `50/50`
