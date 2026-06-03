# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Ứng dụng nhận diện vật thể realtime bằng YOLO chạy trên desktop Python + OpenCV. Dự án có giao diện terminal tiếng Việt, tự dò cấu hình máy trước khi chạy, tự gợi ý mức phù hợp, có pipeline train riêng và có thể chụp mẫu train trực tiếp từ webcam.

## Tổng quan

Dự án hiện dùng họ model `YOLO26` và đã hỗ trợ đủ 5 mức:

| Mức model | File |
|---|---|
| Nhẹ nhất | `yolo26n.pt` |
| Cân bằng | `yolo26s.pt` |
| Khá mạnh | `yolo26m.pt` |
| Mạnh | `yolo26l.pt` |
| Mạnh nhất | `yolo26x.pt` |

Khi chạy `run_app.py` hoặc `run_detect.py`, hệ thống sẽ:

1. dò cấu hình máy thật
2. đọc `CPU`, `RAM`, `GPU`, `VRAM`, `PyTorch`, `CUDA`
3. hiển thị gợi ý trong terminal
4. map lựa chọn người dùng sang model phù hợp nhất mà máy còn chạy ổn định

Ví dụ logic hiện tại:

- máy rất mạnh, `VRAM >= 12GB` -> có thể lên `yolo26x.pt`
- máy mạnh, `VRAM >= 8GB` -> ưu tiên `yolo26l.pt`
- máy tầm trung, `VRAM khoảng 4GB` -> thường hợp `yolo26s.pt` hoặc `yolo26m.pt`
- máy yếu hoặc CPU-only -> hạ xuống `yolo26n.pt`

Mục tiêu của hệ thống không phải ép model lớn nhất bằng mọi giá, mà là chọn mức cao nhất máy còn chạy ổn định.

## Cách chạy nhanh

Luôn chạy bằng Python trong `.venv`:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

### Chạy camera

```powershell
.\\.venv\\Scripts\\python run_app.py
```

### Chạy detect

```powershell
.\\.venv\\Scripts\\python run_detect.py
```

### Chạy train

```powershell
.\\.venv\\Scripts\\python run_train.py
```

### Chạy test

```powershell
.\\.venv\\Scripts\\python run_tests.py
```

### Chạy với mode cố định

```powershell
.\\.venv\\Scripts\\python run_app.py --mode high
.\\.venv\\Scripts\\python run_app.py --mode medium
.\\.venv\\Scripts\\python run_app.py --mode low
```

### Đổi camera index

```powershell
.\\.venv\\Scripts\\python run_app.py --camera-index 1
.\\.venv\\Scripts\\python run_detect.py --camera-index 1
```

## 3 mức người dùng sẽ thấy

Menu người dùng hiện được rút về 3 mức:

- `Cao nhất`
- `Trung bình`
- `Yếu`

Nhưng bên dưới mỗi mức, hệ thống sẽ tự tính model thật theo cấu hình máy.

Ví dụ:

- `Cao nhất` trên máy mạnh có thể là `yolo26x.pt`
- `Cao nhất` trên máy 4GB VRAM có thể tự hạ còn `yolo26m.pt`
- `Trung bình` trên RTX 3050 Ti 4GB thường là `yolo26s.pt`
- `Yếu` trên máy yếu hoặc CPU-only sẽ về `yolo26n.pt`

## Màu trong terminal

- xanh lá: chạy được, trạng thái tốt
- vàng: trung gian, cảnh báo, hoặc cần chú ý
- đỏ: lỗi hoặc không đủ điều kiện chạy

Nếu lệnh không chạy được, terminal sẽ hiện rõ:

- `LÝ DO`
- `Lý do không chạy`
- `GỢI Ý`
- `LỆNH THỬ` hoặc `LỆNH NHANH`

## Chụp mẫu train từ camera

Khi camera đang chạy:

- bấm `T` để vào chế độ chụp mẫu
- hệ thống kiểm tra độ ổn định khung hình trong `5` giây
- nếu rung/lắc thì đếm lại
- đủ ổn định thì mở bảng nhập tên mẫu
- `Enter` để lưu
- `Backspace` để xóa
- `Esc` để hủy

Dữ liệu lưu tại:

- `dataset/sample/images/`
- `dataset/sample/labels/`

`dataset/sample/` chỉ là nơi gom mẫu nhanh. Nguồn train chính thức vẫn là `dataset/raw/`.

## 5 model đang nằm ở đâu

Model local hiện được ưu tiên tìm trong:

- `models/trained/`
- `models/pretrained/`

Trạng thái hiện tại của dự án:

- đã có đủ `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt` trong `models/pretrained/`

File model train xong đẹp nhất sẽ nằm ở:

- `models/trained/best.pt`

## Hướng dẫn huấn luyện

Toàn bộ lệnh dưới đây chạy từ thư mục gốc dự án:

```powershell
PS D:\YOLO>
```

### Bước 1: chuẩn bị thư mục

```powershell
.\\.venv\\Scripts\\python training/prepare_dataset.py
```

Lệnh này tạo sẵn:

- `dataset/raw/images`
- `dataset/raw/labels`
- `dataset/processed/...`
- `dataset/sample/...`
- `models/pretrained`
- `models/trained`
- `models/exported`

### Bước 2: đưa dữ liệu raw vào đúng chỗ

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

### Bước 3: kiểm tra dataset raw

```powershell
.\\.venv\\Scripts\\python training/validate_dataset.py
```

Lệnh này sẽ báo:

- tổng số ảnh raw
- ảnh hợp lệ để train
- ảnh thiếu label
- label rỗng
- label lỗi
- label mồ côi

### Bước 4: chia train / val / test

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

### Bước 5: train model

```powershell
.\\.venv\\Scripts\\python run_train.py
```

Lệnh này sẽ:

1. đọc `training/train_config.yaml`
2. kiểm tra dataset đã split
3. load model train chính
4. fallback sang model nhẹ hơn nếu cần
5. copy `best.pt` về `models/trained/best.pt`

### Bước 6: validate model

```powershell
.\\.venv\\Scripts\\python training/validate_model.py
```

### Bước 7: export model

```powershell
.\\.venv\\Scripts\\python training/export_model.py
```

### Chuỗi lệnh đầy đủ

```powershell
.\\.venv\\Scripts\\python training/prepare_dataset.py
.\\.venv\\Scripts\\python training/validate_dataset.py
.\\.venv\\Scripts\\python training/split_dataset.py
.\\.venv\\Scripts\\python run_train.py
.\\.venv\\Scripts\\python training/validate_model.py
.\\.venv\\Scripts\\python training/export_model.py
```

## Khi nào từng lệnh không chạy được

| Lệnh | Lý do thường gặp |
|---|---|
| `training/prepare_dataset.py` | Hiếm khi lỗi, chủ yếu chỉ tạo thư mục |
| `training/validate_dataset.py` | Chưa có ảnh trong `dataset/raw/images` |
| `training/split_dataset.py` | Dataset raw rỗng hoặc raw không hợp lệ |
| `run_train.py` | Chưa có `dataset/processed/images/train` hoặc `val` |
| `training/validate_model.py` | Chưa có `dataset/processed/images/val` |
| `training/export_model.py` | Chưa có `models/trained/best.pt` |
| `run_app.py` / `run_detect.py` | Không mở được webcam, không load được model, hoặc CUDA không sẵn sàng |

## Cấu trúc thư mục chính

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

### Giải thích nhanh

- `app/`: entry helper cho camera app
- `config/`: YAML cấu hình
- `core/`: lõi detect, load model, chọn runtime
- `dataset/`: raw, processed, sample
- `docs/`: tài liệu phụ
- `models/`: pretrained, trained, exported
- `output/`: ảnh, log, video sinh ra
- `runs/`: artifact từ Ultralytics
- `tests/`: test tự động
- `training/`: pipeline huấn luyện
- `utils/`: helper dùng chung

## File quan trọng

- `core/hardware_info.py`: đọc CPU, RAM, GPU, VRAM, CUDA
- `core/model_selector.py`: chọn model và runtime theo cấu hình máy
- `core/model_loader.py`: load model local
- `core/camera_runner.py`: chạy webcam realtime
- `training/train_model.py`: train model
- `training/validate_model.py`: validate model
- `training/export_model.py`: export ONNX
- `utils/console_ui.py`: giao diện terminal và panel lỗi
- `run_app.py`: chạy app camera chính
- `run_detect.py`: chạy detect camera
- `run_train.py`: chạy pipeline train
- `run_tests.py`: chạy test dashboard

## Trạng thái hệ thống hiện tại

- terminal tiếng Việt có dấu
- menu chọn mode đã gợi ý theo cấu hình máy
- đủ bộ `YOLO26 n/s/m/l/x`
- `run_tests.py` hiện pass `50/50`

## Gợi ý sử dụng thực tế

- máy rất mạnh: chọn `Cao nhất`
- máy tầm trung như RTX 3050 Ti 4GB: nên ưu tiên `Trung bình`
- máy yếu hoặc CPU-only: chọn `Yếu`

Nếu muốn ép `Cao nhất` trên máy yếu, hệ thống vẫn sẽ tự hạ về model hợp lý nhất còn chạy được.
