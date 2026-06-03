# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Ứng dụng nhận diện vật thể realtime bằng YOLO, chạy thuần Python + OpenCV, có chọn mode trong terminal, có fallback theo phần cứng, có pipeline train riêng, và có chức năng chụp mẫu train trực tiếp từ camera.

---

## Mục lục

- [1. Tổng quan](#1-tổng-quan)
- [2. Cài đặt môi trường](#2-cài-đặt-môi-trường)
- [3. Cách chạy dự án](#3-cách-chạy-dự-án)
- [4. Hướng dẫn huấn luyện](#4-hướng-dẫn-huấn-luyện)
- [5. Cấu trúc thư mục và file](#5-cấu-trúc-thư-mục-và-file)
- [6. Khắc phục lỗi nhanh](#6-khắc-phục-lỗi-nhanh)

---

## 1. Tổng quan

### Dự án làm gì

Dự án mở webcam, chạy YOLO để nhận diện vật thể theo thời gian thực, rồi hiển thị kết quả trực tiếp trên cửa sổ OpenCV.

Luồng cơ bản:

```text
Webcam -> OpenCV -> YOLO -> Python -> Bounding boxes -> Camera window
```

### Tính năng chính

- Chạy local trên desktop, không phụ thuộc web UI
- Có 2 entrypoint camera:
  - `run_app.py`
  - `run_detect.py`
- Có 4 mode runtime:
  - `auto`
  - `high`
  - `medium`
  - `low`
- Có dashboard terminal hiển thị:
  - CPU
  - RAM
  - GPU
  - VRAM
  - PyTorch
  - CUDA
- Có fallback runtime và fallback model
- Có pipeline train / validate / export riêng
- Có test dashboard `run_tests.py`
- Có chức năng bấm `T` để chụp mẫu train từ camera
- Có giao diện terminal màu:
  - xanh lá: chạy được
  - vàng: cảnh báo / trạng thái trung gian
  - đỏ: lỗi hoặc không đủ điều kiện chạy

### Camera hiển thị gì

Trên cửa sổ camera hiện có:

- bounding box
- tên class
- confidence
- tọa độ box ở góc trái dưới của box
- màu khác nhau theo từng label

### Kích thước camera hiện tại

Cả 4 mode hiện dùng chung:

- `1200 x 750`

### Chọn model local như thế nào

Việc load model được xử lý bởi `core/model_loader.py` và `config/model_config.yaml`.

Thứ tự ưu tiên hiện tại:

1. `models/trained/best.pt`
2. model trong `models/pretrained/`
3. file model local cùng tên ở thư mục gốc

---

## 2. Cài đặt môi trường

### Yêu cầu

- Windows + PowerShell
- Python `3.10+`
- Webcam hoạt động bình thường
- Nếu muốn dùng GPU NVIDIA: cần PyTorch bản CUDA phù hợp

### Tạo và kích hoạt môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Cài thư viện

```powershell
pip install -r requirements.txt
```

### Cài PyTorch CPU

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

### Cài PyTorch CUDA

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

### Kiểm tra PyTorch và CUDA

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Nếu `torch.cuda.is_available()` là `False` thì môi trường hiện tại chưa chạy CUDA được.

### Thư mục được tạo tự động

Khi chạy app hoặc training, `utils/file_utils.py` sẽ đảm bảo có sẵn:

- `dataset/raw/images`
- `dataset/raw/labels`
- `dataset/processed/images/train`
- `dataset/processed/images/val`
- `dataset/processed/images/test`
- `dataset/processed/labels/train`
- `dataset/processed/labels/val`
- `dataset/processed/labels/test`
- `dataset/sample/images`
- `dataset/sample/labels`
- `models/pretrained`
- `models/trained`
- `models/exported`
- `output/logs`
- `output/screenshots`
- `output/videos`
- `runs/train`
- `runs/detect`
- `runs/val`

---

## 3. Cách chạy dự án

### Chạy app camera chính

```powershell
.\.venv\Scripts\python run_app.py
```

### Chạy detect camera kiểu CLI

```powershell
.\.venv\Scripts\python run_detect.py
```

### Chạy với mode cố định

```powershell
.\.venv\Scripts\python run_app.py --mode high
.\.venv\Scripts\python run_app.py --mode medium
.\.venv\Scripts\python run_detect.py --mode low
```

### Chạy với camera index khác

```powershell
.\.venv\Scripts\python run_app.py --camera-index 1
.\.venv\Scripts\python run_detect.py --camera-index 1
```

### `run_app.py` và `run_detect.py` khác nhau gì

| File | Vai trò |
|---|---|
| `run_app.py` | Entry point desktop chính |
| `run_detect.py` | Entry point detect kiểu CLI |

Cả hai đều:

- gọi `detect_hardware()`
- gọi `select_runtime_config()`
- in dashboard terminal
- gọi `run_camera_session()` trong `core/camera_runner.py`

### Camera session làm gì

`core/camera_runner.py` chịu trách nhiệm:

1. nhận runtime hiện tại
2. load model local
3. mở webcam
4. đọc frame liên tục
5. chạy `model.predict(...)`
6. vẽ box và label lên frame
7. fallback nếu inference lỗi

### Chụp mẫu train bằng phím `T`

Khi camera đang chạy:

- bấm `T` để vào chế độ chụp mẫu
- hệ thống kiểm tra độ ổn định khung hình trong `5` giây
- nếu rung/lắc vượt ngưỡng, bộ đếm sẽ reset
- đủ ổn định thì hiện cửa sổ `YOLO Capture Assistant`
- nhập tên mẫu:
  - `Enter` để lưu
  - `Backspace` để xóa
  - `Esc` để hủy

Dữ liệu sẽ lưu vào:

- `dataset/sample/images/`
- `dataset/sample/labels/`

Lưu ý:

- `dataset/sample/` là nơi gom mẫu nhanh
- không phải nguồn train chính thức

### Chạy test toàn bộ hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

Trạng thái hiện tại:

- `45 / 45 PASS`

### Các lệnh nhanh

| Tác vụ | Lệnh |
|---|---|
| Chạy app camera | `.\.venv\Scripts\python run_app.py` |
| Chạy detect CLI | `.\.venv\Scripts\python run_detect.py` |
| Chạy train | `.\.venv\Scripts\python run_train.py` |
| Chạy test | `.\.venv\Scripts\python run_tests.py` |
| Kiểm tra dataset raw | `.\.venv\Scripts\python training/validate_dataset.py` |
| Chia dataset | `.\.venv\Scripts\python training/split_dataset.py` |
| Validate model | `.\.venv\Scripts\python training/validate_model.py` |
| Export model | `.\.venv\Scripts\python training/export_model.py` |

---

## 4. Hướng dẫn huấn luyện

### Cần đứng ở đâu để chạy lệnh

Tất cả lệnh training trong README này được chạy từ thư mục gốc của dự án:

```powershell
D:\YOLO
```

Tức là trước khi chạy, bạn nên thấy prompt đang ở:

```powershell
PS D:\YOLO>
```

Nếu đang ở thư mục khác thì di chuyển về thư mục gốc:

```powershell
cd D:\YOLO
```

Sau đó kích hoạt môi trường:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Huấn luyện lấy dữ liệu từ đâu

Nguồn train chính thức là:

- `dataset/raw/images/`
- `dataset/raw/labels/`

Không train trực tiếp từ:

- `dataset/sample/`
- video gốc
- ảnh nằm ở chỗ khác mà chưa copy vào `dataset/raw/`

Nếu bạn chụp mẫu bằng camera với phím `T`, dữ liệu sẽ vào `dataset/sample/`. Lúc đó bạn nên kiểm tra lại rồi chuyển mẫu tốt sang `dataset/raw/` để train thật.

### Dữ liệu cần chuẩn bị như thế nào

Mỗi ảnh phải có file `.txt` cùng tên.

Ví dụ đúng:

```text
dataset/
`-- raw/
    |-- images/
    |   |-- frame_001.jpg
    |   `-- frame_002.jpg
    `-- labels/
        |-- frame_001.txt
        `-- frame_002.txt
```

Nội dung mỗi dòng trong file label theo format YOLO:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ:

```text
0 0.512 0.438 0.220 0.310
1 0.300 0.620 0.180 0.250
```

Ý nghĩa:

- số đầu tiên: `class_id`
- 4 số sau: tọa độ box đã normalize trong khoảng `0 -> 1`

### Trình tự huấn luyện đúng

Huấn luyện trong dự án này nên chạy đúng thứ tự sau:

1. Tạo sẵn thư mục dataset
2. Bỏ ảnh + label vào `dataset/raw/`
3. Kiểm tra dataset raw
4. Chia train / val / test
5. Chỉnh class trong `training/data.yaml`
6. Chỉnh cấu hình train trong `training/train_config.yaml`
7. Chạy train
8. Validate model
9. Export model nếu cần

### Bước 1: tạo sẵn thư mục dataset

```powershell
.\.venv\Scripts\python training/prepare_dataset.py
```

Lệnh này chỉ làm 1 việc:

- tạo sẵn khung thư mục cho dataset, models, output, runs

Lệnh này không làm:

- không train
- không chia dữ liệu
- không kiểm tra label

Sau khi chạy xong, tối thiểu bạn sẽ có:

- `dataset/raw/images`
- `dataset/raw/labels`
- `dataset/processed/images/train`
- `dataset/processed/images/val`
- `dataset/processed/images/test`
- `dataset/processed/labels/train`
- `dataset/processed/labels/val`
- `dataset/processed/labels/test`

### Bước 2: bỏ dữ liệu vào đúng chỗ

Bạn tự copy dữ liệu vào:

- ảnh vào `dataset/raw/images/`
- label `.txt` vào `dataset/raw/labels/`

Ví dụ:

```text
dataset/raw/images/anh_001.jpg
dataset/raw/labels/anh_001.txt
```

Lưu ý quan trọng:

- tên ảnh và tên label phải trùng nhau, chỉ khác đuôi file
- không được để ảnh có mà không có label
- không được để label dùng class id không có trong `training/data.yaml`

### Bước 3: kiểm tra dataset raw

```powershell
.\.venv\Scripts\python training/validate_dataset.py
```

Lệnh này sẽ kiểm tra:

- tổng số ảnh raw
- bao nhiêu ảnh hợp lệ để train
- ảnh nào thiếu label
- label nào rỗng
- label nào sai format
- label nào mồ côi, có file txt nhưng không có ảnh

Nếu panel đỏ hiện ra, nghĩa là dữ liệu raw chưa đạt yêu cầu. Lúc đó nhìn vào:

- `LY DO`
- `Lý do không chạy`

để biết chính xác vì sao chưa đi tiếp được.

### Bước 4: chia train / val / test

```powershell
.\.venv\Scripts\python training/split_dataset.py
```

Lệnh này:

- đọc dữ liệu hợp lệ từ `dataset/raw/`
- copy sang `dataset/processed/`
- chia theo tỉ lệ:
  - `train = 70%`
  - `val = 15%`
  - `test = 15%`
- xóa dữ liệu cũ trong `dataset/processed/` trước khi chia lại

Sau khi chạy xong, dữ liệu train thật sẽ nằm ở:

- `dataset/processed/images/train`
- `dataset/processed/images/val`
- `dataset/processed/images/test`
- `dataset/processed/labels/train`
- `dataset/processed/labels/val`
- `dataset/processed/labels/test`

### Bước 5: cập nhật class trong `training/data.yaml`

File này là nơi map `class_id` sang tên class thật.

Ví dụ:

```yaml
path: ../dataset/processed
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: car
  2: motorbike
  3: helmet
```

Ý nghĩa:

- nếu trong label có dòng bắt đầu bằng `0`, model sẽ hiểu đó là `person`
- nếu trong label có dòng bắt đầu bằng `3`, model sẽ hiểu đó là `helmet`

Nếu file label dùng `class_id` không khớp file này, kết quả train sẽ sai.

### Bước 6: kiểm tra `training/train_config.yaml`

File này là cấu hình huấn luyện.

Các trường quan trọng:

- `model`: model chính để train
- `fallback_model`: model fallback nếu cấu hình chính lỗi
- `epochs`: số vòng train
- `imgsz`: kích thước ảnh đưa vào model
- `batch`: batch size
- `device`: GPU / CPU
- `project`: thư mục runs
- `name`: tên lần train

Nếu máy yếu hoặc thiếu VRAM, hãy giảm:

- `imgsz`
- `batch`

### Bước 7: chạy train

```powershell
.\.venv\Scripts\python run_train.py
```

Lệnh này sẽ:

1. Đọc `training/train_config.yaml`
2. Kiểm tra `dataset/processed/images/train`
3. Kiểm tra `dataset/processed/images/val`
4. Load model chính
5. Train
6. Nếu lỗi thì fallback model nhẹ hơn
7. Copy `best.pt` về `models/trained/best.pt`

Nếu không chạy được, panel đỏ sẽ ghi rõ:

- `Lý do không chạy`
- đang thiếu dữ liệu raw hay processed
- cần chạy lại lệnh nào

### Bước 8: validate model

```powershell
.\.venv\Scripts\python training/validate_model.py
```

Lệnh này dùng tập:

- `dataset/processed/images/val`

để đo chất lượng model sau train.

Nếu lệnh này không chạy được, lý do thường là:

- chưa có dữ liệu `val`
- chưa split dataset

### Bước 9: export model

```powershell
.\.venv\Scripts\python training/export_model.py
```

Lệnh này:

- lấy `models/trained/best.pt`
- export sang ONNX

Điều kiện để chạy:

- phải train xong trước
- phải có `models/trained/best.pt`

### Quy trình đầy đủ để copy và chạy

Nếu đã có sẵn ảnh và label trong `dataset/raw/`, thì chạy theo đúng thứ tự này:

```powershell
.\.venv\Scripts\python training/prepare_dataset.py
.\.venv\Scripts\python training/validate_dataset.py
.\.venv\Scripts\python training/split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training/validate_model.py
.\.venv\Scripts\python training/export_model.py
```

### Khi nào từng lệnh không chạy được

| Lệnh | Lý do thường gặp |
|---|---|
| `training/prepare_dataset.py` | Hiếm khi lỗi, vì chủ yếu chỉ tạo thư mục |
| `training/validate_dataset.py` | Chưa có ảnh trong `dataset/raw/images` |
| `training/split_dataset.py` | Dataset raw rỗng hoặc raw không hợp lệ |
| `run_train.py` | Chưa có `dataset/processed/images/train` hoặc `val` |
| `training/validate_model.py` | Chưa có `dataset/processed/images/val` |
| `training/export_model.py` | Chưa có `models/trained/best.pt` |

### Sau khi train xong app có tự dùng model mới không

Có.

`core/model_loader.py` luôn ưu tiên:

- `models/trained/best.pt`

---

## 5. Cấu trúc thư mục và file

### Cây thư mục chính

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

### Giải thích thư mục

| Thư mục | Vai trò |
|---|---|
| `app/` | Entry helper cho app camera |
| `config/` | Toàn bộ YAML cấu hình |
| `core/` | Lõi detect realtime và runtime |
| `dataset/` | Dữ liệu raw, processed, sample |
| `docs/` | Tài liệu phụ |
| `models/` | Model pretrained, trained, exported |
| `output/` | Output phụ của hệ thống |
| `runs/` | Artifact do Ultralytics sinh ra |
| `tests/` | Test tự động |
| `training/` | Pipeline train |
| `utils/` | Helper dùng chung |

### `dataset/`

| Đường dẫn | Vai trò |
|---|---|
| `dataset/raw/` | Nguồn dữ liệu chính thức để train |
| `dataset/processed/` | Dataset đã chia train/val/test |
| `dataset/sample/` | Mẫu chụp nhanh từ camera |

### `models/`

| Đường dẫn | Vai trò |
|---|---|
| `models/pretrained/` | Model local sẵn có |
| `models/trained/` | Model train xong, quan trọng nhất là `best.pt` |
| `models/exported/` | Nơi dành cho model export |

### `core/`

| File | Vai trò |
|---|---|
| `core/hardware_info.py` | Đọc CPU, RAM, GPU, VRAM, CUDA |
| `core/model_selector.py` | Chọn runtime theo mode và phần cứng |
| `core/model_loader.py` | Load model local |
| `core/fallback_manager.py` | Sinh chuỗi fallback runtime |
| `core/camera_runner.py` | Chạy camera realtime |

### `training/`

| File | Vai trò |
|---|---|
| `training/prepare_dataset.py` | Tạo sẵn cấu trúc dataset |
| `training/validate_dataset.py` | Kiểm tra dataset raw |
| `training/split_dataset.py` | Chia train/val/test |
| `training/train_model.py` | Logic train chính |
| `training/validate_model.py` | Validate model |
| `training/export_model.py` | Export ONNX |
| `training/model_paths.py` | Resolve đường dẫn model và data |
| `training/_training_bootstrap.py` | Bootstrap import path |
| `training/terminal_ui.py` | Giao diện terminal cho pipeline train |
| `training/train_config.yaml` | Hyperparameter train |
| `training/data.yaml` | Dataset config cho YOLO |

### `utils/`

| File | Vai trò |
|---|---|
| `utils/file_utils.py` | Tạo thư mục, đọc/ghi YAML |
| `utils/logger.py` | Logger dùng chung |
| `utils/console_ui.py` | Prompt mode, dashboard terminal, panel lỗi |
| `utils/draw_utils.py` | Vẽ detect lên frame |

### `docs/`

| File | Vai trò |
|---|---|
| `docs/install_guide.md` | Ghi chú cài đặt |
| `docs/project_overview.md` | Mô tả tổng quan dự án |
| `docs/training_guide.md` | Ghi chú riêng về training |

---

## 6. Khắc phục lỗi nhanh

### Có GPU nhưng vẫn chạy CPU

Kiểm tra:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### Không mở được webcam

- thử `--camera-index 1`
- đóng ứng dụng khác đang dùng webcam
- kiểm tra webcam ở ứng dụng khác

### Camera lag

- thử `--mode medium`
- nếu vẫn lag, chuyển `--mode low`

### Không detect được vật thể

- tăng ánh sáng
- đưa vật thể gần camera hơn
- kiểm tra model đang dùng
- nếu dùng model tự train, xác nhận `models/trained/best.pt` là file đúng

### Lỗi khi train

- kiểm tra `training/data.yaml`
- kiểm tra `dataset/raw/` đã có ảnh và label chưa
- chạy `training/validate_dataset.py`
- chạy `training/split_dataset.py`
- nếu thiếu VRAM, giảm `imgsz` và `batch`

### Lỗi không load được model local

Kiểm tra:

- `models/pretrained/`
- `models/trained/best.pt`
- `config/model_config.yaml`

### Khi terminal báo đỏ

Nếu một lệnh không chạy được, terminal hiện đã ghi rõ:

- `LY DO`
- `Lý do không chạy`
- `GOI Y`
- `LENH THU` hoặc `LENH NHANH`

Tức là bạn không cần đọc traceback dài mới biết lỗi.

---

## Ghi chú

- Nên chạy toàn bộ dự án bằng Python trong `.venv`
- Nếu muốn dùng GPU NVIDIA, cần cài đúng bản PyTorch có CUDA
- Sau khi train xong, app sẽ tự ưu tiên `models/trained/best.pt`
- Trạng thái test hệ thống hiện tại: `45/45 PASS`
