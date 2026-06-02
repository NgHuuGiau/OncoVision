# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Ứng dụng nhận diện vật thể realtime bằng YOLO, chạy thuần Python + OpenCV, có menu chọn mode trong terminal, có cơ chế fallback theo phần cứng và có sẵn pipeline train, validate, export model.

---

## Mục lục

- [1. Tổng quan dự án](#1-tổng-quan-dự-án)
- [2. Cài đặt môi trường](#2-cài-đặt-môi-trường)
- [3. Cách chạy dự án](#3-cách-chạy-dự-án)
- [4. Hướng dẫn huấn luyện đầy đủ](#4-hướng-dẫn-huấn-luyện-đầy-đủ)
- [5. Giải thích rõ từng thư mục và file](#5-giải-thích-rõ-từng-thư-mục-và-file)
- [6. Khắc phục lỗi nhanh](#6-khắc-phục-lỗi-nhanh)

---

## 1. Tổng quan dự án

### 1.1. Dự án dùng để làm gì

Dự án mở webcam, chạy YOLO để nhận diện vật thể theo thời gian thực và hiển thị kết quả trực tiếp bằng cửa sổ OpenCV.

Luồng chạy thực tế:

```text
Webcam -> OpenCV -> YOLO -> Python -> Bounding boxes -> Cửa sổ camera
```

### 1.2. Những gì đang có trong mã nguồn

- Chạy desktop local, không dùng web UI.
- Có 2 entrypoint chạy detect: `run_app.py` và `run_detect.py`.
- Có menu chọn mode `auto`, `high`, `medium`, `low` ngay trong terminal.
- Có dashboard terminal hiển thị CPU, RAM, GPU, VRAM, PyTorch, CUDA và cấu hình runtime đang dùng.
- Có cơ chế fallback khi model hoặc runtime hiện tại không chạy được.
- Có sẵn model local trong `models/pretrained/`.
- Có pipeline train, validate, export.
- Có bộ test riêng trong terminal qua `run_tests.py`.

### 1.3. Những gì camera đang hiển thị

Khung hình camera hiện tại tập trung vào phần detect:

- Bounding boxes
- Tên class và độ tin cậy do hàm vẽ của dự án render lên ảnh

### 1.4. Các mode đang có

| Phím | Mode | Ý nghĩa |
|---|---|---|
| `1` | `auto` | Tự đọc phần cứng rồi chọn profile phù hợp |
| `2` | `high` | Ưu tiên chất lượng, nặng hơn |
| `3` | `medium` | Cân bằng giữa tốc độ và độ chính xác |
| `4` | `low` | Ưu tiên ổn định, nhẹ hơn |
| `0` | `exit` | Thoát ở menu |

### 1.5. Logic chọn runtime trong dự án

Dự án tách rõ 2 khái niệm:

- `requested`: cấu hình người dùng muốn chạy
- `resolved`: cấu hình máy thực tế chạy được

Ví dụ:

- Bạn chọn `high`
- Nhưng môi trường PyTorch là `CPU-only`
- Hệ thống sẽ tự rơi về CPU profile thay vì cố ép CUDA

### 1.6. Thứ tự ưu tiên load model local

Việc load model được điều khiển bởi `config/model_config.yaml` và `core/yolo_loader.py`.

Thứ tự ưu tiên hiện tại:

1. `models/trained/best.pt`
2. `models/pretrained/yolo26s.pt`
3. file local cùng tên model ở thư mục gốc
4. `yolo11s.pt`
5. `yolov8s.pt`

Lưu ý:

- Nếu profile yêu cầu model khác, loader vẫn build danh sách candidate phù hợp với profile đó.
- Dự án hiện ưu tiên dùng model local, không thiết kế để tải model từ internet trong lúc chạy.

---

## 2. Cài đặt môi trường

### 2.1. Yêu cầu

- Windows + PowerShell
- Python `3.10+`
- Webcam hoạt động bình thường
- Nếu muốn dùng GPU NVIDIA: cần cài đúng bản PyTorch có CUDA

### 2.2. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2.3. Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2.4. Cài thư viện cơ bản

```powershell
pip install -r requirements.txt
```

### 2.5. Các thư viện chính trong `requirements.txt`

Runtime:

- `ultralytics`
- `opencv-python`
- `numpy`
- `pillow`
- `psutil`
- `GPUtil`
- `PyYAML`
- `torch`
- `torchvision`
- `torchaudio`

Training và xử lý dữ liệu:

- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`

### 2.6. Cài PyTorch theo loại máy

#### Máy chỉ dùng CPU

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

#### Máy có GPU NVIDIA

Cấu hình README đang hướng tới:

- `torch 2.10.0`
- `torchvision 0.25.0`
- `torchaudio 2.10.0`
- `CUDA 12.6`

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

### 2.7. Kiểm tra PyTorch và CUDA

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Nếu kết quả là:

- version có hậu tố `+cpu`
- `torch.version.cuda = None`
- `torch.cuda.is_available() = False`

thì môi trường hiện tại đang là CPU-only.

### 2.8. Thư mục được tạo tự động

Mỗi lần chạy app hoặc training, hàm `ensure_project_directories()` trong `utils/file_utils.py` sẽ đảm bảo các thư mục sau tồn tại:

- `dataset/raw/images`
- `dataset/raw/labels`
- `dataset/processed/images/train`
- `dataset/processed/images/val`
- `dataset/processed/images/test`
- `dataset/processed/labels/train`
- `dataset/processed/labels/val`
- `dataset/processed/labels/test`
- `dataset/sample`
- `models/pretrained`
- `models/trained`
- `models/exported`
- `output/screenshots`
- `output/videos`
- `output/logs`
- `runs/train`
- `runs/detect`
- `runs/val`

---

## 3. Cách chạy dự án

### 3.1. Chạy desktop chính

```powershell
.\.venv\Scripts\python run_app.py
```

Đây là entrypoint chính cho người dùng cuối. Luồng chạy:

1. Tạo đủ thư mục cần thiết
2. Nhận mode từ tham số hoặc menu terminal
3. Detect phần cứng
4. Chọn runtime config
5. In dashboard terminal
6. Mở camera và bắt đầu detect

### 3.2. Chạy detect kiểu CLI

```powershell
.\.venv\Scripts\python run_detect.py
```

Đây là phiên bản gọn hơn nhưng vẫn dùng cùng lõi detect trong `core/camera_detector.py`.

### 3.3. Chạy với mode cố định

```powershell
.\.venv\Scripts\python run_app.py --mode high
.\.venv\Scripts\python run_app.py --mode medium
.\.venv\Scripts\python run_detect.py --mode low
```

Các giá trị hợp lệ:

- `auto`
- `high`
- `medium`
- `low`

### 3.4. Chạy với camera khác

```powershell
.\.venv\Scripts\python run_app.py --camera-index 1
.\.venv\Scripts\python run_detect.py --camera-index 1
```

Dùng khi máy có nhiều camera hoặc webcam mặc định bị chiếm.

### 3.5. `run_app.py` và `run_detect.py` khác nhau gì

| File | Vai trò |
|---|---|
| `run_app.py` | Entry point desktop chính, phù hợp dùng như ứng dụng mặc định |
| `run_detect.py` | Entry point detect dạng CLI, phù hợp chạy nhanh hoặc test |

Hiện tại cả hai đều:

- gọi `detect_hardware()`
- gọi `select_runtime_config()`
- in dashboard terminal
- gọi `run_camera_session()` trong `core/camera_detector.py`

### 3.6. Bên trong camera session xảy ra gì

`core/camera_detector.py` làm các việc chính sau:

1. Chọn runtime hiện tại
2. Load model YOLO local
3. Mở webcam
4. Đọc frame liên tục
5. Gọi `model.predict(...)`
6. Vẽ bounding boxes lên ảnh
7. Nếu inference lỗi, tự thử fallback runtime an toàn hơn

### 3.7. Thoát camera

- Nhấn `Esc`

### 3.8. Chạy toàn bộ test hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

`run_tests.py` sẽ:

- discover toàn bộ test trong `tests/`
- chạy lần lượt từng test
- hiển thị `PASS / FAIL / ERROR / SKIP`
- có progress bar trong terminal
- in tổng kết cuối phiên

### 3.9. Lệnh nhanh

| Tác vụ | Lệnh |
|---|---|
| Chạy app chính | `.\.venv\Scripts\python run_app.py` |
| Chạy detect CLI | `.\.venv\Scripts\python run_detect.py` |
| Chạy app mode medium | `.\.venv\Scripts\python run_app.py --mode medium` |
| Chạy detect mode low | `.\.venv\Scripts\python run_detect.py --mode low` |
| Chạy camera index 1 | `.\.venv\Scripts\python run_app.py --camera-index 1` |
| Chạy train | `.\.venv\Scripts\python run_train.py` |
| Validate model | `.\.venv\Scripts\python training/validate_model.py` |
| Export model | `.\.venv\Scripts\python training/export_model.py` |
| Chạy test hệ thống | `.\.venv\Scripts\python run_tests.py` |

---

## 4. Hướng dẫn huấn luyện đầy đủ

Phần này mô tả đúng luồng code hiện tại trong thư mục `training/`.

### 4.1. Cấu trúc dữ liệu YOLO đầu vào

Bạn cần chuẩn bị dataset theo kiểu YOLO:

- ảnh nằm trong `dataset/raw/images/`
- file label `.txt` nằm trong `dataset/raw/labels/`
- mỗi ảnh phải có file label cùng tên

Ví dụ:

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

### 4.2. Bước 0: tạo sẵn thư mục dataset nếu cần

```powershell
.\.venv\Scripts\python training/prepare_dataset.py
```

Script này không chia dữ liệu. Nó chỉ đảm bảo các thư mục cần cho dataset đã tồn tại.

### 4.3. Bước 1: đưa dữ liệu gốc vào `dataset/raw/`

Bạn tự chép dữ liệu vào:

- `dataset/raw/images/`
- `dataset/raw/labels/`

Yêu cầu:

- nhãn đúng format YOLO
- tên file label trùng stem với ảnh
- ảnh có đuôi `.jpg`, `.jpeg`, `.png`, `.bmp`

### 4.4. Bước 2: chia train / val / test

```powershell
.\.venv\Scripts\python training/split_dataset.py
```

Script này làm gì:

- đọc toàn bộ ảnh trong `dataset/raw/images/`
- tìm label tương ứng trong `dataset/raw/labels/`
- trộn dữ liệu với seed cố định `42`
- chia theo tỷ lệ:
  - `train = 70%`
  - `val = 15%`
  - `test = 15%`
- copy sang:
  - `dataset/processed/images/train`
  - `dataset/processed/images/val`
  - `dataset/processed/images/test`
  - `dataset/processed/labels/train`
  - `dataset/processed/labels/val`
  - `dataset/processed/labels/test`

Lưu ý quan trọng:

- Script hiện tại `copy` dữ liệu sang `dataset/processed/`, không xóa dữ liệu cũ trước.
- Nếu bạn thay dataset rồi chạy lại nhiều lần, nên tự dọn `dataset/processed/` để tránh lẫn file cũ.

### 4.5. Bước 3: cập nhật class names trong `training/data.yaml`

File hiện tại:

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

Bạn cần sửa phần `names` cho đúng dataset thực tế.

Ý nghĩa:

- `path`: thư mục dataset đã xử lý
- `train`: đường dẫn tương đối tới ảnh train
- `val`: đường dẫn tương đối tới ảnh val
- `test`: đường dẫn tương đối tới ảnh test
- `names`: mapping class id sang tên class

### 4.6. Bước 4: kiểm tra cấu hình train trong `training/train_config.yaml`

File hiện tại:

```yaml
model: yolo26s.pt
fallback_model: yolo26n.pt
data: training/data.yaml
epochs: 80
imgsz: 512
batch: 4
device: 0
workers: 2
cache: false
amp: true
patience: 20
project: runs/train
name: yolo_camera_rtx3050ti_4gb
```

Ý nghĩa các trường chính:

- `model`: model chính để train
- `fallback_model`: model fallback nếu cấu hình chính lỗi
- `data`: file dataset YAML cho Ultralytics
- `epochs`: số epoch
- `imgsz`: kích thước ảnh train
- `batch`: batch size
- `device`: thiết bị train, `0` thường nghĩa là GPU đầu tiên
- `workers`: số worker load data
- `cache`: có cache dataset hay không
- `amp`: mixed precision
- `patience`: early stopping patience
- `project`: thư mục gốc để Ultralytics lưu artifact
- `name`: tên run train

Khuyến nghị thực tế:

- Nếu máy yếu hoặc VRAM thấp, hãy giảm `imgsz` và `batch`.
- Nếu train bằng CPU, `device: 0` có thể không phù hợp; khi đó nên điều chỉnh lại theo môi trường thực tế.

### 4.7. Bước 5: chạy train

```powershell
.\.venv\Scripts\python run_train.py
```

`run_train.py` chỉ gọi `training.train_yolo.main()`.

Luồng train thực tế trong `training/train_yolo.py`:

1. Gọi `ensure_project_directories()`
2. Đọc config từ `training/train_config.yaml`
3. Tạo model bằng `YOLO(model_name)`
4. Gọi `model.train(**config)`
5. Nếu lỗi ở cấu hình chính:
   - log cảnh báo
   - đổi sang `fallback_model`
   - giảm `imgsz` xuống tối đa `416`
   - giảm `batch` xuống tối đa `4`
   - train lại
6. Sau khi train xong:
   - lấy `weights/best.pt` trong thư mục run của Ultralytics
   - copy sang `models/trained/best.pt`

### 4.8. Kết quả sau khi train nằm ở đâu

Sau khi train thành công, bạn sẽ có:

- artifact đầy đủ của Ultralytics trong `runs/train/...`
- model tốt nhất chuẩn hóa lại ở `models/trained/best.pt`

Đây là file model mà phần detect và validate sẽ ưu tiên dùng.

### 4.9. Bước 6: validate model

```powershell
.\.venv\Scripts\python training/validate_model.py
```

Script này:

- ưu tiên dùng `models/trained/best.pt`
- nếu chưa có thì fallback sang `yolo11n.pt`
- gọi `model.val(data="training/data.yaml", project="runs/val", name="validation")`
- lưu kết quả vào `runs/val`

### 4.10. Bước 7: export model

```powershell
.\.venv\Scripts\python training/export_model.py
```

Script này:

- bắt buộc phải có `models/trained/best.pt`
- load model đó
- gọi `model.export(format="onnx")`

Lưu ý:

- Code hiện tại không chỉ rõ output export phải nằm trong `models/exported/`.
- Vì vậy sau khi export, bạn nên kiểm tra file ONNX thực tế được Ultralytics sinh ra ở đâu trong lần chạy của bạn.

### 4.11. Quy trình huấn luyện đầy đủ nên chạy theo thứ tự nào

```powershell
.\.venv\Scripts\python training/prepare_dataset.py
.\.venv\Scripts\python training/split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training/validate_model.py
.\.venv\Scripts\python training/export_model.py
```

### 4.12. Nếu train xong muốn dùng ngay trong app detect

Không cần sửa code.

Lý do:

- `core/yolo_loader.py` ưu tiên `models/trained/best.pt`
- nên sau khi train xong, lần chạy detect kế tiếp sẽ tự ưu tiên model đã train nếu file này tồn tại

---

## 5. Giải thích rõ từng thư mục và file

### 5.1. Cây thư mục chính

```text
YOLO/
|-- .venv/
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
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- run_app.py
|-- run_detect.py
|-- run_tests.py
`-- run_train.py
```

### 5.2. Giải thích rõ từng thư mục

| Thư mục | Giải thích rõ vai trò |
|---|---|
| `.venv/` | Môi trường ảo của dự án. Toàn bộ Python package của dự án nên nằm ở đây để tránh xung đột với Python toàn máy. |
| `app/` | Tầng ứng dụng mỏng. Nơi đặt các thành phần hỗ trợ phần hiển thị hoặc tổ chức app-level. |
| `config/` | Nơi chứa toàn bộ cấu hình YAML của dự án: mode runtime, camera preset, thứ tự ưu tiên model và setting chung. |
| `core/` | Lõi nghiệp vụ detect realtime. Đây là phần quan trọng nhất của ứng dụng khi chạy camera. |
| `dataset/` | Toàn bộ dữ liệu phục vụ train: dữ liệu gốc, dữ liệu đã chia và dữ liệu mẫu. |
| `docs/` | Tài liệu kỹ thuật phụ, ghi chú hoặc mô tả thêm cho dự án. |
| `models/` | Nơi quản lý model theo từng trạng thái: model preload, model train xong, model export. |
| `output/` | Output phụ của hệ thống như log, ảnh chụp hoặc video do dự án tự tạo ra. |
| `runs/` | Artifact do Ultralytics sinh ra khi train, validate hoặc detect. Dùng để theo dõi lịch sử các lần chạy. |
| `tests/` | Bộ test tự động cho các module chính. |
| `training/` | Toàn bộ pipeline huấn luyện: chuẩn bị dữ liệu, chia dataset, train, validate, export và các file YAML liên quan. |
| `utils/` | Hàm tiện ích dùng chung: đọc YAML, tạo thư mục, logger, prompt runtime, vẽ kết quả detect. |

### 5.3. Giải thích chi tiết `config/`

| File | Vai trò thực tế |
|---|---|
| `config/settings.yaml` | File cấu hình runtime quan trọng nhất. Chứa mode `high/medium/low`, auto profile, confidence, camera preset, fallback profile. |
| `config/model_config.yaml` | Chứa thứ tự ưu tiên khi load model local. |
| `config/camera_config.yaml` | Chứa preset camera theo kích thước hiển thị. Hiện dự án chọn preset chủ yếu từ `settings.yaml`. |

### 5.4. Giải thích chi tiết `core/`

| File | Vai trò thực tế |
|---|---|
| `core/hardware_detector.py` | Đọc CPU, RAM, GPU, VRAM, PyTorch, CUDA để biết máy hiện tại chạy được gì. |
| `core/model_selector.py` | Biến mode người dùng chọn thành `RuntimeConfig` thật sự có thể chạy trên máy hiện tại. |
| `core/yolo_loader.py` | Tìm và load model local theo thứ tự ưu tiên. Đây là nơi quyết định app sẽ lấy model từ `models/trained/`, `models/pretrained/` hay file local khác. |
| `core/fallback_manager.py` | Sinh ra chuỗi fallback runtime để hệ thống thử cấu hình an toàn hơn khi cấu hình hiện tại lỗi. |
| `core/camera_detector.py` | Lõi nhận diện realtime: mở webcam, đọc frame, gọi YOLO predict, vẽ kết quả, xử lý lỗi inference và recovery. |

### 5.5. Giải thích chi tiết `dataset/`

| Thư mục | Vai trò thực tế |
|---|---|
| `dataset/raw/` | Dữ liệu gốc bạn tự đưa vào trước khi train. Đây là đầu vào của `training/split_dataset.py`. |
| `dataset/raw/images/` | Ảnh gốc để train. |
| `dataset/raw/labels/` | Label YOLO gốc tương ứng với ảnh. |
| `dataset/processed/` | Dữ liệu đã được chia train/val/test để Ultralytics sử dụng. |
| `dataset/processed/images/train` | Ảnh train. |
| `dataset/processed/images/val` | Ảnh validation. |
| `dataset/processed/images/test` | Ảnh test. |
| `dataset/processed/labels/train` | Label train. |
| `dataset/processed/labels/val` | Label validation. |
| `dataset/processed/labels/test` | Label test. |
| `dataset/sample/` | Nơi để dữ liệu mẫu hoặc dữ liệu thử nhanh. Hiện chưa đi vào pipeline train chính. |

### 5.6. Giải thích chi tiết `models/`

| Thư mục | Vai trò thực tế |
|---|---|
| `models/pretrained/` | Model YOLO preload sẵn để app detect chạy ngay cả khi chưa train model riêng. |
| `models/trained/` | Model sinh ra sau train. File chuẩn app ưu tiên dùng là `models/trained/best.pt`. |
| `models/exported/` | Thư mục dành cho model export. Tuy nhiên code export hiện chưa ép output vào đây, nên cần xem output thực tế sau khi export. |

### 5.7. Giải thích chi tiết `training/`

| File | Vai trò thực tế |
|---|---|
| `training/prepare_dataset.py` | Tạo sẵn các thư mục dataset cần dùng. Không chia dữ liệu. |
| `training/split_dataset.py` | Chia dữ liệu từ `dataset/raw/` sang `dataset/processed/` theo tỷ lệ train/val/test. |
| `training/train_yolo.py` | Logic train chính. Có fallback model nếu train cấu hình chính thất bại và có bước copy `best.pt` sang `models/trained/`. |
| `training/validate_model.py` | Validate model đã train hoặc fallback sang model mặc định nếu chưa có `best.pt`. |
| `training/export_model.py` | Export model đã train sang ONNX. |
| `training/train_config.yaml` | Cấu hình hyperparameter và đường dẫn train. |
| `training/data.yaml` | Cấu hình dataset để Ultralytics biết train/val/test nằm ở đâu và class names là gì. |
| `training/README_TRAINING.md` | Tài liệu ngắn riêng cho pipeline training. |

### 5.8. Giải thích chi tiết `utils/`

| File | Vai trò thực tế |
|---|---|
| `utils/file_utils.py` | Tạo thư mục, đọc YAML, ghi YAML và cache YAML. |
| `utils/logger.py` | Tạo logger dùng chung cho toàn dự án. |
| `utils/runtime_prompt.py` | In menu chọn mode, progress boot và dashboard terminal. |
| `utils/visualization.py` | Vẽ bounding boxes, label và phần hiển thị detect lên frame. |

### 5.9. Giải thích các file gốc ở thư mục root

| File | Vai trò thực tế |
|---|---|
| `run_app.py` | Entry point desktop chính. |
| `run_detect.py` | Entry point detect kiểu CLI. |
| `run_train.py` | Entry point train model. |
| `run_tests.py` | Entry point chạy toàn bộ test với dashboard terminal đẹp hơn mặc định. |
| `requirements.txt` | Danh sách package Python cần cài. |
| `README.md` | Tài liệu sử dụng và mô tả dự án. |
| `LICENSE` | Giấy phép MIT của dự án. |

### 5.10. Giải thích `tests/`

Thư mục `tests/` chứa test cho các nhóm chức năng chính:

- detect camera
- file utilities
- hardware detection
- runtime selection
- prompt terminal
- entrypoints
- training pipeline
- model loading

Mục đích:

- giảm rủi ro sửa code làm hỏng detect hoặc training
- kiểm tra nhanh toàn bộ hệ thống bằng `run_tests.py`

---

## 6. Khắc phục lỗi nhanh

### Có GPU nhưng vẫn chạy CPU

Nguyên nhân thường gặp:

- `.venv` đang dùng bản `torch +cpu`
- PyTorch không có CUDA build
- cài sai bản `torch`

Kiểm tra:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### Không mở được webcam

- Thử `--camera-index 1`
- Đóng ứng dụng khác đang dùng webcam
- Kiểm tra webcam có hoạt động ở ứng dụng khác không

### Camera lag

- Chạy `--mode medium`
- Nếu vẫn lag, chuyển `--mode low`
- Giảm độ phân giải camera hoặc dùng profile nhẹ hơn trong config

### Không detect được vật thể

- Kiểm tra ánh sáng
- Đưa vật thể gần camera hơn
- Kiểm tra class names và model đang dùng
- Nếu dùng model tự train, xác nhận `models/trained/best.pt` là model đúng

### Lỗi khi train

- Kiểm tra `training/data.yaml`
- Kiểm tra `dataset/raw/` đã có đủ ảnh và label chưa
- Kiểm tra `dataset/processed/` có được tạo sau bước split chưa
- Nếu thiếu VRAM, giảm `imgsz` và `batch` trong `training/train_config.yaml`

### Lỗi không load được model local

Kiểm tra:

- `models/pretrained/`
- `models/trained/best.pt`
- `config/model_config.yaml`

---

## Ghi chú

- Nên chạy toàn bộ dự án bằng Python trong `.venv`.
- Nếu muốn dùng GPU NVIDIA, cần cài đúng bản PyTorch có CUDA, không dùng bản `+cpu`.
- Sau khi train xong, app detect sẽ tự ưu tiên `models/trained/best.pt` nếu file này tồn tại.
- README này mô tả theo code hiện tại trong repo, không mô tả tính năng chưa có trong mã nguồn.
