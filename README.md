# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Ứng dụng nhận diện vật thể realtime bằng YOLO chạy trên desktop Python + OpenCV. Dự án có:

- giao diện terminal tiếng Việt
- tự dò cấu hình máy trước khi chạy
- tự gợi ý mức phù hợp theo `CPU`, `RAM`, `GPU`, `VRAM`, `CUDA`
- hỗ trợ chụp mẫu train trực tiếp từ camera
- pipeline train riêng
- test dashboard để kiểm tra toàn hệ thống

Lưu ý quan trọng:

- repo này chỉ chứa code
- repo này không kèm model `.pt`
- repo này không kèm dataset
- trước khi chạy camera hoặc train, bạn phải tự đặt model vào `models/pretrained/`

## 1. Dự án này làm gì

Luồng chính của dự án:

```text
Webcam -> OpenCV -> YOLO -> Python -> Kết quả detect -> Cửa sổ camera
```

Dự án có 4 nhóm chức năng chính:

1. chạy camera realtime để detect vật thể
2. tự chọn cấu hình phù hợp theo máy
3. chụp mẫu train trực tiếp từ webcam
4. train / validate / export model

## 2. Các model dự án dùng

Dự án hiện dùng họ model `YOLO26` với 5 mức:

| Mức | Model | Ý nghĩa |
|---|---|---|
| Nhẹ nhất | `yolo26n.pt` | nhẹ nhất, dễ chạy nhất |
| Cân bằng | `yolo26s.pt` | cân bằng tốc độ và độ chính xác |
| Khá mạnh | `yolo26m.pt` | nặng hơn, chính xác hơn |
| Mạnh | `yolo26l.pt` | dùng cho GPU mạnh |
| Mạnh nhất | `yolo26x.pt` | dùng cho GPU rất mạnh |

Hệ thống sẽ không ép model lớn nhất bằng mọi giá. Nó sẽ cố chọn mức cao nhất mà máy vẫn còn chạy ổn định.

Ví dụ:

- máy rất mạnh, VRAM lớn -> có thể lên `yolo26x.pt`
- máy tầm trung như RTX 3050 Ti 4GB -> thường hợp `yolo26s.pt` hoặc `yolo26m.pt`
- máy yếu hoặc CPU-only -> về `yolo26n.pt`

## 3. Trước khi cài

Bạn nên chuẩn bị:

- Windows + PowerShell
- Python `3.10+`
- webcam hoạt động bình thường
- nếu muốn chạy GPU NVIDIA: cài driver và dùng đúng bản PyTorch CUDA

Kiểm tra Python:

```powershell
python --version
```

Nếu lệnh này không chạy, bạn cần cài Python trước.

## 4. Cài đặt từ đầu

### Bước 1: clone dự án

```powershell
git clone <repo-url>
cd D:\YOLO
```

Nếu bạn đã có thư mục dự án sẵn rồi thì chỉ cần:

```powershell
cd D:\YOLO
```

### Bước 2: tạo môi trường ảo

```powershell
python -m venv .venv
```

Ý nghĩa:

- tạo Python riêng cho dự án
- không làm bẩn Python hệ thống
- tránh lỗi lệch version package

### Bước 3: kích hoạt môi trường

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

Khi thành công, terminal sẽ có dạng:

```powershell
(.venv) PS D:\YOLO>
```

### Bước 4: cài thư viện

```powershell
pip install -r requirements.txt
```

`requirements.txt` hiện gồm các nhóm chính:

- `ultralytics`
- `opencv-python`
- `numpy`
- `pillow`
- `psutil`
- `GPUtil`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `PyYAML`
- `torch`
- `torchvision`
- `torchaudio`

### Bước 5: cài PyTorch đúng loại

#### Nếu chỉ chạy CPU

```powershell
.\\.venv\\Scripts\\python -m pip uninstall -y torch torchvision torchaudio
.\\.venv\\Scripts\\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Nếu chạy GPU NVIDIA CUDA

```powershell
.\\.venv\\Scripts\\python -m pip uninstall -y torch torchvision torchaudio
.\\.venv\\Scripts\\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Bước 6: kiểm tra PyTorch và CUDA

```powershell
.\\.venv\\Scripts\\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Ý nghĩa:

- `torch.__version__`: version PyTorch
- `torch.version.cuda`: bản CUDA build cùng PyTorch
- `torch.cuda.is_available()`: máy có dùng CUDA thật được không

Nếu `False`:

- có thể bạn đang dùng bản CPU
- hoặc driver/CUDA chưa khớp
- hoặc máy không có GPU NVIDIA phù hợp

## 5. Tạo thư mục dự án cần thiết

Chạy:

```powershell
.\\.venv\\Scripts\\python training/prepare_dataset.py
```

Lệnh này sẽ tạo sẵn các thư mục như:

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
- `output/`
- `runs/`

Lệnh này chỉ chuẩn bị cấu trúc thư mục, chưa train gì cả.

## 6. Đặt model vào đúng chỗ

Repo không kèm model. Bạn phải tự chép model vào:

```text
models/pretrained/
```

Nên có đủ:

- `models/pretrained/yolo26n.pt`
- `models/pretrained/yolo26s.pt`
- `models/pretrained/yolo26m.pt`
- `models/pretrained/yolo26l.pt`
- `models/pretrained/yolo26x.pt`

Bạn không bắt buộc phải có đủ 5 file mới chạy được, nhưng nếu muốn hệ thống auto chọn đúng mọi mức thì nên chuẩn bị đủ.

Thứ tự ưu tiên load model hiện tại:

1. `models/trained/best.pt`
2. model local trong `models/pretrained/`
3. file cùng tên nếu bạn đặt ở chỗ khác trong working directory

## 7. Chạy test trước khi dùng

```powershell
.\\.venv\\Scripts\\python run_tests.py
```

Ý nghĩa:

- kiểm tra toàn bộ hệ thống có lỗi import, logic, dashboard, training pipeline hay không
- nếu test pass thì môi trường code đang ổn

Trạng thái hiện tại của dự án:

- `50/50 PASS`

## 8. Chạy camera realtime

### Chạy app camera chính

```powershell
.\\.venv\\Scripts\\python run_app.py
```

### Chạy detect camera

```powershell
.\\.venv\\Scripts\\python run_detect.py
```

Hai file này đều:

- dò phần cứng máy
- chọn hoặc gợi ý runtime
- load model
- mở webcam
- chạy detect realtime

Khác nhau chủ yếu ở vai trò entrypoint và cách trình bày luồng chạy.

## 9. Menu chọn cấu hình hoạt động ra sao

Người dùng hiện thấy 3 mức:

- `Cao nhất`
- `Trung bình`
- `Yếu`

Nhưng bên trong hệ thống sẽ suy luận tiếp dựa trên cấu hình thật của máy.

Ví dụ:

- `Cao nhất`
  - máy rất mạnh -> có thể dùng `yolo26x.pt`
  - máy 8GB VRAM -> có thể dùng `yolo26l.pt`
  - máy 4GB VRAM -> có thể tự hạ xuống `yolo26m.pt`
- `Trung bình`
  - máy 4GB VRAM -> thường là `yolo26s.pt`
- `Yếu`
  - máy yếu hoặc CPU-only -> `yolo26n.pt`

Mục tiêu là:

- máy mạnh thì chạy cấu hình cao
- máy trung bình thì chạy mức vừa
- máy yếu thì chạy mức nhẹ
- nếu người dùng ép mức quá cao, hệ thống vẫn tự hạ để tránh crash hoặc lag nặng

## 10. Màu trong terminal nghĩa là gì

- xanh lá: chạy được, trạng thái tốt
- vàng: cảnh báo, trung gian, hoặc còn giới hạn
- đỏ: lỗi hoặc không đủ điều kiện chạy

Nếu một lệnh không chạy được, terminal sẽ hiện:

- `LÝ DO`
- `Lý do không chạy`
- `GỢI Ý`
- `LỆNH THỬ` hoặc `LỆNH NHANH`

Tức là bạn không cần phải đọc traceback dài mới biết lỗi gì.

## 11. Chạy với mode cố định

```powershell
.\\.venv\\Scripts\\python run_app.py --mode high
.\\.venv\\Scripts\\python run_app.py --mode medium
.\\.venv\\Scripts\\python run_app.py --mode low
```

Ý nghĩa:

- `--mode high`: yêu cầu mức cao
- `--mode medium`: yêu cầu mức cân bằng
- `--mode low`: yêu cầu mức nhẹ

Nhưng hệ thống vẫn có quyền tự hạ nếu phần cứng không chịu nổi.

## 12. Đổi camera index

Nếu máy có nhiều camera hoặc camera mặc định không đúng:

```powershell
.\\.venv\\Scripts\\python run_app.py --camera-index 1
.\\.venv\\Scripts\\python run_detect.py --camera-index 1
```

## 13. Chụp mẫu train trực tiếp từ camera

Khi camera đang chạy:

- bấm `T` để vào chế độ chụp mẫu
- hệ thống sẽ kiểm tra độ ổn định trong `5` giây
- nếu rung/lắc thì đếm lại
- nếu đủ ổn định thì hiện bảng nhập tên mẫu

Phím dùng trong lúc đặt tên:

- `Enter`: lưu
- `Backspace`: xóa ký tự
- `Esc`: hủy

Mẫu sẽ lưu vào:

- `dataset/sample/images/`
- `dataset/sample/labels/`

Lưu ý:

- `dataset/sample/` chỉ là nơi gom mẫu nhanh
- nguồn train chính thức vẫn là `dataset/raw/`

## 14. Huấn luyện từ đầu

Toàn bộ lệnh dưới đây chạy từ:

```powershell
PS D:\YOLO>
```

### Bước 1: chuẩn bị thư mục

```powershell
.\\.venv\\Scripts\\python training/prepare_dataset.py
```

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

Ý nghĩa:

- `class_id`: id class
- 4 số sau: box đã normalize trong khoảng `0 -> 1`

### Bước 3: kiểm tra dataset raw

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

### Bước 5: kiểm tra `training/data.yaml`

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

Nếu label dùng `class_id` không khớp file này thì train sẽ sai.

### Bước 6: kiểm tra `training/train_config.yaml`

Các mục quan trọng:

- `model`: model chính để train
- `fallback_model`: model fallback nếu cấu hình chính lỗi
- `epochs`: số vòng train
- `imgsz`: kích thước ảnh train
- `batch`: batch size
- `device`: GPU hoặc CPU
- `project`: thư mục `runs`
- `name`: tên lần train

Nếu máy yếu hoặc thiếu VRAM, nên giảm:

- `imgsz`
- `batch`

### Bước 7: chạy train

```powershell
.\\.venv\\Scripts\\python run_train.py
```

Lệnh này sẽ:

1. đọc `training/train_config.yaml`
2. kiểm tra dữ liệu đã split
3. load model train chính
4. fallback sang model nhẹ hơn nếu cần
5. copy `best.pt` về `models/trained/best.pt`

### Bước 8: validate model

```powershell
.\\.venv\\Scripts\\python training/validate_model.py
```

Lệnh này dùng tập `val` để đo chất lượng model sau train.

### Bước 9: export model

```powershell
.\\.venv\\Scripts\\python training/export_model.py
```

Lệnh này lấy `models/trained/best.pt` và export sang ONNX.

### Chuỗi lệnh đầy đủ

```powershell
.\\.venv\\Scripts\\python training/prepare_dataset.py
.\\.venv\\Scripts\\python training/validate_dataset.py
.\\.venv\\Scripts\\python training/split_dataset.py
.\\.venv\\Scripts\\python run_train.py
.\\.venv\\Scripts\\python training/validate_model.py
.\\.venv\\Scripts\\python training/export_model.py
```

## 15. Khi nào từng lệnh không chạy được

| Lệnh | Lý do thường gặp |
|---|---|
| `training/prepare_dataset.py` | hiếm khi lỗi, chủ yếu chỉ tạo thư mục |
| `training/validate_dataset.py` | chưa có ảnh trong `dataset/raw/images` |
| `training/split_dataset.py` | dataset raw rỗng hoặc raw không hợp lệ |
| `run_train.py` | chưa có `dataset/processed/images/train` hoặc `val` |
| `training/validate_model.py` | chưa có `dataset/processed/images/val` |
| `training/export_model.py` | chưa có `models/trained/best.pt` |
| `run_app.py` / `run_detect.py` | không mở được webcam, không load được model, hoặc CUDA không sẵn sàng |

## 16. Cấu trúc thư mục chính

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

- `app/`: helper entry cho camera app
- `config/`: YAML cấu hình
- `core/`: lõi detect, chọn runtime, load model
- `dataset/`: dữ liệu raw, processed, sample
- `docs/`: tài liệu phụ
- `models/`: pretrained, trained, exported
- `output/`: ảnh, log, video sinh ra
- `runs/`: artifact do Ultralytics tạo
- `tests/`: test tự động
- `training/`: pipeline huấn luyện
- `utils/`: helper dùng chung

## 17. File quan trọng

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

## 18. Gợi ý dùng thực tế

- máy rất mạnh: chọn `Cao nhất`
- máy tầm trung như RTX 3050 Ti 4GB: nên ưu tiên `Trung bình`
- máy yếu hoặc CPU-only: chọn `Yếu`

Nếu bạn ép `Cao nhất` trên máy yếu, hệ thống vẫn sẽ tự hạ về model hợp lý nhất còn chạy được.

## 19. Trạng thái hiện tại

- terminal tiếng Việt có dấu
- menu chọn mode đã gợi ý theo cấu hình máy
- hỗ trợ đủ logic cho `YOLO26 n/s/m/l/x`
- `run_tests.py` hiện pass `50/50`
