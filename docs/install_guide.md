# Hướng dẫn cài đặt

Tài liệu này mô tả quy trình cài đặt dự án YOLO trên Windows, kiểm tra môi trường và xử lý các lỗi thường gặp trước khi chạy camera realtime hoặc training.

## 1. Yêu cầu hệ thống

- Windows 10 hoặc Windows 11.
- Python 3.10 trở lên.
- Webcam nếu muốn chạy `run_app.py`.
- GPU NVIDIA là tùy chọn, nhưng nếu có thì nên cài đúng bản PyTorch CUDA để tăng tốc inference và training.

## 2. Tạo môi trường ảo

```powershell
cd D:\YOLO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Chuẩn bị thư mục dự án

```powershell
.\.venv\Scripts\python training\prepare_dataset.py
.\.venv\Scripts\python training\download_models.py
```

Hai lệnh trên sẽ:

- Tạo khung thư mục `dataset/`, `models/`, `output/` nếu chưa có.
- Tải các model YOLO pretrained cần thiết vào `models/pretrained/`.

## 4. Kiểm tra cài đặt sau khi setup

```powershell
.\.venv\Scripts\python run_doctor.py
```

`run_doctor.py` sẽ kiểm tra:

- CPU, RAM, GPU, VRAM.
- PyTorch và CUDA.
- Camera thật.
- Model local.
- Dataset raw và train/val split.

Nếu muốn bỏ qua bước check camera:

```powershell
.\.venv\Scripts\python run_doctor.py --skip-camera-check
```

## 5. Kiểm tra toàn bộ test

```powershell
.\.venv\Scripts\python run_tests.py --skip-camera-check
```

Nếu muốn coi camera thật là điều kiện bắt buộc:

```powershell
.\.venv\Scripts\python run_tests.py --strict-camera
```

## 6. Chạy thử từng công cụ

### Menu tổng

```powershell
.\.venv\Scripts\python run_menu.py
```

### Camera realtime

```powershell
.\.venv\Scripts\python run_app.py
```

Ví dụ chọn mode và camera index:

```powershell
.\.venv\Scripts\python run_app.py --mode medium --camera-index 0
```

Ví dụ dùng model custom:

```powershell
.\.venv\Scripts\python run_app.py --model models/trained/best.pt
```

### Công cụ tư vấn runtime

```powershell
.\.venv\Scripts\python run_tools.py
```

### Training

```powershell
.\.venv\Scripts\python run_train.py
```

## 7. Cấu hình UTF-8 và tiếng Việt trên Windows

Project hiện đã chủ động:

- Chuyển code page terminal sang `65001`.
- Reconfigure `stdout` và `stderr` sang UTF-8.
- Ghi file log bằng UTF-8.

Nếu bạn vẫn thấy lỗi font trong terminal cũ, hãy ưu tiên dùng:

- Windows Terminal
- PowerShell 7
- VS Code Terminal

## 8. Kiểm tra GPU / CUDA

Nếu `run_doctor.py` báo đang chạy CPU dù máy có GPU NVIDIA:

1. Kiểm tra driver NVIDIA.
2. Kiểm tra `torch.cuda.is_available()`.
3. Cài lại PyTorch đúng bản CUDA của máy.

Ví dụ:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 9. Xử lý lỗi camera

Nếu camera không mở được hoặc không trả frame:

- Đóng app camera khác đang chiếm webcam.
- Thử đổi `--camera-index 1` hoặc `--camera-index 2`.
- Kiểm tra quyền truy cập camera trong Windows Settings.

Ví dụ:

```powershell
.\.venv\Scripts\python run_app.py --camera-index 1
```

## 10. Xử lý lỗi model

Nếu camera mở được nhưng không nhận diện:

- Kiểm tra `models/pretrained/` đã có file YOLO chưa.
- Kiểm tra `run_app.py` có đang dùng model đúng mục đích hay không.
- Nếu muốn detect class custom, hãy truyền `--model models/trained/best.pt`.
- Kiểm tra lại ngưỡng trong `config/settings.yaml`.

## 11. Kiểm tra cuối sau cài đặt

Sau khi hoàn tất, nên chạy theo thứ tự:

```powershell
.\.venv\Scripts\python run_doctor.py --skip-camera-check
.\.venv\Scripts\python run_tests.py --skip-camera-check
.\.venv\Scripts\python run_app.py
```

Nếu cả 3 bước trên đều ổn, môi trường đã sẵn sàng cho runtime camera và training.
