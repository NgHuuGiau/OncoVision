# Hướng Dẫn Cài Đặt

Tài liệu này mô tả cách cài OncoVision trên Windows và cách kiểm tra môi trường trước khi chạy camera hoặc training.

## Yêu cầu hệ thống

- Windows 10 hoặc Windows 11
- Python 3.10 trở lên
- Webcam nếu muốn chạy camera realtime
- GPU NVIDIA là tùy chọn, nhưng nên có nếu muốn tăng tốc inference và training

## Tạo môi trường ảo

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

## Chuẩn bị thư mục dự án

```powershell
.\.venv\Scripts\python run_menu.py
```

Mở menu một lần là hệ thống sẽ tự tạo đủ cây thư mục cần thiết.

## Kiểm tra cài đặt

```powershell
.\.venv\Scripts\python run_doctor.py
```

Nếu muốn bỏ qua camera:

```powershell
.\.venv\Scripts\python run_doctor.py --skip-camera-check
```

## Kiểm tra nhanh

```powershell
.\.venv\Scripts\python run_tests.py
.\.venv\Scripts\python run_smoke.py
```

## Chạy thử từng công cụ

### Menu tổng

```powershell
.\.venv\Scripts\python run_menu.py
```

### Camera realtime

```powershell
.\.venv\Scripts\python run_app.py
.\.venv\Scripts\python run_app.py --mode medium --camera-index 0
.\.venv\Scripts\python run_app.py --model models/trained/best.pt
.\.venv\Scripts\python run_app.py --advisor-only
```

### Chat UI

```powershell
.\.venv\Scripts\python run_chat.py
.\.venv\Scripts\python run_chat.py --check-only
```

### Training

```powershell
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python run_train.py --check-only
```

## Xử lý lỗi thường gặp

- Camera không mở được: thử `--camera-index 1` hoặc `2`
- Thiếu model local: kiểm tra `models/pretrained/`
- Lỗi CUDA: kiểm tra driver NVIDIA và bản PyTorch
- Terminal hiển thị lỗi font: ưu tiên Windows Terminal hoặc PowerShell 7

## Kiểm tra cuối sau khi cài đặt

```powershell
.\.venv\Scripts\python run_doctor.py --skip-camera-check
.\.venv\Scripts\python run_smoke.py
.\.venv\Scripts\python run_menu.py
```

Nếu ba bước trên chạy ổn, môi trường đã sẵn sàng.
