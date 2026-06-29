# Hướng Dẫn Cài Đặt

Tài liệu này hướng dẫn cài đặt OncoVision trên Windows, khởi tạo thư mục dự án và kiểm tra môi trường trước khi đưa vào sử dụng.

## 1. Yêu Cầu Hệ Thống

### Bắt buộc

- Windows 10 hoặc Windows 11
- Python 3.10 trở lên
- Quyền tạo virtual environment
- Quyền ghi trong thư mục dự án

### Khuyến nghị

- GPU NVIDIA nếu muốn tối ưu train và inference
- Webcam nếu muốn chạy `run_app.py`
- Windows Terminal hoặc PowerShell 7 để hiển thị Unicode tốt hơn

## 2. Tạo Môi Trường Ảo

```powershell
cd D:\OncoVision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu bị chặn script trong PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Khởi Tạo Thư Mục Dự Án

Lần chạy đầu tiên nên dùng:

```powershell
python run_menu.py
```

Lệnh này giúp tạo và đồng bộ các nhóm thư mục như:

```text
dataset/
models/
output/
runs/
```

Sau đó các luồng `medical`, `training`, `chat`, `camera` sẽ tự bổ sung những thư mục còn cần thiết.

## 4. Kiểm Tra Sau Cài Đặt

### Kiểm tra tổng quát an toàn

```powershell
python run_doctor.py --skip-camera-check
```

Mục đích:

- kiểm tra dependency quan trọng,
- kiểm tra model,
- kiểm tra dataset,
- kiểm tra output directories,
- không yêu cầu webcam thật.

### Kiểm tra chuỗi entrypoint

```powershell
python run_smoke.py
```

Nếu đang chạy trong CI hoặc muốn check nhẹ:

```powershell
python run_smoke.py --ci-safe --stop-on-fail
```

### Kiểm tra unit test

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## 5. Chạy Thử Từng Chức Năng

### Menu tổng

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

## 6. Cấu Trúc Dependency

Repo hiện có ba nhóm dependency để dễ vận hành:

| File | Mục đích |
|---|---|
| `requirements.txt` | Bộ dependency đầy đủ cho runtime chính |
| `requirements-ci.txt` | Dependency tối thiểu để chạy workflow CI |
| `requirements-dev.txt` | Runtime đầy đủ + công cụ phát triển như `ruff`, `mypy` |

## 7. Xử Lý Lỗi Thường Gặp

### Không mở được camera

Thử:

```powershell
python run_app.py --mode low --camera-index 1
```

Nếu vẫn lỗi:

- kiểm tra app khác có đang giữ webcam không,
- chạy `run_doctor.py --skip-camera-check` để xem cảnh báo hệ thống,
- đổi `camera-index` sang `0`, `1`, `2`.

### Thiếu model

Kiểm tra:

```text
models/pretrained/
models/trained/
```

Nếu cần object detection pretrained model, xem các script trong `training/download_models.py`.

### Lỗi CUDA hoặc torch

Kiểm tra:

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
```

Hai lệnh này sẽ cho biết:

- GPU có được nhận không,
- CUDA có sẵn sàng không,
- torch đang build theo CPU hay CUDA.

### Giao diện chat chưa sẵn sàng

Dùng:

```powershell
python run_chat.py --check-only --auto-fix-icons
```

Lệnh này giúp:

- check icon,
- check module bắt buộc,
- check medical model status,
- tự tạo icon nếu đang thiếu.

## 8. Checklist Sau Khi Cài Đặt Xong

Môi trường có thể xem là sẵn sàng khi các mục sau đều ổn:

1. `python run_doctor.py --skip-camera-check` chạy xong không có lỗi nghiêm trọng.
2. `python run_smoke.py` hoặc `python run_smoke.py --ci-safe` pass.
3. `python run_app.py --advisor-only` in được khuyến nghị runtime.
4. `python run_chat.py --check-only` báo trạng thái sẵn sàng.
5. `python run_train.py --check-only` không bị fail do thiếu dependency.

## 9. Khuyến Nghị Cho Team

- Không chạy thẳng training hoặc camera trên máy mới mà chưa chạy doctor/smoke.
- Nếu chỉ muốn review code trên CI, ưu tiên `requirements-ci.txt`.
- Nếu debug local feature đầy đủ, dùng `requirements.txt` hoặc `requirements-dev.txt`.
