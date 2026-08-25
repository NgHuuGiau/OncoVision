# Hướng Dẫn Cài Đặt

Tài liệu hướng dẫn cài đặt OncoVision trên Windows và kiểm tra môi trường trước khi sử dụng.

---

## 1. Yêu cầu hệ thống

### Bắt buộc

- Windows 10 hoặc 11
- Python 3.10 trở lên
- Quyền tạo virtual environment
- Quyền ghi trong thư mục dự án

### Khuyến nghị

- GPU NVIDIA (CUDA) — tối ưu train và inference
- Webcam — nếu sử dụng camera realtime
- PowerShell 7 — hiển thị Unicode tốt hơn

---

## 2. Cài đặt

```powershell
cd D:\OncoVision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu bị chặn script PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

---

## 3. Kiểm tra sau cài đặt

### Quét tổng thể

```powershell
python run_doctor.py --skip-camera-check
```

### Smoke check

```powershell
python run_smoke.py
python run_smoke.py --ci-safe --stop-on-fail
```

### Unit test

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

---

## 4. Chạy thử

| Mục đích | Lệnh |
|---|---|
| Menu tổng | `python run_menu.py` |
| Gợi ý runtime | `python run_app.py --advisor-only` |
| Camera realtime | `python run_app.py` |
| Chat UI desktop | `python run_chat.py` |
| Chat UI web | `python web_app.py` hoặc `python -m uvicorn web_app:app --host 127.0.0.1 --port 8000` |
| Medical CLI | `python run_medical.py status` |

---

## 5. Checklist sẵn sàng

1. `python run_doctor.py --skip-camera-check` — không còn lỗi nghiêm trọng
2. `python run_smoke.py` — pass
3. `python run_app.py --advisor-only` — in được gợi ý runtime
4. `python run_chat.py --check-only` — báo sẵn sàng
5. Web chat mở được tại `http://localhost:8000`

---

## 6. Xử lý lỗi cài đặt

### Không mở được camera

```powershell
python run_app.py --mode low --camera-index 1
```

Kiểm tra app khác có đang dùng webcam, đổi `camera-index` (0, 1, 2).

### Thiếu model

Kiểm tra `models/pretrained/` và `models/trained/`. Chạy `training/download_models.py` nếu cần tải pretrained.

### Lỗi CUDA / torch

Chạy `python run_app.py --advisor-only` và `python run_doctor.py --skip-camera-check` để xem GPU/CUDA có được nhận không.

### Chat UI chưa sẵn sàng

```powershell
python run_chat.py --check-only --auto-fix-icons
```

---

## 7. Khuyến nghị

- Luôn chạy `run_doctor.py` hoặc `run_smoke.py` trên máy mới trước khi train hoặc chạy camera.
- Mọi môi trường đều dùng `requirements.txt` duy nhất.
- Nếu gặp lỗi chưa rõ, xem [troubleshooting.md](troubleshooting.md) trước khi thay đổi cấu hình.
