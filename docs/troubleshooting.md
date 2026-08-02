# Xử Lý Sự Cố

Các lỗi thường gặp và hướng dẫn xử lý.

---

## 1. Camera không mở được

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
python run_app.py --mode low --camera-index 1
```

Nếu vẫn lỗi:

- Kiểm tra ứng dụng khác có đang dùng webcam không
- Thử camera-index khác (0, 1, 2)
- Xem `core/camera_runner.py` để debug

---

## 2. Model không có sẵn

Kiểm tra:

- `models/pretrained/` — model tiền huấn luyện
- `models/trained/` — model đã train

Nếu cần tải pretrained: `python training/download_models.py`.

---

## 3. Chat UI chưa sẵn sàng

```powershell
python run_chat.py --check-only --auto-fix-icons
```

Nếu thất bại, xem:

- `utils/entrypoint_checks.py`
- `medical/system_status.py`
- `app/chat_ui/`

---

## 4. Train preflight fail

```powershell
python run_train.py --check-only
```

Nếu fail: kiểm tra dataset raw, dataset split, xem `training/validate_dataset.py` và `training/split_dataset.py`.

---

## 5. Medical status sai

Xem:

- `medical/system_status.py`
- `medical/model_policy.py`
- `medical/training.py`
- `run_medical.py`

---

## 6. CI fail

Xem theo thứ tự:

1. `.github/workflows/test.yml`
2. `run_smoke.py`
3. `ci-logs/04-ruff.txt`
4. `ci-logs/05-mypy-type-check.txt`
5. `ci-logs/07-smoke-check.txt`

---

## 7. Lỗi Ubuntu nhưng Windows xanh

Nguyên nhân thường gặp:

- Khác biệt dependency giữa hai nền tảng
- mypy quét type debt cũ
- Smoke check phụ thuộc dataset mẫu

Luôn xem log Ubuntu trước — đó là chuẩn debug.

---

## 8. Web Chat UI không mở được

```powershell
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
# Mở http://localhost:8000
```

Nếu lỗi:

- Kiểm tra đã cài dependencies: `pip install -r requirements.txt`
- Kiểm tra port 8000 có bị chiếm: `netstat -ano | findstr :8000`
- Thử port khác: `python -m uvicorn web_app:app --host 0.0.0.0 --port 8080`
- Xem log server để tìm lỗi cụ thể
- Admin DB: `http://localhost:8000/admin/db`
