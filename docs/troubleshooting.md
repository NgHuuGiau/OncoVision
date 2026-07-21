# Troubleshooting

Đây là các lỗi hay gặp và nơi cần xem đầu tiên.

## 1. Camera không mở được

Thử:

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
python run_app.py --mode low --camera-index 1
```

Nếu vẫn lỗi:

- kiểm tra app khác có đang dùng webcam không
- thử `camera-index` khác
- xem `core/camera_runner.py`

## 2. Model không có sẵn

Kiểm tra:

```text
models/pretrained/
models/trained/
```

Nếu cần tải model pretrained, xem `training/download_models.py`.

## 3. Chat UI chưa sẵn sàng

Thử:

```powershell
python run_chat.py --check-only --auto-fix-icons
```

Nếu thất bại, xem:

- `utils/entrypoint_checks.py`
- `medical/system_status.py`
- `app/chat_ui/`

## 4. Train preflight fail

Thử:

```powershell
python run_train.py --check-only
```

Nếu fail:

- kiểm tra dataset raw và split
- xem `training/validate_dataset.py`
- xem `training/split_dataset.py`

## 5. Medical status sai

Xem:

- `medical/system_status.py`
- `medical/model_policy.py`
- `medical/training.py`
- `run_medical.py`

## 6. CI fail

Xem theo thứ tự:

1. `.github/workflows/test.yml`
2. `run_smoke.py`
3. `ci-logs/04-ruff.txt`
4. `ci-logs/05-mypy-type-check.txt`
5. `ci-logs/07-smoke-check.txt`

## 7. Lỗi trên Ubuntu nhưng Windows vẫn xanh

Thường do:

- khác biệt dependency
- mypy quét type debt cũ
- smoke check phụ thuộc dataset mẫu

Khi gặp trường hợp này, xem log Ubuntu trước, không dựa trên Windows.

## 8. Web Chat UI không mở được

Thử:

```powershell
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
```

Mở trình duyệt: `http://localhost:8000`

Nếu vẫn lỗi:

- kiểm tra port 8000 có bị chiếm không: `netstat -ano | findstr :8000`
- thử port khác: `python -m uvicorn web_app:app --host 0.0.0.0 --port 8080`
- xem log server để tìm lỗi cụ thể
- admin DB viewer: `http://localhost:8000/admin/db`
