# Troubleshooting

Day la cac loi hay gap va noi can xem dau tien.

## 1. Camera khong mo duoc

Thu:

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
python run_app.py --mode low --camera-index 1
```

Neu van loi:

- kiem tra app khac co dang dung webcam khong
- thu `camera-index` khac
- xem `core/camera_runner.py`

## 2. Model khong co san

Kiem tra:

```text
models/pretrained/
models/trained/
```

Neu can tai model pretrained, xem `training/download_models.py`.

## 3. Chat UI chua san sang

Thu:

```powershell
python run_chat.py --check-only --auto-fix-icons
```

Neu that bai, xem:

- `utils/entrypoint_checks.py`
- `medical/system_status.py`
- `app/chat_ui/`

## 4. Train preflight fail

Thu:

```powershell
python run_train.py --check-only
```

Neu fail:

- kiem tra dataset raw va split
- xem `training/validate_dataset.py`
- xem `training/split_dataset.py`

## 5. Medical status sai

Xem:

- `medical/system_status.py`
- `medical/model_policy.py`
- `medical/training.py`
- `run_medical.py`

## 6. CI fail

Xem theo thu tu:

1. `.github/workflows/test.yml`
2. `run_smoke.py`
3. `ci-logs/04-ruff.txt`
4. `ci-logs/05-mypy-type-check.txt`
5. `ci-logs/07-smoke-check.txt`

## 7. Loi tren Ubuntu nhung Windows van xanh

Thuong do:

- khac biet dependency
- mypy quet type debt cu
- smoke check phu thuoc dataset mau

Khi gap truong hop nay, xem log Ubuntu truoc, khong dua tren Windows.
