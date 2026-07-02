# Lệnh Nhanh

Tài liệu này là bảng lệnh dùng hằng ngày. Nếu bạn đã hiểu repo, đây là file mở đầu nhanh nhất.

## 1. Kiểm Tra Môi Trường

```powershell
python run_doctor.py --skip-camera-check
python run_smoke.py
python run_smoke.py --ci-safe --stop-on-fail
python -m unittest discover -s tests -p "test_*.py"
```

## 2. Runtime / Camera

```powershell
python run_app.py --advisor-only
python run_app.py
python run_app.py --mode medium
python run_app.py --camera-index 1
python run_app.py --model models/trained/best.pt
```

## 3. Chat UI

```powershell
python run_chat.py --check-only
python run_chat.py --check-only --auto-fix-icons
python run_chat.py
python run_chat.py --cleanup-output --older-than-days 30
```

## 4. Medical CLI

```powershell
python run_medical.py init-dataset
python run_medical.py status
python run_medical.py ready
python run_medical.py sources
python run_medical.py cancer
```

`init-dataset` chỉ in layout mong đợi, không tự tạo dữ liệu trong `dataset/`.

## 5. Training Object Detection

```powershell
python run_train.py --check-only
python training\prepare_dataset.py
python training\validate_dataset.py
python training\split_dataset.py
python run_train.py
python training\validate_model.py
python training\export_model.py
```

## 6. Giải Thích Nhanh

| Lệnh | Dùng khi nào |
|---|---|
| `run_doctor.py --skip-camera-check` | Muốn doctor scan an toàn, không cần webcam |
| `run_smoke.py` | Muốn check nhanh chuỗi entrypoint |
| `run_smoke.py --ci-safe` | Muốn check nhẹ, phù hợp CI hoặc máy chưa đủ data |
| `run_app.py --advisor-only` | Muốn biết mode runtime phù hợp trước khi mở camera |
| `run_chat.py --check-only` | Muốn biết chat UI và medical preflight đã sẵn sàng chưa |
| `run_train.py --check-only` | Muốn xác minh train pipeline có thể chạy được hay không |

## 7. Trình Tự Khuyến Nghị Trên Máy Mới

```powershell
python run_menu.py
python run_doctor.py --skip-camera-check
python run_smoke.py --ci-safe --stop-on-fail
python run_app.py --advisor-only
python run_chat.py --check-only
python run_train.py --check-only
```
