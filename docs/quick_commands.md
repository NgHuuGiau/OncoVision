# Lệnh Nhanh

Bảng lệnh sử dụng hàng ngày. Nếu đã hiểu hệ thống, đây là file tra cứu nhanh nhất.

---

## 1. Kiểm tra môi trường

```powershell
python run_doctor.py --skip-camera-check
python run_smoke.py
python run_smoke.py --ci-safe --stop-on-fail
python -m unittest discover -s tests -p "test_*.py"
```

## 2. Chat AI

### Desktop

```powershell
python run_chat.py --check-only
python run_chat.py --check-only --auto-fix-icons
python run_chat.py
python run_chat.py --cleanup-output --older-than-days 30
```

### Web

```powershell
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
# Mở http://localhost:8000
# Admin DB: http://localhost:8000/admin/db
```

## 3. Camera realtime

```powershell
python run_app.py --advisor-only
python run_app.py
python run_app.py --mode medium
python run_app.py --camera-index 1
python run_app.py --model models/trained/best.pt
```

## 4. Medical CLI

```powershell
python run_medical.py status
python run_medical.py ready
python run_medical.py sources
python run_medical.py cancer
python run_medical.py init-dataset
python run_medical.py train-modality
```

## 5. Huấn luyện

### YOLO detection

```powershell
python run_train.py
```

### CNN classifier

```powershell
# 7 ung thư (cấu hình cao: convnext_tiny @512, epochs 30) — GPU 4GB, ~190k ảnh
python run_train_7cancers_high.py

# Não (4 sub-label, convnext_tiny @512, epochs 35)
python run_train_brain_high.py

# Modality (8 loại hình ảnh, resnet18 @320, epochs 20)
python run_medical.py train-modality
```

## 6. Giải thích nhanh

| Lệnh | Mục đích |
|---|---|
| `run_doctor.py --skip-camera-check` | Quét tổng thể, không cần webcam |
| `run_smoke.py` | Kiểm tra nhanh chuỗi entrypoint |
| `run_smoke.py --ci-safe` | Kiểm tra nhẹ, phù hợp CI |
| `run_app.py --advisor-only` | Gợi ý runtime trước khi mở camera |
| `run_chat.py --check-only` | Kiểm tra chat UI và medical sẵn sàng |
| `run_chat.py --cleanup-output` | Dọn file output cũ |
| `run_train.py --check-only` | Xác minh pipeline train có thể chạy |

## 7. Trình tự trên máy mới

```powershell
python run_menu.py
python run_doctor.py --skip-camera-check
python run_smoke.py --ci-safe --stop-on-fail
python run_app.py --advisor-only
python run_chat.py --check-only
python run_train.py --check-only
```
