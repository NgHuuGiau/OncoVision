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
python web_app.py
# hoặc
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
# Mở http://127.0.0.1:8000
# Admin DB: http://127.0.0.1:8000/admin/db
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
python run_medical.py analyze --image path/to/ảnh.jpg --patient-code BN001
```

## 5. Model & phân tích

Hệ thống chỉ phân tích ảnh bằng model đã train sẵn (đặt trong `models/pretrained/`):

```powershell
# Kiểm tra đã đủ model chưa
python run_doctor.py --skip-camera-check

# Các model cần có trong models/pretrained/:
#   medical_7_cancers_cnn.pt   → 7 ung thư (gan, phổi, vú, dạ dày, đại trực tràng, tiền liệt, tử cung)
#   brain_classifier.pt        → Não (4 sub-label)
#   modality_classifier.pt     → Modality (8 loại ảnh y tế)
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
| `run_medical.py analyze` | Phân tích 1 ảnh y khoa |

## 7. Trình tự trên máy mới

```powershell
python run_menu.py
python run_doctor.py --skip-camera-check
python run_smoke.py --ci-safe --stop-on-fail
python run_app.py --advisor-only
python run_chat.py --check-only
```
