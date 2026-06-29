# Lệnh Nhanh

```powershell
python run_app.py --advisor-only
python run_chat.py --check-only
python run_doctor.py --skip-camera-check
python run_train.py --check-only
python run_medical.py ready
python run_smoke.py
python -m unittest discover -v
```

- `run_doctor.py --skip-camera-check` kiểm tra tổng thể mà không cần webcam thật
- `run_medical.py ready` cho biết skin lesion đã sẵn sàng train hay chưa
- `run_smoke.py` dùng để kiểm tra nhanh chuỗi entrypoint quan trọng
