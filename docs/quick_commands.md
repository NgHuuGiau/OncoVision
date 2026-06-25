# Lệnh Nhanh

Dùng các lệnh này để kiểm tra nhanh dự án:

```powershell
python run_app.py --advisor-only
python run_chat.py --check-only
python run_train.py --check-only
python run_smoke.py
python -m unittest discover -v
```

- `run_app.py --advisor-only` in khuyến nghị runtime mà không mở camera
- `run_chat.py --check-only` kiểm tra thư viện, icon, output và medical model
- `run_train.py --check-only` kiểm tra training dependencies, model và dataset
- `run_smoke.py` chạy chuỗi entrypoint an toàn dùng cho CI
- `python -m unittest discover -v` chạy toàn bộ test suite
