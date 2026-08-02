# CI Và Quality Gate

Tài liệu tóm tắt cách CI hoạt động và các bước kiểm tra khi pipeline thất bại.

---

## 1. CI chạy những gì

Workflow chính tại `.github/workflows/test.yml`, chạy trên `ubuntu-latest` và `windows-latest`.

**Trình tự các bước:**

1. Checkout code
2. Setup Python 3.10
3. Cài đặt dependencies
4. `compileall` — kiểm tra biên dịch
5. `ruff` — kiểm tra code style
6. `mypy` — kiểm tra kiểu dữ liệu
7. Xác minh entrypoint help
8. Smoke check (`run_smoke.py`)
9. Unit test

---

## 2. Bước hard fail

Các bước có thể làm CI đỏ:

- Cài đặt dependencies
- `compileall`
- `ruff`
- Xác minh entrypoint help
- Smoke check
- Unit test

`mypy` hiện đang để `continue-on-error: true` (không chặn CI) nhưng vẫn ghi log.

---

## 3. Phạm vi mypy

Chỉ kiểm tra các module đang bảo trì:

- `core`
- `medical`
- `training`
- `utils`
- `run_*.py`

---

## 4. Smoke check

`run_smoke.py` có hai chế độ:

- **Mặc định**: cảnh báo và fail nếu check thất bại
- **`--ci-safe`**: chỉ chạy check nhẹ, phù hợp CI (training-preflight hạ từ fail xuống warn)

---

## 5. Khi CI đỏ — xem gì trước

Kiểm tra theo thứ tự:

1. `ci-logs/04-ruff.txt`
2. `ci-logs/05-mypy-type-check.txt`
3. `ci-logs/07-smoke-check.txt`
4. `ci-logs/08-unit-tests.txt`

### Lệnh chạy local

```powershell
python run_smoke.py --ci-safe --stop-on-fail
python run_train.py --check-only
python -m unittest discover -s tests -p "test_*.py"
```

---

## 6. Lưu ý

- Nếu muốn bật mypy thành gate cứng, cần dọn type debt trước.
- Ubuntu và Windows có thể khác biệt do dependency, log Ubuntu là chuẩn để debug.
- Mục tiêu hiện tại: CI xanh, ổn định, không chặn vì type debt.
