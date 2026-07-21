# CI and Quality

[![CI](https://img.shields.io/badge/CI-Test%20Python%20Compatibility-2F855A)](../.github/workflows/test.yml)

Tep nay tom tat cach CI hoat dong va noi nao can xem khi muon giai thich vi sao pipeline do.

## 1. CI dang chay gi

Workflow chinh nam o [`.github/workflows/test.yml`](../.github/workflows/test.yml) va chay tren:

- `ubuntu-latest`
- `windows-latest`

Thu tu step:

1. checkout code
2. setup Python 3.10
3. install dependencies
4. `compileall`
5. `ruff`
6. `mypy`
7. verify entrypoint help
8. smoke check
9. unit tests

## 2. Step nao la hard fail

Mac dinh nhung step nay co the lam CI do:

- install dependencies
- `compileall`
- `ruff`
- verify entrypoint help
- smoke check
- unit tests

Step `mypy` đang được để `continue-on-error: true` để không chặn CI, nhưng vẫn ghi log trong artifact.

## 3. Phạm vi mypy hiện tại

`mypy` chỉ soi các module đang được bảo trì:

- `core`
- `medical`
- `training`
- `utils`
- `run_*.py`

HieraChain đã bị loại ra khỏi scope CI.

## 4. Smoke check

`run_smoke.py` có 2 chế độ:

- mặc định: cảnh báo và fail nếu một check fail
- `--ci-safe`: chỉ chạy các check nhẹ, phù hợp CI

Trong `--ci-safe`, `training-preflight` được hạ từ fail sang warn để tránh fail do dataset mẫu không tồn tại trên runner.

## 5. Khi CI do thi xem gi truoc

Thu tu kiem tra nhanh:

1. `ci-logs/04-ruff.txt`
2. `ci-logs/05-mypy-type-check.txt`
3. `ci-logs/07-smoke-check.txt`
4. `ci-logs/08-unit-tests.txt`

## 6. Lenh chay local

```powershell
python run_smoke.py --ci-safe --stop-on-fail
python run_train.py --check-only
python -m unittest discover -s tests -p "test_*.py"
```

## 7. Ghi chú

- Nếu muốn bật mypy quay lại thành gate cứng, cần dọn type debt trước.
- Nếu chỉ cần CI xanh và ổn định, cách hiện tại là đúng mục tiêu.
