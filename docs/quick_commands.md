# Lenh Nhanh

Tai lieu nay la bang lenh de dung hang ngay. Neu ban da hieu repo, day la file mo dau tien de thao tac nhanh.

## 1. Kiem Tra Moi Truong

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
python run_medical.py tcia-download --dry-run
python run_medical.py verify-tcia --collections-file training/tcia_collections_5.json
```

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

## 6. Giai Thich Nhanh

| Lenh | Dung khi nao |
|---|---|
| `run_doctor.py --skip-camera-check` | Muon doctor scan an toan, khong can webcam |
| `run_smoke.py` | Muon check nhanh chuoi entrypoint |
| `run_smoke.py --ci-safe` | Muon check nhe, phu hop CI hoac may chua du data |
| `run_app.py --advisor-only` | Muon biet mode runtime phu hop truoc khi mo camera |
| `run_chat.py --check-only` | Muon biet chat UI va medical preflight da san sang chua |
| `run_train.py --check-only` | Muon xac minh train pipeline co the chay duoc hay khong |

## 7. Trinh Tu Khuyen Nghi Tren May Moi

```powershell
python run_menu.py
python run_doctor.py --skip-camera-check
python run_smoke.py --ci-safe --stop-on-fail
python run_app.py --advisor-only
python run_chat.py --check-only
python run_train.py --check-only
```
