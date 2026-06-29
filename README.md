# OncoVision

OncoVision la bo cong cu AI noi bo gom hai nhanh chinh:

- `Y duoc`: quan ly dataset y khoa, phan tich anh, theo doi trang thai model, va ho tro quy trinh TCIA.
- `Vat the`: train, danh gia, va chay YOLO realtime cho bai toan nhan dien vat the bang camera.

README nay duoc viet theo huong giup nguoi moi vao du an co the:

1. hieu nhanh du an dang lam gi,
2. biet can chay lenh nao truoc,
3. tim dung tai lieu chi tiet trong thu muc `docs/`.

## Diem Noi Bat

| Nhom | Gia tri |
|---|---|
| Realtime camera | Chay camera desktop voi che do runtime `auto`, `high`, `medium`, `low` |
| Medical workflow | Quan ly skin lesion dataset, TCIA collections, output reports, va trang thai model |
| Entrypoint ro rang | Moi tac vu lon deu co file `run_*.py` rieng |
| Kiem tra nhanh | Co `run_doctor.py`, `run_smoke.py`, `run_tests.py` de soat loi som |
| Tai lieu van hanh | Co bo `docs/` cho cai dat, training, runtime, medical va tong quan kien truc |

## Ban Do Nhanh Cua Du An

```text
run_menu.py      -> menu tong hop
run_app.py       -> camera realtime / runtime advisor
run_chat.py      -> chat UI / medical preflight / cleanup output
run_doctor.py    -> doctor scan tong the he thong
run_train.py     -> train YOLO object detection
run_medical.py   -> CLI quan ly luong medical
run_smoke.py     -> smoke check entrypoint
run_tests.py     -> dashboard unit test
```

## Khi Nao Nen Dung File Nao

| Ban muon lam gi | Entrypoint nen dung |
|---|---|
| Xem toan bo chuc nang dang co | `python run_menu.py` |
| Kiem tra may nen chay runtime nao | `python run_app.py --advisor-only` |
| Mo camera realtime | `python run_app.py` |
| Kiem tra chat UI co san sang khong | `python run_chat.py --check-only` |
| Kiem tra tong the moi truong | `python run_doctor.py --skip-camera-check` |
| Train YOLO object detection | `python run_train.py` |
| Kiem tra nhanh luong y duoc | `python run_medical.py status` |
| Chay smoke check an toan | `python run_smoke.py` |
| Chay unit test | `python -m unittest discover -s tests -p "test_*.py"` |

## Cau Truc Luong Nghiep Vu

### 1. Nhanh Vat The

Muc tieu:

- quan ly dataset object detection,
- train model YOLO custom,
- validate model,
- dua model vao camera realtime.

Thu muc du lieu lien quan:

```text
dataset/object_detection/raw/
dataset/object_detection/processed/
models/pretrained/
models/trained/
```

Luong co ban:

```powershell
python run_train.py --check-only
python training\prepare_dataset.py
python training\validate_dataset.py
python training\split_dataset.py
python run_train.py
python training\validate_model.py
python run_app.py --model models/trained/best.pt
```

### 2. Nhanh Y Duoc

Muc tieu:

- to chuc dataset skin lesion va TCIA,
- theo doi do san sang cua model medical,
- chay cac lenh khoi tao, status, ready, sources, verify,
- phuc vu chat UI va pipeline phan tich medical.

Thu muc du lieu lien quan:

```text
dataset/medical/skin_lesion/
dataset/medical/tcia/
output/medical/
```

Luong co ban:

```powershell
python run_medical.py init-dataset
python run_medical.py status
python run_medical.py sources
python run_medical.py ready
python run_chat.py --check-only
```

## Cai Dat Nhanh

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_menu.py
python run_doctor.py --skip-camera-check
python run_smoke.py
```

Neu PowerShell chan script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Bo Tai Lieu Trong `docs/`

| File | Noi dung |
|---|---|
| [docs/install_guide.md](/abs/path/D:/OncoVision/docs/install_guide.md:1) | Huong dan cai dat, khoi tao moi truong, va checklist sau cai dat |
| [docs/project_overview.md](/abs/path/D:/OncoVision/docs/project_overview.md:1) | Tong quan kien truc, cay thu muc, va vai tro tung module |
| [docs/quick_commands.md](/abs/path/D:/OncoVision/docs/quick_commands.md:1) | Lenh nhanh cho van hanh hang ngay |
| [docs/runtime_tool_guide.md](/abs/path/D:/OncoVision/docs/runtime_tool_guide.md:1) | Giai thich `run_app.py --advisor-only` va runtime modes |
| [docs/training_guide.md](/abs/path/D:/OncoVision/docs/training_guide.md:1) | Huong dan chi tiet cho object detection training |
| [docs/medical_imaging_guide.md](/abs/path/D:/OncoVision/docs/medical_imaging_guide.md:1) | Huong dan chi tiet cho luong y duoc |

## Checklist Cho Nguoi Moi Vao Du An

1. Doc [project_overview.md](/abs/path/D:/OncoVision/docs/project_overview.md:1) de nam bo cuc repo.
2. Chay `python run_doctor.py --skip-camera-check` de xem may con thieu gi.
3. Chay `python run_smoke.py` de test nhanh cac entrypoint quan trong.
4. Neu lam object detection, doc [training_guide.md](/abs/path/D:/OncoVision/docs/training_guide.md:1).
5. Neu lam y duoc, doc [medical_imaging_guide.md](/abs/path/D:/OncoVision/docs/medical_imaging_guide.md:1).

## Ghi Chu Van Hanh

- `run_smoke.py --ci-safe` duoc thiet ke cho CI: chi chay nhung check nhe, it phu thuoc camera va dataset cuc bo.
- `run_chat.py --check-only` dung de preflight giao dien chat ma khong mo GUI.
- `run_doctor.py --skip-camera-check` la lenh an toan nhat de ra soat tong quat tren may moi.
- Logger hien da co fallback neu file log bi khoa, nhung van nen giu thu muc `output/` co quyen ghi.

## Muc Tieu Cua Repo

OncoVision khong chi la mot app don le. Repo nay dong vai tro:

- bo cong cu van hanh,
- bo thu vien helper,
- noi luu tru quy trinh huan luyen,
- va la tai lieu ky thuat cho ca nhom.

Vi vay, cach su dung tot nhat la di tu entrypoint `run_*.py`, sau do moi di sau vao `medical/`, `training/`, `core/`, va `utils/`.
