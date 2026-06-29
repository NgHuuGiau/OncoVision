# Tong Quan Du An

Tai lieu nay mo ta bo cuc repo OncoVision o muc kien truc: entrypoint nao dung cho tac vu nao, thu muc nao phu trach phan nao, va luong du lieu di qua he thong ra sao.

## 1. Toan Canh

OncoVision la mot monorepo ky thuat gom:

- bo entrypoint `run_*.py` de van hanh,
- module `core/` cho camera realtime,
- module `medical/` cho nhanh y duoc,
- module `training/` cho object detection va downloader,
- `app/` cho runtime camera va chat UI,
- `utils/` cho helper dung chung,
- `tests/` cho unit test va hoi quy.

## 2. Cay Thu Muc Du An

```text
OncoVision/
|-- .github/
|   `-- workflows/
|       `-- test.yml
|-- app/
|   |-- camera_runtime/
|   `-- chat_ui/
|-- assets/
|-- config/
|-- core/
|-- dataset/
|   |-- medical/
|   `-- object_detection/
|-- docs/
|-- medical/
|-- models/
|   |-- pretrained/
|   `-- trained/
|-- output/
|   |-- captures/
|   |-- chat_captures/
|   |-- logs/
|   `-- medical/
|-- runs/
|-- scripts/
|-- tests/
|-- training/
|-- utils/
|-- run_app.py
|-- run_chat.py
|-- run_doctor.py
|-- run_medical.py
|-- run_menu.py
|-- run_smoke.py
`-- run_train.py
```

## 3. Mo Ta Chi Tiet Tung Thu Muc

### `.github/`

- chua workflow CI cho repo,
- hien tai workflow chinh la `workflows/test.yml`,
- co log artifact va smoke mode an toan cho CI.

### `app/`

Noi chua cac lop o muc ung dung, gan voi entrypoint va UI hon la voi thuat toan nen.

#### `app/camera_runtime/`

Dung de:

- build argument parser cho `run_app.py`,
- chon runtime mode,
- bootstrap luong khoi dong camera,
- giu logic dieu phoi giua phan cung, runtime va luong launch.

File dang chu y:

- `cli.py`: parser cho mode, camera-index, model
- `bootstrap.py`: tao `StartOptions`, resolve runtime
- `launching.py`: boot progress va flow chay camera

#### `app/chat_ui/`

Dung de:

- cung cap giao dien chat,
- luu hoi thoai,
- quan ly icon, paths, output,
- ket noi chat UI voi logic medical.

File dang chu y:

- `cli.py`: parser cho `run_chat.py`
- `window.py`: launch giao dien chat
- `storage.py`: luu hoi thoai sqlite
- `medical_controller.py`: state giua UI va medical service
- `voice_worker.py`: thu am va chuyen giong noi thanh text

### `assets/`

- noi de tai nguyen tinh,
- co the chua icon, anh mau, hoac static asset phuc vu UI / demo.

### `config/`

- chua file YAML cho settings runtime va medical,
- la noi can xem dau tien neu can doi default path, recording, confidence, iou, output.

### `core/`

Day la lop xu ly camera realtime va object detection runtime.

Thanh phan chinh:

- `camera_runner.py`: vong lap camera, doc frame, detect, overlay, record, capture
- `model_loader.py`: nap YOLO model va fallback
- `hardware_info.py`: doc CPU/GPU/CUDA/PyTorch
- `frame_processing.py`: tien xu ly frame, low-light, motion
- `tracking/`: logic gan track, smooth, filter detection
- `recorder.py`, `frame_capture.py`: quay video va chup frame

Noi dung `core/` rat quan trong voi:

- `run_app.py`
- `run_doctor.py`
- `run_tests.py`

### `dataset/`

Chua du lieu van hanh cua du an, tach lam 2 nhanh ro rang.

#### `dataset/medical/`

- skin lesion dataset,
- TCIA dataset,
- va nhung artifact du lieu medical lien quan.

#### `dataset/object_detection/`

- raw images / labels cho object detection,
- processed images sau split train/val/test,
- la dau vao cho pipeline training YOLO.

### `docs/`

- bo tai lieu van hanh va huong dan cho thanh vien nhom,
- duoc to chuc theo chu de: install, runtime, training, medical, quick commands, project overview.

### `medical/`

Day la package nghiep vu cho nhanh y duoc.

Trach nhiem chinh:

- mo ta catalog ung thu,
- quan ly dataset structure medical,
- quan ly model medical,
- tong hop system status,
- luu case DB,
- report / output / service phuc vu chat UI.

Nhung file quan trong:

- `system_status.py`: gom status model + data + output + DB
- `dataset.py`: tao cau truc dataset
- `pipeline.py`: xu ly / phan tich anh medical
- `storage.py`: medical case database
- `chat_service.py`: logic phan hoi cho chat UI
- `model_policy.py`: resolve runtime medical model

### `models/`

Chua weights model.

#### `models/pretrained/`

- model co san dung de khoi dong nhanh,
- thuong dung cho object detection baseline.

#### `models/trained/`

- model custom sau khi train noi bo,
- la noi `best.pt` thuong duoc dua vao runtime hoac validate.

### `output/`

Chua ket qua sinh ra trong qua trinh chay.

So do tong quat:

```text
output/
|-- captures/          # snapshot camera realtime
|-- recordings/        # video recording neu co bat
|-- chat_captures/     # capture / attachment tu chat
|-- logs/              # app.log va log runtime
`-- medical/           # report, normalized, overlay, exports, db
```

### `runs/`

- ket qua huan luyen / validate sinh boi YOLO va training scripts,
- thuong chua artifact theo tung lan train.

### `scripts/`

- script phu de verify, maintenance, hoac tooling nho,
- vi du `verify_entrypoints_help.py` duoc workflow CI goi de check `--help`.

### `tests/`

- unit test va regression test,
- bao gom test cho runtime, medical, training, UI logic, status, smoke support.

Nhom test quan trong:

- `test_run_smoke.py`
- `test_medical_system_status.py`
- `test_camera_detector.py`
- `test_runtime_prompt.py`
- `test_training_pipeline.py`

### `training/`

Day la package training object detection va downloader phu tro.

Vai tro chinh:

- chuan bi dataset,
- validate dataset,
- split train/val/test,
- chay train model,
- validate model,
- export model,
- quan ly TCIA collections va downloader.

File quan trong:

- `prepare_dataset.py`
- `validate_dataset.py`
- `split_dataset.py`
- `train_model.py`
- `validate_model.py`
- `export_model.py`
- `tcia_downloader.py`
- `download_models.py`

### `utils/`

Chua helper dung chung cho toan repo.

Nhom chuc nang:

- `console_ui.py`: in bang, dashboard, terminal rendering
- `entrypoint_checks.py`: preflight checks cho chat / runtime / training
- `file_utils.py`: helper file / YAML / path
- `logger.py`: logger co fallback
- `camera_utils.py`: wrapper mo camera
- `cleanup_utils.py`: don dep output
- `sqlite_utils.py`: helper sqlite

## 4. Vai Tro Cua Tung Entrypoint Goc

| File | Trach nhiem |
|---|---|
| `run_menu.py` | Cua vao tong hop cho nguoi van hanh |
| `run_app.py` | Runtime advisor va camera realtime |
| `run_chat.py` | Chat UI, preflight chat, cleanup output |
| `run_doctor.py` | Doctor scan tong quat cho moi truong |
| `run_medical.py` | CLI quan ly nhanh y duoc |
| `run_train.py` | Entrypoint object detection training |
| `run_smoke.py` | Smoke check entrypoint |
| `run_tests.py` | Dashboard unit test voi camera check tuy chon |

## 5. Luong Du Lieu Tong Quan

### Luong camera realtime

```text
run_app.py
-> app/camera_runtime/*
-> core/hardware_info.py
-> core/model_loader.py
-> core/camera_runner.py
-> output/captures | output/recordings
```

### Luong object detection training

```text
dataset/object_detection/raw
-> training/prepare_dataset.py
-> training/validate_dataset.py
-> training/split_dataset.py
-> run_train.py / training/train_model.py
-> models/trained/best.pt
-> run_app.py --model models/trained/best.pt
```

### Luong medical

```text
dataset/medical/*
-> medical/dataset.py
-> medical/system_status.py
-> run_medical.py
-> output/medical/*
-> run_chat.py --check-only / launch chat
```

## 6. Thu Muc Nao Nen Mo Dau Tien Khi Debug

| Van de | Thu muc / file nen mo dau tien |
|---|---|
| Camera khong chay | `run_app.py`, `core/camera_runner.py`, `utils/camera_utils.py` |
| Runtime goi y sai | `core/hardware_info.py`, `core/runtime_advisor.py`, `app/camera_runtime/bootstrap.py` |
| Chat UI khong san sang | `run_chat.py`, `utils/entrypoint_checks.py`, `app/chat_ui/` |
| Medical status sai | `medical/system_status.py`, `medical/model_policy.py`, `medical/storage.py` |
| Train fail | `run_train.py`, `training/train_model.py`, `training/validate_dataset.py` |
| CI fail | `.github/workflows/test.yml`, `run_smoke.py`, `requirements-ci.txt` |

## 7. Nguyen Tac Kien Truc Dang The Hien Trong Repo

- Entry point ro rang, module nghiep vu tach rieng.
- Package import dang duoc toi uu de tranh side effect qua som.
- CI uu tien smoke mode an toan, khong co gang mo tat ca feature nang.
- Medical va object detection tach layout du lieu de tranh lan nghiep vu.

## 8. Cach Dung Tai Lieu Nay

Neu ban moi vao repo:

1. Doc file nay truoc.
2. Sau do doc `install_guide.md`.
3. Neu phu trach object detection, doc tiep `training_guide.md`.
4. Neu phu trach luong y duoc, doc tiep `medical_imaging_guide.md`.
