# Huong Dan Luong Y Duoc

Tai lieu nay giai thich toan bo nhanh `medical` trong OncoVision: du lieu nao dang duoc quan ly, cac lenh CLI de lam gi, output tao ra nam o dau, va cach kiem tra do san sang cua pipeline.

## 1. Muc Tieu Cua Nhanh Medical

Nhanh `medical` phuc vu cac bai toan:

- quan ly dataset skin lesion,
- quan ly danh sach nguon du lieu ung thu,
- tai va xac minh du lieu TCIA,
- theo doi model medical va fallback model,
- sinh report, output, va ho tro chat UI / phan tich anh.

## 2. Thu Muc Du Lieu Chinh

```text
dataset/medical/
  skin_lesion/
  tcia/

output/medical/
  reports/
  normalized_images/
  processed_images/
  exports/
  medical_cases.db
```

### Y nghia tung nhom

| Thu muc | Vai tro |
|---|---|
| `dataset/medical/skin_lesion/` | Dataset noi bo cho luong da lieu / skin lesion |
| `dataset/medical/tcia/` | Du lieu va collection lay tu TCIA |
| `output/medical/reports/` | Bao cao JSON, text, hoac report output |
| `output/medical/normalized_images/` | Anh da chuan hoa truoc / sau pipeline |
| `output/medical/processed_images/` | Anh da ve overlay, ket qua trung gian |
| `output/medical/exports/` | Goi export, zip, hoac artifact tong hop |
| `output/medical/medical_cases.db` | Co so du lieu nho luu case metadata |

## 3. Entrypoint Lien Quan

| File | Vai tro |
|---|---|
| `run_medical.py` | CLI chinh cho luong medical |
| `run_chat.py` | Chat UI co su dung trang thai medical |
| `run_doctor.py` | Hien thong tin tong quan medical model, dataset, output |
| `run_smoke.py` | Co the goi mot so preflight medical trong luong day du |

## 4. Cac Lenh Quan Trong Nhieu Nhat

### Khoi tao dataset

```powershell
python run_medical.py init-dataset
```

Dung khi:

- may moi chua co khung thu muc medical,
- can dong bo lai layout dataset chuan,
- muon bat dau tu pipeline y duoc truoc khi import/tai data.

### Xem trang thai tong quat

```powershell
python run_medical.py status
```

Lenh nay cho biet:

- model config dang tro vao dau,
- model runtime da resolve ra file nao,
- fallback model co bat hay khong,
- dataset root,
- so anh raw/train/val/test,
- so report / overlay / export / case db.

### Xem do san sang de train / van hanh

```powershell
python run_medical.py ready
```

Dung de tra loi nhanh:

- dataset da init chua,
- raw dataset da co chua,
- processed dataset da san sang chua,
- model medical da san sang chua,
- luong full medical da du dieu kien chua.

### Xem nguon du lieu / cancer targets

```powershell
python run_medical.py sources
python run_medical.py cancer
```

Hai lenh nay giup:

- biet repo dang theo doi nhung cancer nao,
- biet tung nguon du lieu dang o trang thai nao,
- biet da co du lieu local chua.

## 5. Workflow TCIA

### Chay thuoc tinh toan truoc khi tai that

```powershell
python run_medical.py tcia-download --dry-run
```

Nen chay truoc de:

- xem ke hoach download,
- kiem tra collection file,
- tranh mo full download khi chua chac chan layout dung.

### Tai du lieu theo file collection

Vi du:

```powershell
python run_medical.py tcia-download --collections-file training/tcia_collections_5.json
```

### Xac minh sau khi tai

```powershell
python run_medical.py verify-tcia --collections-file training/tcia_collections_5.json
```

### Xem log mot collection

```powershell
python run_medical.py tcia-log --collection "CBIS-DDSM / TCGA-BRCA"
```

## 6. Moi Quan He Giua Chat UI Va Medical

`run_chat.py --check-only` khong chi check giao dien, ma con kiem:

- module bat buoc,
- icon giao dien,
- model medical co san sang khong,
- output / capture directories.

Vi vay, neu chat UI bao chua san sang, kha nang cao la luong medical van con thieu:

- model,
- output directory,
- hoac data/chinh sach fallback.

## 7. Cac Module Chinh Trong Thu Muc `medical/`

| File | Vai tro |
|---|---|
| `dataset.py` | Tao / dam bao cau truc dataset medical |
| `pipeline.py` | Pipeline xu ly va phan tich anh y khoa |
| `system_status.py` | Tong hop trang thai model, data, output, DB |
| `model_policy.py` | Chon model runtime va fallback model |
| `storage.py` | Lam viec voi `medical_cases.db` |
| `reporting.py` | Tao report va artifact |
| `status_helpers.py` | Ham dem file, tong hop metric nho |
| `cancer_catalog.py` | Danh sach cancer labels va target |
| `cancer_dataset_registry.py` | Dang ky nguon du lieu va metadata nguon |
| `chat_service.py` | Logic phuc vu hoi dap / thao tac medical cho chat UI |

## 8. Quy Trinh Van Hanh Khuyen Nghi

### A. Khoi tao mot may moi

```powershell
python run_medical.py init-dataset
python run_medical.py status
python run_doctor.py --skip-camera-check
python run_chat.py --check-only
```

### B. Chuan bi TCIA

```powershell
python run_medical.py sources
python run_medical.py tcia-download --dry-run
python run_medical.py tcia-download --collections-file training/tcia_collections_5.json
python run_medical.py verify-tcia --collections-file training/tcia_collections_5.json
```

### C. Kiem tra kha nang dua vao su dung

```powershell
python run_medical.py ready
python run_medical.py status
python run_chat.py --check-only
```

## 9. Output Va Don Dep

Neu output chat hoac medical phat sinh qua nhieu:

```powershell
python run_chat.py --cleanup-output --older-than-days 30
```

Lenh nay giup:

- xoa file output cu,
- giai phong dung luong,
- giu workspace sach de debug va CI de theo doi hon.

## 10. Luu Y Nghiep Vu

- Repo nay ho tro sang loc, nghien cuu, va van hanh ky thuat; khong thay the danh gia y khoa chuyen mon.
- Neu dung anh that cua benh nhan, can co quy trinh an danh va kiem soat truy cap ro rang.
- Khong nen xem `model_ready=True` la bang chung cho do chinh xac lam sang; do chi la tin hieu cho thay pipeline co the van hanh.

## 11. Khi Nao Nen Debug Module Nao

| Trieu chung | Di debug dau |
|---|---|
| `run_chat.py --check-only` fail | `utils/entrypoint_checks.py`, `medical/system_status.py` |
| `status` bao model chua san sang | `medical/model_policy.py`, `models/` |
| Counts raw/train/val khong dung | `medical/training.py`, `medical/status_helpers.py` |
| TCIA verify fail | `training/tcia_downloader.py`, `training/verify_tcia_downloads.py` |
| Bao cao / export khong sinh | `medical/reporting.py`, `medical/output_management.py` |
