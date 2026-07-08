# Hướng Dẫn Luồng Y Dược

[![Medical](https://img.shields.io/badge/Docs-Medical%20Workflow-00A6A6?logo=readthedocs&logoColor=white)](medical_imaging_guide.md)

Tài liệu này giải thích luồng medical của OncoVision: dataset nào được đọc, entrypoint nào điều phối, và khi nào nên đi theo nhánh nào để debug hay huấn luyện.

> Nếu bạn làm phần medical, đây là chỗ nên mở đầu tiên sau README.

## Tóm Tắt Nhanh

| Mảng | Vai trò |
|---|---|
| `medical/` | Nơi gom logic y dược cốt lõi |
| `run_medical.py` | CLI chính cho luồng medical |
| `run_chat.py` | Kiểm tra UI và trạng thái liên quan medical |
| `run_doctor.py` | Quét tổng quan model, dataset, output |
| `output/medical/` | Nơi chứa kết quả và dữ liệu đầu ra |

## 1. Mục Tiêu

Nhánh y dược của OncoVision phục vụ 4 việc chính:

- quản lý dataset skin lesion,
- theo dõi trạng thái model và output medical,
- phân tích ảnh y khoa và lưu case,
- hỗ trợ chat UI kiểm tra trạng thái y dược.

## 2. Thư Mục Chính

```text
dataset/medical/
output/medical/
medical/
app/chat_ui/
```

Dataset medical hiện được đọc từ `dataset/medical/`, đặc biệt là nhánh `skin_lesion/`.

## 3. Entrypoint Liên Quan

| File | Vai trò |
|---|---|
| `run_medical.py` | CLI chính cho luồng medical |
| `run_chat.py` | Chat UI dùng trạng thái medical |
| `run_doctor.py` | Kiểm tra tổng quan model, dataset, output |

## 4. Command Quan Trọng

```powershell
python run_medical.py init-dataset
python run_medical.py status
python run_medical.py ready
python run_medical.py sources
python run_medical.py cancer
python run_medical.py train-all
python run_chat.py --check-only
```

`init-dataset` chỉ in layout mong đợi, không tự tạo dữ liệu trong `dataset/`.

## 5. Liên Hệ Với Chat UI

`run_chat.py --check-only` không chỉ kiểm tra giao diện, mà còn xác nhận:

- thư viện bắt buộc đã sẵn sàng,
- icon UI có đủ,
- medical model có thể dùng cho chat hay chưa.

Model medical được tìm theo thứ tự: `medical_7_cancers.pt` ở root, `medical/medical_7_cancers.pt`, rồi `fallback_model` nếu bật trong `config/medical_settings.yaml`.

![Medical status chi tiết](../images/Ảnh%20run_doctor.py%20--skip-camera-check%202.png)

## 6. Các Module Chính Trong `medical/`

- `medical/dataset.py`: tạo và kiểm tra layout dataset
- `medical/system_status.py`: tổng hợp trạng thái medical
- `medical/training.py`: audit, split, train, validate
- `medical/output_management.py`: dọn output medical
- `medical/storage.py`: lưu và truy vấn case DB
- `medical/cli_helpers.py`: helper in trạng thái dùng chung cho CLI

## 7. Quy Trình Khuyến Nghị

### A. Khởi tạo máy mới

```powershell
python run_medical.py init-dataset
python run_medical.py status
python run_doctor.py --skip-camera-check
python run_chat.py --check-only
```

### B. Kiểm tra khả năng đưa vào sử dụng

```powershell
python run_medical.py ready
python run_medical.py status
python run_chat.py --check-only
```

## 8. Output Và Dọn Dẹp

Medical output thường nằm trong:

```text
output/medical/
```

Khi cần dọn nhanh:

```powershell
python run_chat.py --cleanup-output --older-than-days 30
```

## 9. Lưu Ý Nghiệp Vụ

- `run_medical.py status` là lệnh xem trạng thái tổng quan nhanh nhất.
- `run_medical.py ready` cho biết luồng medical đã đủ điều kiện train hay chưa.
- `run_medical.py train-all` chạy split, train và validate theo luồng medical.
- `run_medical.py` không tự bịa dữ liệu, nên dataset thiếu thì sẽ báo thiếu rõ ràng.

## 10. Khi Nào Nên Debug Module Nào

| Triệu chứng | Mở đầu tiên |
|---|---|
| `run_chat.py --check-only` fail | `utils/entrypoint_checks.py`, `medical/system_status.py` |
| Medical status sai | `medical/system_status.py`, `medical/model_policy.py`, `medical/storage.py` |
| Counts raw/train/val không đúng | `medical/training.py`, `medical/status_helpers.py` |
| Train medical fail | `medical/training.py`, `training/train_model.py` |
## 11. Medical Inputs Hỗ Trợ

| Nhóm | Ảnh/volume thường dùng |
|---|---|
| Gan | Siêu âm, CT, MRI, đôi khi PET/CT |
| Phổi | X-quang ngực, CT ngực, PET/CT |
| Vú | Mammogram, siêu âm vú, MRI vú |
| Dạ dày | Nội soi, CT, MRI, PET, EUS |
| Đại trực tràng | Nội soi đại tràng, CT ngực-bụng-chậu, MRI trực tràng, PET |
| Tuyến tiền liệt | MRI tuyến tiền liệt, siêu âm, PET/CT |
| Cổ tử cung | MRI, CT, PET/CT |

- `Pap/HPV`, soi cổ tử cung và sinh thiết là đầu vào lâm sàng, không phải file ảnh để upload trực tiếp.
- Chat UI có preset chọn nhóm bệnh để lọc nguồn ảnh ngay từ đầu.
- Chat UI có thêm chọn modality theo nhóm bệnh để file picker bám đúng loại ảnh cần dùng.
- File picker sẽ ưu tiên đuôi ảnh/volume phù hợp với modality đã chọn.
- Folder DICOM series và volume `.nii/.nii.gz` có thể xem từng lát trong preview.
