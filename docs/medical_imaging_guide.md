# Hướng Dẫn Luồng Y Dược

Tài liệu này giải thích toàn bộ nhánh `medical` trong OncoVision: dữ liệu nào đang được quản lý, các lệnh CLI để làm gì, output tạo ra nằm ở đâu, và cách kiểm tra độ sẵn sàng của pipeline.

## 1. Mục Tiêu Của Nhánh Medical

Nhánh `medical` phục vụ các bài toán:

- quản lý dataset skin lesion,
- quản lý danh sách nguồn dữ liệu ung thư,
- tải và xác minh dữ liệu TCIA,
- theo dõi model medical và fallback model,
- sinh report, output, và hỗ trợ chat UI / phân tích ảnh.

## 2. Thư Mục Dữ Liệu Chính

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

### Ý nghĩa từng nhóm

| Thư mục | Vai trò |
|---|---|
| `dataset/medical/skin_lesion/` | Dataset nội bộ cho luồng da liễu / skin lesion |
| `dataset/medical/tcia/` | Dữ liệu và collection lấy từ TCIA |
| `output/medical/reports/` | Báo cáo JSON, text, hoặc report output |
| `output/medical/normalized_images/` | Ảnh đã chuẩn hóa trước / sau pipeline |
| `output/medical/processed_images/` | Ảnh đã vẽ overlay, kết quả trung gian |
| `output/medical/exports/` | Gói export, zip, hoặc artifact tổng hợp |
| `output/medical/medical_cases.db` | Cơ sở dữ liệu nhỏ lưu case metadata |

## 3. Entrypoint Liên Quan

| File | Vai trò |
|---|---|
| `run_medical.py` | CLI chính cho luồng medical |
| `run_chat.py` | Chat UI có sử dụng trạng thái medical |
| `run_doctor.py` | Hiện thông tin tổng quan medical model, dataset, output |
| `run_smoke.py` | Có thể gọi một số preflight medical trong luồng đầy đủ |

## 4. Các Lệnh Quan Trọng Nhất

### Kiểm tra layout dataset

```powershell
python run_medical.py init-dataset
```

Dùng khi:

- muốn xem layout dataset mong đợi,
- muốn kiểm tra đường dẫn raw/processed/metadata/reports,
- muốn xác nhận repo không tự tạo hay tải dữ liệu hộ.

### Xem trạng thái tổng quát

```powershell
python run_medical.py status
```

Lệnh này cho biết:

- model config đang trỏ vào đâu,
- model runtime đã resolve ra file nào,
- fallback model có bật hay không,
- dataset root,
- số ảnh raw/train/val/test,
- số report / overlay / export / case db.

### Xem độ sẵn sàng để train / vận hành

```powershell
python run_medical.py ready
```

Dùng để trả lời nhanh:

- dataset đã init chưa,
- raw dataset đã có chưa,
- processed dataset đã sẵn sàng chưa,
- model medical đã sẵn sàng chưa,
- luồng full medical đã đủ điều kiện chưa.

### Xem nguồn dữ liệu / cancer targets

```powershell
python run_medical.py sources
python run_medical.py cancer
```

Hai lệnh này giúp:

- biết repo đang theo dõi những cancer nào,
- biết từng nguồn dữ liệu đang ở trạng thái nào,
- biết đã có dữ liệu local chưa.

## 5. Workflow TCIA

### Chạy tính toán trước khi tải thật

```powershell
python run_medical.py tcia-download --dry-run
```

Nên chạy trước để:

- xem kế hoạch download,
- kiểm tra collection file,
- tránh mở full download khi chưa chắc chắn layout đúng.

### Tải dữ liệu theo file collection

Ví dụ:

```powershell
python run_medical.py tcia-download --collections-file training/tcia_collections_5.json
```

### Xác minh sau khi tải

```powershell
python run_medical.py verify-tcia --collections-file training/tcia_collections_5.json
```

### Xem log một collection

```powershell
python run_medical.py tcia-log --collection "CBIS-DDSM / TCGA-BRCA"
```

## 6. Mối Quan Hệ Giữa Chat UI Và Medical

`run_chat.py --check-only` không chỉ check giao diện, mà còn kiểm:

- module bắt buộc,
- icon giao diện,
- model medical có sẵn sàng không,
- output / capture directories.

Vì vậy, nếu chat UI báo chưa sẵn sàng, khả năng cao là luồng medical vẫn còn thiếu:

- model,
- output directory,
- hoặc data/chính sách fallback.

## 7. Các Module Chính Trong Thư Mục `medical/`

| File | Vai trò |
|---|---|
| `dataset.py` | Tạo / đảm bảo cấu trúc dataset medical |
| `pipeline.py` | Pipeline xử lý và phân tích ảnh y khoa |
| `system_status.py` | Tổng hợp trạng thái model, data, output, DB |
| `model_policy.py` | Chọn model runtime và fallback model |
| `storage.py` | Làm việc với `medical_cases.db` |
| `reporting.py` | Tạo report và artifact |
| `status_helpers.py` | Hàm đếm file, tổng hợp metric nhỏ |
| `cancer_catalog.py` | Danh sách cancer labels và target |
| `cancer_dataset_registry.py` | Đăng ký nguồn dữ liệu và metadata nguồn |
| `chat_service.py` | Logic phục vụ hỏi đáp / thao tác medical cho chat UI |

## 8. Quy Trình Vận Hành Khuyến Nghị

### A. Khởi tạo một máy mới

```powershell
python run_medical.py init-dataset
python run_medical.py status
python run_doctor.py --skip-camera-check
python run_chat.py --check-only
```

`init-dataset` ở đây chỉ kiểm tra layout mong đợi, không tự tạo `dataset/medical/...`.

### B. Chuẩn bị TCIA

```powershell
python run_medical.py sources
python run_medical.py tcia-download --dry-run
python run_medical.py tcia-download --collections-file training/tcia_collections_5.json
python run_medical.py verify-tcia --collections-file training/tcia_collections_5.json
```

### C. Kiểm tra khả năng đưa vào sử dụng

```powershell
python run_medical.py ready
python run_medical.py status
python run_chat.py --check-only
```

## 9. Output Và Dọn Dẹp

Nếu output chat hoặc medical phát sinh quá nhiều:

```powershell
python run_chat.py --cleanup-output --older-than-days 30
```

Lệnh này giúp:

- xóa file output cũ,
- giải phóng dung lượng,
- giữ workspace sạch để debug và CI dễ theo dõi hơn.

## 10. Lưu Ý Nghiệp Vụ

- Repo này hỗ trợ sàng lọc, nghiên cứu, và vận hành kỹ thuật; không thay thế đánh giá y khoa chuyên môn.
- Nếu dùng ảnh thật của bệnh nhân, cần có quy trình ẩn danh và kiểm soát truy cập rõ ràng.
- Không nên xem `model_ready=True` là bằng chứng cho độ chính xác lâm sàng; đó chỉ là tín hiệu cho thấy pipeline có thể vận hành.

## 11. Khi Nào Nên Debug Module Nào

| Triệu chứng | Đi debug đâu |
|---|---|
| `run_chat.py --check-only` fail | `utils/entrypoint_checks.py`, `medical/system_status.py` |
| `status` báo model chưa sẵn sàng | `medical/model_policy.py`, `models/` |
| Counts raw/train/val không đúng | `medical/training.py`, `medical/status_helpers.py` |
| TCIA verify fail | `training/tcia_downloader.py`, `training/verify_tcia_downloads.py` |
| Báo cáo / export không sinh | `medical/reporting.py`, `medical/output_management.py` |
