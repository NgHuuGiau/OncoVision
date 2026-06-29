# Medical Imaging Guide

Tài liệu này mô tả workflow y dược của dự án OncoVision theo layout mới:

```text
dataset/medical/
  skin_lesion/
  tcia/
```

## Luồng chính

- `skin_lesion`: dữ liệu da liễu để train nội bộ
- `tcia`: dữ liệu 5 ung thư lấy từ TCIA theo danh sách collection

## Lệnh chính

```powershell
python run_medical.py init-dataset
python run_medical.py tcia-download --collections-file training/tcia_collections_5.json --dry-run --limit 1 --manifest
python run_medical.py verify-tcia --collections-file training/tcia_collections_5.json
python run_medical.py ready
```

## Lưu ý

- Đây là hệ thống hỗ trợ sàng lọc, không thay thế bác sĩ.
- Nếu dùng ảnh bệnh nhân thật, cần ẩn danh và kiểm soát truy cập.
- Với TCIA, nên tải thử bằng `--dry-run` trước khi chạy full.
