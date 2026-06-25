# Hướng Dẫn Medical Imaging

Tài liệu này mô tả workflow phân tích ảnh y khoa của dự án OncoVision.

## Mục tiêu

- nhận ảnh y khoa từ file upload
- chuẩn hóa ảnh trước khi suy luận
- phát hiện vùng nghi ngờ bằng model YOLO
- gán mức nguy cơ `low`, `medium`, `high`
- tạo report JSON và Markdown
- lưu lịch sử ca phân tích vào SQLite

## Công nghệ dùng trong luồng medical

- Python
- PySide6 cho UI chat
- OpenCV và Pillow để xử lý ảnh
- YOLO / Ultralytics để inference
- SQLite để lưu lịch sử

## Dataset medical đã được chuẩn hóa sẵn

Khởi tạo cấu trúc mặc định:

```powershell
python run_medical.py init-dataset
```

Cấu trúc chính:

```text
dataset/medical_skin_lesion/
  raw/images
  raw/labels
  processed/images/train
  processed/images/val
  processed/images/test
  processed/labels/train
  processed/labels/val
  processed/labels/test
  metadata
  reports
  data.yaml
```

## Quy trình medical đầy đủ

1. Đưa ảnh và label vào:

```text
dataset/medical_skin_lesion/raw/images
dataset/medical_skin_lesion/raw/labels
```

2. Kiểm tra dataset:

```powershell
python run_medical.py audit-dataset
```

3. Chia train/val/test:

```powershell
python run_medical.py split-dataset
```

4. Train model:

```powershell
python run_medical.py train
```

5. Validate model:

```powershell
python run_medical.py validate
```

6. Hoặc chạy trọn gói:

```powershell
python run_medical.py train-all
```

Sau khi train xong, `config/medical_settings.yaml` sẽ được cập nhật để giao diện chat và CLI dùng cùng model.

## Phân tích ảnh

```powershell
python run_medical.py analyze --image path\to\image.jpg --patient-code BN001
```

Đầu ra thường gồm:

- `output/medical/normalized_images/`
- `output/medical/processed_images/`
- `output/medical/reports/`
- `output/medical/medical_cases.db`

## Phân tích trong giao diện chat

Sau khi mở:

```powershell
python run_chat.py
```

bạn có thể:

- chọn ảnh
- tải ảnh y khoa lên khung chat
- gửi ảnh để phân tích

Luồng chat sẽ:

- gọi medical pipeline
- tạo ảnh overlay đánh dấu vùng nghi ngờ
- lưu report JSON và Markdown
- lưu hồ sơ ca vào SQLite
- trả kết quả ngay trong hội thoại

## Xem lịch sử

```powershell
python run_medical.py history --limit 10
```

## Tính metric y khoa

```powershell
python run_medical.py metrics --truths "[true,false,true]" --predictions "[true,false,false]"
```

Metric hiện có:

- sensitivity
- specificity
- precision
- recall
- accuracy
- f1_score

## Lưu ý quan trọng

- Đây là hệ thống hỗ trợ sàng lọc, không thay thế bác sĩ
- Nếu dùng ảnh bệnh nhân thật, cần ẩn danh dữ liệu và quản lý quyền truy cập
- Nếu đổi bài toán sang MRI, CT hoặc X-ray, cần thêm bước tiền xử lý chuyên biệt
