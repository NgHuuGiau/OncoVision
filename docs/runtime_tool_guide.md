# Hướng Dẫn Runtime Tool

`run_app.py --advisor-only` là công cụ giúp bạn biết máy hiện tại nên chạy cấu hình nào trước khi mở camera thật.
Chế độ này không mở webcam, không chạy inference và không huấn luyện model.

## Mục đích

Runtime Tool được dùng để:

- phân tích phần cứng hiện tại
- gợi ý mode `high`, `medium`, `low`
- hiển thị model local đang có
- đề xuất cấu hình phù hợp cho camera realtime

## Công nghệ liên quan

- Python
- PyTorch
- Ultralytics YOLO
- `psutil` để đọc thông tin CPU/RAM
- `GPUtil` để kiểm tra GPU/VRAM
- `PyYAML` để đọc config

## Lệnh chạy

```powershell
python run_app.py --advisor-only
```

## Runtime Tool hiển thị gì

- CPU, RAM, GPU, VRAM
- Trạng thái CUDA và PyTorch
- Danh sách model YOLO local
- Các mode runtime: `high`, `medium`, `low`
- `model`, `device`, `imgsz`, `max_det` của từng mode
- Mode khuyến nghị theo máy hiện tại

## Cách đọc kết quả

### `high`

- ưu tiên chất lượng nhận diện
- hợp khi máy đủ mạnh
- có thể nặng hơn, FPS thấp hơn

### `medium`

- cân bằng giữa FPS và độ chính xác
- thường là lựa chọn an toàn cho camera realtime

### `low`

- ưu tiên tốc độ và độ ổn định
- hợp cho máy yếu hoặc khi hệ thống đang tải cao

## Khi nào nên dùng

Chạy Runtime Tool trước khi mở camera nếu bạn muốn:

- biết vì sao máy được đề xuất mode nào
- so sánh chất lượng và hiệu năng giữa các mode
- xem model local nào đang sẵn sàng
- tránh mở camera với cấu hình quá nặng

## Liên quan tới `run_doctor.py`

`run_app.py --advisor-only`:

- tập trung vào logic chọn runtime
- dùng để quyết định cấu hình trước khi chạy thật

`run_doctor.py`:

- tập trung vào kiểm tra sức khỏe hệ thống
- kiểm tra camera, model, dữ liệu, icon và môi trường

## Quy trình khuyến nghị

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
python run_app.py --mode medium
```

Sau đó quan sát:

- FPS có ổn không
- Box có bị bỏ sót không
- Có nên đổi model hoặc tăng `imgsz` không

## Tác động tới chất lượng nhận diện

Runtime Tool giúp bạn trả lời các câu hỏi sau:

- Máy có đang chạy đúng mode không
- Model có quá nhẹ hay quá nặng không
- Có nên chuyển sang `models/trained/best.pt` không
- Có nên giữ `medium` hay hạ về `low`

## Tham số cấu hình thường liên quan

Bạn có thể tinh chỉnh thêm trong `config/settings.yaml`:

- `inference.confidence`
- `inference.iou`
- `inference.display_confidence`
- `inference.person_confidence`
- `inference.phone_confidence`
- `inference.enhance_low_light`

## Ghi chú thực tế

- Máy có CUDA mạnh thường sẽ hợp `high` hoặc `medium`
- Máy chỉ có CPU nên ưu tiên `low`
- Nếu model custom tốt hơn pretrained thì nên dùng `models/trained/best.pt`
