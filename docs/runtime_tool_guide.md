# Hướng Dẫn Runtime Advisor

Tài liệu này giải thích công cụ `run_app.py --advisor-only`: nó làm gì, đọc kết quả ra sao, và dùng công cụ này thế nào để chọn runtime mode phù hợp trước khi mở camera thật.

## 1. Runtime Advisor Là Gì

`run_app.py --advisor-only` là chế độ:

- không mở webcam,
- không chạy detection realtime,
- không mở giao diện camera,
- chỉ phân tích hệ thống và đưa ra gợi ý runtime.

Lệnh:

```powershell
python run_app.py --advisor-only
```

## 2. Mục Tiêu Của Công Cụ

Công cụ này giúp trả lời 4 câu hỏi:

1. máy hiện tại đang mạnh đến đâu,
2. có GPU / CUDA / torch sẵn sàng hay không,
3. nên ưu tiên model nào,
4. nên bắt đầu với `high`, `medium`, `low`, hay `auto`.

## 3. Nội Dung Đầu Ra Thường Gặp

Runtime advisor thường hiện:

- CPU, RAM, GPU, VRAM
- torch version
- CUDA build
- danh sách model local đang có
- mode runtime dự kiến
- `imgsz`, `max_det`, `device`, model đề xuất

## 4. Ý Nghĩa Từng Mode

### `high`

Phù hợp khi:

- máy có GPU tốt,
- ưu tiên độ chính xác,
- chấp nhận tải cao hơn.

### `medium`

Phù hợp khi:

- muốn cân bằng FPS và độ chính xác,
- muốn bắt đầu với lựa chọn an toàn,
- đây là mode hay hợp với đa số máy dev.

### `low`

Phù hợp khi:

- máy yếu,
- đang chạy CPU,
- webcam / hệ thống không ổn định,
- cần ưu tiên tốc độ và độ bền.

### `auto`

Không phải mode cố định, mà là cơ chế chọn mode theo máy đang dùng. Thường advisor sẽ đưa ra đề xuất trên cơ sở:

- có CUDA hay không,
- VRAM nhiều hay ít,
- model nào sẵn có trong `models/`.

## 5. Cách Đọc Kết Quả

Ví dụ nếu advisor báo:

```text
medium: model=yolo11s.pt, device=cuda:0, imgsz=512, max_det=120
```

Bạn có thể hiểu:

- mode khởi động hợp lý là `medium`,
- model ưu tiên là `yolo11s.pt`,
- sẽ chạy bằng GPU `cuda:0`,
- kích thước ảnh input 512,
- giới hạn detection mỗi frame là 120.

## 6. Cách Dùng Cùng Các Lệnh Khác

Quy trình khuyến nghị:

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
python run_app.py --mode medium
```

Nếu đã có model custom:

```powershell
python run_app.py --advisor-only
python run_app.py --model models/trained/best.pt --mode medium
```

## 7. Liên Quan Tới Các Module Khác

Runtime advisor phụ thuộc nhiều vào:

- `core/hardware_info.py`
- `core/runtime_advisor.py`
- `app/camera_runtime/bootstrap.py`
- `config/settings.yaml`

Nếu gợi ý không hợp lý, đây là nhóm file nên debug đầu tiên.

## 8. Khi Nào Nên Chạy Runtime Advisor

Nên chạy trong các trường hợp:

- máy mới vừa cài repo,
- vừa thay GPU / driver / torch,
- vừa đổi model local,
- camera realtime đang lag,
- muốn so sánh giữa `medium` và `low`,
- trước khi demo trên máy lạ.

## 9. Vấn Đề Thường Gặp

### Advisor báo CUDA nhưng runtime vẫn chậm

Lý do có thể là:

- model quá nặng,
- `imgsz` quá cao,
- GPU đang bị app khác chiếm,
- webcam output quá lớn.

Thử:

```powershell
python run_app.py --mode low
python run_app.py --mode medium
```

### Advisor thấy model local nhưng kết quả nhận diện kém

Cần tách rõ:

- advisor chỉ gợi ý mode/runtime,
- không đánh giá chất lượng nghiệp vụ của model.

Nếu muốn model tốt hơn, cần xem tiếp:

- `training_guide.md`
- `models/trained/best.pt`
- dataset object detection thực tế.

## 10. Mục Tiêu Khi Sử Dụng Đúng Cách

Runtime advisor giúp team:

- mở camera với cấu hình hợp lý hơn,
- giảm bug do chọn sai mode,
- dễ hướng dẫn máy mới,
- dễ debug vấn đề nặng / lag / FPS thấp một cách có hệ thống.
