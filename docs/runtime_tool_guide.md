# Hướng dẫn dùng Runtime Tool

`run_app.py --advisor-only` là công cụ phân tích và giải thích runtime phù hợp với máy hiện tại. Chế độ này không mở camera, không train và không chạy detect trực tiếp.

## 1. Khi nào nên dùng

Bạn nên chạy `run_app.py --advisor-only` khi muốn:

- Biết vì sao máy được khuyến nghị chạy `high`, `medium` hay `low`.
- So sánh chất lượng và độ ổn định giữa các mode.
- Xem model local nào đang có sẵn.
- Kiểm tra logic chọn model của dự án trước khi mở camera thật.

## 2. Lệnh chạy

```powershell
.\.venv\Scripts\python run_app.py --advisor-only
```

## 3. Công cụ này hiển thị những gì

- CPU, RAM, GPU, VRAM.
- Trạng thái CUDA và PyTorch.
- Các model YOLO local đang có.
- 3 mức runtime `high`, `medium`, `low`.
- `model`, `device`, `imgsz`, `max_det` của từng mức.
- Mức đề xuất nên dùng ngay tại thời điểm chạy.

## 4. Cách đọc kết quả

### `high`

- Mức mạnh nhất máy còn gánh được.
- Ưu tiên chất lượng nhận diện.
- Có thể nặng hơn về FPS.

### `medium`

- Mức cân bằng nhất để dùng thường xuyên.
- Thường là lựa chọn phù hợp nhất cho camera realtime thực chiến.

### `low`

- Mức nhẹ nhất.
- Ưu tiên độ mượt và ổn định.
- Phù hợp khi máy yếu hoặc đang tải cao.

## 5. Phân biệt với `run_doctor.py`

`run_app.py --advisor-only`:

- Tập trung vào giải thích logic chọn runtime.
- Phù hợp để ra quyết định trước khi chạy camera.

`run_doctor.py`:

- Tập trung vào sức khỏe hệ thống.
- Kiểm tra camera, model, dữ liệu, icon và môi trường runtime.

## 6. Phân biệt với `run_app.py`

`run_app.py`:

- Mở webcam thật.
- Chạy YOLO inference.
- Hiển thị box nhận diện, FPS và panel trạng thái.

`run_app.py --advisor-only`:

- Chỉ phân tích.
- Không mở webcam.

## 7. Vì sao kết quả của Runtime Tool quan trọng

Chọn sai mode có thể dẫn đến:

- FPS cao nhưng nhận diện kém do `imgsz` quá thấp.
- Model quá nặng khiến camera giật hoặc timeout.
- GPU chưa được khai thác đúng dù máy có CUDA.

Runtime Tool giúp nhìn ra điều đó trước khi bạn vào camera thật.

## 8. Những yếu tố ảnh hưởng đến khuyến nghị

Runtime Tool dựa trên:

- Loại GPU và VRAM.
- Có CUDA hay không.
- Mức tải CPU/GPU/VRAM hiện tại.
- Danh sách model YOLO đang có trên máy.

## 9. Liên hệ với chất lượng nhận diện

Khi camera nhận diện chưa tốt, bạn nên dùng Runtime Tool để kiểm tra:

- Máy đang chạy mode nào.
- Có đang bị ép xuống model quá nhẹ không.
- Có còn dư chỗ để tăng `imgsz` hoặc đổi model không.
- Có nên chuyển sang `models/trained/best.pt` cho bài toán custom không.

## 10. Ví dụ cách dùng thực tế

### Trường hợp 1: Máy có RTX 3050 Ti 4 GB

Runtime Tool có thể gợi ý:

- `high`: `yolo11s.pt / cuda:0 / imgsz 640`
- `medium`: `yolo11s.pt / cuda:0 / imgsz 512`
- `low`: `yolo11n.pt / cuda:0 / imgsz 416`

Nếu bạn cần chất lượng tốt hơn mà FPS vẫn ổn, nên thử `medium` trước.

### Trường hợp 2: Máy chỉ có CPU

Bạn thường sẽ được kéo về:

- `low`
- `fallback_cpu`
- hoặc `fallback_cpu_weak`

Lúc này mục tiêu nên là ổn định trước, rồi mới nghĩ đến tăng chất lượng.

## 11. Kết hợp với `config/settings.yaml`

Sau khi xem gợi ý từ Runtime Tool, bạn có thể tinh chỉnh thêm:

- `inference.confidence`
- `inference.iou`
- `inference.display_confidence`
- `inference.person_confidence`
- `inference.phone_confidence`
- `inference.enhance_low_light`

## 12. Quy trình khuyến nghị

Để tối ưu camera theo cách có kiểm soát:

```powershell
.\.venv\Scripts\python run_app.py --advisor-only
.\.venv\Scripts\python run_doctor.py --skip-camera-check
.\.venv\Scripts\python run_app.py --mode medium
```

Sau đó quan sát:

- FPS có đủ ổn không.
- Box có bị bỏ sót không.
- Có cần chuyển model hoặc dùng `best.pt` hay không.
