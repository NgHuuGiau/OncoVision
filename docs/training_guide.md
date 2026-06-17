# Hướng dẫn training

Tài liệu này mô tả đầy đủ luồng chuẩn bị dữ liệu, huấn luyện, đánh giá và dùng model custom trong dự án YOLO.

## 1. Cấu trúc dữ liệu đầu vào

Pipeline training hiện dùng trực tiếp dữ liệu trong:

- `dataset/raw/images`
- `dataset/raw/labels`

Project không còn dùng luồng trung gian `dataset/sample`.

## 2. Định dạng label YOLO

Mỗi file label có dạng:

```text
class_id x_center y_center width height
```

Trong đó:

- `class_id`: chỉ số lớp.
- `x_center`, `y_center`, `width`, `height`: giá trị chuẩn hóa trong khoảng `0..1`.

## 3. Cấu hình class hiện tại

File `training/data.yaml` hiện đang khai báo:

```yaml
names:
  0: person
```

Điều này có nghĩa:

- Dataset hiện được cấu hình tối thiểu cho class `person`.
- Nếu bạn muốn train thêm `phone`, `face`, `helmet` hoặc class khác, bạn phải cập nhật đồng thời ảnh và label trong `dataset/raw/`, cùng mapping class trong `training/data.yaml`.

## 4. Luồng training khuyến nghị

```powershell
.\.venv\Scripts\python training\prepare_dataset.py
.\.venv\Scripts\python training\validate_dataset.py
.\.venv\Scripts\python training\split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python training\export_model.py
```

## 5. Ý nghĩa từng bước

### `training/prepare_dataset.py`

- Tạo khung thư mục cần thiết.
- Chuẩn hóa môi trường dataset trước khi làm việc tiếp.

### `training/validate_dataset.py`

- Kiểm tra file ảnh thiếu.
- Kiểm tra file label thiếu hoặc sai format.
- Kiểm tra class id không khớp với `training/data.yaml`.

### `training/split_dataset.py`

- Chia dữ liệu thành `train`, `val`, `test`.
- Ghi kết quả vào `dataset/processed/`.

### `run_train.py`

- Gọi pipeline huấn luyện chính.
- Sinh ra weights trong thư mục huấn luyện.

### `training/validate_model.py`

- Đánh giá model sau khi train.
- Dùng để so sánh chất lượng trước khi đưa vào runtime camera.

### `training/export_model.py`

- Xuất model phục vụ deploy hoặc dùng ở bước khác.
- Đồng bộ model đích về `models/trained/best.pt` nếu pipeline yêu cầu.

## 6. Phân biệt model pretrained và model custom

Project hiện có hai nhóm model:

### `models/pretrained/*.pt`

- Dùng cho nhận diện tổng quát trên webcam.
- Phù hợp khi bạn muốn nhận diện các object phổ biến như `person`, `cell phone`, `bottle`, `car`, `chair`, v.v.

### `models/trained/best.pt`

- Là model custom do bạn train từ dataset riêng.
- Dùng khi bài toán của bạn có class chuyên biệt hoặc cần logic riêng.

## 7. Khi nào nên dùng `best.pt` trong camera

Nên chạy camera với model custom nếu:

- Bạn train cho class không có sẵn trong COCO.
- Bạn cần độ chính xác cao trên bài toán hẹp.
- Bạn chấp nhận đánh đổi khả năng detect object tổng quát để lấy chất lượng trên class riêng.

Ví dụ:

```powershell
.\.venv\Scripts\python run_app.py --model models/trained/best.pt
```

## 8. Vì sao camera có thể “không nhận diện đúng”

Những nguyên nhân thường gặp:

- Dùng model pretrained cho bài toán cần class custom.
- `training/data.yaml` không khớp với label thật.
- Dataset quá ít, lệch góc chụp hoặc thiếu ánh sáng.
- Ảnh train không giống điều kiện webcam thực tế.
- `imgsz` quá thấp để giữ FPS nên box nhỏ bị bỏ sót.

## 9. Khuyến nghị dữ liệu để model ổn định hơn

- Chụp đủ nhiều góc quay: chính diện, nghiêng trái, nghiêng phải, gần, xa.
- Có nhiều mức sáng khác nhau.
- Có cả ảnh nền sạch và nền phức tạp.
- Gắn label nhất quán giữa mọi file.
- Không để class map thay đổi giữa các lần train.

## 10. Kiểm tra model sau training

Sau khi train, nên trả lời 4 câu hỏi:

1. Model có nhận diện đúng class mong muốn không?
2. Có bỏ sót object nhỏ không?
3. Có sinh nhiều false positive không?
4. Khi chạy webcam thật, model có còn ổn dưới ánh sáng yếu không?

Nếu câu trả lời chưa tốt, hãy ưu tiên cải thiện dataset trước khi tăng model quá lớn.

## 11. Gợi ý chiến lược chọn model

- Máy có GPU khá: bắt đầu với `yolo11s.pt` hoặc `yolo11m.pt`.
- Máy yếu hơn: bắt đầu với `yolo11n.pt`.
- Nếu class của bạn ít nhưng khó, hãy ưu tiên dataset tốt trước khi tăng model size.

## 12. Sau training nên làm gì

Quy trình khép kín nên là:

```powershell
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python run_doctor.py --skip-camera-check
.\.venv\Scripts\python run_app.py --model models/trained/best.pt
```

Mục tiêu là kiểm tra model custom ngay trong điều kiện camera thật, thay vì chỉ nhìn metric offline.
