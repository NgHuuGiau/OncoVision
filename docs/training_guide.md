# Hướng Dẫn Training

Tài liệu này mô tả đầy đủ luồng chuẩn bị dữ liệu, huấn luyện, đánh giá và xuất model custom trong dự án.

## Công nghệ dùng trong training

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- PyYAML
- tqdm

## Dữ liệu đầu vào

Pipeline training sử dụng trực tiếp:

- `dataset/raw/images`
- `dataset/raw/labels`

Mỗi ảnh trong `images` nên có file `.txt` tương ứng trong `labels`.

## Định dạng label YOLO

Mỗi dòng label có dạng:

```text
class_id x_center y_center width height
```

Trong đó:

- `class_id`: số lớp
- `x_center`, `y_center`, `width`, `height`: giá trị chuẩn hóa trong khoảng `0..1`

## Cấu hình class hiện tại

File `training/data.yaml` định nghĩa mapping class của dự án.
Nếu bạn muốn train class riêng, hãy cập nhật đồng thời:

- ảnh trong `dataset/raw/`
- label trong `dataset/raw/`
- mapping class trong `training/data.yaml`

## Luồng training khuyến nghị

```powershell
python training\prepare_dataset.py
python training\validate_dataset.py
python training\split_dataset.py
python run_train.py
python training\validate_model.py
python training\export_model.py
```

## Ý nghĩa từng bước

### `training/prepare_dataset.py`

- tạo khung thư mục cần thiết
- chuẩn hóa môi trường dataset

### `training/validate_dataset.py`

- kiểm tra ảnh thiếu
- kiểm tra label thiếu hoặc sai format
- kiểm tra class id không khớp

### `training/split_dataset.py`

- chia dữ liệu thành `train`, `val`, `test`
- ghi kết quả vào `dataset/processed/`

### `run_train.py`

- chạy pipeline huấn luyện chính
- sinh weights trong thư mục train

### `training/validate_model.py`

- đánh giá model sau khi train
- dùng để so sánh với runtime camera

### `training/export_model.py`

- xuất model phục vụ deploy
- đồng bộ model đích về `models/trained/best.pt` nếu cần

## Pretrained và custom model

### `models/pretrained/*.pt`

- dùng cho nhận diện tổng quát
- hợp với bài toán camera realtime phổ biến

### `models/trained/best.pt`

- là model custom sau khi bạn train từ dataset riêng
- hợp khi bài toán có class chuyên biệt

## Khi nào nên dùng `best.pt`

Nên dùng model custom nếu:

- bạn train class không có trong COCO
- bạn cần độ chính xác cao cho domain riêng
- bạn muốn tối ưu cho dữ liệu thực tế của dự án

Ví dụ:

```powershell
python run_app.py --model models/trained/best.pt
```

## Vì sao camera có thể nhận diện chưa đúng

Nguyên nhân thường gặp:

- dùng pretrained model cho bài toán cần class custom
- `training/data.yaml` không khớp với label thật
- dataset quá ít hoặc quá lệch góc chụp
- ảnh train khác quá xa điều kiện webcam thực tế
- `imgsz` quá thấp để giữ FPS

## Cách làm model ổn định hơn

- chụp nhiều góc
- có nhiều mức sáng
- có cả nền sạch và nền phức tạp
- gắn label nhất quán
- giữ class map ổn định giữa các lần train

## Kiểm tra sau training

Sau khi train xong, nên trả lời các câu hỏi:

1. Model có nhận đúng class không?
2. Có bỏ sót object nhỏ không?
3. Có quá nhiều false positive không?
4. Chạy webcam thật có ổn dưới ánh sáng yếu không?

## Gợi ý chọn model

- Máy có GPU khá: bắt đầu với `yolo11s.pt` hoặc `yolo11m.pt`
- Máy yếu hơn: bắt đầu với `yolo11n.pt`
- Class khó nhưng nhỏ: ưu tiên dataset tốt trước khi tăng model size

## Sau training nên làm gì

```powershell
python training\validate_model.py
python run_doctor.py --skip-camera-check
python run_app.py --model models/trained/best.pt
```

Mục tiêu là kiểm tra model custom ngay trong điều kiện camera thật.
