# Hướng Dẫn Training Object Detection

Tài liệu này mô tả đầy đủ luồng train YOLO object detection trong OncoVision, từ dữ liệu raw đến model `best.pt` đưa vào runtime camera.

## 1. Mục Tiêu

Nhánh training được dùng để:

- chuẩn bị dataset object detection,
- validate dữ liệu và label,
- split train/val/test,
- train model YOLO custom,
- validate và export model,
- đưa model vào `run_app.py`.

## 2. Thư Mục Và Tệp Liên Quan

```text
dataset/object_detection/raw/
  images/
  labels/

dataset/object_detection/processed/
  images/
    train/
    val/
    test/

training/
  data.yaml
  train_config.yaml
  prepare_dataset.py
  validate_dataset.py
  split_dataset.py
  train_model.py
  validate_model.py
  export_model.py
```

## 3. Đầu Vào Chuẩn

### Ảnh gốc

- đặt trong `dataset/object_detection/raw/images/`

### Label YOLO

- đặt trong `dataset/object_detection/raw/labels/`
- tên file phải khớp tên ảnh

Ví dụ:

```text
images/
  sample_001.jpg
labels/
  sample_001.txt
```

## 4. Định Dạng Label YOLO

Mỗi dòng label:

```text
class_id x_center y_center width height
```

Trong đó:

- `class_id`: chỉ số lớp
- `x_center`, `y_center`, `width`, `height`: giá trị chuẩn hóa trong khoảng `0..1`

## 5. File Cấu Hình Quan Trọng

### `training/data.yaml`

Dùng để:

- khai báo `train`, `val`, `test`,
- map class names,
- đưa cho YOLO biết dataset đang dùng class nào.

Nếu đổi class map, phải đổi đồng thời:

- raw labels,
- dataset split,
- `data.yaml`,
- logic train / validate liên quan.

### `training/train_config.yaml`

Dùng để:

- chọn model mặc định,
- fallback model,
- epoch, batch, image size, output setup nếu dự án đang sử dụng.

## 6. Các Script Trong `training/`

| Script | Vai trò |
|---|---|
| `prepare_dataset.py` | Tạo / đảm bảo khung dataset |
| `validate_dataset.py` | Soát lỗi ảnh, label, class id |
| `split_dataset.py` | Chia train / val / test |
| `train_model.py` | Logic train nội bộ |
| `validate_model.py` | Đánh giá model sau train |
| `export_model.py` | Đồng bộ / xuất model sau train |
| `download_models.py` | Tải pretrained models nếu cần |

## 7. Luồng Training Khuyến Nghị

```powershell
python run_train.py --check-only
python training\prepare_dataset.py
python training\validate_dataset.py
python training\split_dataset.py
python run_train.py
python training\validate_model.py
python training\export_model.py
```

## 8. Ý Nghĩa Từng Bước

### Bước 1. `run_train.py --check-only`

Dùng để trả lời:

- dependency `ultralytics` / `torch` có sẵn không,
- pretrained model có tồn tại không,
- raw data đã có chưa,
- processed train/val đã sẵn sàng chưa.

### Bước 2. `prepare_dataset.py`

Dùng để:

- tạo khung thư mục dataset cần thiết,
- đồng bộ layout local.

### Bước 3. `validate_dataset.py`

Dùng để:

- phát hiện ảnh thiếu label,
- phát hiện label sai format,
- phát hiện `class_id` không hợp lệ,
- phát hiện dữ liệu xấu trước khi train.

### Bước 4. `split_dataset.py`

Dùng để:

- chia dữ liệu sang `train`, `val`, `test`,
- đưa dữ liệu vào layout mà YOLO có thể dùng.

### Bước 5. `run_train.py`

Dùng để:

- kích hoạt training pipeline chính,
- sinh artifact train trong `runs/`,
- tạo model custom mới.

### Bước 6. `validate_model.py`

Dùng để:

- đánh giá model sau train,
- xác minh model mới có dùng được cho deployment / runtime hay không.

### Bước 7. `export_model.py`

Dùng để:

- xuất model được chọn,
- đồng bộ về `models/trained/best.pt` nếu quy trình dự án cần.

## 9. Model Nào Nên Dùng

### `models/pretrained/*.pt`

Dùng khi:

- cần baseline nhanh,
- chưa có dataset custom đủ tốt,
- đang debug pipeline runtime.

### `models/trained/best.pt`

Dùng khi:

- đã train xong bộ dữ liệu nội bộ,
- cần tăng độ chính xác cho class riêng,
- muốn demo bằng model của dự án thay vì pretrained.

## 10. Cách Đưa Model Vào Runtime

```powershell
python run_app.py --model models/trained/best.pt
```

Có thể kết hợp với mode:

```powershell
python run_app.py --model models/trained/best.pt --mode medium
```

## 11. Dấu Hiệu Dataset Chưa Tốt

Thường gặp:

- label thiếu hoặc sai format,
- class map không đồng nhất,
- quá ít ảnh,
- ảnh train khác xa môi trường webcam thật,
- object nhỏ nhưng `imgsz` thấp,
- góc chụp quá ít biến thể.

## 12. Cách Làm Model Ổn Định Hơn

- chụp nhiều góc khác nhau,
- có ánh sáng yếu và ánh sáng mạnh,
- có background sạch và background phức tạp,
- gán nhãn nhất quán,
- không đổi class order tùy tiện giữa các lần train.

## 13. Kiểm Tra Sau Training

Sau khi train xong, nên chạy:

```powershell
python training\validate_model.py
python run_doctor.py --skip-camera-check
python run_app.py --model models/trained/best.pt
```

Mục đích:

- xác minh model tồn tại,
- kiểm tra runtime vẫn mở được,
- test trong bối cảnh webcam thật.

## 14. Các Câu Hỏi Nên Tự Trả Lời Sau Mỗi Lần Train

1. Model có nhận đúng class chính không?
2. Có bỏ sót object nhỏ không?
3. Có false positive quá nhiều không?
4. Khi chạy webcam thật, FPS còn chấp nhận được không?
5. Model custom có thực sự tốt hơn pretrained không?

## 15. Kiểm Lỗi Nhanh Theo Triệu Chứng

| Triệu chứng | Nơi nên debug |
|---|---|
| `run_train.py --check-only` fail | `training/train_config.yaml`, `training/model_paths.py`, dependency local |
| Dataset split xong nhưng count sai | `training/split_dataset.py`, `training/dataset_ops.py` |
| Train chạy nhưng model kém | dataset raw, class map, điều kiện chụp, `data.yaml` |
| Runtime camera nhận diện khác training kỳ vọng | `run_app.py`, `config/settings.yaml`, model đã nạp, image size |

## 16. Liên Quan Tới Nhánh Medical

Nhánh object detection và nhánh medical tách biệt về dữ liệu:

- object detection: `dataset/object_detection/`
- medical: `dataset/medical/`

Không nên trộn 2 layout này vào nhau. Nếu cần huấn luyện medical riêng, hãy đi theo hướng dẫn ở `medical_imaging_guide.md`.
