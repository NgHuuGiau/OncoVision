# Hướng Dẫn Training Object Detection

[![Training](https://img.shields.io/badge/Docs-Training%20Pipeline-FFB000?logo=readthedocs&logoColor=white)](training_guide.md)

Tài liệu này mô tả đầy đủ luồng train YOLO object detection trong OncoVision, từ dữ liệu raw đến model `best.pt` đưa vào runtime camera.

> Nếu bạn đang chuẩn bị train model mới, file này là bản đồ nhanh nhất của luồng dữ liệu và script liên quan.

## Tóm Tắt Nhanh

| Bước | Script chính |
|---|---|
| Chuẩn bị dữ liệu | `scripts/scripts_run_yolo_train.py` |
| Kiểm tra dữ liệu | `scripts/scripts_run_yolo_train.py` |
| Chia tập | Đã có sẵn trong `dataset/processed/` |
| Huấn luyện | `scripts/scripts_run_yolo_train.py` |
| Đánh giá | `python -m ultralytics yolo detect val model=models/trained/medical_yolo_detect.pt data=dataset/processed/data.yaml` |
| Xuất model | Tự động lưu vào `models/trained/medical_yolo_detect.pt` |

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
dataset/processed/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml

scripts/
  scripts_run_yolo_train.py

runs/detect/medical_yolo/
  weights/
    best.pt
    last.pt

models/
  pretrained/
    yolo11s.pt
  trained/
    medical_yolo_detect.pt
```

## 3. Đầu Vào Chuẩn

### Ảnh gốc

- đặt trong `dataset/processed/images/`

### Label YOLO

- đặt trong `dataset/processed/labels/`
- tên file phải khớp tên ảnh

## 4. Định Dạng Label YOLO

Mỗi dòng label:

```text
class_id x_center y_center width height
```

Trong đó:

- `class_id`: chỉ số lớp
- `x_center`, `y_center`, `width`, `height`: giá trị chuẩn hóa trong khoảng `0..1`

## 5. File Cấu Hình Quan Trọng

### `dataset/processed/data.yaml`

Dùng để:

- khai báo `train`, `val`, `test`,
- map class names,
- đưa cho YOLO biết dataset đang dùng class nào.

Nếu đổi class map, phải đổi đồng thời:

- raw labels,
- dataset split,
- `data.yaml`,
- logic train / validate liên quan.

## 6. Các Script Trong `scripts/`

| Script | Vai trò |
|---|---|
| `scripts_run_yolo_train.py` | Huấn luyện YOLO detection với config tối ưu |

## 7. Luồng Training Khuyến Nghị

```powershell
python scripts/scripts_run_yolo_train.py
```

Script sẽ:

- Tự động resume từ `runs/detect/medical_yolo/weights/last.pt` nếu có
- Train 50 epochs với imgsz=512, batch=4, AdamW optimizer
- Lưu best model vào `models/trained/medical_yolo_detect.pt`

## 8. Cấu Hình Training Hiện Tại

| Tham số | Giá trị | Mục đích |
|---------|---------|----------|
| Model | yolo11s | Nhẹ, phù hợp 4GB VRAM |
| Epochs | 50 | Train sâu để hội tụ tốt |
| Imgsz | 512 | Tăng chi tiết, cải thiện mAP |
| Batch | 4 | An toàn với 4GB VRAM |
| Optimizer | AdamW | Ổn định hơn SGD |
| LR0 | 0.001 | Learning rate khởi đầu |
| LRF | 0.0001 | Learning rate cuối |
| Warmup | 5 epochs | Tránh shock ở đầu |
| Mosaic | 0.3 | Giảm nhiễu, giữ giải phẫu |
| Mixup | 0.1 | Tăng diversity nhẹ |
| Copy-paste | 0.1 | Tăng diversity nhẹ |
| Flip LR | 0.5 | Lật ngang 50% |
| Flip UD | 0.0 | Giữ hướng giải phẫu |
| Degrees | 15 | Xoay ảnh |
| HSV-H | 0.015 | Màu nhẹ |
| HSV-S | 0.5 | Saturation vừa |
| HSV-V | 0.3 | Brightness vừa |

## 9. Model Nào Nên Dùng

### `models/pretrained/yolo11s.pt`

Dùng khi:

- cần baseline nhanh,
- chưa có dataset custom đủ tốt,
- đang debug pipeline runtime.

### `models/trained/medical_yolo_detect.pt`

Dùng khi:

- đã train xong bộ dữ liệu nội bộ,
- cần tăng độ chính xác cho class riêng,
- muốn demo bằng model của dự án thay vì pretrained.

## 10. Cách Đưa Model Vào Runtime

```powershell
python run_app.py --model models/trained/medical_yolo_detect.pt
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
python scripts/scripts_run_yolo_train.py
python run_doctor.py --skip-camera-check
python run_app.py --model models/trained/medical_yolo_detect.pt
```

Mục đích:

- xác minh model tồn tại,
- kiểm tra runtime vẫn mở được,
- test trong bối cảnh webcam thật.

## 14. Câu Hỏi Nên Tự Trả Lời Sau Mỗi Lần Train

1. Model có nhận đúng class chính không?
2. Có bỏ sót object nhỏ không?
3. Có false positive quá nhiều không?
4. Khi chạy webcam thật, FPS còn chấp nhận được không?
5. Model custom có thực sự tốt hơn pretrained không?

## 15. Kiểm Lỗi Nhanh Theo Triệu Chứng

| Triệu chứng | Nơi nên debug |
|---|---|
| `scripts/scripts_run_yolo_train.py` fail | `dataset/processed/data.yaml`, dependency local, VRAM |
| Train chạy nhưng model kém | dataset raw, class map, điều kiện chụp, `data.yaml` |
| Runtime camera nhận diện khác training kỳ vọng | `run_app.py`, `config/medical_settings.yaml`, model đã nạp, image size |

## 16. Liên Quan Tới Nhánh Medical

Nhánh object detection và nhánh medical tách biệt về dữ liệu:

- object detection: `dataset/object_detection/`
- medical: `dataset/medical/`

Không nên trộn 2 layout này vào nhau. Nếu cần huấn luyện medical riêng, hãy đi theo hướng dẫn ở `medical_imaging_guide.md`.

![Luồng training](../images/Ảnh%20luồng%20training.png)
