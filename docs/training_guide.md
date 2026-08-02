# Hướng Dẫn Training Object Detection

Tài liệu mô tả pipeline huấn luyện YOLO detection: từ dữ liệu raw đến model `best.pt` triển khai trên runtime camera.

---

## 1. Tổng quan

Pipeline gồm các bước:

1. Chuẩn bị dữ liệu (raw → processed)
2. Kiểm tra dữ liệu và nhãn
3. Chia tập train/val/test
4. Huấn luyện model YOLO
5. Đánh giá và xuất model
6. Triển khai lên camera

---

## 2. Cấu trúc thư mục

```
dataset/processed/
├── images/                    # Ảnh
│   ├── train/
│   ├── val/
│   └── test/
├── labels/                    # Nhãn YOLO
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml                  # Cấu hình dataset



runs/detect/medical_yolo/
└── weights/
    ├── best.pt                # Model tốt nhất
    └── last.pt                # Model checkpoint cuối

models/
├── pretrained/
│   └── yolo11s.pt             # Model tiền huấn luyện
└── trained/
    └── medical_yolo_detect.pt # Model đã train
```

---

## 3. Định dạng dữ liệu

### Ảnh

Đặt trong `dataset/processed/images/`, không có yêu cầu kích thước cố định.

### Nhãn YOLO

Mỗi ảnh có một file `.txt` tương ứng trong `dataset/processed/labels/`, định dạng:

```
class_id x_center y_center width height
```

Trong đó `x_center`, `y_center`, `width`, `height` là giá trị chuẩn hóa (0–1).

---

## 4. Cấu hình dataset

File `dataset/processed/data.yaml`:

```yaml
train: dataset/processed/images/train
val: dataset/processed/images/val
test: dataset/processed/images/test
nc: <số_lớp>
names: ["class_0", "class_1", ...]
```

Nếu thay đổi danh sách lớp, cần cập nhật đồng thời nhãn raw, dataset split, `data.yaml` và logic train.

---

## 5. Huấn luyện

```powershell
python run_train.py
```

Script tự động:

- Resume từ `runs/detect/medical_yolo/weights/last.pt` nếu có
- Train 50 epochs với cấu hình tối ưu
- Lưu best model vào `models/trained/medical_yolo_detect.pt`

### Tham số training

| Tham số | Giá trị | Mục đích |
|---|---|---|
| Model | yolo11s | Nhẹ, phù hợp 4GB VRAM |
| Epochs | 50 | Hội tụ tốt |
| Imgsz | 512 | Chi tiết cao, cải thiện mAP |
| Batch | 4 | An toàn VRAM |
| Optimizer | AdamW | Ổn định hơn SGD |
| LR0 | 0.001 | Learning rate đầu |
| LRF | 0.0001 | Learning rate cuối |
| Warmup | 5 epochs | Tránh shock |
| Mosaic | 0.3 | Augment giữ giải phẫu |
| Mixup | 0.1 | Tăng diversity |
| Copy-paste | 0.1 | Tăng diversity |
| Lật ngang | 50% | Augment |
| Lật dọc | 0% | Giữ hướng giải phẫu |
| Xoay | 15° | Augment |

---

## 6. Lựa chọn model

- **`models/pretrained/yolo11s.pt`**: dùng baseline nhanh, chưa có dataset custom, đang debug pipeline
- **`models/trained/medical_yolo_detect.pt`**: dùng sau khi train xong, tăng độ chính xác cho lớp riêng

### Triển khai

```powershell
python run_app.py --model models/trained/medical_yolo_detect.pt
```

---

## 7. Dấu hiệu dataset chưa tốt

- Nhãn thiếu hoặc sai định dạng
- Class map không đồng nhất
- Quá ít ảnh
- Ảnh train khác xa môi trường webcam thật
- Vật thể nhỏ nhưng `imgsz` thấp
- Góc chụp thiếu biến thể

### Cải thiện

- Chụp nhiều góc, nhiều điều kiện ánh sáng
- Gán nhãn nhất quán
- Có cả background sạch và phức tạp
- Không đổi class order giữa các lần train

---

## 8. Kiểm tra sau training

```powershell
python run_train.py
python run_doctor.py --skip-camera-check
python run_app.py --model models/trained/medical_yolo_detect.pt
```

### Câu hỏi tự kiểm tra

1. Model nhận đúng class chính không?
2. Có bỏ sót vật thể nhỏ không?
3. False positive có quá nhiều không?
4. FPS trên webcam thật còn chấp nhận được không?
5. Model custom có thực sự tốt hơn pretrained không?

---

## 9. Debug nhanh

| Triệu chứng | Nơi debug |
|---|---|
| Script train fail | `dataset/processed/data.yaml`, dependency, VRAM |
| Train chạy nhưng model kém | Dataset raw, class map, điều kiện chụp |
| Camera nhận diện khác kỳ vọng | `run_app.py`, model đã nạp, image size |

### Phân tách nhánh

- **Object detection**: `dataset/object_detection/`, `run_train.py`
- **Medical**: `dataset/medical/`, `run_medical.py`

Không trộn lẫn hai luồng. Xem `medical_imaging_guide.md` cho nhánh y dược.
