# OncoVision

OncoVision hiện có 2 luồng chính:

- `Y dược`: phân tích ảnh y khoa, skin lesion và dữ liệu TCIA
- `Vật thể`: huấn luyện và kiểm tra YOLO cho bài toán nhận diện vật thể

## Khi chạy `run_menu.py`

Menu sẽ tự tạo đủ cây thư mục chuẩn:

- `dataset/`
- `models/`
- `output/`
- `runs/`

Từ đó, hệ thống sẽ bổ sung thêm các thư mục con cần thiết cho từng luồng.

## Luồng Y dược

Khung dữ liệu chuẩn:

- `dataset/medical/skin_lesion`
- `dataset/medical/tcia`

Lệnh chính:

```powershell
python run_medical.py init-dataset
python run_medical.py status
python run_medical.py ready
python run_medical.py sources
python run_medical.py cancer
python run_medical.py tcia-download --dry-run
python run_medical.py verify-tcia
python run_medical.py tcia-log --collection "CBIS-DDSM / TCGA-BRCA"
```

## Luồng Vật thể

Khung dữ liệu chuẩn:

- `dataset/object_detection/raw`
- `dataset/object_detection/processed`

Lệnh chính:

```powershell
python run_medical.py audit-dataset
python run_medical.py split-dataset
python run_medical.py train
python run_medical.py validate
python run_medical.py train-all
```

## Checklist sẵn sàng train

- Có ảnh và nhãn hợp lệ trong `dataset/object_detection/raw`
- Đã split ra `train / val / test`
- Model local hoặc fallback model tồn tại
- `run_medical.py ready` báo `ready_for_train_skin=True`
- Nếu cần y dược mở rộng, TCIA đã được tải đủ theo mục tiêu

