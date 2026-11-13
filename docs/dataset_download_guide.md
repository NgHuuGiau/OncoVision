# Hướng Dẫn Tải Dataset Ảnh Y Khoa

## Tổng quan

Script `scripts/download_datasets.py` tự động tải dataset ảnh y khoa cho 11 loại ung thư từ HuggingFace.

## Sử dụng

### Tải tất cả 11 loại
```bash
python scripts/download_datasets.py --max-images 1000
```

### Tải từng loại riêng
```bash
python scripts/download_datasets.py --cancer brain --max-images 1000
python scripts/download_datasets.py --cancer breast --max-images 1000
python scripts/download_datasets.py --cancer lung --max-images 1000
```

### Xem trước không tải
```bash
python scripts/download_datasets.py --dry-run
```

## Dataset đã tải thành công

| Loại ung thư | Ảnh | Lớp | Modality |
|---|---|---|---|
| **Brain** | ✅ 789 | glioma (250), meningioma (250), pituitary (250), no_tumor (39) | MRI |

## Các loại còn lại (cần tải tiếp)

| Loại ung thư | Dataset | Modality | Trạng thái |
|---|---|---|---|
| Breast | gymprathap/Breast-Cancer-Ultrasound-Images-Dataset | Siêu âm | ⏳ Chưa tải |
| Lung | Kabil007/LungCancer4Types | CT | ⏳ Chưa tải |
| Thyroid | FangDai/Thyroid_Ultrasound_Images | Siêu âm | ⏳ Cần login HF |
| Kidney | abdullahmostafa34/kidney-ct-scans | CT | ⏳ Chưa tải |
| Liver | spsahikotha/Liver-Tumor-Detection | CT | ⏳ Chưa tải |
| Stomach | spsahikotha/Stomach-Cancer-Detection | Nội soi | ⏳ Chưa tải |
| Colorectal | Athsf/ColonDB | Nội soi | ⏳ Chưa tải |
| Cervical | Bala2309/cervical-cancer-classification | Colposcopy | ⏳ Chưa tải |
| Prostate | spsahikotha/Prostate-Cancer-Detection | MRI | ⏳ Chưa tải |
| Pancreas | spsahikotha/Pancreas-Cancer-Detection | CT | ⏳ Chưa tải |

## Lưu ý quan trọng

### 1. Rate Limit HuggingFace

Khi không có token, HuggingFace giới hạn tốc độ tải (~30 file/phút). Để tải nhanh hơn:

```bash
# Đăng ký tài khoản tại https://huggingface.co/join
# Tạo token tại https://huggingface.co/settings/tokens
# Đặt biến môi trường
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
python scripts/download_datasets.py --max-images 1000
```

### 2. Thyroid cần login

Dataset thyroid (FangDai/Thyroid_Ultrasound_Images) yêu cầu đăng nhập HuggingFace.

### 3. Cấu trúc dữ liệu

```
data/datasets/
├── brain/
│   ├── glioma/
│   │   ├── Te-gl_1.jpg
│   │   ├── Te-gl_2.jpg
│   │   └── ...
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
├── breast/
│   ├── normal/
│   ├── benign/
│   └── malignant/
└── ... (các loại khác)
```

### 4. Tiếp tục tải

Nếu script bị dừng do rate limit, chạy lại với cùng tham số — script sẽ tiếp tục từ nơi dừng:

```bash
python scripts/download_datasets.py --cancer breast,lung,kidney,liver --max-images 1000
```

## Kích thước ước tính

| Loại | Số ảnh | Kích thước ước tính |
|---|---|---|
| Brain | ~1,000 | ~20 MB |
| Breast | ~780 | ~200 MB |
| Lung | ~1,000 | ~50 MB |
| Thyroid | ~900 | ~100 MB |
| Các loại khác | ~1,000 mỗi loại | ~50-200 MB mỗi loại |
| **Tổng** | **~11,000** | **~1-2 GB** |
