# Hướng Dẫn Tải Model OncoVision

## Tổng Quan

OncoVision sử dụng model suy luận (inference-only) đã được chuẩn bị bên ngoài. Model không nằm trong repository (gitignored) và phải được tải hoặc đặt thủ công vào thư mục `models/pretrained/`.

## Cách 1: Tải Tự Động (Khuyến Nghị)

### Tải tất cả model có sẵn

```bash
python scripts/fetch_model.py --all
```

Lệnh này đọc `models/model_catalog.json` và tải tất cả model được liệt kê từ HuggingFace hoặc GitHub.

### Tải model cụ thể

```bash
# Từ HuggingFace
python scripts/fetch_model.py owner/repo --file model.pt --target lung --kind classifier --modality ct

# Từ GitHub
python scripts/fetch_model.py owner/repo --file model.pt --target kidney --kind detector --modality ct --repo-type github
```

### Tùy chọn

| Tùy chọn | Mô tả |
|---|---|
| `--all` | Tải mọi model trong catalog |
| `--force` | Ghi đè file đích nếu đã có |
| `--catalog PATH` | Đường dẫn catalog JSON (mặc định: `models/model_catalog.json`) |
| `--base PATH` | Thư mục đích (mặc định: `models/pretrained/cancers`) |
| `--dry-run` | Xem trước mà không tải |

## Cách 2: Đặt Thủ Công

### Cấu trúc thư mục

```
models/
└── pretrained/
    ├── brain_classifier.pt              # Classifier u não (4 loại)
    ├── modality_classifier.pt           # Classifier modality (8 loại)
    ├── brain_detector.pt                # YOLO detector y khoa
    ├── brain_volume_model.pt2           # Model 3D (torch.export)
    ├── brain_volume_model.json          # Manifest 3D
    └── cancers/
        ├── brain/
        │   ├── classifier.pt
        │   ├── detector.pt
        │   ├── mri/                     # Theo modality
        │   │   ├── classifier.pt
        │   │   └── detector.pt
        │   └── ct/
        ├── lung/
        │   ├── ct/
        │   └── xray/
        ├── kidney/
        └── ... (các nhóm ung thư khác)
```

### Các bước

1. Train model bên ngoài (không train trong dự án này)
2. Đặt file `.pt` vào thư mục tương ứng
3. Chạy kiểm tra:

```bash
python run_doctor.py --skip-camera-check
python run_medical.py status
```

## Kiểm Tra Model

### Xem trạng thái model

```bash
python run_medical.py status
```

### Kiểm tra model đã cài

```bash
python run_doctor.py --skip-camera-check
```

### Kết quả kiểm tra

| Biểu tượng | Ý nghĩa |
|---|---|
| ✅ | Model có sẵn và hợp lệ |
| ⚠️ | Model thiếu |
| 🔒 | Nhóm ung thư chưa có model |

## Model Catalog

File `models/model_catalog.json` chứa danh sách model có sẵn để tải:

```json
{
  "version": 1,
  "description": "Catalog model OncoVision",
  "entries": [
    {
      "repo": "owner/repo",
      "file": "model.pt",
      "target": "lung",
      "kind": "classifier",
      "modality": "ct",
      "sha256": "abc123..."
    }
  ]
}
```

Khi có model mới trên HuggingFace, thêm vào catalog và chạy `fetch_model.py --all`.

## Xử Lý Lỗi Thường Gặp

### Lỗi: "Không tìm thấy model nào"

```bash
# Chạy lần đầu — tải model
python scripts/fetch_model.py --all
```

### Lỗi: "SHA256 không khớp"

Model bị hỏng hoặc tải thiếu. Xóa file và tải lại:

```bash
# Xóa model cũ và tải lại
python scripts/fetch_model.py --all --force
```

### Lỗi: "Không kết nối được HuggingFace"

Kiểm tra kết nối mạng hoặc tải thủ công từ trình duyệt và đặt vào thư mục `models/pretrained/`.

### Lỗi: "PyAudio không khả dụng" (trên Linux/macOS)

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio
```

## Tham Khảo

- [Medical Imaging Guide](medical_imaging_guide.md) — Luồng suy luận y khoa
- [Project Overview](project_overview.md) — Tổng quan kiến trúc
- [Quick Commands](quick_commands.md) — Lệnh nhanh hàng ngày
