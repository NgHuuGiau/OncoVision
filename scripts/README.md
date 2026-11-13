# Scripts OncoVision

> **Cập nhật 2026-11-13**: Scripts đã được phân loại theo mục đích. Các script debug one-off nên được xóa sau khi không còn cần.

## 📋 Phân loại Scripts

### ✅ Production (Sử dụng thường xuyên)

| Script | Mô tả |
|---|---|
| `fetch_model.py` | Tải model từ HuggingFace/GitHub, kiểm tra SHA256, đặt đúng vị trí |
| `import_foreign_model.py` | Import model từ bên ngoài vào hệ thống |
| `organize_models.py` | Tổ chức lại thư mục model theo cấu trúc chuẩn |

**Ví dụ:**
```bash
# Tải tất cả model có sẵn trong catalog
python scripts/fetch_model.py --all

# Xem trạng thái model
python scripts/fetch_model.py --status

# Tải model cụ thể
python scripts/fetch_model.py owner/repo --file model.pt --target lung --kind classifier --modality ct

# Import model từ file local
python scripts/import_foreign_model.py --source path/to/model.pt --target brain --kind classifier
```

### 🔧 Development (Phát triển)

| Script | Mô tả |
|---|---|
| `train_modality_classifier.py` | Train classifier nhận diện modality |
| `train_modality_fixed.py` | Phiên bản sửa lỗi |
| `train_modality_simple.py` | Phiên bản đơn giản để test |
| `generate_volume_models.py` | Generate model 3D volume |
| `generate_volumes.py` | Generate volume data |
| `generate_volumes_v2.py` | Phiên bản 2 generate volume |

> **Lưu ý**: Dự án chỉ suy luận (inference-only). Training phải được thực hiện bên ngoài.

### 🔍 Debug (Tạm thời — nên xóa sau)

| Script | Mô tả | Trạng thái |
|---|---|---|
| `check_models.py` | Kiểm tra trạng thái model | ⚠️ Có thể thay bằng `fetch_model.py --status` |
| `check_models2.py` | Phiên bản 2 | ⚠️ Duplicate |
| `check_models_full.py` | Kiểm tra đầy đủ | ⚠️ Có thể thay bằng `fetch_model.py --status` |
| `check_modalities.py` | Kiểm tra modality | ⚠️ Tạm thời |
| `check_password.py` | Kiểm tra password | ⚠️ Tạm thời |
| `check_status.py` | Kiểm tra trạng thái | ⚠️ Có thể thay bằng `run_doctor.py` |
| `check_users.py` | Kiểm tra user | ⚠️ Tạm thời |
| `get_modalities.py` | Lấy danh sách modality | ⚠️ Tạm thời |
| `probe_liver.py` | Probe liver model | ⚠️ Tạm thời |
| `probe_liver2.py` | Phiên bản 2 | ⚠️ Duplicate |

### 📦 Deploy (HuggingFace search)

| Script | Mô tả |
|---|---|
| `deploy/download_more_models.py` | Tải thêm model |
| `deploy/hf_search_broad_missing.py` | Tìm model thiếu |
| `deploy/hf_search_detectors.py` | Tìm detector |
| `deploy/search_more_*.py` | Tìm kiếm model |

### 🛠️ Maintenance

| Script | Mô tả |
|---|---|
| `maintenance/convert_batch1.py` | Convert model |
| `maintenance/test_new_models.py` | Test model mới |

### 🪟 Windows-only (PowerShell)

| Script | Mô tả |
|---|---|
| `cleanup_main.ps1` | Dọn dẹp output |
| `create_dist.ps1` | Tạo distribution |
| `deploy_production.ps1` | Deploy production |

> **Lưu ý**: Các script PowerShell chỉ chạy trên Windows. Dùng Docker cho cross-platform.

## 🗑️ Scripts đã gộp (có thể xóa)

Các script sau đã được gộp vào `fetch_model.py` hoặc `run_smoke.py`:

- `download_all_models.py` → `fetch_model.py --all`
- `download_models.py` → `fetch_model.py --all`
- `download_missing_models.py` → `fetch_model.py --all`
- `download_volume_models.py` → `fetch_model.py --all`
- `download_final_*.py` → `fetch_model.py --all`
- `final_summary.py` → `fetch_model.py --status`
- `summary_models.py` → `fetch_model.py --status`
- `verify_full_pipeline.py` → `run_smoke.py`

## 📊 Thống kê

- **Production**: 3 scripts
- **Development**: 6 scripts
- **Debug**: 10 scripts (nên dọn dẹp)
- **Deploy**: 7 scripts
- **Maintenance**: 2 scripts
- **Tổng**: ~28 scripts (mục tiêu: giảm xuống ≤15)
