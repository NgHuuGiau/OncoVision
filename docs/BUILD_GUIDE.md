# Hướng dẫn Build OncoVision thành EXE

## Yêu cầu
- Python 3.12 (khuyến nghị) hoặc 3.10/3.11 — **không dùng Python 3.13** (chưa hỗ trợ cx_Freeze/PyInstaller đầy đủ)
- cx_Freeze 8.x: `pip install cx-Freeze`
- Disk space: ~15GB (models 6.95GB + build process)

## Build Desktop Chat App

### Cách 1: Dùng PowerShell script
```powershell
.\scripts\build_exe.ps1
```

### Cách 2: Chạy trực tiếp
```powershell
python scripts/setup_build.py build_exe
```

## Kết quả
- Output: `build/exe.win-amd64-3.12/`
  - `OncoVision.exe` — file chạy chính (~1.36 MB)
  - `lib/` — dependencies Python (~5 GB)
    - `torch/`, `cv2/`, `ultralytics/` — AI libraries
    - `PySide6/` — GUI framework
  - `models/` — model AI (6.95 GB)
    - `pretrained/cancers/` — 266 files cho 11 loại ung thư
  - `medical/`, `core/`, `utils/` — code Python
  - `config/` — cấu hình
  - `assets/` — tài nguyên
  - `templates/` — HTML templates

## Chạy EXE
```powershell
.\build\exe.win-amd64-3.12\OncoVision.exe
```

## Lưu ý quan trọng

### 1. Models đã được bundle
- Tất cả 266 model files (6.95 GB) đã được bundle vào `models/`
- Không cần tải model riêng khi mang exe đi nơi khác

### 2. Đường dẫn tự động phát hiện
- Code tự động phát hiện khi chạy từ exe (`sys.frozen`)
- `paths.py`, `cancer_model_registry.py`, `pipeline.py` đã được fix
- Output/data sẽ lưu trong thư mục build khi chạy từ exe

### 3. Thời gian build
- Build lần đầu: ~30-60 phút (copy models 6.95 GB + PyTorch)
- Build lại: ~10-20 phút (nếu build/ còn cache)

### 4. Kích thước cuối cùng
- `build/exe.win-amd64-3.12/`: ~11.92 GB total
  - `OncoVision.exe`: 1.36 MB
  - `lib/`: ~5 GB (PyTorch, PySide6, ultralytics, v.v.)
  - `models/`: 6.95 GB (266 model files)

### 5. Dọn dẹp build
```powershell
Remove-Item -Path build -Recurse -Force
```

## Lỗi thường gặp

### ModuleNotFoundError: No module named 'torch.cuda'
- **Nguyên nhân**: cx_Freeze exclude `torch.cuda` và `torchvision` khỏi bundle
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — thêm `torch.cuda` và `torchvision` vào `includes`

### ModuleNotFoundError: No module named 'torch.distributed'
- **Nguyên nhân**: cx_Freeze exclude `torch.distributed` khỏi bundle, nhưng `torch.utils.data.dataloader` cần import nó khi chạy
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — xóa `torch.distributed` khỏi `excludes` và thêm vào `includes`
- Nếu gặp lại: kiểm tra `scripts/setup_build.py` có `"torch.distributed"` trong `includes` và **không** trong `excludes`

### AttributeError: module 'torch.jit' has no attribute 'script_if_tracing'
- **Nguyên nhân**: cx_Freeze exclude `torch.jit` khỏi bundle, nhưng `torch.distributions.von_mises` cần `@torch.jit.script_if_tracing` decorator khi runtime
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — xóa `torch.jit` khỏi `excludes`
- Nếu gặp lại: kiểm tra `scripts/setup_build.py` có **không** chứa `"torch.jit"` trong `excludes`

### ModuleNotFoundError: No module named 'torch.fx.experimental._constant_symnode'
- **Nguyên nhân**: cx_Freeze exclude `torch.fx` khỏi bundle, nhưng `torch.nested` cần `torch.fx.experimental._constant_symnode` cho symbolic shapes khi runtime
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — xóa `torch.fx` khỏi `excludes` và thêm vào `includes`
- Nếu gặp lại: kiểm tra `scripts/setup_build.py` có `"torch.fx"` trong `includes` và **không** trong `excludes`

### ImportError: cannot import name 'multiprocessing' from partially initialized module 'torch'
- **Nguyên nhân**: cx_Freeze exclude `torch.multiprocessing` khỏi bundle, nhưng PyTorch cần nó cho nested tensor và symbolic shapes khi runtime, gây circular import
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — xóa `torch.multiprocessing` khỏi `excludes` và thêm vào `includes`
- Nếu gặp lại: kiểm tra `scripts/setup_build.py` có `"torch.multiprocessing"` trong `includes` và **không** trong `excludes`

### ImportError: cannot import name 'ao' from partially initialized module 'torch'
- **Nguyên nhân**: cx_Freeze exclude `torch.ao` khỏi bundle, nhưng PyTorch 2.7+ cần import `torch.ao` trong `torch/__init__.py` khi khởi động, gây circular import
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — xóa `torch.ao` khỏi `excludes`
- Nếu gặp lại: kiểm tra `scripts/setup_build.py` có **không** chứa `"torch.ao"` trong `excludes`

### ModuleNotFoundError: No module named 'html'
- **Nguyên nhân**: cx_Freeze exclude `html` (stdlib module) khỏi bundle, nhưng `torchvision.datasets.flickr` cần `html.parser` khi import
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — xóa `"html"` khỏi `excludes`
- Nếu gặp lại: kiểm tra `scripts/setup_build.py` có **không** chứa `"html"` trong `excludes`
- **Bài học**: Không exclude Python stdlib modules mà third-party libraries có thể cần

### Models không được copy vào build
- **Nguyên nhân**: cx_Freeze 8.x cần đường dẫn tuyệt đối cho `include_files`
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — dùng `PROJECT_ROOT` path tuyệt đối cho tất cả include_files
- Nếu gặp lại: kiểm tra `scripts/setup_build.py` dùng `str(PROJECT_ROOT / "models")` thay vì `"models"`

### Quy tắc chung cho torch modules
- Các torch modules mà PyTorch import động khi runtime **không được exclude** trong cx_Freeze
- Các module đã phải thêm vào includes: `torch.cuda`, `torch.distributed`, `torch.jit`, `torch.fx`, `torch.multiprocessing`, `torch.ao`
- Các module vẫn có thể exclude: `torch.elastic`, `torch.onnx`, `torch.testing`, `torch.utils.tensorboard`

### RecursionError với PyTorch
- cx_Freeze gặp lỗi recursion khi phân tích PyTorch
- **Đã fix**: `scripts/setup_build.py` đã tăng recursion limit lên 5000 và exclude các torch modules không cần thiết

### Permission denied
- File bị lock khi copy
- **Giải pháp**: Dọn dẹp build cũ trước khi build lại

### cx_Freeze 8.x không dùng spec file
- cx_Freeze 8.x không còn hỗ trợ spec file như cũ
- **Giải pháp**: Dùng `scripts/setup_build.py` thay thế

### Python 3.13 không hỗ trợ
- **Nguyên nhân**: cx_Freeze, PyInstaller, PyAudio, PySide6 chưa hỗ trợ đầy đủ Python 3.13
- **Giải pháp**: Dùng Python 3.12 để build EXE. Web app (Docker) cũng dùng Python 3.12.
- **Nếu bắt buộc dùng 3.13**: Build PyInstaller từ source hoặc chờ phiên bản 6.10+; compile PyAudio từ source (cần Visual Studio Build Tools)

## Build Web App (Docker)
```bash
docker-compose up --build
```
