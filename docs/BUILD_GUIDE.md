# Hướng dẫn Build OncoVision thành EXE

## Yêu cầu
- Python 3.10, 3.11, 3.12 hoặc 3.13 (cx_Freeze 8.x đã hỗ trợ Python 3.13)
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
- Output: `build/exe.win-amd64-3.13/` (hoặc 3.12 tùy Python version)
  - `OncoVision.exe` — file chạy chính (~1.4 MB)
  - `lib/` — dependencies Python (~4.98 GB)
    - `torch/`, `cv2/`, `ultralytics/` — AI libraries
    - `PySide6/` — GUI framework
  - `models/` — model AI (6.95 GB)
    - `pretrained/cancers/` — 71 files (47 .pt + 24 .pt2) cho 11 loại ung thư
  - `medical/`, `core/`, `utils/`, `entrypoints/` — code Python
  - `config/` — cấu hình
  - `assets/` — tài nguyên
  - `templates/` — HTML templates
  - Tổng: ~14,342 files, ~11.94 GB

## Chạy EXE
```powershell
.\build\exe.win-amd64-3.13\OncoVision.exe
```

## Test EXE
- Khởi động: mở cửa sổ desktop nhúng giao diện web (QtWebEngine) — **giống hệt web app**
- Đăng nhập bằng form web trong cửa sổ (admin / adminadminad lần đầu, hệ thống bắt đổi mật khẩu) → mở "OncoVision AI Workstation"
- Web server local chạy ở port ngẫu nhiên, tự dừng khi đóng cửa sổ
- RAM: ~458 MB, CPU: ~4s
- Database tự tạo: `output/onco.db`
- Xóa `build\exe.win-amd64-3.13\output` trước khi đóng gói phát hành

## Lưu ý quan trọng

### 1. Models đã được bundle
- Tất cả 71 model files (6.95 GB) đã được bundle vào `models/`
- Không cần tải model riêng khi mang exe đi nơi khác

### 2. Đường dẫn tự động phát hiện
- Code tự động phát hiện khi chạy từ exe (`sys.frozen`)
- `paths.py`, `cancer_model_registry.py`, `pipeline.py` đã được fix
- Output/data sẽ lưu trong thư mục build khi chạy từ exe

### 3. Thời gian build
- Build lần đầu: ~30-60 phút (copy models 6.95 GB + PyTorch)
- Build lại: ~10-20 phút (nếu build/ còn cache)

### 4. Kích thước cuối cùng
- `build/exe.win-amd64-3.13/`: ~11.94 GB total (14,342 files)
  - `OncoVision.exe`: 1.4 MB
  - `lib/`: ~4.98 GB (PyTorch, PySide6, ultralytics, v.v.)
  - `models/`: 6.95 GB (71 model files: 47 .pt + 24 .pt2)

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

### Python 3.13 hỗ trợ
- cx_Freeze 8.x đã hỗ trợ Python 3.13
- Build EXE thành công với Python 3.13.15
- PyAudio có thể cần compile từ source trên Windows nếu không có wheel 3.13
- Web app (Docker) dùng Python 3.12

## Build Web App (Docker)
```bash
docker-compose -f deploy/docker-compose.yml up --build
```

## Lỗi torch.onnx / torch._dynamo.polyfills.fx

### ImportError: No module named 'torch.onnx'
- **Nguyên nhân**: cx_Freeze exclude `torch.onnx` khỏi bundle, nhưng `torchvision` import `torch.onnx` ngay khi nạp → `medical.cnn_classifier` chết ngay từ import
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — **không exclude** `torch.onnx`

### ModuleNotFoundError: No module named 'torch._dynamo.polyfills.fx'
- **Nguyên nhân**: cx_Freeze bỏ sót `torch._dynamo.polyfills.fx` (torch import động qua `importlib.import_module`) → lỗi khi torchvision nạp
- **Giải pháp**: Đã fix trong `scripts/setup_build.py` — thêm `"torch._dynamo.polyfills.fx"` vào `includes`

### TypeError: main() got an unexpected keyword argument 'window_title'
- **Nguyên nhân**: `entrypoints/run_chat.py` import `main` (hàm chết) thay vì `launch_chat_app`; `launch_chat_app` không tạo `QApplication`/gọi `app.exec()`
- **Giải pháp**: Đã fix `app/chat_ui/window.py` — đưa `_run_desktop_login` và khởi động GUI vào đúng trong `launch_chat_app`; xóa `main()` chết

### Quy tắc torch modules (cập nhật)
- Các torch modules phải include: `torch.cuda`, `torch.distributed`, `torch.jit`, `torch.fx`, `torch.multiprocessing`, `torch.ao`, `torch.onnx`, `torch._dynamo.polyfills.fx`
- Các module vẫn có thể exclude: `torch.elastic`, `torch.testing`, `torch.utils.tensorboard`
