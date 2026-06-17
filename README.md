# Dự án YOLO Realtime Camera

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111827)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![PySide6](https://img.shields.io/badge/PySide6-Qt_for_Python-41CD52?logo=qt&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Tùy chọn-76B900?logo=nvidia&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?logo=windows&logoColor=white)
![UTF-8](https://img.shields.io/badge/Terminal-UTF--8-0F766E)

Repo này tập trung vào 3 nhóm chức năng chính:

- Chạy YOLO realtime trên webcam với cấu hình thích nghi theo phần cứng.
- Kiểm tra, tư vấn và chẩn đoán runtime bằng terminal.
- Huấn luyện, đánh giá và xuất model custom từ dataset riêng.

## Ngăn xếp công nghệ

- Ngôn ngữ chính: Python.
- Inference: Ultralytics YOLO, PyTorch, CUDA tùy chọn.
- Xử lý ảnh và camera: OpenCV, NumPy.
- Giao diện desktop: PySide6.
- Terminal và log: UTF-8 trên Windows để hiển thị tiếng Việt đầy đủ.

## Thành phần chính

- `run_menu.py`: menu tổng để mở nhanh các công cụ chính.
- `run_app.py`: camera realtime YOLO, có dashboard phần cứng và nhận diện.
- `run_chat.py`: giao diện desktop/chat.
- `run_doctor.py`: kiểm tra phần cứng, camera, model, dữ liệu và gợi ý runtime.
- `run_app.py --advisor-only`: bộ tư vấn runtime, giải thích vì sao máy nên chạy mức nào mà không mở camera.
- `run_tests.py`: chạy toàn bộ test của repo.
- `run_train.py`: huấn luyện model custom.

## Luồng camera realtime

Khi chạy `run_app.py`:

1. Hệ thống đọc tham số CLI.
2. `app.camera_runtime.bootstrap.resolve_start_bundle()` dò phần cứng và chọn runtime phù hợp.
3. Dashboard khởi động được in ra terminal.
4. `core.camera_runner.run_camera_session()` mở camera và bắt đầu inference.
5. Kết quả nhận diện được lọc, làm mượt box, vẽ trail và hiển thị FPS.

Lưu ý: nhánh hiện tại đã dùng trực tiếp `run_camera_session(...)`. Luồng preview-only cũ không còn là đường chạy chính của ứng dụng.

## Tối ưu nhận diện hiện tại

Các điểm đã được tổ chức lại và tối ưu:

- Runtime được chọn theo phần cứng, thay vì dùng một cấu hình cố định cho mọi máy.
- `confidence`, `IoU`, `imgsz` và các ngưỡng hiển thị được cấu hình trong `config/settings.yaml`.
- Có thể tăng sáng có điều kiện cho khung hình tối trước khi inference.
- FPS được hiển thị thành badge riêng, tránh che khuất nội dung quan trọng trên khung hình.

Nếu bạn cần nhận diện class custom thay vì object tổng quát từ model pretrained, có thể chạy:

```powershell
.\.venv\Scripts\python run_app.py --model models/trained/best.pt
```

Điều này đặc biệt quan trọng khi `best.pt` được train cho class riêng như `face`, `helmet`, `phone`, v.v.

## Cài đặt nhanh

```powershell
cd D:\YOLO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python training\prepare_dataset.py
.\.venv\Scripts\python training\download_models.py
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Cách chạy

Menu tổng:

```powershell
.\.venv\Scripts\python run_menu.py
```

Chạy trực tiếp từng công cụ:

```powershell
.\.venv\Scripts\python run_app.py
.\.venv\Scripts\python run_chat.py
.\.venv\Scripts\python run_doctor.py
.\.venv\Scripts\python run_app.py --advisor-only
.\.venv\Scripts\python run_tests.py
.\.venv\Scripts\python run_train.py
```

Ví dụ chọn mode và camera:

```powershell
.\.venv\Scripts\python run_app.py --mode medium --camera-index 0
```

Ví dụ ép dùng model custom:

```powershell
.\.venv\Scripts\python run_app.py --model models/trained/best.pt
```

## Tinh chỉnh trong `config/settings.yaml`

Các khóa quan trọng cho camera:

- `camera.show_fps`

Các khóa quan trọng cho inference:

- `inference.confidence`
- `inference.iou`
- `inference.display_confidence`
- `inference.person_confidence`
- `inference.phone_confidence`
- `inference.enhance_low_light`
- `inference.low_light_mean_threshold`

## Training

Dataset đầu vào đặt ở:

- `dataset/raw/images`
- `dataset/raw/labels`

Luồng khuyến nghị:

```powershell
.\.venv\Scripts\python training\prepare_dataset.py
.\.venv\Scripts\python training\validate_dataset.py
.\.venv\Scripts\python training\split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python training\export_model.py
```

Hiện tại `training/data.yaml` đang khai báo tối thiểu:

```yaml
names:
  0: person
```

Nếu bạn muốn nhận diện class khác trong camera bằng model custom, cần cập nhật dataset và `training/data.yaml` đồng bộ trước khi train lại.

## Chẩn đoán nhanh

```powershell
.\.venv\Scripts\python run_doctor.py
.\.venv\Scripts\python run_app.py --advisor-only
```

- `run_doctor.py` dùng để kiểm tra hệ thống có đủ điều kiện chạy hay không.
- `run_app.py --advisor-only` dùng để giải thích vì sao máy nên chạy `high`, `medium` hay `low`.

## Tài liệu chi tiết

- [docs/install_guide.md](docs/install_guide.md)
- [docs/training_guide.md](docs/training_guide.md)
- [docs/project_overview.md](docs/project_overview.md)
- [docs/runtime_tool_guide.md](docs/runtime_tool_guide.md)

## Ghi chú vận hành

- Nhấn `Esc` để thoát camera realtime.
- Nếu webcam tối, hãy tăng ánh sáng thực tế trước khi chỉ trông chờ vào tăng sáng bằng phần mềm.
- FPS cao không đồng nghĩa với nhận diện tốt hơn; nếu phải hạ `imgsz` quá thấp để tăng FPS thì độ chính xác có thể giảm.
