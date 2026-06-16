# Dự án YOLO Realtime Camera

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111827)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![PySide6](https://img.shields.io/badge/PySide6-Qt_for_Python-41CD52?logo=qt&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?logo=nvidia&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?logo=windows&logoColor=white)
![UTF-8](https://img.shields.io/badge/Terminal-UTF--8-0F766E)

Repo này tập trung vào 3 nhóm chức năng chính:

- Chạy YOLO realtime trên webcam với cấu hình tự thích nghi theo phần cứng.
- Kiểm tra, tư vấn và chẩn đoán runtime bằng terminal.
- Huấn luyện, đánh giá và xuất model custom từ dataset riêng.

## Ngăn xếp công nghệ

- Ngôn ngữ chính: Python.
- Inference: Ultralytics YOLO, PyTorch, CUDA tùy chọn.
- Xử lý ảnh và camera: OpenCV, NumPy.
- Giao diện desktop: PySide6 (Qt for Python).
- Terminal và log: UTF-8 trên Windows, hiển thị tiếng Việt đầy đủ.

## Điểm nổi bật hiện tại

- `run_app.py` mở camera realtime và chạy nhận diện thật, không còn rơi vào chế độ chỉ preview FPS.
- FPS được hiển thị thành badge rõ ràng, bám theo box nhận diện chính; khi không có box, FPS sẽ tự rơi về góc an toàn trên khung hình.
- Runtime camera hỗ trợ tinh chỉnh `confidence`, `IoU`, `imgsz`, ngưỡng hiển thị và tăng sáng khung hình tối từ `config/settings.yaml`.
- `run_doctor.py` và chế độ tư vấn trong `run_app.py --advisor-only` đã đồng bộ với logic chọn runtime thực tế của `run_app.py`.
- Terminal/log dùng UTF-8 trên Windows để hiển thị tiếng Việt đầy đủ.

## Các script chính

- `run_menu.py`: menu tổng để mở nhanh các công cụ chính.
- `run_app.py`: camera realtime YOLO, có dashboard phần cứng và nhận diện.
- `run_chat.py`: giao diện desktop/chat.
- `run_doctor.py`: kiểm tra phần cứng, camera, model, dữ liệu và gợi ý runtime.
- `run_app.py --advisor-only`: bộ tư vấn runtime, giải thích vì sao máy nên chạy mức nào mà không mở camera.
- `run_tests.py`: chạy toàn bộ test của repo.
- `run_train.py`: huấn luyện model custom.

## Luồng camera realtime

Khi chạy `run_app.py`:

1. Hệ thống dò CPU, RAM, GPU, VRAM, PyTorch và CUDA.
2. Runtime được chọn theo phần cứng bằng bộ tối ưu trong `core/runtime_advisor.py`.
3. YOLO load model local theo `candidate_models` của runtime và thứ tự ưu tiên trong `config/model_config.yaml`.
4. Camera mở bằng luồng đọc frame riêng để giảm trễ.
5. Nếu khung hình tối, pipeline có thể tăng sáng nhẹ trước khi inference.
6. Kết quả nhận diện được lọc, làm mượt box, vẽ trail chuyển động và hiển thị FPS.

### FPS hiển thị như thế nào

- Nếu có detection: badge `FPS` bám theo box có độ tin cậy cao nhất, ưu tiên góc phải phía dưới.
- Nếu box ở sát mép ảnh: badge tự đổi vị trí để không bị cắt.
- Nếu chưa có detection: badge `FPS` rơi về góc an toàn trên khung hình.
- Bật/tắt bằng `camera.show_fps` trong `config/settings.yaml`.

### Vì sao trước đó camera chỉ hiện FPS mà không nhận diện

Nguyên nhân chính ở nhánh hiện tại là `run_app.py` đã từng bị đổi sang `run_camera_preview_session(...)`, nghĩa là chỉ mở camera để đo FPS chứ không gọi YOLO detect. Luồng này đã được khôi phục về `run_camera_session(...)`.

## Tối ưu nhận diện hiện tại

Các điểm đã được rà soát và tối ưu:

- Mô hình: `run_app.py` dùng runtime tối ưu theo máy, thay vì cấu hình camera quá bảo thủ.
- Confidence: vẫn giữ ngưỡng model base ở mức vừa phải, nhưng thêm ngưỡng hiển thị riêng cho `person`, `phone` và object chung.
- IoU: đưa về cấu hình rõ ràng trong `config/settings.yaml` và truyền thẳng vào YOLO predict.
- `imgsz`: chọn theo profile `high/medium/low`, phản ánh đúng năng lực GPU/CPU.
- Tiền xử lý: thêm tăng sáng có điều kiện cho khung hình tối để cải thiện nhận diện webcam.
- FPS: tách FPS ra khỏi panel tĩnh để tránh che góc ảnh và giúp đọc frame dễ hơn.

Nếu bạn cần nhận diện class custom thay vì object tổng quát từ model pretrained, có thể chạy:

```powershell
.\.venv\Scripts\python run_app.py --model models/trained/best.pt
```

Điều này đặc biệt quan trọng nếu `best.pt` của bạn được train cho class riêng như `face`, `helmet`, `phone`, v.v.

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

- `camera.show_fps`: bật/tắt FPS.

Các khóa quan trọng cho inference:

- `inference.confidence`: ngưỡng confidence gốc truyền vào YOLO.
- `inference.iou`: IoU threshold cho NMS.
- `inference.display_confidence`: ngưỡng hiển thị object chung sau inference.
- `inference.person_confidence`: ngưỡng hiển thị `person`/`face`.
- `inference.phone_confidence`: ngưỡng hiển thị `phone`.
- `inference.enhance_low_light`: bật tăng sáng khung hình tối trước inference.
- `inference.low_light_mean_threshold`: ngưỡng sáng trung bình để quyết định có tăng sáng hay không.

## Training

Dataset đầu vào đặt ở:

- `dataset/raw/images`
- `dataset/raw/labels`

Luồng chuẩn:

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

`run_doctor.py` dùng để kiểm tra hệ thống có đủ điều kiện chạy hay không.

`run_app.py --advisor-only` dùng để giải thích vì sao máy nên chạy `high`, `medium` hay `low`.

## Tài liệu chi tiết

- `docs/install_guide.md`
- `docs/training_guide.md`
- `docs/project_overview.md`
- `docs/runtime_tool_guide.md`

## Ghi chú vận hành

- Nhấn `Esc` để thoát camera realtime.
- Nếu webcam tối, hãy bật thêm đèn hoặc xoay mặt/camera về phía có nguồn sáng, vì tăng sáng bằng phần mềm chỉ giúp một phần.
- FPS cao không tự động đồng nghĩa với nhận diện tốt hơn; nếu phải hạ `imgsz` quá thấp để tăng FPS thì độ chính xác có thể giảm.
