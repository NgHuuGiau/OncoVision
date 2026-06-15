# Tổng quan dự án

## 1. Mục tiêu

Dự án này được xây quanh 3 trục chính:

- Camera YOLO realtime trên desktop.
- Bộ công cụ chẩn đoán và tư vấn runtime theo phần cứng.
- Pipeline training cho model custom.

## 2. Điểm vào chính của hệ thống

```text
run_menu.py
  -> run_app.py
  -> run_chat.py
  -> run_doctor.py
  -> run_tests.py
  -> run_train.py
```

## 3. Kiến trúc runtime camera

```text
run_app.py
  -> app.runtime_entry
  -> app.chat_bootstrap
  -> core.runtime_advisor
  -> core.model_selector
  -> core.model_loader
  -> core.camera_runner
  -> utils.draw_utils
```

## 4. Vai trò từng module quan trọng

### `app/runtime_entry.py`

- Xây parser CLI.
- Điều phối luồng camera hoặc UI.
- In dashboard trước khi mở camera.

### `app/chat_bootstrap.py`

- Dò phần cứng hiện tại.
- Xây `StartOptions`.
- Chọn mode mặc định theo target và phần cứng.

### `core/runtime_advisor.py`

- Tạo runtime tối ưu cho `high`, `medium`, `low`.
- Phù hợp hơn với máy thực tế so với cấu hình tối thiểu.
- Là nguồn recommendation chính cho `run_app.py` và `run_doctor.py`.

### `core/model_selector.py`

- Đọc `config/settings.yaml`.
- Tạo `RuntimeConfig`.
- Giữ các tham số inference như `confidence`, `iou`, `imgsz`, `show_fps`, tăng sáng khung hình tối.

### `core/model_loader.py`

- Load model local từ `models/pretrained/`, file local root hoặc `models/trained/best.pt`.
- Tuân theo thứ tự ưu tiên trong `config/model_config.yaml`.

### `core/camera_runner.py`

- Mở camera và đọc frame theo luồng riêng.
- Chạy YOLO predict.
- Lọc detection, làm mượt box, gán track id, vẽ motion trail.
- Tăng sáng frame tối trước inference khi cần.
- Trả FPS và trạng thái runtime.

### `utils/draw_utils.py`

- Vẽ box và label.
- Neo FPS badge vào box nhận diện chính.
- Tự né mép khung hình nếu box nằm sát rìa.

## 5. Cách hệ thống chọn runtime

Khi chạy `run_app.py`:

1. Dò phần cứng bằng `detect_hardware()`.
2. Tạo recommendation bằng `core.runtime_advisor`.
3. Resolve `RuntimeConfig` tương ứng với `high`, `medium`, `low` hoặc `auto`.
4. Dùng runtime đó để load model và mở camera.

### Ý nghĩa của các mode

- `high`: mức chất lượng cao nhất máy còn gánh được.
- `medium`: mức cân bằng để dùng thường xuyên.
- `low`: mức an toàn nhất khi cần mượt và ổn định.

## 6. Vì sao `run_doctor.py` và `run_app.py` phải đồng bộ

Nếu `run_doctor.py` dùng logic chọn runtime cũ còn `run_app.py` dùng logic mới, người dùng sẽ thấy:

- Doctor gợi ý một model.
- Camera thực tế lại chạy model khác.

Hiện tại hai luồng này đã được đồng bộ để cùng phản ánh cấu hình tối ưu thực tế.

## 7. Tinh chỉnh nhận diện

Các tham số hiện có trong `config/settings.yaml`:

- `inference.confidence`
- `inference.iou`
- `inference.display_confidence`
- `inference.person_confidence`
- `inference.phone_confidence`
- `inference.enhance_low_light`
- `inference.low_light_mean_threshold`

Mục tiêu của việc tách các tham số này là:

- Dễ phân tích nguyên nhân nhận diện yếu.
- Dễ tăng recall hoặc giảm false positive theo từng loại object.
- Dễ document và debug hơn.

## 8. Cấu trúc thư mục chính

```text
app/
core/
docs/
models/
tests/
tools/
training/
utils/
config/
run_app.py
run_chat.py
run_doctor.py
run_menu.py
run_tests.py
run_tools.py
run_train.py
```

## 9. Training flow

```text
dataset/raw
  -> training/validate_dataset.py
  -> training/split_dataset.py
  -> run_train.py
  -> training/validate_model.py
  -> training/export_model.py
```

## 10. Model flow

### Runtime camera tổng quát

```text
models/pretrained/*.pt
  -> model_loader
  -> run_app.py
  -> camera realtime
```

### Runtime camera với model custom

```text
models/trained/best.pt
  -> run_app.py --model models/trained/best.pt
  -> camera realtime cho class riêng
```

## 11. Trạng thái hiện tại của dataset

`training/data.yaml` hiện khai báo:

```yaml
names:
  0: person
```

Điều này rất quan trọng khi phân tích lỗi nhận diện:

- Nếu bạn mong đợi class khác ngoài `person`, cần xem lại dataset và model custom.
- Nếu bạn đang dùng model pretrained, camera sẽ nhận diện object tổng quát theo COCO.

## 12. Các điểm cần nhớ khi bảo trì

- Không đổi logic recommendation ở `run_app.py` mà quên cập nhật `run_doctor.py`.
- Không đổi vị trí hiển thị FPS mà quên sửa test visualization.
- Không đổi threshold trong `config/settings.yaml` mà quên document lại README/docs.
- Nếu terminal bị lỗi dấu tiếng Việt, kiểm tra helper UTF-8 trong `utils/terminal_encoding.py`.
