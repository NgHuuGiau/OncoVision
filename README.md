# OncoVision

OncoVision là bộ công cụ AI nội bộ gồm hai nhánh chính:

- `Y dược`: quản lý dataset y khoa, phân tích ảnh, theo dõi trạng thái model và hỗ trợ quy trình TCIA.
- `Vật thể`: train, đánh giá và chạy YOLO realtime cho bài toán nhận diện vật thể bằng camera.

README này được viết theo hướng giúp người mới vào dự án có thể:

1. hiểu nhanh dự án đang làm gì,
2. biết cần chạy lệnh nào trước,
3. tìm đúng tài liệu chi tiết trong thư mục `docs/`.

## Điểm Nổi Bật

| Nhóm | Giá trị |
|---|---|
| Realtime camera | Chạy camera desktop với chế độ runtime `auto`, `high`, `medium`, `low` |
| Medical workflow | Quản lý skin lesion dataset, TCIA collections, output reports và trạng thái model |
| Entrypoint rõ ràng | Mỗi tác vụ lớn đều có file `run_*.py` riêng |
| Kiểm tra nhanh | Có `run_doctor.py`, `run_smoke.py`, `run_tests.py` để soát lỗi sớm |
| Tài liệu vận hành | Có bộ `docs/` cho cài đặt, training, runtime, medical và tổng quan kiến trúc |

## Bản Đồ Nhanh Của Dự Án

```text
run_menu.py      -> menu tổng hợp
run_app.py       -> camera realtime / runtime advisor
run_chat.py      -> chat UI / medical preflight / cleanup output
run_doctor.py    -> doctor scan tổng thể hệ thống
run_train.py     -> train YOLO object detection
run_medical.py   -> CLI quản lý luồng medical
run_smoke.py     -> smoke check entrypoint
run_tests.py     -> dashboard unit test
```

## Khi Nào Nên Dùng File Nào

| Bạn muốn làm gì | Entrypoint nên dùng |
|---|---|
| Xem toàn bộ chức năng đang có | `python run_menu.py` |
| Kiểm tra máy nên chạy runtime nào | `python run_app.py --advisor-only` |
| Mở camera realtime | `python run_app.py` |
| Kiểm tra chat UI có sẵn sàng không | `python run_chat.py --check-only` |
| Kiểm tra tổng thể môi trường | `python run_doctor.py --skip-camera-check` |
| Train YOLO object detection | `python run_train.py` |
| Kiểm tra nhanh luồng y dược | `python run_medical.py status` |
| Chạy smoke check an toàn | `python run_smoke.py` |
| Chạy unit test | `python -m unittest discover -s tests -p "test_*.py"` |

## Cấu Trúc Luồng Nghiệp Vụ

### 1. Nhánh Vật Thể

Mục tiêu:

- quản lý dataset object detection,
- train model YOLO custom,
- validate model,
- đưa model vào camera realtime.

Thư mục dữ liệu liên quan:

```text
dataset/object_detection/raw/
dataset/object_detection/processed/
models/pretrained/
models/trained/
```

Luồng cơ bản:

```powershell
python run_train.py --check-only
python training\prepare_dataset.py
python training\validate_dataset.py
python training\split_dataset.py
python run_train.py
python training\validate_model.py
python run_app.py --model models/trained/best.pt
```

### 2. Nhánh Y Dược

Mục tiêu:

- tổ chức dataset skin lesion và TCIA,
- theo dõi độ sẵn sàng của model medical,
- chạy các lệnh khởi tạo, status, ready, sources, verify,
- phục vụ chat UI và pipeline phân tích medical.

Thư mục dữ liệu liên quan:

```text
dataset/medical/skin_lesion/
dataset/medical/tcia/
output/medical/
```

Luồng cơ bản:

```powershell
python run_medical.py init-dataset
python run_medical.py status
python run_medical.py sources
python run_medical.py ready
python run_chat.py --check-only
```

## Cài Đặt Nhanh

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_menu.py
python run_doctor.py --skip-camera-check
python run_smoke.py
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Bộ Tài Liệu Trong `docs/`

| File | Nội dung |
|---|---|
| [docs/install_guide.md](/abs/path/D:/OncoVision/docs/install_guide.md:1) | Hướng dẫn cài đặt, khởi tạo môi trường và checklist sau cài đặt |
| [docs/project_overview.md](/abs/path/D:/OncoVision/docs/project_overview.md:1) | Tổng quan kiến trúc, cây thư mục và vai trò từng module |
| [docs/quick_commands.md](/abs/path/D:/OncoVision/docs/quick_commands.md:1) | Lệnh nhanh cho vận hành hằng ngày |
| [docs/runtime_tool_guide.md](/abs/path/D:/OncoVision/docs/runtime_tool_guide.md:1) | Giải thích `run_app.py --advisor-only` và runtime modes |
| [docs/training_guide.md](/abs/path/D:/OncoVision/docs/training_guide.md:1) | Hướng dẫn chi tiết cho object detection training |
| [docs/medical_imaging_guide.md](/abs/path/D:/OncoVision/docs/medical_imaging_guide.md:1) | Hướng dẫn chi tiết cho luồng y dược |

## Checklist Cho Người Mới Vào Dự Án

1. Đọc [project_overview.md](/abs/path/D:/OncoVision/docs/project_overview.md:1) để nắm bố cục repo.
2. Chạy `python run_doctor.py --skip-camera-check` để xem máy còn thiếu gì.
3. Chạy `python run_smoke.py` để test nhanh các entrypoint quan trọng.
4. Nếu làm object detection, đọc [training_guide.md](/abs/path/D:/OncoVision/docs/training_guide.md:1).
5. Nếu làm y dược, đọc [medical_imaging_guide.md](/abs/path/D:/OncoVision/docs/medical_imaging_guide.md:1).

## Ghi Chú Vận Hành

- `run_smoke.py --ci-safe` được thiết kế cho CI: chỉ chạy những check nhẹ, ít phụ thuộc camera và dataset cục bộ.
- `run_chat.py --check-only` dùng để preflight giao diện chat mà không mở GUI.
- `run_doctor.py --skip-camera-check` là lệnh an toàn nhất để rà soát tổng quát trên máy mới.
- Logger hiện đã có fallback nếu file log bị khóa, nhưng vẫn nên giữ thư mục `output/` có quyền ghi.

## Mục Tiêu Của Repo

OncoVision không chỉ là một app đơn lẻ. Repo này đóng vai trò:

- bộ công cụ vận hành,
- bộ thư viện helper,
- nơi lưu trữ quy trình huấn luyện,
- và là tài liệu kỹ thuật cho cả nhóm.

Vì vậy, cách sử dụng tốt nhất là đi từ entrypoint `run_*.py`, sau đó mới đi sâu vào `medical/`, `training/`, `core/` và `utils/`.
