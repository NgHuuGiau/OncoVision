# OncoVision

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/Windows-11%2B-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![PowerShell](https://img.shields.io/badge/PowerShell-7%2B-5391FE?logo=powershell&logoColor=white)](https://learn.microsoft.com/powershell/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-8CAAE6)](https://python-pillow.org/)
[![YAML](https://img.shields.io/badge/YAML-Config-CB171E?logo=yaml&logoColor=white)](https://yaml.org/)
[![Medical](https://img.shields.io/badge/Medical-Screening%20%26%20Reports-00A6A6)](docs/medical_imaging_guide.md)
[![Training](https://img.shields.io/badge/Training-YOLO%20Pipeline-FFB000)](docs/training_guide.md)
[![Chat UI](https://img.shields.io/badge/Chat%20UI-Desktop%20Assistant-6C5CE7)](docs/runtime_tool_guide.md)

> OncoVision gom camera realtime, training YOLO và nhánh medical vào một repo duy nhất, nên README này là bản đồ nhanh nhất để vào hệ thống.

## Start Here

1. Xem [Tóm Tắt Nhanh](#tóm-tắt-nhanh) để hiểu repo làm gì trong 30 giây.
2. Mở [Bản Đồ Entrypoint](#bản-đồ-entrypoint) để biết dùng file nào cho việc nào.
3. Chọn một nhánh:
   - [docs/runtime_tool_guide.md](docs/runtime_tool_guide.md) nếu bạn làm camera realtime.
   - [docs/medical_imaging_guide.md](docs/medical_imaging_guide.md) nếu bạn làm medical.
   - [docs/training_guide.md](docs/training_guide.md) nếu bạn train model.

## Demo / Screenshots

| Màn hình | Ý nghĩa |
|---|---|
| ![Menu tổng OncoVision](images/Ảnh%20run_menu.py.png) | Cửa vào tổng hợp để chọn luồng phù hợp |
| ![Runtime advisor](images/Ảnh%20run_app.py%20--advisor-only.png) | Chọn mode trước khi mở camera thật |
| ![Doctor scan hệ thống](images/Ảnh%20run_doctor.py%20--skip-camera-check%201.png) | Kiểm tra tổng quan hệ thống |
| ![Medical status chi tiết](images/Ảnh%20run_doctor.py%20--skip-camera-check%202.png) | Xem trạng thái medical và dataset |
| ![Chat preflight](images/Ảnh%20run_chat.py%20--check-only.png) | Kiểm tra chat UI và phụ thuộc |
| ![Luồng training](images/Ảnh%20luồng%20training.png) | Minh hoạ pipeline training |

## Mục Lục Nhanh

- [Tóm Tắt Nhanh](#tóm-tắt-nhanh)
- [Ngôn Ngữ Và Thư Viện](#ngôn-ngữ-và-thư-viện)
- [Bản Đồ Entrypoint](#bản-đồ-entrypoint)
- [Khi Nào Dùng File Nào](#khi-nào-dùng-file-nào)
- [Luồng Nghiệp Vụ](#luồng-nghiệp-vụ)
- [Cài Đặt Nhanh](#cài-đặt-nhanh)
- [Tài Liệu Trong docs](#tài-liệu-trong-docs)
- [Dành Cho Người Mới](#dành-cho-người-mới)

## Tóm Tắt Nhanh

| Nhóm | Giá trị |
|---|---|
| Realtime camera | Chạy camera desktop với các mode `auto`, `high`, `medium`, `low` |
| Medical workflow | Quản lý skin lesion dataset, report, status model và output medical |
| Entrypoint rõ ràng | Mỗi tác vụ lớn có `run_*.py` riêng |
| Kiểm tra sớm | Có `run_doctor.py`, `run_smoke.py`, `run_tests.py` để rà lỗi nhanh |
| Tài liệu vận hành | Có bộ `docs/` cho cài đặt, training, runtime, medical và kiến trúc |

> Mục tiêu của README là cho bạn nắm được hệ thống trong vài phút, rồi đi đúng file thay vì phải lần mò cả repo.

## Ngôn Ngữ Và Thư Viện

| Ký hiệu | Thành phần |
|---|---|
| 🐍 | Python |
| 🧠 | PyTorch / Ultralytics |
| 📷 | OpenCV |
| 🪟 | PySide6 |
| 🗃️ | SQLite |
| 📄 | YAML / JSON |
| 🧰 | `utils/` helper nội bộ |

> Nếu chỉ nhìn một chỗ, hãy nhìn bảng này cùng badge phía trên để hiểu repo đang dựa vào những gì.

## Bản Đồ Entrypoint

![Menu tổng OncoVision](images/Ảnh%20run_menu.py.png)

```text
run_menu.py      -> menu tổng hợp
run_app.py       -> runtime advisor / camera realtime
run_chat.py      -> chat UI / medical preflight / cleanup output
run_doctor.py    -> doctor scan toàn hệ thống
run_train.py     -> train YOLO object detection
run_medical.py   -> CLI quản lý luồng medical
run_smoke.py     -> smoke check entrypoint
run_tests.py     -> dashboard unit test
```

## Khi Nào Dùng File Nào

| Bạn muốn làm gì | Entrypoint nên dùng |
|---|---|
| Xem toàn bộ chức năng | `python run_menu.py` |
| Kiểm tra máy nên chạy runtime nào | `python run_app.py --advisor-only` |
| Mở camera realtime | `python run_app.py` |
| Kiểm tra chat UI sẵn sàng chưa | `python run_chat.py --check-only` |
| Kiểm tra tổng thể môi trường | `python run_doctor.py --skip-camera-check` |
| Train YOLO object detection | `python run_train.py` |
| Kiểm tra nhanh luồng y dược | `python run_medical.py status` |
| Chạy smoke check an toàn | `python run_smoke.py` |
| Chạy unit test | `python -m unittest discover -s tests -p "test_*.py"` |

## Luồng Nghiệp Vụ

### 1. Nhánh Vật Thể

Mục tiêu:

- quản lý dataset object detection,
- train model YOLO custom,
- validate model,
- đưa model vào camera realtime.

Thư mục liên quan:

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

![Runtime advisor](images/Ảnh%20run_app.py%20--advisor-only.png)

### 2. Nhánh Y Dược

Mục tiêu:

- tổ chức dataset skin lesion,
- theo dõi model medical,
- chạy các lệnh khởi tạo, status, ready, sources, verify,
- phục vụ chat UI và pipeline phân tích medical.

Thư mục liên quan:

```text
dataset/medical/skin_lesion/
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

![Doctor scan hệ thống](images/Ảnh%20run_doctor.py%20--skip-camera-check%201.png)

![Medical status chi tiết](images/Ảnh%20run_doctor.py%20--skip-camera-check%202.png)

![Chat preflight](images/Ảnh%20run_chat.py%20--check-only.png)

![Luồng training](images/Ảnh%20luồng%20training.png)

## Cài Đặt Nhanh

```powershell
git clone <repo-url>
cd OncoVision
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_menu.py
```

## Tài Liệu Trong docs

- [docs/project_overview.md](docs/project_overview.md): bản đồ kiến trúc tổng thể
- [docs/runtime_tool_guide.md](docs/runtime_tool_guide.md): giải thích runtime advisor
- [docs/medical_imaging_guide.md](docs/medical_imaging_guide.md): luồng y dược
- [docs/training_guide.md](docs/training_guide.md): luồng training object detection

## Dành Cho Người Mới

1. Mở [docs/project_overview.md](docs/project_overview.md) để hiểu cấu trúc tổng thể.
2. Mở [docs/runtime_tool_guide.md](docs/runtime_tool_guide.md) nếu bạn muốn chạy camera realtime.
3. Mở [docs/medical_imaging_guide.md](docs/medical_imaging_guide.md) nếu bạn làm nhánh medical.
4. Mở [docs/training_guide.md](docs/training_guide.md) nếu bạn chuẩn bị train model.
