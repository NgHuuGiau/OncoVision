# Đánh Giá Toàn Diện: Thiếu Sót & Cải Tiến OncoVision

## Ngày: 2026-11-13

## Tóm Tắt

Sau khi phân tích kỹ lưỡng toàn bộ codebase (216 file Python, 55 tests, 6 module chính), dự án OncoVision có **kiến trúc tốt, chuyên sâu về y khoa**, nhưng còn nhiều thiếu sót quan trọng cần xử lý. Dưới đây là danh sách đầy đủ các vấn đề theo mức độ ưu tiên.

---

## 1. ❌ Volume 3D — Hoàn toàn không hoạt động (Ưu tiên: CAO)

### Vấn đề
- **Tất cả file `.pt2` không thể load** do không tương thích phiên bản PyTorch
- File được export bằng **PyTorch 2.14.1+cpu** (phiên bản mới hơn)
- Máy hiện tại dùng **PyTorch 2.7.1+cu118**
- `torch.export.load` yêu cầu schema version khớp: `(8, 7)` vs `6`
- Cấu trúc file `.pt2` là **TorchScript archive** (có `models/model.json`), không phải `torch.export` format (thiếu `serialized_exported_program.json`)
- `load_volume_model()` trong `volume_inference.py` chỉ dùng `torch.export.load` — không có fallback

### Ảnh hưởng
- **0/11 nhóm ung thư** có thể chạy suy luận 3D
- Pipeline y khoa không thể phân tích volume NIfTI/DICOM series
- `medical_settings.yaml` trỏ đến `brain_volume_model.pt2` — không hoạt động

### Giải pháp
1. **Ngắn hạn**: Thêm fallback trong `load_volume_model()` để xử lý TorchScript format bằng `torch.jit.load()`
2. **Trung hạn**: Re-export tất cả model 3D bằng PyTorch 2.7.1 hoặc phiên bản tương thích
3. **Dài hạn**: Tạo script tự động kiểm tra và convert format model 3D

---

## 2. ⚠️ Thyroid Detector — Dùng COCO Generic (Ưu tiên: CAO)

### Vấn đề
- File `models/pretrained/cancers/thyroid/ultrasound/detector.pt` là **YOLO COCO tổng quát** (80 lớp: person, bicycle, car, dog...)
- **Không phải** detector chuyên biệt cho thyroid nodule
- Khi chạy inference cho tuyến giáp, detector sẽ phát hiện đối tượng không liên quan

### Ảnh hưởng
- Kết quả detection cho tuyến giáp **hoàn toàn vô nghĩa**
- Người dùng có thể bị误导 bởi kết quả sai

### Giải pháp
1. **Ngắn hạn**: Đánh dấu thyroid detector là không khả dụng trong `cancer_model_registry.py`
2. **Trung hạn**: Huấn luyện lại từ dataset TN5000 (5,000 ảnh, có sẵn trên HuggingFace)
3. **Dài hạn**: Thêm validation tự động để phát hiện detector không khớp với cancer target

---

## 3. ⚠️ Breast Detector — Chỉ 1 Lớp Chung (Ưu tiên: CAO)

### Vấn đề
- File `models/pretrained/cancers/breast/mammogram/detector.pt` chỉ có 1 lớp `item`
- Không phân biệt được loại tổn thương vú (mass, calcification, architectural distortion...)

### Ảnh hưởng
- Detection cho vú **không hữu ích** trong thực tế lâm sàng
- Không thể phân loại loại tổn thương

### Giải pháp
1. Kiểm tra lại chất lượng detector này
2. Cân nhắc huấn luyện lại từ dataset BUSI hoặc DDIM với labels chi tiết hơn

---

## 4. ⚠️ Cancer Catalog Thiếu Spleen (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- Thư mục `models/pretrained/cancers/spleen/` tồn tại trong disk
- Nhưng `medical/cancer_catalog.py` chỉ có 11 nhóm: liver, lung, breast, stomach, colorectal, prostate, cervical, kidney, pancreas, thyroid, brain
- **Không có spleen** trong `COMMON_CANCER_TARGETS`

### Ảnh hưởng
- Model spleen (nếu có) không thể được sử dụng bởi pipeline
- `cancer_model_registry.py` dựa vào `COMMON_CANCER_TARGETS` — bỏ qua spleen

### Giải pháp
1. Kiểm tra xem spleen có model classifier/detector hợp lệ không
2. Nếu có, thêm vào `COMMON_CANCER_TARGETS`
3. Nếu không, xóa thư mục `models/pretrained/cancers/spleen/`

---

## 5. ⚠️ Model Catalog Rỗng (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- File `models/model_catalog.json` có `entries: []` — hoàn toàn rỗng
- File này được thiết kế để catalog tất cả model đã kiểm định
- `scripts/fetch_model.py` có thể điền vào, nhưng hiện tại không có entry nào

### Ảnh hưởng
- Không có cách tự động để biết model nào đã được kiểm định
- Người dùng không thể tự động tải model từ catalog

### Giải pháp
1. Điền `model_catalog.json` với thông tin của tất cả 11 classifier + 10 detector đã kiểm tra
2. Thêm SHA256 checksums cho mỗi model
3. Cập nhật `fetch_model.py` để đọc từ catalog

---

## 6. ⚠️ Clinical Validation Chưa Có Báo Cáo Thực (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- Framework `clinical_validation.py` hoàn chỉnh và có test
- Nhưng `medical_settings.yaml` có `clinical_validation_report:` — **rỗng**
- Không có báo cáo kiểm định lâm sàng độc lập nào được tạo
- `clinical_validation_status()` sẽ luôn trả về `False`

### Ảnh hưởng
- Kết quả chỉ dùng hỗ trợ sàng lọc, không có validation lâm sàng
- Không thể khẳng định độ chính xác theo bệnh nhân

### Giải pháp
1. Tạo cohort kiểm định độc lập cho ít nhất brain classifier
2. Chạy `build_independent_validation_report()` để tạo báo cáo
3. Cập nhật `medical_settings.yaml` với đường dẫn báo cáo

---

## 7. ⚠️ File Lớn Cần Refactor (Ưu tiên: TRUNG BÌNH)

### Vấn đề

| File | Dòng | Vấn đề |
|---|---|---|
| `medical/pipeline.py` | 2242 | Tất cả logic inference, validation, preprocessing, reporting trong 1 file |
| `web_app.py` | 1277 | Tất cả route, auth, upload, analysis trong 1 file |
| `app/chat_ui/window.py` | 1302 | Toàn bộ GUI chat, auth, medical, voice trong 1 file |

### Ảnh hưởng
- Khó bảo trì, khó test, khó review
- Vi phạm Single Responsibility Principle
- Khó tách biệt concern cho team phát triển

### Giải pháp cho pipeline.py
Tách thành các module riêng:
- `pipeline/validator.py` — Validation logic
- `pipeline/preprocessor.py` — Preprocessing dispatch
- `pipeline/inference.py` — Core inference (classifier + detector + volume)
- `pipeline/postprocessor.py` — Post-processing và risk assessment
- `pipeline/reporter.py` — Report generation
- `pipeline/orchestrator.py` — Main orchestration logic

### Giải pháp cho web_app.py
Tách thành:
- `web/routes/` — API routes
- `web/auth/` — Authentication logic
- `web/upload/` — Upload handling
- `web/analysis/` — Analysis endpoint
- `web/middleware/` — Middleware

### Giải pháp cho window.py
Tách thành:
- `chat_ui/views/` — UI components
- `chat_ui/controllers/` — Business logic
- `chat_ui/services/` — Service layer

---

## 8. ⚠️ Scripts Rời Rạc (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- **40 scripts** trong `scripts/` — nhiều script debug one-off
- Các nhóm:
  - `check_*.py` (6 files): Kiểm tra model, modalities, password, users, status
  - `download_*.py` (7 files): Tải model — nhiều phiên bản khác nhau
  - `train_*.py` (3 files): Training modality classifier
  - `generate_*.py` (3 files): Generate volume models
  - `probe_*.py` (2 files): Probe liver model
  - `deploy/` (7 files): Search và download từ HuggingFace
  - `maintenance/` (2 files): Convert và test model mới

### Ảnh hưởng
- Khó bảo trì, khó hiểu mục đích
- Nhiều script có thể đã lỗi thời
- Không có documentation rõ ràng

### Giải pháp
1. Đã có `scripts/README.md` phân loại — tốt
2. Xóa các script debug one-off không còn cần thiết
3. Consolidate các script download thành 1 script chính
4. Di chuyển script production vào `scripts/production/`
5. Di chuyển script dev vào `scripts/dev/` (đã có trong ruff exclude)

---

## 9. ⚠️ Test Coverage Chưa Đầy Đủ (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- **55 tests** (50 integration + 5 unit) — khá tốt nhưng còn thiếu
- Thiếu test cho:
  - `medical/explainability.py` — Grad-CAM (466 dòng, không có test)
  - `medical/uncertainty/quantification.py` — MC Dropout, Deep Ensemble (150 dòng, không có test)
  - `medical/segmentation/models.py` — UNet, AttentionUNet (228 dòng, không có test)
  - `medical/preprocessing/` — 7 preprocessor riêng (không có test)
  - `medical/volume_inference.py` — Chỉ có 1 unit test, không test real inference
  - `app/chat_ui/voice_worker.py` — Voice recognition (không có test)
  - `core/tracking/` — 4 files tracking (không có test)
  - `web_app.py` — Chỉ có test web_auth, không test upload/analysis endpoint

### Ảnh hưởng
- Không thể đảm bảo chất lượng code không có test
- Refactor file lớn sẽ rất rủi ro nếu không có test

### Giải pháp
1. Thêm test cho Grad-CAM trong `explainability.py`
2. Thêm test cho uncertainty quantification
3. Thêm test cho segmentation models
4. Thêm test cho từng preprocessor
5. Thêm test integration cho volume inference với dummy data
6. Thêm test cho voice worker
7. Thêm test cho tracking module
8. Thêm test cho web upload và analysis endpoint

---

## 10. ⚠️ CI/CD Chưa Có Coverage Report (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- CI/CD có sẵn: `.github/workflows/test.yml` + `codeql.yml` — tốt
- Nhưng không có:
  - **Coverage report upload** — không có `codecov` hay `coveralls`
  - **Quality gate** — không có threshold coverage tối thiểu
  - **Build artifact** — không build Docker image trong CI
  - **Security scan** — chỉ có CodeQL, không có dependency vulnerability scan

### Ảnh hưởng
- Không thể theo dõi coverage theo thời gian
- Không có quality gate để ngăn code không có test
- Không thể đảm bảo Docker image build được

### Giải pháp
1. Thêm `codecov` action để upload coverage report
2. Thêm quality gate: `--cov-fail-under=70`
3. Thêm step build Docker image trong CI
4. Thêm `dependabot` hoặc `snyk` để scan dependency vulnerability

---

## 11. ⚠️ Docker Image Cũ (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- `Dockerfile` dùng `nvidia/cuda:12.1.1-runtime-ubuntu22.04` — **cũ**
- CUDA 12.1 đã ra mắt năm 2023, hiện tại là CUDA 12.6+
- Python 3.10 — ổn nhưng có thể nâng cấp lên 3.12
- `Dockerfile.monai` dùng `nvidia/cuda:12.1-devel-ubuntu22.04` — **rất lớn** (~10GB+)

### Ảnh hưởng
- Không tận dụng được cải tiến hiệu năng CUDA mới
- Image lớn, build chậm
- Có thể có lỗ hổng bảo mật trong package cũ

### Giải pháp
1. Nâng cấp lên `nvidia/cuda:12.6.0-runtime-ubuntu22.04`
2. Cân nhắc dùng `python:3.12-slim` làm base image cho web-only
3. Dùng multi-stage build để giảm kích thước image
4. Thêm `.dockerignore` để loại bỏ file không cần thiết

---

## 12. ⚠️ Không Có `.dockerignore` (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- Không có file `.dockerignore` trong project
- `COPY . .` trong Dockerfile sẽ copy **tất cả** file, bao gồm:
  - `__pycache__/`, `.git/`, `.venv/`, `output/`, `*.pyc`
  - Model file lớn (nếu có)
  - Log file, cache file

### Ảnh hưởng
- Docker image lớn hơn cần thiết
- Build chậm hơn
- Có thể leak sensitive data vào image

### Giải pháp
Tạo `.dockerignore`:
```
__pycache__/
*.py[cod]
.git/
.venv/
venv/
output/
runs/
*.log
*.tmp
.env
models/pretrained/**/*.pt
models/pretrained/**/*.pt2
models/pretrained/**/*.safetensors
.ua/
.understand-anything/
```

---

## 13. ⚠️ Security — Default Admin (Ưu tiên: CAO)

### Vấn đề
- `web_app.py` có `auth_db.ensure_default_admin()` — tạo admin/admin mặc định
- `lifespan()` function tự động tạo admin nếu chưa có user
- Mặc dù có force change password ở `require_authenticated()`, nhưng vẫn là rủi ro

### Ảnh hưởng
- Nếu attacker truy cập trước khi admin đổi password → full access
- Vi phạm best practice security

### Giải pháp
1. Không tạo default admin — yêu cầu setup ban đầu
2. Hoặc dùng environment variable để disable trong production
3. Thêm rate limiting cho login endpoint

---

## 14. ⚠️ Security — CSRF Token (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- CSRF token có sẵn — tốt
- Nhưng token được lưu trong session cookie
- Không có `SameSite` attribute được set explicitly

### Ảnh hưởng
- Có thể bị CSRF attack nếu cookie không có SameSite

### Giải pháp
1. Set `SameSite=Lax` cho session cookie
2. Set `Secure=True` trong production
3. Thêm `HttpOnly=True` cho session cookie

---

## 15. ⚠️ Performance — Không Có Caching (Ưu tiên: THẤP)

### Vấn đề
- Không có caching cho model inference
- Mỗi request đều load model từ disk (nếu không có global cache)
- `web_app.py` có `_medical_status_cache` nhưng chỉ cho status, không cho model

### Ảnh hưởng
- Chậm khi có nhiều request đồng thời
- Tốn tài nguyên GPU/CPU

### Giải pháp
1. Thêm model caching — load model 1 lần, reuse
2. Thêm response caching cho kết quả inference không đổi
3. Consider dùng Redis cho cache trong production

---

## 16. ⚠️ Performance — Không Có Async Inference (Ưu tiên: THẤP)

### Vấn đề
- `web_app.py` dùng `threading.BoundedSemaphore` cho concurrent analysis
- Nhưng inference vẫn chạy blocking trong thread
- Không tận dụng được async của FastAPI

### Ảnh hưởng
- Không thể scale tốt với nhiều user đồng thời
- Blocking thread pool

### Giải pháp
1. Chạy inference trong process pool hoặc task queue (Celery/RQ)
2. Dùng WebSocket cho real-time progress update
3. Consider dùng FastAPI's `BackgroundTasks` cho inference

---

## 17. ⚠️ Documentation — Thiếu API Docs (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- README.md tốt — nhưng chỉ hướng dẫn cài đặt
- Không có API documentation cho web endpoints
- Không có Swagger/OpenAPI docs (mặc dù FastAPI có sẵn)
- Thiếu architecture documentation
- Thiếu contributing guide

### Ảnh hưởng
- Khó onboard developer mới
- Khó hiểu API endpoints
- Khó maintain khi team lớn lên

### Giải pháp
1. Bật Swagger UI của FastAPI (`/docs`)
2. Tạo `docs/api.md` — document tất cả endpoints
3. Tạo `docs/architecture.md` — mô tả kiến trúc 6 module
4. Tạo `CONTRIBUTING.md` — hướng dẫn đóng góp
5. Tạo `docs/development.md` — hướng dẫn phát triển

---

## 18. ⚠️ Monitoring — Không Có Logging Centralized (Ưu tiên: THẤP)

### Vấn đề
- Có `utils/logger.py` — tốt
- Nhưng không có centralized logging
- Không có structured logging (JSON format)
- Không có log aggregation (ELK, Grafana...)

### Ảnh hưởng
- Khó debug trong production
- Khó monitor health của hệ thống
- Không có alert khi có lỗi

### Giải pháp
1. Thêm structured logging (JSON format)
2. Thêm log levels rõ ràng cho từng module
3. Consider dùng Sentry cho error tracking
4. Thêm health check endpoint chi tiết hơn

---

## 19. ⚠️ Database — Không Có Migration (Ưu tiên: THẤP)

### Vấn đề
- SQLite dùng trực tiếp — không có migration framework
- Schema thay đổi phải manual
- Không có versioning cho database

### Ảnh hưởng
- Khó upgrade database khi có thay đổi schema
- Rủi ro data loss khi migration manual

### Giải pháp
1. Thêm `alembic` cho SQLite migration
2. Tạo migration cho các bảng hiện có
3. Document quy trình migration

---

## 20. ⚠️ Internationalization — Chỉ Tiếng Việt (Ưu tiên: THẤP)

### Vấn đề
- Toàn bộ UI, message, comment — chỉ tiếng Việt
- Không có i18n framework
- Khó mở rộng cho user quốc tế

### Ảnh hưởng
- Giới hạn user base
- Khó cộng đồng đóng góp

### Giải pháp
1. Thêm i18n framework (babel/flask-babel)
2. Tạo translation files cho English
3. Giữ tiếng Việt làm default, thêm English làm alternative

---

## 21. ⚠️ Preprocessing — Thiếu Colonoscopy (Ưu tiên: THẤP)

### Vấn đề
- `medical/preprocessing/pipeline.py` có 7 preprocessor: ct, mri, xray, mammogram, pet_ct, endoscopy, ultrasound
- Nhưng **colorectal** dùng `colonoscopy` — không có preprocessor riêng
- Sẽ fallback về `_default_preprocessor` — không tối ưu

### Ảnh hưởng
- Kết quả cho colorectal có thể không tối ưu
- Không có contrast boost hay normalization đặc biệt cho colonoscopy

### Giải pháp
1. Tạo `medical/preprocessing/colonoscopy.py`
2. Thêm vào `_MODALITY_PREPROCESSORS`
3. Tune contrast và normalization cho colonoscopy

---

## 22. ⚠️ Preprocessing — Thiếu Colposcopy (Ưu tiên: THẤP)

### Vấn đề
- Tương tự colonoscopy — `cervical` dùng `colposcopy` — không có preprocessor riêng
- Fallback về `_default_preprocessor`

### Giải pháp
1. Tạo `medical/preprocessing/colposcopy.py`
2. Thêm vào `_MODALITY_PREPROCESSORS`

---

## 23. ⚠️ Chat UI — Không Có Conversation Export (Ưu tiên: THẤP)

### Vấn đề
- Chat history lưu trong SQLite — tốt
- Nhưng không có cách export conversation
- Không có cách backup chat history

### Ảnh hưởng
- Khó audit trail
- Khó backup dữ liệu chat

### Giải pháp
1. Thêm endpoint export conversation (JSON/PDF)
2. Thêm CLI command để backup chat history
3. Consider dùng `run_medical.py backup-db` cho cả chat

---

## 24. ⚠️ Voice — Chỉ Windows (Ưu tiên: THẤP)

### Vấn đề
- `pyaudio` chỉ trên Windows — tốt
- `sounddevice` + `soundfile` trên Linux/macOS — tốt
- Nhưng whisper model phải có sẵn — không có auto-download

### Ảnh hưởng
- Voice không hoạt động nếu không có model whisper
- Không có fallback khi voice fail

### Giải pháp
1. Thêm auto-download whisper model nếu không có
2. Thêm message rõ ràng khi voice không khả dụng
3. Consider dùng cloud STT làm fallback

---

## 25. ⚠️ Assets — Chỉ 1 Favicon (Ưu tiên: THẤP)

### Vấn đề
- `static/` chỉ có `favicon.svg`
- Không có CSS custom — dùng inline style
- Không có JavaScript custom — dùng vanilla JS trong template

### Ảnh hưởng
- Khó customize UI
- Khó maintain style

### Giải pháp
1. Thêm `static/css/` cho custom style
2. Thêm `static/js/` cho custom JavaScript
3. Consider dùng CSS framework (Tailwind/Bootstrap)

---

## 26. ⚠️ Templates — Thiếu Error Page Custom (Ưu tiên: THẤP)

### Vấn đề
- Có `404.html` và `500.html` — tốt
- Nhưng không có maintenance page
- Không có page cho "model not ready"

### Giải pháp
1. Thêm maintenance page
2. Thêm page cho "model not ready" với hướng dẫn

---

## 27. ⚠️ Makefile — Duplicate Targets (Ưu tiên: THẤP)

### Vấn đề
- `.PHONY` có duplicate entries
- `test`, `lint`, `type-check` xuất hiện 2 lần
- `test-medical` và `test-web` trỏ đến file không tồn tại

### Ảnh hưởng
- Confusing khi dùng `make`
- Target không hoạt động

### Giải pháp
1. Xóa duplicate entries
2. Sửa path cho `test-medical` và `test-web`
3. Thêm comment cho từng target

---

## 28. ⚠️ Requirements — Không Có Lock File (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- `requirements.txt` có version range (`>=`) — không fixed
- Không có `requirements.lock` hay `poetry.lock`
- CI/CD install có thể fail nếu package mới có breaking change

### Ảnh hưởng
- Không reproducible build
- CI/CD có thể fail unexpectedly

### Giải pháp
1. Thêm `pip-compile` để tạo `requirements.lock`
2. Hoặc chuyển sang `poetry`/`pipenv` cho dependency management
3. Commit lock file vào repo

---

## 29. ⚠️ Config — Không Có Environment-Specific (Ưu tiên: THẤP)

### Vấn đề
- `config/settings.yaml` và `config/medical_settings.yaml` — chung cho tất cả environment
- Không có `config/production.yaml` hay `config/development.yaml`
- Không có environment variable override

### Ảnh hưởng
- Khó configure khác nhau cho dev/staging/production
- Rủi ro dùng config dev trong production

### Giải pháp
1. Thêm environment-specific config files
2. Thêm environment variable override
3. Document quy trình config cho từng environment

---

## 30. ⚠️ Compliance — Chỉ Disclaimer (Ưu tiên: TRUNG BÌNH)

### Vấn đề
- `medical/compliance.py` chỉ có `MEDICAL_DISCLAIMER` — tốt
- Nhưng không có:
  - Audit trail cho inference result
  - Consent management
  - Data privacy (HIPAA/GDPR compliance)
  - Model version tracking trong report

### Ảnh hưởng
- Không đủ cho production y khoa
- Không thể audit ai đã dùng hệ thống khi nào

### Giải pháp
1. Thêm audit trail — log mọi inference request
2. Thêm consent management — user phải đồng ý trước khi phân tích
3. Thêm model version trong report
4. Consider HIPAA/GDPR compliance checklist

---

## Tổng Kết Theo Mức Độ Ưu Tiên

### 🔴 CAO (Cần xử lý ngay)
1. Volume 3D không hoạt động — fix `load_volume_model()`
2. Thyroid detector dùng COCO generic — đánh dấu không khả dụng
3. Breast detector chỉ 1 lớp — kiểm tra lại
4. Security: Default admin — yêu cầu setup ban đầu

### 🟡 TRUNG BÌNH (Nên xử lý sớm)
5. Cancer catalog thiếu spleen — kiểm tra và thêm/xóa
6. Model catalog rỗng — điền thông tin model
7. Clinical validation chưa có báo cáo — tạo cohort kiểm định
8. File lớn cần refactor — pipeline.py, web_app.py, window.py
9. Scripts rời rạc — dọn dẹp và tổ chức
10. Test coverage chưa đầy đủ — thêm test cho module thiếu
11. CI/CD chưa có coverage report — thêm codecov
12. Docker image cũ — nâng cấp CUDA
13. Không có `.dockerignore` — tạo file
14. Security: CSRF token — set SameSite
15. Documentation thiếu API docs — tạo tài liệu
16. Requirements không có lock file — thêm pip-compile
17. Compliance chỉ disclaimer — thêm audit trail

### 🟢 THẤP (Có thể xử lý sau)
18. Performance — không có caching
19. Performance — không có async inference
20. Monitoring — không có centralized logging
21. Database — không có migration
22. Internationalization — chỉ tiếng Việt
23. Preprocessing — thiếu colonoscopy/colposcopy
24. Chat UI — không có conversation export
25. Voice — chỉ Windows (đã có alternative)
26. Assets — chỉ 1 favicon
27. Templates — thiếu error page custom
28. Makefile — duplicate targets
29. Config — không có environment-specific

---

## Đánh Giá Lại

**Từ 8.5/10 → 7/10** sau khi phân tích chi tiết:

- ✅ **Điểm mạnh**: Kiến trúc module hóa tốt, medical pipeline chuyên sâu, test coverage khá tốt, CI/CD có sẵn, Docker-ready, security-aware
- ❌ **Điểm yếu**: Volume 3D không hoạt động, 2 detector có vấn đề, file lớn cần refactor, test coverage chưa đầy đủ, documentation thiếu, security có thể cải thiện

**Khuyến nghị ưu tiên**:
1. Fix Volume 3D — ảnh hưởng lớn nhất đến functionality
2. Xử lý thyroid/breast detector — tránh kết quả sai
3. Refactor file lớn — cải thiện maintainability
4. Thêm test coverage — đảm bảo chất lượng
5. Cải thiện security — bảo vệ user data
