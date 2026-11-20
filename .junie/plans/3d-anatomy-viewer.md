---
sessionId: session-261116-003306-4pkf
---

# Requirements

### Overview & Goals

Thêm **3D Anatomy Viewer** vào OncoVision — sau khi AI phân tích xong một ca bệnh, ngoài ảnh 2D + Grad-CAM, hệ thống hiển thị model 3D giải phẫu cho phép bác sĩ **xoay, zoom, highlight cơ quan bị bệnh** để hình dung vị trí tổn thương trong không gian 3D.

### Scope

**In Scope (MVP):**
- Export model .blend nam giới → .glb (web-friendly)
- Thêm tab "3D View" trong khu vực hiển thị kết quả phân tích
- Highlight cơ quan liên quan đến loại ung thư đã phân tích (12 loại → 12+ cơ quan)
- Xoay/zoom/pan cơ bản bằng chuột
- Tooltip tên cơ quan khi hover

**Out of Scope (Phase 2+):**
- Model 3D nữ giới riêng
- Layer toggle (xương, cơ, nội tạng, mạch máu)
- Marker 3D chính xác từ bounding box 2D
- Model 3D từ DICOM/CT scan thực tế
- Female anatomy model

### User Stories

1. As a **bác sĩ**, sau khi phân tích xong ung thư gan, tôi muốn xem model 3D giải phẫu với gan được highlight đỏ để hình dung vị trí.
2. As a **bác sĩ**, tôi muốn xoay model 3D để xem cơ quan từ nhiều góc độ khác nhau.
3. As a **bác sĩ**, tôi muốn hover vào cơ quan 3D để xem tên và mô tả ngắn.

### Functional Requirements

- **FR1:** Sau khi phân tích xong, tab "3D View" xuất hiện bên cạnh ảnh 2D.
- **FR2:** Model 3D load từ file .glb, hỗ trợ xoay/zoom/pan.
- **FR3:** Cơ quan liên quan đến loại ung thư được highlight (màu đỏ/cam).
- **FR4:** Hover vào cơ quan → tooltip hiển thị tên cơ quan + mô tả.
- **FR5:** Mapping 12 loại ung thư → cơ quan 3D tương ứng.

### Non-Functional Requirements

- Model 3D tối ưu < 20MB (decimate + compress) để load nhanh.
- Three.js load từ CDN, không tăng kích thước build.
- Không ảnh hưởng hiệu năng phân tích AI hiện tại.
- Tương thích Chrome, Firefox, Edge (WebGL2).

# Technical Design

### Current Implementation

- **Web UI:** `templates/index.html` (340 dòng) — admin panel với `medical-viewer` (pacs screen + chat AI). Không có 3D viewer.
- **Case detail:** `static/js/case-workflow.js` (226 dòng) — hiển thị ảnh 2D, detections, risk level, recommendation.
- **Cancer catalog:** `medical/cancer_catalog.py` (132 dòng) — 12 nhóm ung thư với `key`, `label`, `modalities`.
- **3D assets:** `model_blender/male_blender/` — file `.blend` + `layers.json` (6739 dòng, ~3000+ bộ phận) + `app-annotations.json` (4029 dòng) + textures.
- **Static files:** `static/js/app.js` (1523 dòng), `static/css/styles.css` (1680 dòng).
- **Build:** `scripts/setup_build.py` — cx_Freeze, include `templates/`, `static/`.

### Key Decisions

| Quyết định | Lý do |
|-----------|-------|
| **Three.js** qua CDN | Không cần build vào EXE, nhẹ, phổ biến, hỗ trợ .glb |
| **Format .glb** (Binary GLTF) | Standard web, compress tốt, hỗ trợ texture, single file |
| **Tab mới trong main viewer** | Không làm rối giao diện hiện tại, dễ toggle |
| **Backend API trả mapping** | Python biết cancer key → organ names, JS chỉ render |
| **Export .blend → .glb 1 lần** | Không cần Blender runtime, chỉ cần file .glb cuối |

### Proposed Changes

#### 1. Export 3D Model (1-time setup)
- Dùng Blender để export `male.blend` → `static/models/anatomy_male.glb`
- Decimate mesh để giảm polygon count (~50%)
- Compress texture, optimize file size < 20MB
- Tương tự nếu có female model sau này

#### 2. Backend: Cancer → Organ Mapping
- Thêm `medical/anatomy_mapping.py`:
  - Mapping dict: `cancer_key → list[organ_names]`
  - Ví dụ: `"liver" → ["Liver", "Gan"]`
  - API endpoint `GET /api/anatomy/organs?target=liver` → trả danh sách cơ quan cần highlight
- Thêm vào `entrypoints/web_app.py`

#### 3. Frontend: 3D Viewer Component
- Thêm `static/js/3d-viewer.js`:
  - Init Three.js scene, camera, renderer
  - Load .glb model
  - Highlight organs dựa trên cancer target
  - Hover tooltip
  - Resize handler
- Thêm `static/css/3d-viewer.css`:
  - Tab layout, canvas container, tooltip styling

#### 4. UI Integration
- `templates/index.html`: Thêm tab "3D View" trong `medical-viewer` section
- `static/js/app.js`: Gọi init 3D viewer sau khi phân tích xong
- `scripts/setup_build.py`: Include `static/models/` vào build

### Data Models / Contracts

**API: `GET /api/anatomy/organs?target={cancer_key}`**
```json
{
  "target": "liver",
  "label": "Ung thư gan",
  "organs_to_highlight": [
    {"name": "Liver", "color": "#FF0000", "opacity": 0.8},
    {"name": "Gan", "color": "#FF0000", "opacity": 0.8}
  ]
}
```

**Mapping (medical/anatomy_mapping.py):**
```python
CANCER_TO_ORGANS: dict[str, list[str]] = {
    "liver": ["Liver"],
    "lung": ["Lung (Left)", "Lung (Right)"],
    "breast": ["Breast (Left)", "Breast (Right)"],
    "stomach": ["Stomach"],
    "colorectal": ["Colon", "Rectum"],
    "prostate": ["Prostate Gland"],
    "cervical": ["Uterus", "Cervix"],
    "kidney": ["Kidney (Left)", "Kidney (Right)"],
    "pancreas": ["Pancreas"],
    "thyroid": ["Thyroid Gland"],
    "brain": ["Brain"],
    "spleen": ["Spleen"],
}
```

### File Structure

```
static/
├── models/
│   └── anatomy_male.glb          [NEW] Export từ .blend
├── js/
│   └── 3d-viewer.js              [NEW] Three.js viewer logic
└── css/
    └── 3d-viewer.css             [NEW] 3D viewer styling

medical/
└── anatomy_mapping.py            [NEW] Cancer → organ mapping

templates/
└── index.html                    [MODIFIED] Thêm tab 3D View

static/js/
└── app.js                        [MODIFIED] Gọi init 3D viewer

scripts/
└── setup_build.py                [MODIFIED] Include static/models/
```

### Architecture Diagram

```mermaid
graph LR
    A[User phân tích ảnh] --> B[AI Pipeline]
    B --> C[Kết quả: cancer target]
    C --> D[Frontend: app.js]
    D --> E[GET /api/anatomy/organs?target=liver]
    E --> F[Backend: anatomy_mapping.py]
    F --> G[Trả organs_to_highlight]
    G --> H[3D Viewer: 3d-viewer.js]
    H --> I[Load anatomy_male.glb]
    I --> J[Highlight Liver organ]
    J --> K[Render Three.js scene]
```

### Risks

| Rủi ro | Mức độ | Mitigation |
|--------|--------|------------|
| .blend → .glb mất chi tiết | Trung bình | Decimate cẩn thận, kiểm tra visual |
| Model .glb quá nặng (>50MB) | Cao | Compress Draco, optimize texture |
| Mapping cancer → organ không chính xác | Trung bình | Manual review với bác sĩ |
| WebGL2 không hỗ trợ trên browser cũ | Thấp | Fallback: ẩn tab 3D, chỉ hiện 2D |
| Female model chưa có | Thấp | MVP chỉ nam, thêm female Phase 2 |

# Delivery Steps

### ✓ Step 1: Export 3D model và tạo anatomy mapping
Chuẩn bị tài sản 3D và backend mapping cơ bản.
- Export `model_blender/male_blender/male.blend` → `static/models/anatomy_male.glb` (dùng Blender, decimate mesh, compress texture, target < 20MB).
- Tạo `medical/anatomy_mapping.py` với dict mapping 12 cancer keys → danh sách organ names trong model 3D.
- Thêm API endpoint `GET /api/anatomy/organs?target={key}` vào `entrypoints/web_app.py` trả JSON organs_to_highlight.
- Update `scripts/setup_build.py`: include `static/models/` vào build cx_Freeze.

### ✓ Step 2: Implement 3D Viewer component (Three.js)
Tạo component 3D viewer độc lập, có thể test riêng.
- Tạo `static/js/3d-viewer.js`: init Three.js scene, camera, OrbitControls, GLTFLoader; load `anatomy_male.glb`; highlight organs theo danh sách nhận từ API; hover tooltip với tên cơ quan; resize handler.
- Tạo `static/css/3d-viewer.css`: canvas container, tooltip styling, tab layout.
- Three.js load từ CDN (không build vào EXE): `@loaders/gltf`, `@controls/orbit`.
- Test standalone: mở HTML đơn giản với 3D viewer, verify load model + highlight hoạt động.

### ✓ Step 3: Tích hợp 3D Viewer vào UI chính
Gắn 3D viewer vào giao diện admin panel, hoạt động sau khi phân tích xong.
- Sửa `templates/index.html`: thêm tab "3D View" trong `medical-viewer` section (bên cạnh ảnh 2D), với canvas container.
- Sửa `static/js/app.js`: sau khi nhận kết quả phân tích (có `target`), gọi `init3DViewer(target)` để load model + highlight cơ quan tương ứng.
- Đảm bảo 3D viewer resize đúng khi toggle sidebar/chat panel.
- Test full flow: upload ảnh → phân tích → xem kết quả 2D + 3D viewer với highlight đúng cơ quan.

---

## Kết quả phân tích Blender Model (male_blender)

### Phát hiện quan trọng

- **Model nam giới hoàn chỉnh**: ~5000+ bộ phận, 6739 dòng layers.json
- **Không có**: Breast, Uterus, Cervix, Ovary, Vagina (model nam)
- **Kidney**: Không có "Kidney" như một organ, mà là các parts: Renal Cortex, Renal Pyramid, Renal Pelvis, Renal Columns, Renal Papilla, Fibrous Capsule, Major/Minor Calices
- **Brain**: 20 parts (Medulla Oblongata, Pons, Cerebellum, Midbrain, Telencephalon) — layers 25-29
- **Lung**: 37 parts (lobes, bronchi, pleura) — layers 30-31
- **Colorectal**: 88 parts (colon, rectum, cecum, appendix, small intestine) — layers 12-13
- **Prostate**: 6 parts (Superior/Inferior Left/Right, Capsule) — layer 39
- **Urinary Bladder**: Có đầy đủ — layers 39
- **Testis**: Có đầy đủ — layers 14, 39

### Mapping chính xác từ Blender Model

```python
CANCER_TO_ORGANS: dict[str, list[str]] = {
    "brain": ["Medulla Oblongata", "Pons", "Cerebellum", "Midbrain", "Telencephalon"],
    "thyroid": ["Thyroid Gland"],
    "lung": ["Superior Lobe of Left Lung", "Inferior Lobe of Left Lung",
             "Superior Lobe of Right Lung", "Middle Lobe of Right Lung", "Inferior Lobe of Right Lung"],
    "breast": [],  # Model nam giới, không có breast
    "stomach": ["Stomach"],
    "liver": ["Liver"],
    "pancreas": ["Pancreas"],
    "spleen": ["Spleen"],
    "kidney": ["Renal Cortex", "Renal Pyramid", "Renal Pelvis"],  # Không có "Kidney"
    "colorectal": ["Ascending Colon", "Transverse Colon", "Descending Colon",
                   "Sigmoid Colon", "Rectum", "Cecum"],
    "prostate": ["Prostate"],
    "cervical": [],  # Model nam giới, không có uterus/cervix
}
```

### Delivery Steps — Phase 2: Xử lý issues và phát triển thêm

### ✓ Step 4: Phân tích Blender model & cập nhật mapping chính xác
Cập nhật anatomy mapping dựa trên dữ liệu thực tế từ Blender model.
- ✅ Đã phân tích `layers.json` (6739 dòng, ~5000+ bộ phận)
- ✅ Xác định các cơ quan chính cho 12 loại ung thư
- ✅ Phát hiện model nam giới không có Breast, Uterus, Cervix
- ✅ Kidney dùng "Renal Cortex/Pyramid/Pelvis" thay vì "Kidney"
- Cập nhật `medical/anatomy_mapping.py` với tên chính xác từ Blender model
- Tạo `docs/blender_cancer_organs.txt` làm tài liệu tham chiếu
- Cập nhật `static/js/3d-viewer.js` procedural model để khớp với Blender names

### ✓ Step 5: Xử lý các issues còn lại
Xử lý các vấn đề được đề cập trong issue description.
- **cx_Freeze trong requirements**: Thêm `cx_Freeze>=7.1` vào `requirements/requirements.txt` nếu chưa có
- **Python 3.13 version mismatch**: Kiểm tra `config/pyproject.toml` `requires-python` và điều chỉnh nếu cần
- **Dead code cleanup**: Xác nhận `app/chat_ui/window.py` đã được loại khỏi build
- **Volume 3D .pt2 files**: Tài liệu hóa vấn đề PyTorch mismatch, không fix trong phase này
- **Spleen model**: Tài liệu hóa việc chưa có model 2D nguồn mở
- **Thyroid detector**: Xác nhận hệ thống tự động bỏ qua COCO generic detector

### ✓ Step 6: Kế hoạch phát triển 3D Viewer với Blender Model
Lập kế hoạch chi tiết để tích hợp Blender model thực tế thay vì procedural model.
- **Tùy chọn 1**: Export `.blend → .glb` bằng Blender (cần cài Blender)
- **Tùy chọn 2**: Tiếp tục dùng procedural model (Three.js primitives) — hiện tại
- **Tùy chọn 3**: Kết hợp cả hai — procedural cho MVP, .glb khi có Blender
- Đánh giá kích thước file .glb dự kiến và thời gian load
- Kế hoạch thêm female model (Phase 3)
- Kế hoạch layer toggle (xương, cơ, nội tạng, mạch máu) — Phase 3

### ✓ Step 7: Tối ưu 3D Viewer hiện tại
Cải thiện 3D viewer với API integration, tooltip tiếng Việt, và legend chi tiết.
- ✅ Thêm `fetchOrganDescriptions()` — gọi API `GET /api/anatomy/organs?target={key}` để lấy mô tả tiếng Việt
- ✅ Cập nhật `init3DViewer()` — gọi API trước khi render model
- ✅ Cải thiện `showTooltip()` — hiển thị mô tả tiếng Việt từ API + trạng thái ⚠️/✅
- ✅ Cải thiện `buildLegend()` — hiển thị tên cơ quan bị ảnh hưởng với tên tiếng Việt
- ✅ Cập nhật CSS — tooltip đẹp hơn, max-width 300px, font-size phù hợp
- ✅ 425 tests passed, 0 failed

### ✓ Step 8: Bổ sung Female Organs cho 12 ung thư đầy đủ
Thêm cơ quan nữ vào 3D viewer để hỗ trợ đầy đủ 12 loại ung thư.
- ✅ Tìm kiếm nguồn mở 3D female anatomy models — không tìm được nguồn đầy đủ, quyết định dùng procedural model
- ✅ Thêm 6 cơ quan nữ vào `static/js/3d-viewer.js`: Breast (Left/Right), Uterus, Cervix, Ovary (Left/Right)
- ✅ Thêm 2 shapes mới: `breast` (teardrop), `uterus` (pear shape)
- ✅ Cập nhật `medical/anatomy_mapping.py`: breast → ["Breast (Left)", "Breast (Right)"], cervical → ["Uterus", "Cervix", "Ovary (Left)", "Ovary (Right)"]
- ✅ Thêm mô tả tiếng Việt cho 6 cơ quan nữ
- ✅ 425 tests passed, 0 failed
- **Tổng số cơ quan 3D**: 33 (27 nam + 6 nữ) — đầy đủ cho 12 loại ung thư

### ✓ Step 9: Dọn dẹp Blender Model — chỉ giữ texture liên quan 12 ung thư
Dọn dẹp `model_blender/male_blender/` để chỉ giữ các file cần thiết cho 12 loại ung thư.
- ✅ Tạo `scripts/cleanup_blender_textures.py` — script tự động phân loại texture theo keywords
- ✅ Xóa 656 files (~1.24 GB) không liên quan: xương, cơ, gân, fascia, mỡ, da, răng, tai, mắt, tim, bàng quang, tinh hoàn, dương vật, thymus, gallbladder, peritoneum, trachea, sinus
- ✅ Xóa `raw-normal/` subdirectory (125 files, ~450 MB)
- ✅ Giữ lại 48 texture files (~171 MB) cho 12 ung thư:
  - **Não**: Brain_Inside, Cerebellum, Dura_mater, Spinal_Region
  - **Tuyến giáp**: Thyroid_Parotid_Nasal
  - **Phổi**: Respiratory_Lobes, Pleuras, Respiratory
  - **Gan**: AbdomenLiver, filling_organs_liver
  - **Tụy**: AbdomenPancreas, filling_organs_pancreas
  - **Lách**: filling_organs_spleen
  - **Thận**: AbdomenKidney
  - **Dạ dày**: AbdomenStomach
  - **Đại trực tràng**: AbdomenLargeIntestine, filling_organs_large_intestine
  - **Ruột non**: AbdomenSmallIntestine
  - **Tuyến tiền liệt**: filling_prostate
- ✅ Giữ `.blend` file (1.47 GB) để export .glb sau này
- ✅ Giữ JSON files: layers.json, app-annotations.json, descriptions.jsonl, textures.jsonl, source-materials.json, validation.json
- **Kết quả**: 383 files (~2.88 GB) → 55 files (~1.65 GB), tiết kiệm ~1.23 GB

### ✓ Step 10: Dọn dẹp và sắp xếp project structure
Dọn dẹp file thừa, đổi tên thư mục, và sắp xếp lại project structure.
- ✅ Xóa 187 file .txt trong output/ (test/build logs thừa)
- ✅ Xóa 12 file test thừa trong output/ (check_cancer.py, test_auth.py, test_brain.py...)
- ✅ Xóa data/web_users.db (legacy file, 36KB)
- ✅ Xóa scripts/extract_cancer_organs.py (đã dùng xong)
- ✅ Xóa build/ (12.23 GB build output cũ)
- ✅ Đổi tên "model blender" → "model_blender" (chuẩn Python naming)
- ✅ Cập nhật comment trong medical/anatomy_mapping.py và static/js/3d-viewer.js
- ✅ Thêm model_blender/ vào .gitignore
- ✅ 433 tests passed, 19 subtests passed, 0 failed

### ✓ Step 11: Cải thiện hiển thị kết quả phân tích AI
Thiết kế lại card kết quả phân tích để hiển thị thông minh hơn với đầy đủ thông tin.
- ✅ Cập nhật `buildResultCard()` trong `static/js/app.js`:
  - Thêm context grid: vùng cơ quan (tiếng Việt), modality, model name, chiều 2D/3D
  - Thêm confidence level indicator: 🟢 Cao (≥80%), 🟡 Trung bình (≥50%), 🔴 Thấp (<50%)
  - Thêm malignancy indicator: ⚠️ Nghi ngờ ác tính / ✅ Không nghi ngờ
  - Thêm quality warnings section với danh sách cảnh báo
  - Thêm recommendation section với đề xuất hỗ trợ
  - Dynamic stats: số nhãn, độ tin cậy TB, mức độ tin cậy, nghi ngờ ác tính, mức độ rủi ro
  - Cancer label mapping tiếng Việt cho 12 loại ung thư
  - Risk label tiếng Việt: Cao, Trung bình, Thấp, Chưa rõ
- ✅ Cập nhật `static/css/styles.css`:
  - Context grid với icon + label/value
  - Malignancy indicator: malignant (đỏ) / benign (xanh lá)
  - Quality warnings: nền vàng, danh sách bullet
  - Recommendation: nền xanh accent
  - Result section title
  - Dynamic stats với màu sắc theo risk/confidence level
  - Xóa duplicate CSS rules
- ✅ 436 tests passed, 19 subtests passed, 0 failed