# Hướng dẫn phân tích ảnh y khoa

OncoVision là ứng dụng **suy luận**. Người dùng đưa ảnh vào, hệ thống kiểm tra đầu vào, chạy model đã được cung cấp, rồi lưu kết quả sàng lọc và gợi ý bước tiếp theo. Ứng dụng không huấn luyện model; CLI có kiểm tra manifest dữ liệu và chia train/validation/test theo bệnh nhân để chuẩn bị dữ liệu bên ngoài.

## Chạy phân tích

```powershell
python run_chat.py
# hoặc dùng giao diện web
python web_app.py
# hoặc CLI
python run_medical.py analyze --image path/to/image.dcm --patient-code BN001
```

Các lệnh CLI còn lại phục vụ suy luận, chuẩn bị dữ liệu và quản lý ca: `validate-image`, `validate-dataset`, `split-dataset`, `status`, `report`, `history`, `show-case`, `export-case`, `delete-case`, `cleanup-output`.

## Model

Model suy luận được chuẩn bị bên ngoài và đặt trong `models/pretrained/` hoặc đường dẫn cấu hình tại `config/medical_settings.yaml`.

Catalog có 11 nhóm: gan, phổi, vú, dạ dày, đại trực tràng, tuyến tiền liệt, cổ tử cung, thận, tụy, tuyến giáp và não. Giao diện luôn liệt kê đủ catalog nhưng chỉ bật nhóm có model đúng hợp đồng trên máy.

Khi có model mới, đặt file model và metadata cần thiết vào vị trí cấu hình rồi chạy:

```powershell
python run_medical.py status
python run_medical.py analyze --image path/to/image.dcm --patient-code BN001
```

Với model công khai (vd repo HuggingFace), dùng công cụ dev để tải, kiểm runtime và đặt đúng hợp đồng một bước:

```powershell
python scripts/fetch_model.py owner/repo --file model.pt --target brain --kind detector --modality mri
```

File không nạp được (sai định dạng, sai task YOLO) sẽ bị từ chối và `run_medical.py status` cũng báo rõ lý do thay vì báo sẵn sàng giả.

### Hợp đồng model theo nhóm

Đặt model vào `models/pretrained/cancers/<key>/`, với `<key>` là `liver`, `lung`, `breast`, `stomach`, `colorectal`, `prostate`, `cervical`, `kidney`, `pancreas`, `thyroid` hoặc `brain`:

```text
models/pretrained/cancers/<key>/
├── classifier.pt             # tùy chọn: checkpoint classifier 2D của dự án
├── detector.pt               # tùy chọn: Ultralytics YOLO detect/segment
├── volume.pt                 # tùy chọn: TorchScript 3D
├── volume.json               # bắt buộc cùng volume.pt
└── clinical_validation.json  # báo cáo cohort độc lập đã phê duyệt
```

Nếu một cơ quan có nhiều loại ảnh, đặt model trong thư mục modality như `lung/ct/`, `lung/xray/` hoặc `cervical/colposcopy/`. Khi modality đã được xác định, runtime bắt buộc model nằm trong đúng thư mục modality và không rơi về model cấp cơ quan. Các khóa chuẩn gồm `ct`, `mri`, `xray`, `ultrasound`, `mammogram`, `endoscopy`, `colonoscopy`, `colposcopy`, `pet_ct` và `eus`.

Chỉ cần một trong `classifier.pt`, `detector.pt` hoặc cặp `volume.pt + volume.json` để nhóm xuất hiện ở trạng thái có runtime. Khả năng phân loại, tọa độ 2D, volume 3D và kiểm định lâm sàng được báo riêng; có một khả năng không làm các khả năng còn lại thành sẵn sàng.

- `classifier.pt` chứa `model_state_dict`, `class_labels`, `backbone` và `num_classes` theo checkpoint classifier của dự án.
- `detector.pt` là model Ultralytics YOLO có task `detect` hoặc `segment`. Hệ thống chỉ công bố tọa độ khi model trả bbox hợp lệ.
- `volume.pt` nhận tensor `[N,C,D,H,W]` và trả `classification_logits`, `segmentation_logits`, hoặc tuple gồm hai tensor đó.
- `volume.json` khai báo target, task, modality, shape đầu vào, nhãn và ngưỡng mask. Tọa độ segmentation được đổi về volume gốc và xuất cả voxel `(x1,y1,z1,x2,y2,z2)`, mm cùng thể tích `mm³`.
- Mask 3D được lưu dạng NPZ nén với các mảng `mask`, `affine`, `spacing_xyz_mm` và `coordinate_system`, rồi đưa vào metadata và gói xuất ca.
- Các đường dẫn `brain_model`, `yolo_model_path`, `volume_model_path` và `volume_manifest_path` cũ vẫn được dùng làm tương thích ngược cho nhóm não.
- Model phân loại không sinh tọa độ. Các trọng số YOLO nhận diện vật thể phổ thông không phải model ung thư và không được dùng thay model y khoa.

`python run_medical.py status` hiển thị riêng trạng thái phân loại, định vị và kiểm định lâm sàng. “Model chạy được” không đồng nghĩa với “đã kiểm định lâm sàng”.

Manifest 3D tối thiểu:

```json
{
  "task": "volume_multitask",
  "target": "brain",
  "class_labels": ["glioma", "meningioma", "pituitary", "no_tumor"],
  "input_modalities": ["t1", "t1ce", "t2", "flair"],
  "segmentation_labels": ["tumor_core", "whole_tumor", "enhancing_tumor"],
  "segmentation_activation": "sigmoid",
  "input_shape": [128, 128, 128],
  "mask_threshold": 0.5
}
```

Có thể sao chép `config/brain_volume_model.example.json` sang `models/pretrained/cancers/<key>/volume.json`, rồi đổi `target`, nhãn và modality cho đúng model thực tế.

mpMRI tuyến tiền liệt có mẫu riêng tại `config/prostate_mpmri_volume_model.example.json`, theo thứ tự kênh `t2`, `dwi`, `adc`, `dce`. Tên file đầu vào phải chứa đúng token và tất cả volume phải cùng shape/affine.

Các task hợp lệ là `volume_classification`, `volume_segmentation` và `volume_multitask`. Với MRI đa kênh, mỗi file NIfTI trong cùng thư mục phải chứa token modality tương ứng trong tên file và phải có cùng shape/affine. Hệ thống giới hạn kích thước đầu vào model để tránh hết bộ nhớ và chỉ bật trạng thái 3D khi cả model lẫn manifest nạp được.

## Kiểm định cohort độc lập

CSV đánh giá phải có đúng một dòng cho mỗi bệnh nhân với các cột `patient_id`, `truth`, `prediction`. Tạo báo cáo gắn với SHA-256 của model:

```powershell
python run_medical.py validate-cohort `
  --csv external_cohort_predictions.csv `
  --model models/pretrained/cancers/brain/volume.pt `
  --cohort-id external-hospital-01 `
  --output models/pretrained/cancers/brain/clinical_validation.json
```

Báo cáo sinh tự động vẫn ở trạng thái chưa phê duyệt. Sau khi đơn vị lâm sàng độc lập xác nhận nguồn cohort, chuẩn tham chiếu, cỡ mẫu và các chỉ số, báo cáo phải có `approved_by`, `approved_at`, `dataset_provenance`, `ground_truth_method`, `sample_size_justification` và `site_count`. Hệ thống chỉ báo sẵn sàng khi hash trong báo cáo khớp chính xác model đang chạy.

Ước lượng ban đầu số ca dương, âm và tổng cho bài toán nhị phân:

```powershell
python run_medical.py plan-validation `
  --sensitivity 0.90 --specificity 0.85 --prevalence 0.20 `
  --margin 0.05 --confidence 0.95 `
  --output validation_sample_size.json
```

Đây là ước lượng CI nhị thức hai phía để lập kế hoạch. Protocol chính thức vẫn cần statistician và đơn vị lâm sàng phê duyệt.

## Dữ liệu cần đưa vào

| Loại dữ liệu | Tệp tối thiểu | Nhãn khi huấn luyện/kiểm định | Hệ thống xử lý |
|---|---|---|---|
| Ảnh 2D | JPG/PNG/TIFF hoặc một DICOM | `patient_id`, nhãn lớp; bbox/mask nếu cần tọa độ | kiểm tra modality/vùng, chuẩn hóa, classifier hoặc detector |
| Volume 3D | một NIfTI/MHA/MHD hoặc một thư mục DICOM series | nhãn bệnh nhân; mask 3D nếu cần thể tích/tọa độ | giữ spacing/affine, resize cho model, đưa mask về hệ tọa độ gốc |
| MRI đa chuỗi | các NIfTI đã đăng ký, tên chứa token trong `input_modalities` | nhãn và mask cùng không gian ảnh | kiểm tra shape/affine rồi ghép theo kênh |
| Video nội soi/soi cổ tử cung | MP4/AVI/MOV/MKV | nhãn theo ca; timestamp/mask nếu huấn luyện video | lấy mẫu tối đa 12 khung trải đều, tổng hợp điểm và lưu frame/timestamp của phát hiện mạnh nhất |
| Cohort độc lập | CSV `patient_id,truth,prediction` | chuẩn tham chiếu mô bệnh học hoặc hội đồng chuyên gia | tính metric, CI 95%, khóa báo cáo theo SHA-256 model |

Một bệnh nhân phải nằm trọn trong một split để tránh rò rỉ dữ liệu. Dữ liệu đầu vào nên kèm modality, cơ quan, protocol, cơ sở chụp và ground truth; các ca âm tính và ca khó phải được giữ lại trong test ngoài cơ sở.

Manifest CSV chuẩn dùng các cột `patient_id,image_path,cancer_target,modality,truth_label,mask_path,bbox,site_id,protocol,scanner,split`. Chạy `python run_medical.py validate-dataset --csv dataset.csv` để kiểm tra, rồi `python run_medical.py split-dataset --csv dataset.csv --output dataset_split.csv` để chia cố định theo `patient_id`.

Một ca đơn có thể dùng YAML gồm `patient_id`, `cancer_target`, `modality`, `image`, sau đó chạy `python run_medical.py analyze --manifest case.yaml`.

Không có model phù hợp thì ứng dụng phải báo chưa hỗ trợ, không dùng nhầm model khác để đưa ra kết luận.

## Đầu vào

- Ảnh: JPG, PNG, BMP, TIFF, WEBP.
- Ảnh y khoa/volume: DICOM (`.dcm` hoặc thư mục series), NIfTI (`.nii`, `.nii.gz`), MHA/MHD; video nội soi MP4/AVI/MOV/MKV.
- Nhận diện modality và vùng cơ thể dựa trên nội dung ảnh, tên tệp và metadata DICOM.
- Modality do người dùng chọn được đối chiếu với modality nhận diện từ tệp. Nếu hai giá trị không khớp ở mức tin cậy cấu hình, phân tích bị từ chối; nếu chưa xác minh được, kết quả có cảnh báo. Việc đối chiếu này không chứng minh model đã được huấn luyện hoặc kiểm định trên modality đó.
- Ảnh soi cổ tử cung có pipeline 2D riêng; Pap/HPV, kết quả sinh thiết và văn bản lâm sàng không được đưa qua pipeline ảnh.

Model não hiện tại là classifier 2D. Các nhóm khác đã có luồng tiếp nhận và chọn model nhưng chưa có trọng số trong repository. Volume NIfTI/MHA/MHD và thư mục DICOM series chỉ chạy khi có cặp model/manifest 3D đúng target; hệ thống không ghép volume thành montage rồi dùng classifier 2D. DICOM series được khử định danh trước suy luận, còn tọa độ và spacing được giữ trong báo cáo.

Classifier trả về nhãn và điểm dự đoán, không tự xác định tọa độ tổn thương hoặc kết luận ác tính. Tọa độ chỉ được lưu khi model định vị thực sự cung cấp bbox; Grad-CAM chỉ là heatmap giải thích. Nhận định nguy cơ/ác tính chỉ được hiển thị sau khi nhân viên y tế rà soát ca.

Với YOLO segment, hệ thống lưu cả polygon trong JSON và raster hóa mask nhị phân PNG. Mask được đưa vào metadata, báo cáo HTML/Markdown và gói xuất ca.

## Kết quả và an toàn

Ca phân tích, ảnh xử lý và báo cáo được lưu trong `output/medical/`; lịch sử ca nằm trong SQLite. Báo cáo bao gồm mức độ sàng lọc, cảnh báo chất lượng ảnh, giải thích trực quan khi model hỗ trợ và khuyến nghị bước tiếp theo.

Đây là công cụ hỗ trợ sàng lọc, không thay thế bác sĩ, chẩn đoán mô bệnh học hoặc quy trình chuyên môn. Kết quả không chắc chắn cần được chuyển chuyên gia đánh giá.
