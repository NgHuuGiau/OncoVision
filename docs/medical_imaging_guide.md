# Hướng dẫn phân tích ảnh y khoa

OncoVision là ứng dụng **suy luận**. Người dùng đưa ảnh vào, hệ thống kiểm tra đầu vào, chạy model đã được cung cấp, rồi lưu kết quả sàng lọc và gợi ý bước tiếp theo. Ứng dụng không tải dữ liệu huấn luyện, không tạo train/validation/test split và không huấn luyện model.

## Chạy phân tích

```powershell
python run_chat.py
# hoặc dùng giao diện web
python web_app.py
# hoặc CLI
python run_medical.py analyze --image path/to/image.dcm --patient-code BN001
```

Các lệnh CLI còn lại phục vụ suy luận và quản lý ca: `validate-image`, `status`, `report`, `history`, `show-case`, `export-case`, `delete-case`, `cleanup-output`.

## Model

Model suy luận được chuẩn bị bên ngoài và đặt trong `models/pretrained/` hoặc đường dẫn cấu hình tại `config/medical_settings.yaml`.

| Nhóm | Trạng thái |
|---|---|
| Não | Có model hiện tại; hỗ trợ 4 nhãn glioma, meningioma, pituitary, no_tumor |
| Gan, phổi, vú, dạ dày, đại trực tràng, tuyến tiền liệt, cổ tử cung, thận, tụy, tuyến giáp | Đã đăng ký luồng nhận diện đầu vào; chờ model suy luận tương ứng |
| Nhận diện modality | Dùng model có sẵn nếu được cấu hình |

Khi có model mới, đặt file model và metadata cần thiết vào vị trí cấu hình rồi chạy:

```powershell
python run_medical.py status
python run_medical.py analyze --image path/to/image.dcm --patient-code BN001
```

Không có model phù hợp thì ứng dụng phải báo chưa hỗ trợ, không dùng nhầm model khác để đưa ra kết luận.

## Đầu vào

- Ảnh: JPG, PNG, BMP, TIFF, WEBP.
- Ảnh y khoa/volume: DICOM (`.dcm` hoặc thư mục series), NIfTI (`.nii`, `.nii.gz`), MHA/MHD.
- Nhận diện modality và vùng cơ thể dựa trên nội dung ảnh, tên tệp và metadata DICOM.
- Pap/HPV, nội soi cổ tử cung và sinh thiết không phải định dạng ảnh được app phân tích trực tiếp.

## Kết quả và an toàn

Ca phân tích, ảnh xử lý và báo cáo được lưu trong `output/medical/`; lịch sử ca nằm trong SQLite. Báo cáo bao gồm mức độ sàng lọc, cảnh báo chất lượng ảnh, giải thích trực quan khi model hỗ trợ và khuyến nghị bước tiếp theo.

Đây là công cụ hỗ trợ sàng lọc, không thay thế bác sĩ, chẩn đoán mô bệnh học hoặc quy trình chuyên môn. Kết quả không chắc chắn cần được chuyển chuyên gia đánh giá.
