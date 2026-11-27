# Medical Imaging Guide

## Giới thiệu

Hệ thống OncoVision hỗ trợ phân tích các loại ảnh y khoa sau:
- **CT** (Computed Tomography)
- **MRI** (Magnetic Resonance Imaging)
- **PET/CT** (Positron Emission Tomography)
- **X-quang** (X-ray)
- **Siêu âm** (Ultrasound)
- **Mammogram** (X-quang vú)
- **Nội soi** (Endoscopy)
- **EUS** (Endoscopic Ultrasound)

## Định dạng ảnh hỗ trợ

| Định dạng | Mở rộng | Mô tả |
|-----------|---------|-------|
| DICOM | `.dcm` | Tiêu chuẩn ảnh y khoa |
| NIfTI | `.nii`, `.nii.gz` | Neuroimaging Informatics Technology Initiative |
| JPEG | `.jpg`, `.jpeg` | Ảnh y khoa dạng JPEG |
| PNG | `.png` | Ảnh y khoa dạng PNG |

## 11 nhóm ung thư hỗ trợ

| Nhóm | Mã | Modality phổ biến |
|------|-----|-------------------|
| Ung thư não | brain | MRI, CT |
| Ung thư phổi | lung | CT, X-quang |
| Ung thư vú | breast | Mammogram, Siêu âm |
| Ung thư dạ dày | stomach | Nội soi, CT |
| Ung thư đại trực tràng | colorectal | CT, MRI, Nội soi |
| Ung thư tuyến tiền liệt | prostate | MRI, CT |
| Ung thư cổ tử cung | cervical | Nội soi, Siêu âm |
| Ung thư thận | kidney | CT, MRI |
| Ung thư tụy | pancreas | CT, MRI |
| Ung thư tuyến giáp | thyroid | Siêu âm |
| Ung thư gan | liver | CT, MRI, PET/CT |

## Cách phân tích ảnh

1. Chuẩn bị ảnh y khoa (DICOM, NIfTI, JPG, PNG)
2. Chọn nhóm ung thư tương ứng
3. Chọn modality (loại ảnh)
4. Chạy phân tích qua giao diện web hoặc desktop
5. Xem kết quả: mức độ nguy cơ, nhãn dự đoán, heatmap Grad-CAM

## Lưu ý y khoa

> **CẢNH BÁO**: Hệ thống này chỉ hỗ trợ sàng lọc và nghiên cứu. Kết quả không dùng để chẩn đoán lâm sắc. Mọi trường hợp nghi ngờ ung thư cần được bác sĩ chuyên khoa đánh giá và kiểm định thêm.
