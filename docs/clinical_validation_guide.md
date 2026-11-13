# Hướng Dẫn Kiểm Định Lâm Sàng

## Tổng Quan

Kiểm định lâm sàng (clinical validation) xác nhận model hoạt động đúng trên cohort bệnh nhân độc lập — không nằm trong dữ liệu train. OncoVision đã có framework kiểm định hoàn chỉnh trong `medical/clinical_validation.py`.

## Quy Trình 6 Bước

### Bước 1: Ước Lượng Cỡ Mẫu

Sử dụng CLI để tính số ca cần thiết:

```bash
python run_medical.py plan-validation \
  --sensitivity 0.90 \
  --specificity 0.85 \
  --prevalence 0.20
```

Kết quả trả về:
- `minimum_positive_cases`: Số ca dương tính tối thiểu
- `minimum_negative_cases`: Số ca âm tính tối thiểu
- `minimum_total_cases`: Tổng số ca cần thiết

### Bước 2: Chuẩn Bị Cohort Độc Lập

Yêu cầu:
- **Không rò rỉ bệnh nhân**: Bệnh nhân trong cohort kiểm định không được xuất hiện trong dữ liệu train
- **Theo bệnh nhân**: Mỗi bệnh nhân chỉ xuất hiện một lần
- **Đa dạng**: Bao phủ các mức độ bệnh, thiết bị, bệnh viện

Kiểm tra dữ liệu:

```bash
# Kiểm tra CSV dữ liệu
python run_medical.py validate-dataset --csv dataset.csv

# Chia tập dữ liệu không rò rỉ bệnh nhân
python run_medical.py split-dataset --csv dataset.csv --output dataset_split.csv
```

### Bước 3: Chạy Inference Trên Cohort

Chạy phân tích từng ảnh trong cohort:

```bash
# Phân tích từng ảnh
for image in cohort_images/*; do
  python run_medical.py analyze \
    --image "$image" \
    --patient-code "$(basename $image .jpg)"
done
```

Hoặc dùng script tự động để chạy batch inference.

### Bước 4: Tạo CSV Kết Quả

Tạo file CSV với 3 cột bắt buộc:

```csv
patient_id,truth,prediction
BN001,glioma,glioma
BN002,meningioma,meningioma
BN003,pituitary,no_tumor
BN004,no_tumor,no_tumor
```

Yêu cầu:
- `patient_id`: Mã bệnh nhân duy nhất
- `truth`: Chẩn đoán chuẩn (ground truth)
- `prediction`: Kết quả model dự đoán

### Bước 5: Tạo Báo Cáo Validation

```bash
python run_medical.py validate-clinical \
  --csv results.csv \
  --model models/pretrained/brain_classifier.pt \
  --cohort-id "BRAIN-2026-001" \
  --output models/pretrained/cancers/brain/clinical_validation.json
```

Báo cáo bao gồm:
- **Metrics**: Accuracy, precision, recall, F1 (tổng và theo lớp)
- **Confidence intervals**: Wilson score 95% CI cho sensitivity/specificity
- **Confusion matrix**: Ma trận nhầm lẫn
- **Model hash**: SHA256 của model được kiểm định
- **Cohort info**: Số bệnh nhân, số lớp, SHA256 CSV

### Bước 6: Gửi Đơn Vị Lâm Sàng Phê Duyệt

Sao chép `config/clinical_validation_template.json`, điền thông tin:

```json
{
  "schema_version": 1,
  "cohort_id": "BRAIN-2026-001",
  "dataset_provenance": "Bệnh viện XYZ, 2026",
  "ground_truth_method": "Chẩn đoán bằng phẫu thuật + histopathology",
  "sample_size_justification": "Dựa trên plan-validation với sensitivity=0.90",
  "site_count": 2,
  "inclusion_criteria": ["Tuổi 18-75", "MRI não có sẵn"],
  "exclusion_criteria": ["Ảnh nhiễu", "Thiếu hồ sơ"],
  "approved_by": "GS.TS Nguyễn Văn A",
  "approved_at": "2026-11-13T00:00:00+07:00"
}
```

Gửi báo cáo cho đơn vị lâm sàng độc lập phê duyệt.

## Tích Hợp Với Pipeline

Khi có `clinical_validation.json` hợp lệ:

1. **Badge trong báo cáo**: Hiển thị "Đã kiểm định lâm sàng"
2. **Confidence intervals**: Bao gồm CI trong kết quả inference
3. **Cảnh báo hash**: Cảnh báo nếu model hash không khớp với báo cáo

Kiểm tra trạng thái:

```bash
python run_medical.py status
```

## Metrics Được Tính Toán

| Metric | Mô tả |
|---|---|
| **Accuracy** | Tỷ lệ dự đoán đúng tổng thể |
| **Precision** | Tỷ lệ dự đoán dương tính đúng |
| **Recall (Sensitivity)** | Tỷ lệ phát hiện bệnh nhân thực sự |
| **Specificity** | Tỷ lệ phát hiện âm tính đúng |
| **F1 Score** | Trung bình harmonic của precision và recall |
| **95% CI** | Khoảng tin cậy Wilson score |

## Lưu Ý Quan Trọng

1. **Không tự động hóa hoàn toàn**: Clinical validation cần sự tham gia của chuyên gia lâm sàng
2. **Bảo mật dữ liệu**: Tuân thủ quy định bảo mật bệnh nhân (HIPAA, GDPR)
3. **Đa trung tâm**: Khuyến nghị kiểm định trên nhiều bệnh viện
4. **Cập nhật định kỳ**: Re-validate khi có model mới hoặc dữ liệu mới

## Tham Khảo

- [Model Download Guide](model_download.md) — Tải và cài đặt model
- [Medical Imaging Guide](medical_imaging_guide.md) — Luồng suy luận y khoa
- `medical/clinical_validation.py` — Source code framework
