# Hướng Dẫn Clinical Validation — OncoVision

## Tổng Quan

Clinical validation là bước **bắt buộc** để đưa OncoVision vào sản xuất y khoa thực sự. Framework đã được xây dựng sẵn trong `medical/clinical_validation.py` — tài liệu này hướng dẫn cách sử dụng.

## Quy Trình 5 Bước

### Bước 1: Lên Kế Hoạch Cỡ Mẫu

Sử dụng hàm `plan_binary_validation_sample_size()` để tính số ca tối thiểu cần thiết:

```python
from medical.clinical_validation import plan_binary_validation_sample_size

plan = plan_binary_validation_sample_size(
    sensitivity=0.85,      # Độ nhạy mong đợi
    specificity=0.90,      # Độ đặc hiệu mong đợi
    prevalence=0.30,       # Tỷ lệ mắc bệnh trong cohort
    margin=0.05,           # Biên độ sai số CI (±5%)
    confidence=0.95,       # Mức tin cậy 95%
)

print(f"Cần tối thiểu {plan['minimum_total_cases']} ca")
print(f"  - Dương tính: {plan['planned_positive_cases']} ca")
print(f"  - Âm tính: {plan['planned_negative_cases']} ca")
```

**Ví dụ thực tế cho ung thư não (Brain MRI):**

```python
brain_plan = plan_binary_validation_sample_size(
    sensitivity=0.88,
    specificity=0.92,
    prevalence=0.40,  # 4 lớp: glioma, meningioma, pituitary, no_tumor
    margin=0.05,
)
# Kết quả: ~380 ca tối thiểu
```

### Bước 2: Chuẩn Bị Cohort Độc Lập

Cohort validation phải thỏa mãn:

1. **Độc lập**: Không trùng với dữ liệu training
2. **Theo bệnh nhân**: Mỗi bệnh nhân chỉ xuất hiện 1 lần
3. **Đa trung tâm**: Ít nhất 1 cơ sở y tế khác (tốt nhất 2+)
4. **Chuẩn tham chiếu**: Kết quả gold standard từ bác sĩ chuyên khoa

**Định dạng CSV:**

```csv
patient_id,truth,prediction
P001,glioma,glioma
P002,meningioma,meningioma
P003,no_tumor,no_tumor
P004,pituitary,pituitary
P005,glioma,no_tumor
```

**Cột bắt buộc:**
- `patient_id`: Mã bệnh nhân duy nhất
- `truth`: Nhãn gold standard (phải khớp với class_labels của model)
- `prediction`: Kết quả dự đoán của model

### Bước 3: Chạy Model Trên Cohort

Sử dụng pipeline để chạy inference trên toàn bộ cohort:

```python
from medical.pipeline import MedicalImageAnalyzer
import csv

analyzer = MedicalImageAnalyzer()
predictions = []

with open("cohort_images.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        result = analyzer.analyze_image(
            image_path=row["image_path"],
            target_key="brain",
        )
        predictions.append({
            "patient_id": row["patient_id"],
            "truth": row["truth"],
            "prediction": result.prediction_label,
        })

# Ghi kết quả ra CSV
with open("validation_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["patient_id", "truth", "prediction"])
    writer.writeheader()
    writer.writerows(predictions)
```

### Bước 4: Tạo Báo Cáo Kiểm Định

```python
from medical.clinical_validation import build_independent_validation_report

report_path = build_independent_validation_report(
    csv_path="validation_results.csv",
    model_path="models/pretrained/cancers/brain/mri/classifier.pt",
    output_path="output/validation/brain_mri_report.json",
    cohort_id="BRAIN_MRI_2026_Q1",
)

print(f"Báo cáo đã lưu tại: {report_path}")
```

**Báo cáo JSON chứa:**
- Thông tin cohort (số bệnh nhân, SHA256 hash)
- Thông tin model (đường dẫn, SHA256 hash)
- Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
- Per-class metrics với Wilson 95% CI
- Trạng thái phê duyệt (chờ đơn vị lâm sàng)

### Bước 5: Phê Duyệt Báo Cáo

Sau khi có báo cáo, cần điền thông tin phê duyệt:

```python
import json
from pathlib import Path

report = json.loads(Path("output/validation/brain_mri_report.json").read_text())

# Điền thông tin phê duyệt
report["approved_by"] = "GS.TS Nguyễn Văn A - Khoa Ung Bướu, BV Bạch Mai"
report["approved_at"] = "2026-06-15T10:30:00+07:00"
report["dataset_provenance"] = "Dữ liệu từ BV Bạch Mai và BV K Trung ương, 2025-2026"
report["ground_truth_method"] = "Chẩn đoán mô bệnh học + hội chẩn đa chuyên khoa"
report["sample_size_justification"] = "Tính toán theo plan_binary_validation_sample_size, margin ±5%, confidence 95%"
report["site_count"] = 2

Path("output/validation/brain_mri_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
```

## Kiểm Tra Trạng Thái

```python
from medical.clinical_validation import clinical_validation_status

is_valid, reason = clinical_validation_status(
    report_path="output/validation/brain_mri_report.json",
    model_path="models/pretrained/cancers/brain/mri/classifier.pt",
)

if is_valid:
    print("✅ Model đã được kiểm định lâm sàng")
else:
    print(f"❌ {reason}")
```

## Checklist Trước Khi Deploy

- [ ] Cohort validation độc lập, không trùng training data
- [ ] Mỗi bệnh nhân chỉ xuất hiện 1 lần
- [ ] Có chuẩn tham chiếu từ bác sĩ chuyên khoa
- [ ] Báo cáo JSON có đầy đủ: approved_by, approved_at, dataset_provenance, ground_truth_method, sample_size_justification, site_count ≥ 1
- [ ] SHA256 hash model trong báo cáo khớp với model đang chạy
- [ ] Metrics đạt ngưỡng tối thiểu (thỏa hiệp sensitivity/specificity theo yêu cầu lâm sàng)

## Template CSV Validation

Tải template CSV tại: `data/templates/validation_template.csv`

## Lưu Ý Quan Trọng

1. **Không tự phê duyệt**: Báo cáo phải được đơn vị lâm sàng độc lập phê duyệt
2. **Hash model**: Mỗi lần update model, cần chạy lại validation
3. **Đa trung tâm**: Kết quả từ 1 bệnh viện không đủ để generalize
4. **Theo dõi liên tục**: Sau deploy, cần monitoring drift và re-validation định kỳ

## Tham Khảo

- `medical/clinical_validation.py`: Source code framework
- `medical/metrics.py`: Các hàm tính metrics
- `models/README.md`: Quy ước model và hợp đồng runtime
