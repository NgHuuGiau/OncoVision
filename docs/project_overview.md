# Tổng quan kiến trúc OncoVision

OncoVision là ứng dụng suy luận ảnh y khoa. Luồng ứng dụng nhận ảnh từ người dùng, kiểm tra định dạng/chất lượng, chọn model suy luận có sẵn, tạo kết quả sàng lọc và lưu hồ sơ ca bệnh. **Ứng dụng không xử lý bộ dữ liệu huấn luyện và không train model.** Model mới được chuẩn bị bên ngoài rồi bổ sung vào vị trí cấu hình.

## Thành phần chính

| Thành phần | Vai trò |
|---|---|
| `app/chat_ui/` | Giao diện desktop, tải ảnh, lịch sử chat/ca |
| `web_app.py`, `templates/` | Giao diện web upload ảnh và xem kết quả |
| `medical/validator.py` | Kiểm tra đầu vào, modality và vùng cơ thể |
| `medical/dataset.py` | Đọc metadata/volume y khoa và chuẩn hóa ảnh đầu vào; không quản lý train/val/test |
| `medical/pipeline.py` | Điều phối suy luận, giải thích và tạo kết quả |
| `medical/cnn_classifier.py` | Nạp model CNN và chạy dự đoán |
| `medical/explainability.py` | Grad-CAM/heatmap khi model hỗ trợ |
| `medical/reporting.py` | Báo cáo JSON/Markdown/HTML và xuất hồ sơ ca |
| `medical/storage.py` | Lưu lịch sử ca bệnh trong SQLite |
| `medical/cancer_catalog.py` | Nhóm bệnh, modality và trạng thái model hỗ trợ |
| `models/pretrained/` | Model suy luận có sẵn; không chứa pipeline huấn luyện |

## Luồng xử lý

```text
Ảnh người dùng
  → kiểm tra định dạng, metadata, chất lượng
  → chọn loại ảnh/vùng cơ thể
  → nạp model tương ứng từ cấu hình
  → phân tích, độ tin cậy, giải thích và khuyến nghị
  → lưu báo cáo và lịch sử ca
```

Nếu không có model phù hợp, hệ thống báo chưa hỗ trợ thay vì suy luận bằng model sai loại. Model não hiện có thể dùng độc lập; các nhóm ung thư khác cần model suy luận tương ứng được cung cấp sau.

## Entrypoint

| Lệnh | Vai trò |
|---|---|
| `python run_chat.py` | Mở giao diện desktop |
| `python web_app.py` | Mở giao diện web |
| `python run_medical.py analyze --image ... --patient-code ...` | Phân tích ảnh bằng CLI |
| `python run_medical.py status` | Kiểm tra model và số ca đã lưu |
| `python run_doctor.py --skip-camera-check` | Kiểm tra môi trường, model và cấu hình |
| `python run_smoke.py --ci-safe --stop-on-fail` | Smoke test an toàn cho CI |

## Dữ liệu và an toàn

- Ảnh upload được chuẩn hóa vào thư mục output; báo cáo và lịch sử nằm trong `output/`.
- Ảnh y tế có thể chứa thông tin định danh; chỉ dùng dữ liệu được phép và bảo vệ thư mục output.
- Đây là hỗ trợ sàng lọc, không thay thế chẩn đoán của bác sĩ hay xét nghiệm xác nhận.
