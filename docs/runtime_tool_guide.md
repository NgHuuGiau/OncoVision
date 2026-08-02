# Hướng Dẫn Runtime Advisor

Tài liệu giải thích công cụ gợi ý runtime camera — cách hoạt động, đọc kết quả và chọn chế độ phù hợp.

---

## 1. Giới thiệu

`run_app.py --advisor-only` là chế độ phân tích hệ thống không mở webcam, giúp trả lời bốn câu hỏi:

1. Máy hiện tại đang mạnh đến đâu?
2. Có GPU / CUDA sẵn sàng không?
3. Nên ưu tiên model nào?
4. Nên chọn chế độ `high`, `medium`, `low` hay `auto`?

---

## 2. Chế độ runtime

| Chế độ | Phù hợp khi |
|---|---|
| **high** | Máy có GPU tốt, ưu tiên độ chính xác, chấp nhận tải cao |
| **medium** | Cân bằng FPS và độ chính xác — an toàn cho đa số máy dev |
| **low** | Máy yếu, chạy CPU, cần ưu tiên tốc độ và ổn định |
| **auto** | Cơ chế tự chọn dựa trên CUDA, VRAM, model sẵn có |

---

## 3. Cách đọc kết quả

Kết quả mẫu:

```
medium: model=yolo11s.pt, device=cuda:0, imgsz=512, max_det=120
```

Ý nghĩa:

- **medium**: chế độ khởi động phù hợp
- **yolo11s.pt**: model được đề xuất
- **cuda:0**: chạy trên GPU
- **imgsz=512**: kích thước ảnh đầu vào
- **max_det=120**: giới hạn detection mỗi frame

---

## 4. Sử dụng

### Quy trình khuyến nghị

```powershell
python run_app.py --advisor-only
python run_doctor.py --skip-camera-check
python run_app.py --mode medium
```

### Với model custom

```powershell
python run_app.py --advisor-only
python run_app.py --model models/trained/best.pt --mode medium
```

---

## 5. Module liên quan

Runtime advisor phụ thuộc vào:

- `core/hardware_info.py` — đọc cấu hình máy
- `core/runtime_advisor.py` — logic gợi ý
- `app/camera_runtime/bootstrap.py` — khởi tạo runtime
- `config/settings.yaml` — cấu hình mặc định

---

## 6. Khi nào nên chạy

- Máy mới vừa cài repo
- Vừa thay GPU, driver hoặc torch
- Camera realtime đang bị lag
- Trước khi demo trên máy lạ

---

## 7. Vấn đề thường gặp

### Advisor báo CUDA nhưng runtime chậm

Nguyên nhân: model quá nặng, `imgsz` quá cao, GPU bị app khác chiếm.

Khắc phục: thử `--mode low` hoặc `--mode medium`.

### Kết quả nhận diện kém

Advisor chỉ gợi ý cấu hình — không đánh giá chất lượng model. Nếu model kém, cần xem lại dataset và quy trình train.
