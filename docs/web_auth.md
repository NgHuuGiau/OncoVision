# Đăng nhập và phân quyền web

Web app không có tài khoản mặc định và không mở đăng ký công khai. Tài khoản được lưu trong `output/onco.db`, bảng `web_users`. Khi khởi chạy server lần đầu, ứng dụng tự tạo bảng nhưng không tự tạo admin.

## Tạo admin đầu tiên bằng SQL

1. Khởi chạy web app một lần để tạo `output/onco.db` và bảng `web_users`.
2. Tạo hash mật khẩu an toàn bằng lệnh sau; mật khẩu được nhập ẩn và phải dài ít nhất 12 ký tự:

   ```powershell
   python -m app.web_auth hash-password
   ```

3. Mở `output/onco.db` bằng DB Browser for SQLite hoặc công cụ SQLite bạn đang dùng, rồi chạy SQL sau. Thay `<HASH_VUA_TAO>` bằng chuỗi hash từ bước 2:

   ```sql
   INSERT INTO web_users (username, password_hash, role, is_active)
   VALUES ('admin', '<HASH_VUA_TAO>', 'admin', 1);
   ```

4. Đăng nhập tại `/login`. Nếu tên `admin` đã tồn tại, hãy chọn username khác hoặc cập nhật bản ghi hiện có.

Không lưu mật khẩu dạng chữ thường trong SQL. Hash dùng PBKDF2-HMAC-SHA256 với salt ngẫu nhiên.

## Vai trò

| Vai trò | Quyền |
|---|---|
| `admin` | Toàn quyền, gồm tạo/đổi vai trò/khóa tài khoản tại `/admin/users` |
| `clinician` | Xem, tải ảnh, phân tích, quản lý hội thoại và hồ sơ |
| `viewer` | Chỉ xem hồ sơ, hội thoại và báo cáo; không được tải ảnh/phân tích/ghi dữ liệu bệnh nhân. Có thể đổi giao diện riêng trên trình duyệt |

Admin tạo tài khoản sau này ngay trong trang **Quản lý tài khoản**. Hệ thống không cho khóa hoặc hạ quyền admin cuối cùng. Mọi phiên đều kiểm tra trạng thái tài khoản hiện tại; khóa tài khoản sẽ thu hồi quyền ở request kế tiếp.

Phân quyền hiện áp dụng ở mức vai trò cho toàn bộ workspace: các tài khoản `clinician`/`viewer` cùng được xem các ca trong database này. Chưa có phân vùng hồ sơ theo bác sĩ, khoa hoặc cơ sở.

## Triển khai trên máy chủ

- Mặc định ứng dụng chỉ bind `127.0.0.1`. Không đổi sang `0.0.0.0` nếu chưa đặt HTTPS/TLS, giới hạn truy cập mạng và cấu hình bảo vệ dữ liệu phù hợp.
- Đặt `ONCOVISION_SESSION_SECRET` thành secret ngẫu nhiên, ổn định qua các lần khởi động và không đưa vào Git/log. Nếu không đặt, ứng dụng tạo secret ngẫu nhiên cho tiến trình hiện tại; phiên đăng nhập sẽ hết hiệu lực khi server khởi động lại.
- Khi chạy sau HTTPS, đặt `ONCOVISION_COOKIE_SECURE=1` để cookie chỉ gửi qua HTTPS.
- Cookie phiên có `HttpOnly`, `SameSite=Lax`, hết hạn sau 8 giờ; request thay đổi dữ liệu cần CSRF token. Sau 5 lần đăng nhập sai liên tiếp theo username/IP, đăng nhập bị khóa 15 phút.
- API, ca bệnh, hội thoại, ảnh và tệp trong `/output` đều yêu cầu đăng nhập. Swagger/OpenAPI bị tắt trên web app.

Đăng nhập không thay thế mã hóa ổ đĩa/database, backup an toàn, TLS hay quy trình quản lý dữ liệu bệnh nhân khi triển khai thật.
