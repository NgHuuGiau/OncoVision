# Đăng nhập và phân quyền web

Web app không có tài khoản mặc định và không mở đăng ký công khai. Tài khoản được lưu trong `output/onco.db`, bảng `web_users`. Khi khởi chạy server lần đầu, ứng dụng tự tạo bảng nhưng không tự tạo admin.

## Tạo admin đầu tiên bằng SQL

1. Khởi chạy web app một lần để tạo `output/onco.db` và bảng `web_users`.
2. Tạo hash mật khẩu an toàn bằng lệnh sau; mật khẩu được nhập ẩn và phải dài ít nhất 12 ký tự:

   ```powershell
   python -m app.web_auth hash-password
   ```

3. Mở `output/onco.db` bằng DB Browser for SQLite hoặc công cụ SQLite bạn đang dùng, rồi chạy SQL sau. Thay `<HASH_VUA_TAO>` và email admin đã xác minh:

   ```sql
   INSERT INTO web_users (username, password_hash, email, role, is_active)
   VALUES ('admin', '<HASH_VUA_TAO>', 'admin@example.com', 'admin', 1);
   ```

4. Đăng nhập tại `/login`. Nếu tên `admin` đã tồn tại, hãy chọn username khác hoặc cập nhật bản ghi hiện có.
5. Khởi chạy web với cấu hình Gmail SMTP bên dưới. Email admin cần là hộp thư bạn có thể truy cập.

Không lưu mật khẩu dạng chữ thường trong SQL. Hash dùng PBKDF2-HMAC-SHA256 với salt ngẫu nhiên.

## Vai trò

| Vai trò | Quyền |
|---|---|
| `admin` | Toàn quyền, gồm tạo/đổi vai trò/khóa tài khoản tại `/admin/users` |
| `clinician` | Nhân viên y tế: xem, tải ảnh, phân tích, quản lý hội thoại và hồ sơ |
| `viewer` | Chỉ xem hồ sơ, hội thoại và báo cáo; không được tải ảnh/phân tích/ghi dữ liệu bệnh nhân. Có thể đổi giao diện riêng trên trình duyệt |

Admin tạo tài khoản và gán email khôi phục trong trang **Quản lý tài khoản**. Khi quên mật khẩu, người dùng nhập username và email đã đăng ký; hệ thống gửi mã 6 ký tự đến email đó. Mã được lưu dưới dạng hash, dùng một lần, hết hạn sau 10 phút và không hiển thị trên trang. Email phải được xác minh với người dùng trước khi lưu. Tài khoản SQL cũ chưa có email cần được cập nhật trong trang quản trị trước khi khôi phục. Mã khôi phục cũ không có thời hạn bị vô hiệu hóa khi cập nhật cơ sở dữ liệu.

### Cấu hình gửi email bằng Gmail

Tạo App Password trong tài khoản Gmail dùng làm địa chỉ gửi; không dùng mật khẩu Gmail thường. Đặt các biến môi trường sau trong cùng cửa sổ PowerShell trước khi chạy ứng dụng:

```powershell
$env:ONCOVISION_SMTP_HOST = "smtp.gmail.com"
$env:ONCOVISION_SMTP_PORT = "587"
$env:ONCOVISION_SMTP_USERNAME = "sender@gmail.com"
$env:ONCOVISION_SMTP_PASSWORD = "app-password"
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Thay giá trị mẫu bằng thông tin thật. Không gửi App Password cho người khác, không ghi vào mã nguồn và không commit vào Git. Email người nhận là email khôi phục đã lưu ở tài khoản; địa chỉ Gmail cấu hình SMTP là địa chỉ gửi. Nếu SMTP chưa cấu hình hoặc gửi thất bại, giao diện vẫn trả phản hồi chung để không tiết lộ tài khoản, còn máy chủ ghi lỗi để admin kiểm tra.

Sau 5 lần nhập mã sai theo username/IP, thao tác khôi phục bị khóa 15 phút; mỗi IP chỉ được yêu cầu gửi một email khôi phục mỗi phút.

Hệ thống không cho khóa hoặc hạ quyền admin cuối cùng. Mọi phiên đều kiểm tra trạng thái tài khoản hiện tại; khóa tài khoản sẽ thu hồi quyền ở request kế tiếp.

Phân quyền hiện áp dụng ở mức vai trò cho toàn bộ workspace: các tài khoản `clinician`/`viewer` cùng được xem các ca trong database này. Chưa có phân vùng hồ sơ theo bác sĩ, khoa hoặc cơ sở.

`viewer` chưa phải cổng tra cứu riêng theo mã bệnh nhân: họ chỉ xem dữ liệu chung của workspace. Ô tìm kiếm hiện tại lọc lịch sử hội thoại trên giao diện, không cấp quyền truy cập riêng cho một hồ sơ. Không dùng mã bệnh nhân dễ đoán như mật khẩu tra cứu.

## Triển khai trên máy chủ

- Mặc định ứng dụng chỉ bind `127.0.0.1`. Không đổi sang `0.0.0.0` nếu chưa đặt HTTPS/TLS, giới hạn truy cập mạng và cấu hình bảo vệ dữ liệu phù hợp.
- Đặt `ONCOVISION_SESSION_SECRET` thành secret ngẫu nhiên, ổn định qua các lần khởi động và không đưa vào Git/log. Nếu không đặt, ứng dụng tạo secret ngẫu nhiên cho tiến trình hiện tại; phiên đăng nhập sẽ hết hiệu lực khi server khởi động lại.
- Khi chạy sau HTTPS, đặt `ONCOVISION_COOKIE_SECURE=1` để cookie chỉ gửi qua HTTPS.
- Cookie phiên có `HttpOnly`, `SameSite=Lax`, hết hạn sau 8 giờ; request thay đổi dữ liệu cần CSRF token. Sau 5 lần đăng nhập sai liên tiếp theo username/IP, đăng nhập bị khóa 15 phút.
- API, ca bệnh, hội thoại, ảnh và tệp trong `/output` đều yêu cầu đăng nhập. Swagger/OpenAPI bị tắt trên web app.

Đăng nhập không thay thế mã hóa ổ đĩa/database, backup an toàn, TLS hay quy trình quản lý dữ liệu bệnh nhân khi triển khai thật.
