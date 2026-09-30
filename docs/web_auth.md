# Đăng nhập và phân quyền web

Web app không có tài khoản mặc định và không mở đăng ký công khai. Tài khoản được lưu trong `output/onco.db`, bảng `web_users`. Khi khởi chạy server lần đầu, ứng dụng tự tạo bảng nhưng không tự tạo admin.

## Tạo admin đầu tiên bằng CLI

1. Khởi chạy web app một lần để tạo `output/onco.db` và bảng `web_users` (hoặc CLI tự tạo).
2. Chạy lệnh sau; mật khẩu nhập ẩn, ít nhất 12 ký tự:

   ```powershell
   python -m app.web_auth create-admin --username admin --email admin@example.com
   ```

3. Đăng nhập tại `/login`. Tên đã tồn tại thì chọn username khác.

Muốn hash rời để chèn tay thì dùng `python -m app.web_auth hash-password`. Không lưu mật khẩu dạng chữ thường. Hash dùng PBKDF2-HMAC-SHA256 với salt ngẫu nhiên.

## Vai trò

| Vai trò | Quyền |
|---|---|
| `admin` | Tạo/đổi vai trò/khóa tài khoản; tải dữ liệu, chạy phân tích, phân công ca, **duyệt kết quả nhân viên gửi và gán ca cho tài khoản người dùng** |
| `clinician` | Chỉ xem ca được giao; kiểm tra ảnh/kết quả, chỉnh mức nguy cơ và khuyến nghị, rồi **gửi admin duyệt** (chưa phát hành) |
| `viewer` | Xem **danh sách ca được gán cho mình** + tải PDF; vẫn tra cứu được bằng mã 10 ký tự; không xem ca người khác, hội thoại hoặc ảnh gốc chưa duyệt |

Admin tạo tài khoản và gán email khôi phục trong trang **Quản lý tài khoản**. Khi quên mật khẩu, người dùng chỉ nhập username; hệ thống tự tìm email đã lưu và gửi mã 6 ký tự đến đó. Sau khi nhận thư, người dùng nhập mã và mật khẩu mới, không cần nhập lại username/email. Mã được lưu dưới dạng hash, dùng một lần, hết hạn sau 10 phút và không hiển thị trên trang. Email phải được xác minh với người dùng trước khi lưu. Tài khoản SQL cũ chưa có email cần được cập nhật trong trang quản trị trước khi khôi phục. Mã khôi phục cũ không có thời hạn bị vô hiệu hóa khi cập nhật cơ sở dữ liệu.

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

### Luồng xử lý ca bệnh

1. Admin tải ảnh/dữ liệu lên, chạy phân tích và giao ca cho một tài khoản `clinician` đang hoạt động.
2. Nhân viên chỉ thấy các ca được giao cho chính tài khoản đó. Họ rà soát ảnh và kết quả AI, chỉnh mức nguy cơ/khuyến nghị nếu cần, rồi bấm **Lưu và gửi admin duyệt** (chưa phát hành cho người dùng).
3. Admin duyệt ca đã gửi trong modal **Phân công ca bệnh** (nút Duyệt & phát hành), hệ thống tạo mã tra cứu 10 ký tự; sau đó admin **gán ca cho tài khoản `viewer`** của người bệnh. Chuyển ca sang nhân viên khác sẽ đưa ca về chờ duyệt, vô hiệu mã cũ và gỡ link tài khoản.
4. Người dùng đăng nhập bằng vai trò `viewer` xem mục **Ca của tôi** để đọc khuyến nghị và tải PDF (vẫn tra được bằng mã). Tệp PDF dành cho người đọc không chứa đường dẫn nội bộ hay ảnh gốc.

API cũng kiểm tra vai trò và quyền sở hữu ca; ẩn nút trên giao diện không phải biện pháp phân quyền. Mã tra cứu là thông tin cần giữ kín và chỉ gửi cho đúng người nhận. Ứng dụng hiện vẫn phù hợp chạy localhost; trước khi cho nhiều máy truy cập cần bổ sung triển khai HTTPS, sao lưu và quy trình bảo vệ dữ liệu bệnh nhân.

## Triển khai trên máy chủ

- Mặc định ứng dụng chỉ bind `127.0.0.1`. Không đổi sang `0.0.0.0` nếu chưa đặt HTTPS/TLS, giới hạn truy cập mạng và cấu hình bảo vệ dữ liệu phù hợp.
- Đặt `ONCOVISION_SESSION_SECRET` thành secret ngẫu nhiên, ổn định qua các lần khởi động và không đưa vào Git/log. Nếu không đặt, ứng dụng tạo secret ngẫu nhiên cho tiến trình hiện tại; phiên đăng nhập sẽ hết hiệu lực khi server khởi động lại.
- Khi chạy sau HTTPS, đặt `ONCOVISION_COOKIE_SECURE=1` để cookie chỉ gửi qua HTTPS.
- Cookie phiên có `HttpOnly`, `SameSite=Lax`, hết hạn sau 8 giờ; request thay đổi dữ liệu cần CSRF token. Sau 5 lần đăng nhập sai liên tiếp theo username/IP, đăng nhập bị khóa 15 phút.
- API, ca bệnh, hội thoại, ảnh và tệp trong `/output` đều yêu cầu đăng nhập. Swagger/OpenAPI bị tắt trên web app.

Đăng nhập không thay thế mã hóa ổ đĩa/database, backup an toàn, TLS hay quy trình quản lý dữ liệu bệnh nhân khi triển khai thật.
