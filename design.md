# YOLO Chat AI - Design Specification v2.0

## Mục tiêu

Thiết kế giao diện phải bám sát 100% theo mẫu tham chiếu gồm 4 màn hình:

* Giao diện chính Light Mode
* Giao diện chính Dark Mode
* Panel Cài đặt Light Mode
* Panel Cài đặt Dark Mode

Không tự ý thêm hoặc bớt bất kỳ thành phần nào ngoài thiết kế.

---

# 1. Layout tổng thể

Ứng dụng chia thành 3 khu vực:

```
┌──────────────┬───────────────────────────────┬──────────────────────┐
│   Sidebar    │          Main Chat            │   Settings Panel     │
│   240px      │          Flexible             │       420px          │
└──────────────┴───────────────────────────────┴──────────────────────┘
```

Khoảng cách giữa các panel:

* Gap: 16px

Bo góc:

* Border Radius: 20px

Padding:

* 24px

Font:

* Inter
* SF Pro Display
* Segoe UI

Ưu tiên Inter.

---

# 2. Sidebar

Sidebar luôn nằm bên trái.

## Chỉ giữ các thành phần sau

```
YOLO Chat AI

+ Chat mới

Tìm kiếm đoạn chat

( khoảng trống )

Cài đặt >
```

## Không được hiển thị

* Lịch sử chat
* Danh sách conversation
* Recent
* Favorites
* Folder
* YOLO Pro
* Premium
* Avatar chat
* Danh sách AI
* History
* Bất kỳ section phụ nào khác

## Header

Logo bên trái.

Tên:

```
YOLO Chat AI
```

Có nút thu gọn sidebar ở góc phải.

## Nút Chat mới

Button màu xanh.

Chiều cao khoảng:

44px

Bo góc:

12px

Gradient:

```
#2563FF -> #1D4ED8
```

Có icon dấu +.

## Search

Placeholder:

```
Tìm kiếm đoạn chat
```

Có icon Search.

Có icon filter bên phải.

Bo góc lớn.

## Footer

Chỉ còn đúng:

```
⚙ Cài đặt >
```

Không thêm bất kỳ item nào khác.

---

# 3. Main Chat

Main chat là trạng thái Empty State.

Không có hội thoại.

Không có gợi ý.

Không có bubble.

Không có prompt card.

Không có lịch sử.

## Thanh trên

Nằm góc trên bên phải.

Gồm:

```
[Sáng]

[Tối]

[Desktop ▼]
```

Button active có nền xanh.

## Tiêu đề

```
👋 Xin chào, Hữu Giàu!
```

Bên dưới:

```
Hôm nay bạn muốn hỏi điều gì?
```

Canh trái.

## Trung tâm

Robot icon.

Bên dưới:

```
Bắt đầu cuộc trò chuyện
```

Tiếp theo:

```
Đặt câu hỏi cho YOLO Chat AI để nhận câu trả lời hữu ích!
```

Canh giữa.

## Thanh nhập

```
[📎]

Nhập tin nhắn của bạn...

[🎤]

[➤]
```

Button gửi màu xanh.

Mic nằm trước nút gửi.

Input bo tròn.

## Footer

Hiển thị:

```
YOLO Chat AI có thể mắc lỗi.
Hãy kiểm tra lại thông tin quan trọng.
```

---

# 4. Settings Panel

Khi nhấn Cài đặt sẽ mở panel bên phải.

## Header

```
Cài đặt
                           ✕
```

Có nút đóng.

## Sidebar trong Settings

Chỉ có:

```
🏠 Chung
```

Không có:

* AI
* Assistant
* Appearance
* Notification
* Security
* Privacy
* Keyboard
* About
* Advanced
* Labs
* Experimental

## Nội dung

Tiêu đề:

```
Chung
```

Chỉ có đúng 2 Card.

---

# Card 1 - Giao diện

```
Giao diện

[Sáng]

[Tối]

[Hệ thống]
```

Option active:

* Border xanh
* Background xanh nhạt
* Icon xanh

Các option còn lại:

* Border xám
* Background mặc định

---

# Card 2 - Ngôn ngữ

```
Ngôn ngữ

Tiếng Việt ▼
```

Dropdown đơn giản.

Không có thêm setting nào.

---

# 5. Light Theme

Background:

```
#F7F9FD
```

Panel:

```
#FFFFFF
```

Text:

```
#0F172A
```

Secondary:

```
#64748B
```

Border:

```
#E2E8F0
```

Accent:

```
#2563FF
```

Hover:

```
#1D4ED8
```

Shadow:

```
0 12px 40px rgba(15,23,42,.06)
```

---

# 6. Dark Theme

Background:

```
#07111F
```

Panel:

```
#0F172A
```

Panel Soft:

```
#111C2E
```

Border:

```
rgba(255,255,255,.08)
```

Text:

```
#F8FAFC
```

Secondary:

```
#94A3B8
```

Input:

```
rgba(255,255,255,.06)
```

Shadow:

```
0 12px 40px rgba(0,0,0,.35)
```

---

# 7. Border Radius

Container:

20px

Card:

16px

Button:

12px

Input:

14px

Chip:

10px

---

# 8. Typography

Heading:

32px

Subheading:

18px

Normal:

15px

Small:

13px

Weight:

400

500

600

700

---

# 9. Spacing

Container:

24px

Card:

18px

Gap giữa card:

16px

Gap icon-text:

8px

Gap section:

24px

---

# 10. Animation

Hover:

150ms ease

Transition Theme:

250ms ease

Button:

scale(0.98) khi click

Không sử dụng animation phức tạp.

---

# 11. Responsive

Desktop là ưu tiên.

Sidebar:

240px

Settings:

420px

Main:

Flexible

Khi thu nhỏ:

Sidebar có thể collapse.

Settings trượt từ phải vào.

---

# 12. Quy tắc bắt buộc

## Sidebar

* Không có lịch sử chat.
* Không có conversation.
* Không có YOLO Pro.
* Không có Recent.
* Không có Favorites.

## Main Chat

* Không có gợi ý nhanh.
* Không có bubble.
* Không có prompt.
* Không có lịch sử.
* Chỉ hiển thị Empty State.

## Settings

Chỉ có:

```
Cài đặt
 └── Chung
       ├── Giao diện
       │      ├── Sáng
       │      ├── Tối
       │      └── Hệ thống
       │
       └── Ngôn ngữ
              └── Tiếng Việt
```

Không thêm bất kỳ menu hoặc setting nào khác.

---

# 13. Mục tiêu cuối cùng

Toàn bộ giao diện phải giống mẫu tham chiếu gần như 100%, bao gồm:

* Bố cục
* Khoảng cách
* Màu sắc
* Typography
* Bo góc
* Kích thước
* Icon
* Trạng thái chọn
* Light Mode
* Dark Mode

Không tự ý sáng tạo thêm thành phần mới hoặc thay đổi cấu trúc nếu không có yêu cầu.
