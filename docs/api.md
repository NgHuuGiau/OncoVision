# OncoVision API Documentation

## Overview

OncoVision provides a REST API for medical image analysis via FastAPI. The API is accessible at `/api/v1/` endpoints.

**Base URL**: `http://localhost:8000` (development)

**Authentication**: Session-based with CSRF protection. Most endpoints require login.

## Authentication Endpoints

### POST /login
Login with username and password.
- **Form data**: `username`, `password`, `csrf_token`
- **Response**: Redirect to `/` on success, error on failure

### POST /logout
Logout current user. Clears session.

### GET /forgot-password
Show password recovery form.

### POST /forgot-password/request
Request password recovery email.
- **Form data**: `email`, `csrf_token`

### POST /forgot-password
Reset password with recovery code.
- **Form data**: `recovery_code`, `new_password`, `confirm_password`, `csrf_token`

## Medical Analysis Endpoints

### POST /api/v1/analyze
Upload and analyze a medical image.
- **Content-Type**: `multipart/form-data`
- **Form data**:
  - `file`: Image file (JPG, PNG, DICOM, NIfTI)
  - `target_key`: Cancer target (e.g., `brain`, `liver`, `lung`)
  - `csrf_token`: CSRF token from session
- **Response**: JSON with analysis results
- **Max file size**: 200MB

### GET /api/v1/status
Get medical system status (model availability, hardware info).
- **Response**: JSON with system status

### GET /api/v1/cancers
List available cancer screening targets.
- **Response**: JSON list of cancer targets with modality info

### GET /api/v1/cases
List medical cases for authenticated user.
- **Response**: JSON list of cases

### GET /api/v1/cases/{case_id}
Get a specific case by ID.
- **Response**: JSON with case details

### GET /api/v1/cases/{case_id}/pdf
Export case report as PDF.
- **Response**: PDF file

### GET /api/v1/cases/{case_id}/export
Export case data as JSON.
- **Response**: JSON with case data

### DELETE /api/v1/cases/{case_id}
Delete a case.
- **Response**: 200 on success

## Chat Endpoints

### POST /api/v1/chat
Send a message to the medical chat service.
- **Content-Type**: `application/json`
- **Body**: `{"message": "your message", "image_path": "optional/path"}`
- **Response**: JSON with chat response

### GET /api/v1/chat/history
Get chat history for authenticated user.
- **Query params**: `limit` (default 50), `offset` (default 0)
- **Response**: JSON list of chat messages

### DELETE /api/v1/chat/history
Delete chat history.
- **Response**: 200 on success

## Health & System

### GET /healthz
Health check endpoint. No authentication required.
- **Response**: `{"status": "ok"}`

### GET /openapi.json
OpenAPI schema (development only).

### GET /docs
Swagger UI (development only).

### GET /redoc
ReDoc UI (development only).

## Error Responses

| Status Code | Meaning |
|---|---|
| 400 | Bad request (invalid file, missing params) |
| 401 | Unauthorized (not logged in) |
| 403 | Forbidden (CSRF token mismatch) |
| 404 | Not found |
| 413 | File too large (>200MB) |
| 500 | Internal server error |

## Rate Limiting

- Max concurrent analyses: 1 (configurable via `ONCOVISION_MAX_CONCURRENT_ANALYSIS`)
- Max upload size: 200MB
- Max form size: 1MB

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ONCOVISION_ENV` | `development` | Environment (development/production) |
| `ONCOVISION_MAX_CONCURRENT_ANALYSIS` | `1` | Max concurrent analysis threads |
| `ONCOVISION_ALLOW_DEFAULT_ADMIN` | `` | Allow default admin in production |
| `ONCOVISION_LOG_FORMAT` | `` | Log format (`json` for structured logging) |
| `ONCOVISION_DB_PATH` | `output/onco.db` | Database path |

## CSRF Protection

Tất cả các yêu cầu POST/PUT/DELETE đều yêu cầu CSRF token để bảo vệ chống lại tấn công Cross-Site Request Forgery.

### Cách lấy CSRF token

1. **GET /login** — Trang đăng nhập trả về CSRF token trong form hidden field:
   ```html
   <input type="hidden" name="csrf_token" value="...">
   ```

2. **Sau khi đăng nhập thành công** — Trang chủ (`/`) trả về CSRF token trong meta tag:
   ```html
   <meta name="csrf-token" content="...">
   ```

### Cách sử dụng CSRF token

**Cách 1: Truyền qua header `X-CSRF-Token`** (khuyến nghị cho API calls):
```python
import requests

session = requests.Session()

# Bước 1: Lấy CSRF token từ trang login
login_page = session.get("http://localhost:8000/login")
import re
csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text)
csrf_token = csrf_match.group(1)

# Bước 2: Đăng nhập với CSRF token
session.post("http://localhost:8000/login", data={
    "username": "admin",
    "password": "Admin@123456",
    "csrf_token": csrf_token
})

# Bước 3: Lấy CSRF token mới sau khi đăng nhập
main_page = session.get("http://localhost:8000/")
csrf_match = re.search(r'name="csrf-token" content="([^"]+)"', main_page.text)
csrf_token = csrf_match.group(1)

# Bước 4: Upload file với CSRF token trong header
with open("scan.png", "rb") as f:
    resp = session.post(
        "http://localhost:8000/api/upload",
        files={"file": ("scan.png", f, "image/png")},
        headers={"X-CSRF-Token": csrf_token}
    )
    print(resp.json())

# Bước 5: Analyze với CSRF token
resp = session.post(
    "http://localhost:8000/api/analyze",
    data={
        "image_path": "output/web_uploads/...",
        "target_key": "lung",
        "csrf_token": csrf_token
    },
    headers={"X-CSRF-Token": csrf_token}
)
print(resp.json())
```

**Cách 2: Truyền qua form field `csrf_token`**:
```python
session.post(
    "http://localhost:8000/api/conversations",
    data={"csrf_token": csrf_token}
)
```

### Lưu ý quan trọng

- CSRF token được lưu trong session cookie và thay đổi sau mỗi lần đăng nhập.
- Sau khi đăng nhập thành công, luôn lấy lại CSRF token mới từ trang chủ.
- Một số endpoint đặc biệt (`/logout`, `/admin/users/create`, `/admin/users/{id}/update`) cho phép truyền CSRF token qua form field thay vì header.
- Nếu nhận được lỗi `403 Forbidden` với thông báo "CSRF token không hợp lệ", hãy lấy lại token mới.

## Example Usage

```python
import requests
import re

session = requests.Session()
base = "http://localhost:8000"

# Bước 1: Lấy CSRF token từ trang login
login_page = session.get(f"{base}/login")
csrf_match = re.search(r'name="csrf_token" value="([^"]+)"', login_page.text)
csrf_token = csrf_match.group(1)

# Bước 2: Đăng nhập
session.post(f"{base}/login", data={
    "username": "admin",
    "password": "Admin@123456",
    "csrf_token": csrf_token
})

# Bước 3: Lấy CSRF token mới sau đăng nhập
main_page = session.get(f"{base}/")
csrf_match = re.search(r'name="csrf-token" content="([^"]+)"', main_page.text)
csrf_token = csrf_match.group(1)

# Bước 4: Upload ảnh
with open("scan.png", "rb") as f:
    resp = session.post(
        f"{base}/api/upload",
        files={"file": ("scan.png", f, "image/png")},
        headers={"X-CSRF-Token": csrf_token}
    )
    upload = resp.json()
    print(f"Upload ID: {upload['upload_id']}")

# Bước 5: Phân tích
resp = session.post(
    f"{base}/api/analyze",
    data={
        "image_path": upload["stored_path"],
        "target_key": "lung",
        "csrf_token": csrf_token
    },
    headers={"X-CSRF-Token": csrf_token}
)
print(resp.json())
```
