<#
.SYNOPSIS
    Tạo package production OncoVision - sẵn sàng deploy
.DESCRIPTION
    Copy tất cả file cần thiết vào thư mục OncoVision_Production
    Loại bỏ file dev, cache, test
.NOTES
    Chạy từ D:\Project\OncoVision
    Kết quả: D:\Project\OncoVision_Production
#>

param(
    [string]$SourcePath = "D:\Project\OncoVision",
    [string]$DestPath = "D:\Project\OncoVision_Production"
)

Write-Host "=== ONCOVISION PRODUCTION DEPLOY ===" -ForegroundColor Cyan
Write-Host "Source: $SourcePath" -ForegroundColor Gray
Write-Host "Dest:   $DestPath" -ForegroundColor Gray
Write-Host ""

# Xóa destination cũ
if (Test-Path $DestPath) {
    Write-Host "Xóa thư mục cũ..." -ForegroundColor Yellow
    Remove-Item -Path $DestPath -Recurse -Force -ErrorAction SilentlyContinue
}

# Tạo thư mục đích
New-Item -Path $DestPath -ItemType Directory -Force | Out-Null
