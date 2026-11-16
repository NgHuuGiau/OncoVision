# Build OncoVision Desktop EXE
# Tu dong build OncoVision thanh file .exe standalone.
# Su dung cx_Freeze 8.x voi setup_build.py (thay the spec file cu).
#
# Usage:
#     .\build_exe.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== OncoVision Build EXE ===" -ForegroundColor Cyan

# Kiem tra cx_Freeze
try {
    python -c "import cx_Freeze" 2>&1 | Out-Null
} catch {
    Write-Host "Loi: cx_Freeze khong co. Cai dat: pip install cx-Freeze" -ForegroundColor Red
    exit 1
}

# Kiem tra models
$modelsDir = "models\pretrained\cancers"
if (-not (Test-Path $modelsDir)) {
    Write-Host "Loi: Khong tim thay models tai $modelsDir" -ForegroundColor Red
    exit 1
}
$modelCount = (Get-ChildItem -Path $modelsDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "Tim thay $modelCount model files" -ForegroundColor Green

# Xoa build cu
Write-Host "Dọn dẹp build cũ..." -ForegroundColor Yellow
Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue

# Build
Write-Host "Build OncoVision.exe..." -ForegroundColor Yellow
Write-Host "Thoi gian uoc tinh: 30-60 phut" -ForegroundColor Yellow
python scripts/setup_build.py build_exe

if ($LASTEXITCODE -eq 0) {
    $buildDir = (Get-ChildItem -Path "build" -Directory -Filter "exe.*" | Select-Object -First 1).FullName
    $exePath = Join-Path $buildDir "OncoVision.exe"
    if (Test-Path $exePath) {
        $size = [math]::Round((Get-Item $exePath).Length / 1MB, 2)
        Write-Host ""
        Write-Host "=== Build thanh cong! ===" -ForegroundColor Green
        Write-Host "File: $exePath ($size MB)" -ForegroundColor Green
        Write-Host ""
        $totalSize = [math]::Round(((Get-ChildItem -Path $buildDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB), 2)
        Write-Host "Kich thuoc tong: $totalSize GB" -ForegroundColor Green
        Write-Host ""
        Write-Host "Chay: .\$exePath" -ForegroundColor Cyan
    }
} else {
    Write-Host "Build that bai!" -ForegroundColor Red
    exit 1
}
