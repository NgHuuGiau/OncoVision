<#
.SYNOPSIS
    Dọn dẹp thư mục gốc OncoVision - sắp xếp file code chính
.DESCRIPTION
    Di chuyển file test/debug vào thư mục tests/
    Di chuyển script build vào scripts/
    Xóa file rác, cache, log
.NOTES
    Chạy từ D:\Project\OncoVision
#>

Write-Host "=== ONCOVISION MAIN DIRECTORY CLEANUP ===" -ForegroundColor Cyan
Write-Host ""

# 1. Tạo thư mục tests nếu chưa có
$testDir = "D:\Project\OncoVision\tests\manual"
if (-not (Test-Path $testDir)) {
    New-Item -Path $testDir -ItemType Directory -Force | Out-Null
    Write-Host "Created $testDir" -ForegroundColor Green
}

# 2. Di chuyển các file test/debug thủ công vào tests/manual
$manualTestFiles = @(
    "test_*.py",
    "test_*.txt",
    "debug_*.py",
    "check_*.py",
    "auto_open_downloads.py",
    "clear_login_attempts.py",
    "create_manual_guide.py",
    "create_part1.py",
    "fix_demo_mode.py",
    "test_desktop_import.py",
    "test_e2e.py",
    "test_final2.py",
    "test_production.py",
    "test_prod_simple.py",
    "test_web.py",
    "test_web_comprehensive.py",
    "test_web_manual.py",
    "test_web_manual_full.py",
    "run_app.py",
    "run_medical.py",
    "run_menu.py",
    "run_smoke.py",
    "run_tests.py",
    "set_demo_false.py",
    "check_password.py",
    "check_users.py",
    "get_modalities.py",
    "set_demo_false.py"
)

foreach ($pattern in $manualTestFiles) {
    Get-ChildItem "D:\Project\OncoVision" -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
        $dest = Join-Path "D:\Project\OncoVision\tests\manual" $_.Name
        Write-Host "Moving $($_.Name) -> tests\manual\" -ForegroundColor Yellow
        Move-Item $_.FullName -Destination $dest -Force -ErrorAction SilentlyContinue
    }
}

# 3. Di chuyển script build vào scripts/
$buildScripts = @("build_exe*.bat", "create_recovery_branch.ps1", "run_recovery.ps1", "organize_models.py", "download_models.py")
foreach ($pattern in $buildScripts) {
    Get-ChildItem "D:\Project\OncoVision" -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
        $dest = Join-Path "D:\Project\OncoVision\scripts" $_.Name
        Write-Host "Moving $($_.Name) -> scripts\" -ForegroundColor Yellow
        Move-Item $_.FullName -Destination $dest -Force -ErrorAction SilentlyContinue
    }
}

# 4. Xóa file rác, cache, log
$toRemove = @(
    "*.log", "*.tmp", "*.zip", "*.cache", "__pycache__", 
    ".pytest_cache", ".mypy_cache", ".ruff_cache", 
    "build_log.txt", "kidney.zip", "pancreas.zip"
)

foreach ($pattern in $toRemove) {
    Get-ChildItem "D:\Project\OncoVision" -Filter $pattern -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) {
            Write-Host "Removing directory: $($_.FullName)" -ForegroundColor Red
            Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Write-Host "Removing file: $($_.FullName)" -ForegroundColor Red
            Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
        }
    }
}

# 5. Xóa thư mục build cũ, runs, output, bundles, docs, data
$dirsToRemove = @("build", "runs", "output", "bundles", "docs", "data", "dist", ".freebuff")
foreach ($dir in $dirsToRemove) {
    $path = "D:\Project\OncoVision\$dir"
    if (Test-Path $path) {
        Write-Host "Removing directory: $path" -ForegroundColor Red
        Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 6. Giữ lại file quan trọng ở root
Write-Host "" 
Write-Host "=== FILES KEPT AT ROOT ===" -ForegroundColor Cyan
$keepFiles = @(
    "web_app.py", "run_chat.py", "run_doctor.py", "run_recovery.ps1",
    "OncoVisionChat.spec", "requirements.txt", "requirements-web.txt",
    "pytest.ini", "ruff.toml", "Makefile", "LICENSE", "README.md",
    "docker-compose.yml", "Dockerfile", "Dockerfile.monai",
    ".gitignore", ".gitattributes", ".dockerignore", ".env.example",
    "DOWNLOAD_GUIDE.md", "MONAI_MODELS_SUMMARY.md", "MANUAL_DOWNLOAD_GUIDE.txt",
    "deploy_production.ps1"
)

foreach ($f in $keepFiles) {
    if (Test-Path "D:\Project\OncoVision\$f") {
        Write-Host "  ✓ $f" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $f (missing)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "=== CLEANUP COMPLETE ===" -ForegroundColor Cyan