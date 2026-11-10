# Create OncoVision distribution package
$sourceDir = "D:\Project\OncoVision"
$distDir = "D:\Project\OncoVision\dist\OncoVision"
$launcherBuild = "D:\Project\OncoVision\dist\OncoVisionLauncher"

Write-Host "Creating distribution package..." -ForegroundColor Cyan

# Clean up
if (Test-Path $distDir) {
    Remove-Item -Path $distDir -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -Path $distDir -ItemType Directory -Force | Out-Null

# Copy Python runtime from launcher build
Write-Host "Copying Python runtime..." -ForegroundColor Yellow
Copy-Item -Path "$launcherBuild\_internal\*" -Destination $distDir -Recurse -Force -ErrorAction SilentlyContinue

# Copy application files
Write-Host "Copying application files..." -ForegroundColor Yellow
$appDirs = @("app", "medical", "core", "utils", "config", "assets", "models", "static", "templates", "tests")
foreach ($dir in $appDirs) {
    if (Test-Path "$sourceDir\$dir") {
        Copy-Item -Path "$sourceDir\$dir" -Destination "$distDir\$dir" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  Copied $dir" -ForegroundColor Gray
    }
}

# Copy entry point scripts
Write-Host "Copying entry point scripts..." -ForegroundColor Yellow
$scripts = @("run_chat.py", "run_doctor.py", "run_medical.py", "run_app.py", "run_menu.py", "run_smoke.py", "run_tests.py", "web_app.py", "requirements.txt")
foreach ($script in $scripts) {
    if (Test-Path "$sourceDir\$script") {
        Copy-Item -Path "$sourceDir\$script" -Destination "$distDir\$script" -Force -ErrorAction SilentlyContinue
    }
}

# Copy updated launcher.py
Write-Host "Copying updated launcher..." -ForegroundColor Yellow
Copy-Item -Path "$sourceDir\launcher.py" -Destination "$distDir\launcher.py" -Force -ErrorAction SilentlyContinue

# Rename the launcher EXE to OncoVision.exe
Write-Host "Renaming EXE..." -ForegroundColor Yellow
if (Test-Path "$distDir\OncoVisionLauncher.exe") {
    Rename-Item -Path "$distDir\OncoVisionLauncher.exe" -NewName "OncoVision.exe" -Force -ErrorAction SilentlyContinue
}

Write-Host "Distribution package created at $distDir" -ForegroundColor Green
Write-Host "Contents:" -ForegroundColor Cyan
Get-ChildItem -Path $distDir | Select-Object Name, Length | Format-Table -AutoSize