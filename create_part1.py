content = '''<#>
.SYNOPSIS
    OncoVision Setup - Part 1: 3D Models & Colorectal
.DESCRIPTION
    Run as Administrator. Downloads 3D models for Pancreas, Kidney and Colorectal detector.
    Run from D:\Project\OncoVision directory as Administrator.
#>

$ErrorActionPreference = 'Stop'

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "ONCOVISION SETUP - PART 1: 3D MODELS & COLORECTAL" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Run as Administrator!" -ForegroundColor Yellow
Write-Host ""

# Cleanup
Write-Host "`n=== CLEANUP ===" -ForegroundColor Cyan
Write-Host "Cleaning up..." -ForegroundColor Yellow
Get-Process -Name "powershell" -ErrorAction SilentlyContinue | Where-Object {$_.Id -ne $PID} | Stop-Process -Force -ErrorAction SilentlyContinue
Remove-Item -Force -ErrorAction SilentlyContinue pancreas.zip, temp_pancreas, kidney.zip, temp_kidney 2>$null
Write-Host "✅ Cleanup done" -ForegroundColor Green

# PANCREAS 3D
Write-Host "`n=== PANCREAS 3D (~1GB) ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "models/pretrained/cancers/pancreas" | Out-Null
Write-Host "Downloading Pancreas 3D (~1GB)..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/pancreas_ct_dints_segmentation_v0.4.3.zip" -OutFile "pancreas.zip" -TimeoutSec 1800
Expand-Archive -Path "pancreas.zip" -DestinationPath "temp_pancreas" -Force
Copy-Item "temp_pancreas/pancreas_ct_dints_segmentation/models/model.pt" "models/pretrained/cancers/pancreas/volume.pt" -Force
Copy-Item "temp_pancreas/pancreas_ct_dints_segmentation/configs/inference.json" "models/pretrained/cancers/pancreas/volume.json" -Force
Remove-Item "pancreas.zip", "temp_pancreas" -Recurse -Force
Write-Host "✅ Pancreas 3D done!" -ForegroundColor Green

# KIDNEY 3D
Write-Host "`n=== KIDNEY 3D (~325MB) ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "models/pretrained/cancers/kidney" | Out-Null
Write-Host "Downloading Kidney 3D (~325MB)..." -ForegroundColor Yellow
Invoke-WebRequest -Uri "https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/renalStructures_UNEST_segmentation_v0.2.2.zip" -OutFile "kidney.zip" -TimeoutSec 1800
Expand-Archive -Path "kidney.zip" -DestinationPath "temp_kidney" -Force
Copy-Item "temp_kidney/renalStructures_UNEST_segmentation/models/model.pt" "models/pretrained/cancers/kidney/volume.pt" -Force
Copy-Item "temp_kidney/renalStructures_UNEST_segmentation/configs/inference.json" "models/pretrained/cancers/kidney/volume.json" -Force
Remove-Item "kidney.zip", "temp_kidney" -Recurse -Force
Write-Host "✅ Kidney 3D done!" -ForegroundColor Green

# COLORECTAL DETECTOR
Write-Host "`n=== COLORECTAL DETECTOR ===" -ForegroundColor Cyan
if (-not (Test-Path "models/pretrained/cancers/colorectal/colonoscopy/detector.pt")) {
    try {
        $local = python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='goktug14/yolov5_kvasir_polyp', filename='weights/best.pt', local_dir='models/pretrained/cancers/colorectal/colonoscopy', local_dir_use_symlinks=False))"
        Move-Item $local.Trim() "models/pretrained/cancers/colorectal/colonoscopy/detector.pt" -Force
        Write-Host "✅ Colorectal detector downloaded" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed. Manual: https://universe.roboflow.com/goktug14/yolov5_kvasir_polyp" -ForegroundColor Red
    }
} else {
    Write-Host "✅ Already exists"
}

Write-Host "`n=== PART 1 COMPLETE ===" -ForegroundColor Green
Write-Host "Run Part 2 next for Roboflow detectors and classifiers." -ForegroundColor Yellow