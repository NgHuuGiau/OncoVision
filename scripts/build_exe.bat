@echo off
cd /d D:\Project\OncoVision
D:\Project\OncoVision\.venv\Scripts\pyinstaller.exe ^
  --name OncoVisionChat ^
  --windowed ^
  --icon=assets\icons\icon.svg ^
  --add-data "models\modality_classifier.pt;models" ^
  --add-data "config;config" ^
  --add-data "static;static" ^
  --add-data "templates;templates" ^
  --add-data "assets;assets" ^
  --add-data "medical;medical" ^
  --add-data "core;core" ^
  --add-data "utils;utils" ^
  --add-data "app;app" ^
  run_chat.py