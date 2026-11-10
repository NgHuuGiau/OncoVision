# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['run_chat.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('static', 'static'),
        ('templates', 'templates'),
        ('assets', 'assets'),
        ('medical', 'medical'),
        ('core', 'core'),
        ('utils', 'utils'),
        ('app', 'app'),
        ('models', 'models'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'matplotlib.*',
        'PIL', 'PIL.*',
        'cv2', 'torch', 'torchvision',
        'ultralytics', 'numpy',
        'pydicom', 'sqlalchemy',
        'fastapi', 'uvicorn', 'starlette',
        'PySide6', 'PySide6.*',
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OncoVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,  # Console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='OncoVision',
)