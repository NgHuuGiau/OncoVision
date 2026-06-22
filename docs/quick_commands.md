# Quick Commands

Use these commands for fast verification and CI-style checks:

```powershell
.\.venv\Scripts\python run_app.py --advisor-only
.\.venv\Scripts\python run_chat.py --check-only
.\.venv\Scripts\python run_train.py --check-only
.\.venv\Scripts\python run_smoke.py
```

- `run_app.py --advisor-only` prints runtime recommendations and never opens the camera.
- `run_chat.py --check-only` validates chat dependencies, icons, output paths, and the medical model.
- `run_train.py --check-only` validates training dependencies, pretrained models, and dataset readiness.
- `run_smoke.py` runs the safe end-to-end entrypoint sequence used by CI.
