from __future__ import annotations

"""Health & status routes: healthz, health, v1_models, api/status."""

import time

from fastapi import HTTPException, Request

from entrypoints.web.shared import (
    MEDICAL_DISCLAIMER,
    _cached_medical_status,
    get_db,
    get_medical_system_status,
)


def register_health_routes(app):
    """Register health and status routes."""

    _STATUS_CACHE_TTL_SECONDS = 300.0
    _status_cache: tuple[float, dict] | None = None

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok"}

    @app.get("/health", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @app.get("/v1/models", include_in_schema=False)
    def v1_models():
        medical = _cached_medical_status()
        models = []
        for target_key, caps in medical.target_capabilities.items():
            if caps.get("runtime_ready"):
                models.append({
                    "id": f"oncovision-{target_key}",
                    "name": target_key,
                    "ready": True,
                })
        return {
            "models": models,
            "total": len(models),
        }

    @app.get("/api/status")
    def api_status():
        nonlocal _status_cache
        if _status_cache is not None and time.monotonic() - _status_cache[0] < _STATUS_CACHE_TTL_SECONDS:
            return _status_cache[1]

        from medical.pipeline import build_default_medical_analyzer_config
        config = build_default_medical_analyzer_config()
        demo_mode = config.demo_mode

        medical = get_medical_system_status()
        db_stats = get_db().get_db_stats()

        if demo_mode:
            model_ready = True
            model_message = "Demo mode đang bật - chạy mô phỏng không cần model thực tế"
            classification_ready = True
            localization_ready = True
            volume_ready = True
            clinical_validation_ready = True
            all_targets_ready = True
        else:
            model_ready = medical.model_ready
            model_message = medical.model_message
            classification_ready = medical.classification_ready
            localization_ready = medical.localization_ready
            volume_ready = medical.volume_ready
            clinical_validation_ready = medical.clinical_validation_ready
            all_targets_ready = medical.all_targets_ready

        payload = {
            "ok": True,
            "model_ready": model_ready,
            "model_message": model_message,
            "demo_mode": demo_mode,
            "classification_ready": classification_ready,
            "localization_ready": localization_ready,
            "volume_ready": volume_ready,
            "clinical_validation_ready": clinical_validation_ready,
            "all_targets_ready": all_targets_ready,
            "target_capabilities": medical.target_capabilities,
            "case_count": medical.case_count,
            "analyzed_cancers": list(medical.analyzed_cancers),
            "analyzed_modalities": list(medical.analyzed_modalities),
            "db_stats": db_stats,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

        if demo_mode:
            payload["demo_warning"] = "⚠️ DEMO MODE: Hệ thống đang chạy mô phỏng. Kết quả không phản ánh thực tế và không dùng cho chẩn đoán y khoa."

        _status_cache = (time.monotonic(), payload)
        return payload
