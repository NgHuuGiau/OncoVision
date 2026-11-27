from __future__ import annotations

"""Anatomy API routes for 3D viewer organ mapping."""

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from medical.anatomy_mapping import (
    get_available_targets,
    get_full_mapping,
    validate_target,
)

router = APIRouter(prefix="/api/anatomy", tags=["anatomy"])


@router.get("/organs")
async def get_organs_for_target(
    request: Request,
    target: str = Query(..., description="Cancer target key (e.g., liver, lung)"),
    lang: str = Query("vi", description="Language for labels (vi/en)"),
) -> JSONResponse:
    """
    Get organs to highlight for a specific cancer target.

    Returns:
        JSON with target, label, and organs_to_highlight list
    """
    # Validate target
    if not validate_target(target):
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "detail": f"Unknown cancer target: {target}",
                "available_targets": get_available_targets(),
            },
        )

    # Validate language
    if lang not in ("vi", "en"):
        lang = "vi"

    mapping = get_full_mapping(target, lang)

    return JSONResponse(
        content={
            "ok": True,
            **mapping,
        }
    )


@router.get("/targets")
async def get_anatomy_targets() -> JSONResponse:
    """Get list of available cancer targets for 3D viewer."""
    targets = get_available_targets()

    # Include labels for both languages
    from medical.anatomy_mapping import (
        CANCER_LABELS_EN,
        CANCER_LABELS_VI,
    )

    target_list = [
        {
            "key": t,
            "label_vi": CANCER_LABELS_VI.get(t, t),
            "label_en": CANCER_LABELS_EN.get(t, t),
        }
        for t in targets
    ]

    return JSONResponse(
        content={
            "ok": True,
            "targets": target_list,
        }
    )


def register_anatomy_routes(app):
    """Register anatomy routes with the FastAPI app."""
    app.include_router(router)