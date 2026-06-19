from __future__ import annotations


def resolve_legacy_start_options(
    *,
    requested_mode: str | None,
    requested_model: str | None,
    resolve_start_bundle_fn,
) -> tuple[str, str]:
    start_options = resolve_start_bundle_fn(
        requested_mode=requested_mode,
        requested_model=requested_model,
        requested_target="ui",
        preferred_target="ui",
    )
    return start_options.selected_mode, start_options.selected_model
