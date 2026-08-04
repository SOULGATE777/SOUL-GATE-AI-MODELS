"""Validation helpers for optional white-BG Form overrides (torch/FastAPI-free)."""

from typing import Optional

# Legacy Form values (ignored by Photoroom path) + "photoroom" so admin echo of
# pipeline rembg_model_name does not 400.
ALLOWED_REMBG_MODELS = frozenset({"u2net", "isnet-general-use", "photoroom"})


def check_white_bg_overrides(
    rembg_model: Optional[str],
    rembg_edge_margin_frac: Optional[float],
    rembg_edge_margin_min_px: Optional[int],
    face_protect_core_frac: Optional[float],
) -> Optional[str]:
    """Return an error detail string if white-BG overrides are invalid, else None."""
    if rembg_model is not None and rembg_model not in ALLOWED_REMBG_MODELS:
        return f"rembg_model must be one of {sorted(ALLOWED_REMBG_MODELS)}"
    if rembg_edge_margin_frac is not None and not 0.0 <= rembg_edge_margin_frac <= 0.25:
        return "rembg_edge_margin_frac must be between 0.0 and 0.25"
    if rembg_edge_margin_min_px is not None and not 0 <= rembg_edge_margin_min_px <= 64:
        return "rembg_edge_margin_min_px must be between 0 and 64"
    if face_protect_core_frac is not None and not 0.1 <= face_protect_core_frac <= 0.8:
        return "face_protect_core_frac must be between 0.1 and 0.8"
    return None
