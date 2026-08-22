"""Photoroom Remove Background API client (solid BG).

Keep in sync with the twin copy in the other preprocess service
(profile_prod/preprocesamiento and frontal_prod/preprocesamiento).

Privacy: face/head crops are sent to Photoroom (third-party egress) when
BG removal runs — accepted product risk; ops must cover DPA/retention.

Fail-open: returns None on missing key, non-200, or any exception.
Never logs the API key or raw response body.

bg_color allowlist: white | #a6a6a6 (admin Analysis Testing gray).
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Fixed allowlist — do not honor arbitrary PHOTOROOM_API_URL (SSRF / key exfil).
PHOTOROOM_API_URL = "https://sdk.photoroom.com/v1/segment"
ALLOWED_PHOTOROOM_API_URL = PHOTOROOM_API_URL
REQUEST_TIMEOUT_SEC = 30

# Photoroom form bg_color values we accept from callers (gateway / admin testing).
_ALLOWED_BG = {
    "white": "white",
    "#ffffff": "white",
    "#a6a6a6": "#a6a6a6",
}
WHITE_BG_RGB: Tuple[int, int, int] = (255, 255, 255)
GRAY_BG_RGB: Tuple[int, int, int] = (166, 166, 166)


def normalize_photoroom_bg_color(raw: Optional[str]) -> str:
    """Return allowlisted Photoroom bg_color; unknown/empty → white."""
    if raw is None:
        return "white"
    key = str(raw).strip().lower()
    if not key:
        return "white"
    return _ALLOWED_BG.get(key, "white")


def bg_color_to_rgb(bg_color: Optional[str]) -> Tuple[int, int, int]:
    """RGB tuple for margin/letterbox fills matching Photoroom bg_color."""
    normalized = normalize_photoroom_bg_color(bg_color)
    if normalized == "#a6a6a6":
        return GRAY_BG_RGB
    return WHITE_BG_RGB


def is_configured() -> bool:
    """True when PHOTOROOM_API_KEY env is non-empty."""
    key = os.environ.get("PHOTOROOM_API_KEY", "")
    return bool(key and key.strip())


def _resolved_api_url() -> Optional[str]:
    """Return the allowlisted URL, or None if env override is invalid/unsafe."""
    raw = os.environ.get("PHOTOROOM_API_URL", "").strip()
    if not raw:
        return ALLOWED_PHOTOROOM_API_URL
    if raw.rstrip("/") != ALLOWED_PHOTOROOM_API_URL.rstrip("/"):
        logger.warning(
            "Photoroom: PHOTOROOM_API_URL rejected (not allowlisted); skipping white-BG"
        )
        return None
    parsed = urlparse(raw)
    if parsed.scheme != "https" or not parsed.netloc:
        logger.warning("Photoroom: PHOTOROOM_API_URL must be https; skipping white-BG")
        return None
    return ALLOWED_PHOTOROOM_API_URL


def remove_background_white(
    image_rgb: np.ndarray, bg_color: str = "white"
) -> Optional[np.ndarray]:
    """
    Call Photoroom segment API with allowlisted bg_color; return RGB uint8 or None.

    Resizes output to input HxW when dimensions differ. Fail-open on errors.
    Default bg_color is white (platform); admin testing may pass #a6a6a6.
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        logger.warning("Photoroom: invalid image input; skipping")
        return None
    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        logger.warning("Photoroom: expected HxWx3 RGB; skipping")
        return None

    api_key = os.environ.get("PHOTOROOM_API_KEY", "")
    if not api_key or not api_key.strip():
        logger.warning("Photoroom: PHOTOROOM_API_KEY not set; skipping white-BG")
        return None

    api_url = _resolved_api_url()
    if api_url is None:
        return None

    resolved_bg = normalize_photoroom_bg_color(bg_color)
    h, w = image_rgb.shape[:2]

    try:
        pil_image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)

        files = {"image_file": ("image.png", buf, "image/png")}
        data = {
            "bg_color": resolved_bg,
            "format": "png",
            "size": "full",
        }
        headers = {"x-api-key": api_key.strip()}

        response = requests.post(
            api_url,
            headers=headers,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT_SEC,
            verify=True,
            allow_redirects=False,
        )

        if response.status_code != 200:
            logger.warning(
                "Photoroom: HTTP %s; skipping white-BG",
                response.status_code,
            )
            return None

        content = response.content
        if not content:
            logger.warning("Photoroom: empty response; skipping white-BG")
            return None

        arr = np.frombuffer(content, dtype=np.uint8)
        decoded_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            logger.warning("Photoroom: failed to decode response image; skipping")
            return None

        out_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

        if out_rgb.shape[0] != h or out_rgb.shape[1] != w:
            out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        return out_rgb.astype(np.uint8)

    except Exception as e:
        logger.warning(
            "Photoroom: request failed (fail-open): %s", type(e).__name__
        )
        return None
