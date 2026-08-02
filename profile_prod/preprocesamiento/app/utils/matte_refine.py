"""Person-matte refinement (pure OpenCV/numpy — no torch)."""
from typing import Optional, Tuple

import cv2
import numpy as np

# Luma/chroma gate for subject-aware silhouette protect (BGR or RGB same weights).
_LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _background_seed(keep: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (x, y) on background, or None if the silhouette covers the canvas."""
    if keep[0, 0] == 0:
        return (0, 0)
    ys, xs = np.where(keep == 0)
    if len(xs) == 0:
        return None
    return (int(xs[0]), int(ys[0]))


def _clip_rect(
    rect: Tuple[int, int, int, int], w: int, h: int
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def subject_like_mask(image_rgb: np.ndarray) -> np.ndarray:
    """True where pixels look like subject (skin/hair/clothes), not light BG.

    Light tent/sky/wall inside a face bbox must stay removable; dark lips/beard/
    nose silhouette must be restorable when rembg zeros them.
    """
    img = image_rgb.astype(np.float32)
    luma = img @ _LUMA_W
    chroma = img.max(axis=2) - img.min(axis=2)
    # Near-white / washed BG: high luma and low chroma.
    bg_like = (luma >= 200.0) & (chroma <= 40.0)
    return ~bg_like


def refine_person_matte(
    mask: np.ndarray,
    protect_rect: Optional[Tuple[int, int, int, int]] = None,
    image_rgb: Optional[np.ndarray] = None,
    silhouette_rect: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Clean a soft person matte and optionally enforce face guards.

    Keeps rembg-style soft fringe while fixing residual QA failures:
    close → open → keep-largest → drop solid FG outside keep → fill holes.

    Guards (applied last):
    - ``silhouette_rect`` + ``image_rgb``: subject-aware restore for profile
      nose/lips/chin at the bbox edge (force opaque only on non-BG-like pixels).
    - ``protect_rect``: hard central core — always force opaque (catastrophic
      backstop; must stay small so light BG inside the head bbox is not locked).
    """
    alpha = mask.astype(np.float32)
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)

    binary = (alpha >= 0.5).astype(np.uint8)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = None
    if num_labels > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        alpha[(labels > 0) & (labels != largest)] = 0.0
        keep = (labels == largest).astype(np.uint8)
    elif num_labels == 2:
        keep = (labels == 1).astype(np.uint8)

    if keep is not None and keep.any():
        # Solid FG left outside keep after morph-open (chin sticks) → drop.
        # Soft fringe (α < 0.9, often ~0.5) stays so rembg AA edges survive.
        alpha[(alpha >= 0.9) & (keep == 0)] = 0.0

        h, w = keep.shape
        seed = _background_seed(keep)
        if seed is not None:
            flood = keep.copy()
            mask_ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood, mask_ff, seed, 1)
            interior_holes = flood == 0
            if interior_holes.any():
                alpha[interior_holes] = 1.0

    h, w = alpha.shape[:2]

    # Subject-aware silhouette shell (profile nose/mouth/chin at bbox edge).
    # Only restore near existing FG so dark walls rembg already removed stay out;
    # light tent/sky inside the head bbox stays removable via subject_like_mask.
    if (
        silhouette_rect is not None
        and image_rgb is not None
        and image_rgb.shape[:2] == alpha.shape[:2]
    ):
        clipped = _clip_rect(silhouette_rect, w, h)
        if clipped is not None:
            sx1, sy1, sx2, sy2 = clipped
            fg = (alpha >= 0.45).astype(np.uint8)
            # ~15px band: enough for profile lips/chin on a tight detector edge.
            band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            near_fg = cv2.dilate(fg, band_k, iterations=1)
            roi_subject = subject_like_mask(image_rgb[sy1:sy2, sx1:sx2])
            roi_near = near_fg[sy1:sy2, sx1:sx2] > 0
            roi_alpha = alpha[sy1:sy2, sx1:sx2].copy()
            restore = roi_subject & roi_near & (roi_alpha < 0.90)
            roi_alpha[restore] = 1.0
            alpha[sy1:sy2, sx1:sx2] = roi_alpha

    # Hard central core — always opaque.
    if protect_rect is not None:
        clipped = _clip_rect(protect_rect, w, h)
        if clipped is not None:
            px1, py1, px2, py2 = clipped
            alpha[py1:py2, px1:px2] = 1.0

    return alpha
