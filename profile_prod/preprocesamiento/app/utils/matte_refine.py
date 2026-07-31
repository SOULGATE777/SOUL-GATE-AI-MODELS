"""Person-matte refinement (pure OpenCV/numpy — no torch)."""
from typing import Optional, Tuple

import cv2
import numpy as np


def _background_seed(keep: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (x, y) on background, or None if the silhouette covers the canvas."""
    if keep[0, 0] == 0:
        return (0, 0)
    ys, xs = np.where(keep == 0)
    if len(xs) == 0:
        return None
    return (int(xs[0]), int(ys[0]))


def refine_person_matte(
    mask: np.ndarray,
    protect_rect: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Clean a soft person matte and optionally enforce a face-core guard.

    Keeps rembg-style soft fringe while fixing residual QA failures:
    close → open → keep-largest → drop solid FG outside keep → fill holes.
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

    if protect_rect is not None:
        px1, py1, px2, py2 = protect_rect
        h, w = alpha.shape[:2]
        px1 = max(0, min(int(px1), w))
        px2 = max(0, min(int(px2), w))
        py1 = max(0, min(int(py1), h))
        py2 = max(0, min(int(py2), h))
        if px2 > px1 and py2 > py1:
            alpha[py1:py2, px1:px2] = 1.0

    return alpha
