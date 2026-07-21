"""Unit tests for GrabCut white-background cleaning (profile preprocess)."""

import numpy as np
import cv2
import pytest

from app.utils.image_processing import (
    WHITE_BG,
    composite_on_white,
    apply_white_background_grabcut,
    ImageProcessor,
)


def test_white_bg_constant():
    assert WHITE_BG == (255, 255, 255)


def test_composite_on_white_full_fg():
    """Full foreground mask leaves image unchanged."""
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    mask = np.ones((40, 40), dtype=np.float32)
    out = composite_on_white(image, mask)
    assert out.shape == image.shape
    np.testing.assert_array_equal(out, image)


def test_composite_on_white_full_bg():
    """Zero mask yields solid white."""
    image = np.full((40, 40, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.float32)
    out = composite_on_white(image, mask)
    assert out.shape == image.shape
    assert np.all(out == 255)


def test_composite_on_white_soft_blend():
    """Half mask blends halfway toward white."""
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.full((20, 20), 0.5, dtype=np.float32)
    out = ImageProcessor.composite_on_white(image, mask)
    # 0 * 0.5 + 255 * 0.5 = 127.5 → 127 or 128
    assert out.shape == image.shape
    assert np.all(out >= 127) and np.all(out <= 128)


def test_grabcut_preserves_shape_and_returns_bool():
    """GrabCut returns bool success and preserves HxWx3 shape."""
    img = np.full((120, 120, 3), 40, dtype=np.uint8)
    cv2.circle(img, (60, 60), 35, (200, 80, 60), -1)
    result, applied = apply_white_background_grabcut(img)
    assert isinstance(applied, bool)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_grabcut_corners_closer_to_white():
    """Solid color circle on contrasting bg → outer corners nearer white after."""
    bg_color = (30, 30, 30)
    fg_color = (180, 90, 50)
    img = np.full((160, 160, 3), bg_color, dtype=np.uint8)
    cv2.circle(img, (80, 80), 45, fg_color, -1)

    result, applied = apply_white_background_grabcut(img)
    assert applied is True
    assert result.shape == img.shape

    # Sample four outer corners (should trend toward white vs original dark bg)
    corners_before = np.array(
        [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], dtype=np.float32
    )
    corners_after = np.array(
        [result[0, 0], result[0, -1], result[-1, 0], result[-1, -1]],
        dtype=np.float32,
    )
    mean_before = float(corners_before.mean())
    mean_after = float(corners_after.mean())
    assert mean_after > mean_before, (
        f"Expected corners closer to white: before={mean_before:.1f}, after={mean_after:.1f}"
    )


def test_grabcut_fail_open_tiny_image():
    """Very small images fail-open without crashing."""
    tiny = np.full((8, 8, 3), 128, dtype=np.uint8)
    result, applied = apply_white_background_grabcut(tiny)
    assert applied is False
    np.testing.assert_array_equal(result, tiny)
