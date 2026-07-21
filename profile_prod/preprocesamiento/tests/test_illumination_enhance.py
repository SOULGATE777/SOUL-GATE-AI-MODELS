"""Unit tests for conditional dark-photo illumination (profile preprocess)."""

from unittest.mock import patch

import numpy as np
import pytest

from app.utils.image_processing import (
    MEAN_L_DARK_THRESHOLD,
    maybe_enhance_dark,
    ImageProcessor,
)


def test_mean_l_dark_threshold_constant():
    assert MEAN_L_DARK_THRESHOLD == 90


def test_maybe_enhance_dark_flag_true_on_dark_image():
    """Dark image (mean LAB L < 90) → illumination_enhanced True."""
    dark = np.full((64, 64, 3), 20, dtype=np.uint8)
    out, enhanced = maybe_enhance_dark(dark)
    assert enhanced is True
    assert out.shape == dark.shape
    assert out.dtype == np.uint8


def test_maybe_enhance_dark_flag_false_on_bright_image():
    """Well-lit image (mean LAB L >= 90) → no-op, flag False."""
    bright = np.full((64, 64, 3), 200, dtype=np.uint8)
    out, enhanced = maybe_enhance_dark(bright)
    assert enhanced is False
    np.testing.assert_array_equal(out, bright)


def test_maybe_enhance_dark_fail_open_invalid_input():
    """Invalid input fails open without raising."""
    out, enhanced = maybe_enhance_dark(None)
    assert enhanced is False
    assert out is None

    gray = np.zeros((32, 32), dtype=np.uint8)
    out2, enhanced2 = maybe_enhance_dark(gray)
    assert enhanced2 is False
    np.testing.assert_array_equal(out2, gray)


def test_maybe_enhance_dark_fail_open_on_exception():
    """Unexpected errors fail-open: original image, flag False."""
    img = np.full((40, 40, 3), 30, dtype=np.uint8)
    with patch.object(
        ImageProcessor,
        "enhance_image_quality",
        side_effect=RuntimeError("boom"),
    ):
        out, enhanced = ImageProcessor.maybe_enhance_dark(img)
    assert enhanced is False
    np.testing.assert_array_equal(out, img)


def test_module_level_delegates_to_static():
    """Module-level helper mirrors ImageProcessor static method."""
    img = np.full((48, 48, 3), 25, dtype=np.uint8)
    a_out, a_flag = maybe_enhance_dark(img)
    b_out, b_flag = ImageProcessor.maybe_enhance_dark(img)
    assert a_flag == b_flag
    np.testing.assert_array_equal(a_out, b_out)
