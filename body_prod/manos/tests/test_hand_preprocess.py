"""Unit tests for palm white-BG + conditional dark CLAHE preprocess."""

import numpy as np
import cv2

from app.utils.image_processing import (
    MEAN_L_DARK_THRESHOLD,
    maybe_enhance_dark,
    apply_white_background_grabcut,
    prepare_image_for_analysis,
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


def test_maybe_enhance_dark_fail_open_none():
    """None input fails open without raising."""
    out, enhanced = maybe_enhance_dark(None)
    assert enhanced is False
    assert out is None


def test_grabcut_returns_true_on_synthetic():
    """Synthetic circle on contrasting bg → GrabCut succeeds."""
    img = np.full((120, 120, 3), 40, dtype=np.uint8)
    cv2.circle(img, (60, 60), 35, (200, 80, 60), -1)
    result, applied = apply_white_background_grabcut(img)
    assert applied is True
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_grabcut_fail_open_none():
    """None input fails open without raising."""
    out, applied = apply_white_background_grabcut(None)
    assert applied is False
    assert out is None


def test_prepare_image_for_analysis_order_and_flags():
    """prepare_image_for_analysis returns (image, white_bg, illum) flags."""
    dark = np.full((120, 120, 3), 20, dtype=np.uint8)
    cv2.circle(dark, (60, 60), 35, (80, 40, 30), -1)
    out, white_bg, illum = prepare_image_for_analysis(dark)
    assert out.shape == dark.shape
    assert isinstance(white_bg, bool)
    assert illum is True
