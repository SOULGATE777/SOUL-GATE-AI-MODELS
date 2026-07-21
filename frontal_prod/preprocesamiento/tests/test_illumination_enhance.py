"""Tests for conditional dark-photo illumination enhancement."""
from unittest.mock import patch

import cv2
import numpy as np

from app.utils.image_processing import MEAN_L_DARK_THRESHOLD, maybe_enhance_dark


def _mean_l(image_rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    return float(np.mean(lab[:, :, 0]))


def test_dark_synthetic_enhances():
    """Dark synthetic (mean L ~40) → illumination_enhanced True."""
    # LAB L≈40: uniform dark gray RGB around (35, 35, 35)
    image = np.full((64, 64, 3), 35, dtype=np.uint8)
    assert _mean_l(image) < MEAN_L_DARK_THRESHOLD
    assert abs(_mean_l(image) - 40) < 15

    out, enhanced = maybe_enhance_dark(image)

    assert enhanced is True
    assert out is not None
    assert out.shape == image.shape


def test_bright_synthetic_noop():
    """Bright synthetic (mean L ~150) → False and near-identity."""
    # LAB L≈150: mid-bright gray RGB around (148, 148, 148)
    image = np.full((64, 64, 3), 148, dtype=np.uint8)
    assert _mean_l(image) >= MEAN_L_DARK_THRESHOLD
    assert abs(_mean_l(image) - 150) < 15

    out, enhanced = maybe_enhance_dark(image)

    assert enhanced is False
    np.testing.assert_array_equal(out, image)


def test_fail_open_on_enhance_error():
    """Monkeypatch enhance to raise → original image + False."""
    image = np.full((64, 64, 3), 30, dtype=np.uint8)
    original = image.copy()

    with patch(
        "app.utils.image_processing.ImageProcessor.enhance_image_quality",
        side_effect=RuntimeError("enhance boom"),
    ):
        out, enhanced = maybe_enhance_dark(image)

    assert enhanced is False
    np.testing.assert_array_equal(out, original)
