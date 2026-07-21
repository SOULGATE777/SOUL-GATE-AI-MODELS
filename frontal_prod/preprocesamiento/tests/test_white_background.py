"""Tests for white-background cleaning in frontal preprocess."""
from unittest.mock import MagicMock

import numpy as np

from app.utils.image_processing import composite_on_white
from app.models.frontal_preprocessing_pipeline import FrontalPreprocessingPipeline


def test_composite_on_white_hard_mask_background_is_white():
    """Red subject on blue; hard mask → background pixels ~255."""
    h, w = 20, 30
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:] = (0, 0, 255)  # blue background
    image[5:15, 10:20] = (255, 0, 0)  # red subject

    mask = np.zeros((h, w), dtype=np.float32)
    mask[5:15, 10:20] = 1.0

    out = composite_on_white(image, mask)

    # Subject stays red
    assert np.allclose(out[5:15, 10:20], (255, 0, 0), atol=1)

    # Background becomes white
    bg = out.copy()
    bg[5:15, 10:20] = 0
    bg_pixels = bg[bg.sum(axis=2) > 0]
    assert bg_pixels.shape[0] > 0
    assert np.allclose(bg_pixels, 255, atol=1)


def test_composite_on_white_invalid_shapes_returns_unchanged():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:] = 42
    bad_mask = np.zeros((4, 4), dtype=np.float32)
    out = composite_on_white(image, bad_mask)
    np.testing.assert_array_equal(out, image)


def test_apply_white_background_fail_open():
    """Segmenter raises → image unchanged and False (fail-open)."""
    pipeline = object.__new__(FrontalPreprocessingPipeline)
    segmenter = MagicMock()
    segmenter.process.side_effect = RuntimeError("segmenter boom")
    pipeline.selfie_segmenter = segmenter

    image = np.full((16, 16, 3), 77, dtype=np.uint8)
    original = image.copy()

    out, applied = pipeline.apply_white_background(image)

    assert applied is False
    np.testing.assert_array_equal(out, original)
    segmenter.process.assert_called_once()
