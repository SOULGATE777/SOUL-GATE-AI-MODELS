"""Unit tests for white-background cleaning (profile preprocess).

Covers the shared ``composite_on_white`` helper, the retained GrabCut utility,
and the live MediaPipe Selfie Segmentation path on the pipeline. Pipeline tests
lazily import ``ProfilePreprocessingPipeline`` behind ``importorskip('torch')``
so they run in CI/Docker (where torch is installed) and skip locally.
"""

from unittest.mock import MagicMock

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


def test_grabcut_uniform_image_does_not_crash():
    """Uniform image: seeded mask may still apply; must not crash or alter shape."""
    img = np.full((100, 100, 3), 90, dtype=np.uint8)
    result, applied = apply_white_background_grabcut(img)
    assert isinstance(applied, bool)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_grabcut_preserves_center_foreground():
    """Tall centered FG (profile-like) keeps center pixels non-white after cut."""
    bg = (40, 40, 40)
    fg = (160, 100, 70)
    img = np.full((200, 160, 3), bg, dtype=np.uint8)
    # Tall oval covering face + hair crown area
    cv2.ellipse(img, (80, 100), (45, 75), 0, 0, 360, fg, -1)

    result, applied = apply_white_background_grabcut(img)
    assert applied is True
    # Center of ellipse should remain close to FG color (not washed to white)
    center = result[100, 80].astype(np.float32)
    assert float(center.mean()) < 200, f"Center washed out: {center}"
    # Top of oval (crown) should still have substantial FG
    crown = result[40, 80].astype(np.float32)
    assert float(crown.mean()) < 220, f"Crown clipped to white: {crown}"


def test_grabcut_fail_open_bad_ndim():
    """Non-RGB array fail-opens."""
    gray = np.full((40, 40), 128, dtype=np.uint8)
    result, applied = apply_white_background_grabcut(gray)
    assert applied is False
    np.testing.assert_array_equal(result, gray)


def test_grabcut_removes_detached_fg_blob():
    """Detached blob near a corner is dropped (keep-largest-component)."""
    bg = (35, 35, 35)
    fg = (170, 110, 80)
    img = np.full((200, 160, 3), bg, dtype=np.uint8)
    # Main head-like blob in the center
    cv2.ellipse(img, (80, 100), (46, 74), 0, 0, 360, fg, -1)
    # Small detached blob near top-left, separated from the main blob
    cv2.rectangle(img, (12, 12), (34, 34), fg, -1)

    result, applied = apply_white_background_grabcut(img)
    assert applied is True
    # Main center is preserved
    center = result[100, 80].astype(np.float32)
    assert float(center.mean()) < 200, f"Center washed out: {center}"
    # Detached corner blob region should be removed → near white
    corner = result[23, 23].astype(np.float32)
    assert float(corner.mean()) > 210, f"Detached blob not removed: {corner}"


# --- Live MediaPipe Selfie Segmentation path (pipeline.apply_white_background) ---
# Guarded by importorskip('torch'): the pipeline module imports torch at module
# scope, so these run in CI/Docker and skip in a torch-less local env.


def _make_pipeline_with_segmenter(segmenter):
    """Build a ProfilePreprocessingPipeline without loading the torch model."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    pipeline = object.__new__(ProfilePreprocessingPipeline)
    pipeline.selfie_segmenter = segmenter
    return pipeline


def test_apply_white_background_fail_open():
    """Segmenter raises → image unchanged and False (fail-open)."""
    segmenter = MagicMock()
    segmenter.process.side_effect = RuntimeError("segmenter boom")
    pipeline = _make_pipeline_with_segmenter(segmenter)

    image = np.full((16, 16, 3), 77, dtype=np.uint8)
    original = image.copy()

    out, applied = pipeline.apply_white_background(image)

    assert applied is False
    np.testing.assert_array_equal(out, original)
    segmenter.process.assert_called_once()


def test_apply_white_background_none_mask_fail_open():
    """No segmentation mask → image unchanged and False (fail-open)."""
    result = MagicMock()
    result.segmentation_mask = None
    segmenter = MagicMock()
    segmenter.process.return_value = result
    pipeline = _make_pipeline_with_segmenter(segmenter)

    image = np.full((16, 16, 3), 99, dtype=np.uint8)
    original = image.copy()

    out, applied = pipeline.apply_white_background(image)

    assert applied is False
    np.testing.assert_array_equal(out, original)


def test_apply_white_background_composites_on_white():
    """A hard mask makes background white and keeps subject; applied is True."""
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    result = MagicMock()
    result.segmentation_mask = mask
    segmenter = MagicMock()
    segmenter.process.return_value = result
    pipeline = _make_pipeline_with_segmenter(segmenter)

    image = np.full((40, 40, 3), (10, 20, 30), dtype=np.uint8)
    out, applied = pipeline.apply_white_background(image)

    assert applied is True
    # Subject core preserved (edge is feathered, so assert on the interior)
    assert np.allclose(out[15:25, 15:25], (10, 20, 30), atol=1)
    # A background corner is white
    assert np.all(out[0, 0] == 255)


# --- Mask refinement (_refine_segmentation_mask) ---


def test_refine_mask_kills_midalpha_ghosting():
    """Uniform mid-range probability (model uncertainty) collapses to background.

    This is the fix for translucent background "ghosting": a soft 0.4 alpha over
    the whole frame must not survive as a half-transparent blend.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.full((32, 32), 0.4, dtype=np.float32)
    alpha = ProfilePreprocessingPipeline._refine_segmentation_mask(mask)

    assert alpha.shape == (32, 32)
    assert float(alpha.max()) == 0.0


def test_refine_mask_keeps_largest_component():
    """A big subject blob is kept; a small detached blob is dropped."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((60, 60), dtype=np.float32)
    mask[10:50, 10:40] = 1.0   # large subject
    mask[2:6, 52:56] = 1.0     # small detached background blob
    alpha = ProfilePreprocessingPipeline._refine_segmentation_mask(mask, feather_px=0)

    assert float(alpha[30, 25]) == 1.0   # subject core kept
    assert float(alpha[4, 54]) == 0.0    # detached blob removed


def test_refine_mask_feathers_edge_smoothly():
    """Feather produces a smooth 0→1 ramp at the edge, not a hard blocky step."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((40, 40), dtype=np.float32)
    mask[8:32, 8:32] = 1.0
    alpha = ProfilePreprocessingPipeline._refine_segmentation_mask(mask, feather_px=2)

    assert float(alpha[20, 20]) == 1.0                 # interior solid
    assert 0.0 < float(alpha[20, 7]) < 1.0             # feathered edge is partial
