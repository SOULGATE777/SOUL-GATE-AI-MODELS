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


def test_refine_mask_default_threshold_trims_soft_halo():
    """A uniform 0.55 probability is below the 0.6 default → dropped as background.

    Locks the 0.5→0.6 default: the uncertain halo the segmenter bleeds onto the
    background (the "too much background left" symptom) must not survive.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.full((32, 32), 0.55, dtype=np.float32)
    alpha = ProfilePreprocessingPipeline._refine_segmentation_mask(mask)
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


def test_refine_mask_open_severs_background_bridge():
    """Attached background (thin mask bridge) is dropped; without open it survives.

    This is the fix for "profiles leave too much background": reflective glass /
    fences / banners stay above threshold and touch the subject through a thin
    mask bridge, so keep-largest-component keeps them. The morphological open
    severs that bridge so the blob is dropped.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((80, 80), dtype=np.float32)
    mask[8:72, 8:40] = 1.0        # subject (largest component)
    mask[33:47, 58:74] = 1.0      # attached background blob
    mask[38:41, 40:58] = 1.0      # thin bridge linking blob → subject

    # Open disabled: the bridge keeps them one component → background survives.
    bridged = ProfilePreprocessingPipeline._refine_segmentation_mask(
        mask, feather_px=0, open_frac=0.0
    )
    assert float(bridged[40, 65]) == 1.0   # background blob still present

    # Open enabled: bridge severed → blob dropped by keep-largest, subject kept.
    cleaned = ProfilePreprocessingPipeline._refine_segmentation_mask(
        mask, feather_px=0, open_frac=0.05
    )
    assert float(cleaned[40, 20]) == 1.0   # subject preserved
    assert float(cleaned[40, 65]) == 0.0   # attached background removed


def test_refine_mask_default_open_severs_bridge_at_scale():
    """The DEFAULT open_frac (0.015) severs a resolution-scaled bridge.

    Proves production defaults (not just a strong open) drop attached background:
    on a realistic ~400px crop the default kernel (~6px) breaks a thin bridge and
    keep-largest drops the blob, while the subject is preserved.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((400, 400), dtype=np.float32)
    mask[40:360, 40:200] = 1.0     # subject (largest)
    mask[180:260, 300:370] = 1.0   # attached background blob
    mask[216:220, 200:300] = 1.0   # thin 4px bridge

    alpha = ProfilePreprocessingPipeline._refine_segmentation_mask(mask, feather_px=0)
    assert float(alpha[200, 120]) == 1.0   # subject preserved
    assert float(alpha[220, 335]) == 0.0   # attached background removed at default


def test_refine_mask_open_never_clips_protected_face():
    """The bridge-break open must not defeat the face guard: rect stays opaque."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # Segmenter says almost everything is background; only a thin sliver is FG.
    mask = np.zeros((60, 60), dtype=np.float32)
    mask[0:60, 0:2] = 1.0
    alpha = ProfilePreprocessingPipeline._refine_segmentation_mask(
        mask, feather_px=2, open_frac=0.05, protect_rect=(20, 20, 40, 40)
    )
    assert float(alpha[20:40, 20:40].min()) == 1.0


# --- Face guard: the face region must NEVER be clipped ---


def test_refine_mask_protect_rect_forces_face_opaque():
    """Even a fully-background mask keeps the ENTIRE protected rect exactly opaque.

    Guards the hard requirement: the face is never clipped, so every pixel inside
    the protect rect (edges and corners included) must be exactly 1.0 — the feather
    blur must not soften the rect interior.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # Segmenter is completely wrong: says everything is background.
    mask = np.zeros((60, 60), dtype=np.float32)
    protect = (20, 20, 40, 40)

    for feather in (0, 2, 4):
        alpha = ProfilePreprocessingPipeline._refine_segmentation_mask(
            mask, feather_px=feather, protect_rect=protect
        )
        # EVERY pixel in the rect is fully opaque (min == 1.0), not just the center.
        assert float(alpha[20:40, 20:40].min()) == 1.0, f"feather={feather}"
    # Far outside stays background.
    assert float(alpha[2, 2]) == 0.0


def test_refine_mask_protect_rect_tiny_face_opaque():
    """A very small protect rect is still exactly opaque (no edge dip from blur)."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((40, 40), dtype=np.float32)
    protect = (18, 18, 22, 22)  # 4x4 face core
    alpha = ProfilePreprocessingPipeline._refine_segmentation_mask(
        mask, feather_px=2, protect_rect=protect
    )
    assert float(alpha[18:22, 18:22].min()) == 1.0


def test_face_protect_rect_expands_and_clamps():
    """Protect rect expands the face bbox (more on top) and clamps to the crop."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # Face bbox (10,10)-(50,60) inside a 100x100 crop.
    rect = ProfilePreprocessingPipeline._face_protect_rect(10, 10, 50, 60, 100, 100)
    assert rect is not None
    px1, py1, px2, py2 = rect
    # Expanded outward on every side, and clamped within [0, crop].
    assert px1 < 10 and px2 > 50
    assert py1 < 10 and py2 > 60
    assert px1 >= 0 and py1 >= 0 and px2 <= 100 and py2 <= 100
    # Top expansion (forehead) is larger than the chin expansion.
    assert (10 - py1) > (py2 - 60)


def test_face_protect_rect_degenerate_returns_none():
    """A zero/negative-area bbox yields no protection rect."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    assert ProfilePreprocessingPipeline._face_protect_rect(10, 10, 10, 10, 100, 100) is None
    assert ProfilePreprocessingPipeline._face_protect_rect(50, 50, 10, 10, 100, 100) is None


def test_apply_white_background_never_clips_face():
    """End-to-end: segmenter drops the face, protect_rect restores it."""
    pytest.importorskip("torch")

    # Mask marks the face region as background (alpha 0) — worst case.
    mask = np.ones((40, 40), dtype=np.float32)
    mask[10:30, 10:30] = 0.0
    result = MagicMock()
    result.segmentation_mask = mask
    segmenter = MagicMock()
    segmenter.process.return_value = result
    pipeline = _make_pipeline_with_segmenter(segmenter)

    image = np.full((40, 40, 3), (12, 34, 56), dtype=np.uint8)
    out, applied = pipeline.apply_white_background(image, protect_rect=(12, 12, 28, 28))

    assert applied is True
    # Face pixels survive (not whitened) thanks to the protection rect.
    assert np.allclose(out[20, 20], (12, 34, 56), atol=1)
