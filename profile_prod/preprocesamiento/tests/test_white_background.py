"""Unit tests for white-background cleaning (profile preprocess).

Covers the shared ``composite_on_white`` helper, the retained GrabCut utility,
and the rembg (u2net) matting path on the pipeline. Pipeline tests lazily import
``ProfilePreprocessingPipeline`` behind ``importorskip('torch')`` so they run in
CI/Docker (where torch is installed) and skip locally, and they monkeypatch the
module-level ``rembg_remove`` so no ONNX model is needed to test the wiring.
"""

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


# --- rembg (u2net) matting path (pipeline.apply_white_background) ---
# Guarded by importorskip('torch'): the pipeline module imports torch at module
# scope, so these run in CI/Docker and skip in a torch-less local env. The
# module-level ``rembg_remove`` is monkeypatched so no ONNX model is needed.


def _make_pipeline():
    """Build a ProfilePreprocessingPipeline without loading the torch model."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    pipeline = object.__new__(ProfilePreprocessingPipeline)
    # Sentinel session so _get_rembg_session() short-circuits (no real ONNX init).
    pipeline.rembg_model_name = "isnet-general-use"
    pipeline._rembg_sessions = {"isnet-general-use": object()}
    return pipeline


def _patch_rembg(monkeypatch, mask_or_exc):
    """Patch module-level rembg_remove to return a mask (or raise)."""
    import app.models.profile_preprocessing_pipeline as mod

    def fake_remove(image, session=None, only_mask=False):
        if isinstance(mask_or_exc, Exception):
            raise mask_or_exc
        return mask_or_exc

    monkeypatch.setattr(mod, "rembg_remove", fake_remove)


def test_apply_white_background_fail_open(monkeypatch):
    """rembg raises → image unchanged and False (fail-open)."""
    pipeline = _make_pipeline()
    _patch_rembg(monkeypatch, RuntimeError("rembg boom"))

    image = np.full((16, 16, 3), 77, dtype=np.uint8)
    original = image.copy()

    out, applied = pipeline.apply_white_background(image)

    assert applied is False
    np.testing.assert_array_equal(out, original)


def test_apply_white_background_none_mask_fail_open(monkeypatch):
    """No matte returned → image unchanged and False (fail-open)."""
    pipeline = _make_pipeline()
    _patch_rembg(monkeypatch, None)

    image = np.full((16, 16, 3), 99, dtype=np.uint8)
    original = image.copy()

    out, applied = pipeline.apply_white_background(image)

    assert applied is False
    np.testing.assert_array_equal(out, original)


def test_apply_white_background_empty_mask_fail_open(monkeypatch):
    """Empty (size 0) matte → image unchanged and False (fail-open)."""
    pipeline = _make_pipeline()
    _patch_rembg(monkeypatch, np.empty((0, 0), dtype=np.uint8))

    image = np.full((16, 16, 3), 55, dtype=np.uint8)
    original = image.copy()

    out, applied = pipeline.apply_white_background(image)

    assert applied is False
    np.testing.assert_array_equal(out, original)


def test_apply_white_background_composites_on_white(monkeypatch):
    """A hard matte makes background white and keeps subject; applied is True."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    pipeline = _make_pipeline()
    _patch_rembg(monkeypatch, mask)

    image = np.full((40, 40, 3), (10, 20, 30), dtype=np.uint8)
    out, applied = pipeline.apply_white_background(image)

    assert applied is True
    # Subject core preserved
    assert np.allclose(out[15:25, 15:25], (10, 20, 30), atol=1)
    # A background corner is white
    assert np.all(out[0, 0] == 255)


# --- Matte refinement (_refine_matte) ---


def test_refine_matte_preserves_soft_edge():
    """Mid-fringe stays soft (0 < α < 1); core stays solid after edge AA blur.

    Locks the key behaviour change vs the old MediaPipe path: no hard threshold
    re-hardening. Gaussian edge AA may shift the fringe value but must not
    force it to 0 or 1.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((40, 40), dtype=np.float32)
    mask[8:32, 8:32] = 1.0
    mask[8:32, 7] = 0.5   # one soft edge column
    alpha = ProfilePreprocessingPipeline._refine_matte(mask)

    assert float(alpha[20, 20]) == 1.0            # interior solid, untouched
    fringe = float(alpha[20, 7])
    assert 0.0 < fringe < 1.0, f"soft fringe hardened: {fringe}"


def test_refine_matte_accepts_uint8_and_normalises():
    """A 0–255 uint8 matte is normalised to [0, 1]."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    alpha = ProfilePreprocessingPipeline._refine_matte(mask)
    assert float(alpha.max()) == 1.0
    assert float(alpha[15, 15]) == 1.0
    assert float(alpha[0, 0]) == 0.0


def test_refine_matte_drops_detached_speck():
    """A big subject blob is kept; a small detached speck is dropped."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((60, 60), dtype=np.float32)
    mask[10:50, 10:40] = 1.0   # large subject
    mask[2:6, 52:56] = 1.0     # small detached speck
    alpha = ProfilePreprocessingPipeline._refine_matte(mask)

    assert float(alpha[30, 25]) == 1.0   # subject core kept
    assert float(alpha[4, 54]) == 0.0    # detached speck removed


def test_refine_matte_preserves_soft_edge_when_keeplargest_fires():
    """Soft subject fringe survives even when keep-largest drops a speck.

    Regression for the bug where multiplying the soft alpha by the largest
    component zeroed the subject's anti-aliased fringe (labelled background)
    whenever a detached speck triggered keep-largest. The speck must be dropped
    AND the soft edge column must keep its 0.5 value.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((60, 60), dtype=np.float32)
    mask[10:50, 10:40] = 1.0   # large subject
    mask[10:50, 9] = 0.5       # soft (anti-aliased) fringe column, alpha < 0.5
    mask[2:6, 52:56] = 1.0     # detached speck (forces keep-largest)
    alpha = ProfilePreprocessingPipeline._refine_matte(mask)

    assert float(alpha[30, 25]) == 1.0            # subject core kept
    fringe = float(alpha[30, 9])
    assert 0.0 < fringe < 1.0, f"soft fringe hardened: {fringe}"
    assert float(alpha[4, 54]) == 0.0             # detached speck dropped


def test_refine_matte_single_component_untouched():
    """A single clean component is returned intact (keep-largest is a no-op)."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((50, 50), dtype=np.float32)
    mask[10:40, 10:40] = 1.0
    alpha = ProfilePreprocessingPipeline._refine_matte(mask)
    assert float(alpha[25, 25]) == 1.0
    assert float(alpha.sum()) == float(mask.sum())  # nothing zeroed


def test_refine_matte_fills_interior_hair_hole():
    """Enclosed zero pocket inside the subject becomes opaque (hair hole fill)."""
    from app.utils.matte_refine import refine_person_matte

    mask = np.zeros((60, 60), dtype=np.float32)
    mask[10:50, 10:50] = 1.0
    mask[25:35, 25:35] = 0.0  # interior hole
    alpha = refine_person_matte(mask)
    assert float(alpha[30, 30]) == 1.0
    # Deep interior stays opaque; near-edge may be slightly <1 after edge AA blur.
    assert float(alpha[20, 20]) >= 0.99
    assert float(alpha[2, 2]) == 0.0


def test_refine_matte_open_drops_thin_protrusion():
    """Thin chin/jaw FG stick detached by morph-open is dropped."""
    from app.utils.matte_refine import refine_person_matte

    mask = np.zeros((80, 80), dtype=np.float32)
    mask[20:60, 20:55] = 1.0  # head
    # Thin stick into empty space (pen / pink blob after open)
    mask[40:42, 55:75] = 1.0
    alpha = refine_person_matte(mask)
    assert float(alpha[40, 40]) == 1.0
    assert float(alpha[40, 70]) == 0.0


def test_refine_matte_preserves_soft_edge_without_torch():
    """Soft AA fringe survives morph when no face guards (no Gaussian AA)."""
    from app.utils.matte_refine import refine_person_matte

    mask = np.zeros((40, 40), dtype=np.float32)
    mask[8:32, 8:32] = 1.0
    mask[8:32, 7] = 0.5
    alpha = refine_person_matte(mask)
    assert float(alpha[20, 20]) == 1.0
    fringe = float(alpha[20, 7])
    assert 0.0 < fringe < 1.0, f"soft fringe hardened: {fringe}"


def test_refine_matte_corner_fg_does_not_paint_background():
    """If subject touches (0,0), hole-fill must not force all BG opaque."""
    from app.utils.matte_refine import refine_person_matte

    mask = np.zeros((50, 50), dtype=np.float32)
    mask[0:30, 0:30] = 1.0  # touches corner
    mask[10:18, 10:18] = 0.0  # interior hole
    alpha = refine_person_matte(mask)
    assert float(alpha[14, 14]) == 1.0  # hole filled
    assert float(alpha[45, 45]) == 0.0  # far BG stays BG


# --- Face guard: the face region must NEVER be clipped ---


def test_refine_matte_protect_rect_forces_face_opaque():
    """Even a fully-background matte keeps the ENTIRE protected rect exactly opaque.

    Guards the hard requirement: the face is never clipped, so every pixel inside
    the protect rect (edges and corners included) must be exactly 1.0.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # Matte is completely wrong: says everything is background.
    mask = np.zeros((60, 60), dtype=np.float32)
    protect = (20, 20, 40, 40)

    alpha = ProfilePreprocessingPipeline._refine_matte(mask, protect_rect=protect)
    # EVERY pixel in the rect is fully opaque (min == 1.0), not just the center.
    assert float(alpha[20:40, 20:40].min()) == 1.0
    # Far outside stays background.
    assert float(alpha[2, 2]) == 0.0


def test_refine_matte_protect_rect_tiny_face_opaque():
    """A very small protect rect is still exactly opaque."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    mask = np.zeros((40, 40), dtype=np.float32)
    protect = (18, 18, 22, 22)  # 4x4 face core
    alpha = ProfilePreprocessingPipeline._refine_matte(mask, protect_rect=protect)
    assert float(alpha[18:22, 18:22].min()) == 1.0


def test_face_protect_rect_is_central_core_inside_bbox():
    """Protect rect is a small central core strictly INSIDE the face bbox.

    Regression for the bug where an outward-expanded guard clamped to the whole
    crop and forced the background opaque (killing background removal). The guard
    must stay on-subject: centered on, and contained within, the detected bbox.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # Face bbox (10,10)-(50,60), fw=40 fh=50, center (30,35), half=(10,12.5).
    rect = ProfilePreprocessingPipeline._face_protect_rect(10, 10, 50, 60, 100, 100)
    # Exact central core (Python round: 22.5→22, 47.5→48).
    assert rect == (20, 22, 40, 48)
    px1, py1, px2, py2 = rect
    # ≈50% of each bbox side (core is clearly smaller than the bbox).
    assert (px2 - px1) == 20 and (py2 - py1) == 26


def test_face_protect_rect_bbox_fills_crop_leaves_corners_free():
    """The production failure shape: bbox ≈ fills the crop → guard must NOT cover
    the crop corners (else the whole frame is forced opaque and no bg is removed).

    Direct regression for the shipped bug where the outward-expanded guard clamped
    to the entire crop. The central core must leave a wide unprotected margin.
    """
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # Detected head bbox nearly fills a 200x200 crop (crop = bbox + tiny padding).
    cw, ch = 200, 200
    rect = ProfilePreprocessingPipeline._face_protect_rect(4, 4, 196, 196, cw, ch)
    assert rect is not None
    px1, py1, px2, py2 = rect
    # Corners are OUTSIDE the guard on every side (background removable there).
    assert px1 > 0 and py1 > 0 and px2 < cw and py2 < ch
    margin_frac = px1 / cw
    assert margin_frac > 0.2, f"guard margin too thin: {margin_frac:.2f}"
    # Guard covers well under half the crop area (rembg removes the rest).
    guard_area = (px2 - px1) * (py2 - py1)
    assert guard_area < 0.30 * (cw * ch), f"guard covers too much: {guard_area}"


def test_face_protect_rect_degenerate_returns_none():
    """A zero/negative-area bbox yields no protection rect."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    assert ProfilePreprocessingPipeline._face_protect_rect(10, 10, 10, 10, 100, 100) is None
    assert ProfilePreprocessingPipeline._face_protect_rect(50, 50, 10, 10, 100, 100) is None


def test_apply_white_background_never_clips_face(monkeypatch):
    """End-to-end: the matte drops the face, protect_rect restores it."""
    pipeline = _make_pipeline()

    # Matte marks the face region as background (alpha 0) — worst case.
    mask = np.full((40, 40), 255, dtype=np.uint8)
    mask[10:30, 10:30] = 0
    _patch_rembg(monkeypatch, mask)

    image = np.full((40, 40, 3), (12, 34, 56), dtype=np.uint8)
    out, applied = pipeline.apply_white_background(image, protect_rect=(12, 12, 28, 28))

    assert applied is True
    # Face pixels survive (not whitened) thanks to the protection rect.
    assert np.allclose(out[20, 20], (12, 34, 56), atol=1)


def test_silhouette_subject_aware_restores_dark_profile_edge():
    """Dark nose/lips at bbox edge survive even when rembg zeros them.

    Regression for perfilder4: raising face_protect_core_frac locked a light-BG
    rectangle; subject-aware silhouette restore keeps dark face and drops light BG.
    Soft restore (>= ~0.85) is OK — hard 1.0 on the silhouette creates jagged edges.
    """
    from app.utils.matte_refine import refine_person_matte

    h, w = 80, 120
    image = np.full((h, w, 3), 255, dtype=np.uint8)  # light BG
    # Dark profile edge strip on the right (lips/chin stand-in), adjacent to FG
    image[25:55, 58:72] = (40, 30, 25)
    # Light tent-like BG inside expanded head area (must stay removable)
    image[10:22, 20:50] = (235, 230, 220)
    # Dark wall far from FG (must stay removable — not in dilate band).
    # Band radius is ~31px; place wall ≥40px past FG end (col 55 → col 100+).
    image[5:15, 100:110] = (45, 45, 50)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[20:60, 20:55] = 1.0  # rembg kept head core, dropped right silhouette

    silhouette = (15, 5, 78, 65)  # covers dark strip + light tent (not far wall)
    hard_core = (30, 30, 50, 50)

    alpha = refine_person_matte(
        mask,
        protect_rect=hard_core,
        image_rgb=image,
        silhouette_rect=silhouette,
    )
    # Dark lips/chin strip soft-restored (adjacent to FG; sample mid-strip)
    restored = float(alpha[40, 62])
    assert restored >= 0.85
    # Soft restore must not hard-force 1.0 (stair-step / pixelation regression).
    assert restored < 1.0, f"silhouette restore hardened to 1.0: {restored}"
    # Light tent inside silhouette NOT force-locked
    assert float(alpha[16, 35]) < 0.15
    # Far dark wall NOT restored (outside FG dilate band + outside silhouette)
    assert float(alpha[10, 105]) < 0.15
    # Hard core still forced
    assert float(alpha[40, 40]) == 1.0


def test_silhouette_restores_deep_profile_gap():
    """rembg often zeros >15px of nose/mouth when the face sits on the frame edge."""
    from app.utils.matte_refine import refine_person_matte

    h, w = 200, 200
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    fg_end = 100
    gap = 32
    image[60:140, fg_end : fg_end + gap] = (50, 40, 35)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[50:150, 40:fg_end] = 1.0
    silhouette = (30, 40, 180, 160)

    alpha = refine_person_matte(
        mask, image_rgb=image, silhouette_rect=silhouette
    )
    mid = float(alpha[100, fg_end + gap // 2])
    tip = float(alpha[100, fg_end + gap - 2])
    assert mid >= 0.85, f"mid-strip not restored: {mid}"
    assert tip >= 0.85, f"profile tip not restored after blur: {tip}"
    assert tip < 1.0


def test_frozen_near_fg_post_blur_does_not_grow_restore_band():
    """Post-blur soft restore must reuse frozen near_fg — not dilate from the shell.

    Subject-like pixels just outside the pre-restore dilate band stay unrestored.
    Recomputing near_fg after soft restore would grow the band (~31px) into them.
    """
    from app.utils.matte_refine import refine_person_matte, _silhouette_near_fg

    h, w = 200, 220
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    fg_end = 100
    # In-band dark tip (within ~31px dilate) — should soft-restore.
    image[70:130, fg_end : fg_end + 18] = (50, 40, 35)
    # Outside pre-restore band, but inside a band grown from the soft shell.
    outside_col = fg_end + 40  # 140
    image[70:130, outside_col : outside_col + 12] = (48, 38, 32)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[50:150, 40:fg_end] = 1.0
    frozen = _silhouette_near_fg(mask)
    assert int(frozen[100, fg_end + 10]) > 0
    assert int(frozen[100, outside_col + 5]) == 0

    silhouette = (30, 40, 210, 160)
    alpha = refine_person_matte(
        mask, image_rgb=image, silhouette_rect=silhouette
    )
    assert float(alpha[100, fg_end + 10]) >= 0.85
    # Freeze invariant: second pass must not pull outside-band subject into matte.
    outside = float(alpha[100, outside_col + 5])
    assert outside < 0.15, f"restore band grew past freeze: alpha={outside}"


def test_gaussian_edge_aa_softens_fringe_keeps_protect_core():
    """Gaussian edge AA leaves mid-fringe soft; protect core stays ~1.0."""
    from app.utils.matte_refine import refine_person_matte

    mask = np.zeros((60, 60), dtype=np.float32)
    mask[15:45, 15:45] = 1.0  # hard block — blur should soften the edge
    protect = (25, 25, 35, 35)
    alpha = refine_person_matte(mask, protect_rect=protect)

    assert float(alpha[30, 30]) == 1.0  # protect core re-hardened
    # Sample just outside the hard block edge (column 14 neighbors FG at 15)
    fringe = float(alpha[30, 14])
    assert 0.0 < fringe < 1.0, f"expected soft mid-fringe, got {fringe}"


def test_yellowish_tent_inside_silhouette_not_restored():
    """Yellowish light strip (luma~205, chroma~45) inside silhouette is BG-like."""
    from app.utils.matte_refine import refine_person_matte, subject_like_mask

    h, w = 80, 80
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    # Yellowish tent/wall: R=230 G=200 B=185 → luma≈207, chroma=45
    image[25:55, 58:72] = (230, 200, 185)

    # Confirm subject_like_mask classifies it as BG
    strip = image[40:41, 62:63]
    assert not bool(subject_like_mask(strip)[0, 0])

    mask = np.zeros((h, w), dtype=np.float32)
    mask[20:60, 20:55] = 1.0
    silhouette = (15, 5, 78, 65)

    alpha = refine_person_matte(
        mask,
        image_rgb=image,
        silhouette_rect=silhouette,
    )
    # Must NOT be silhouette-restored into the matte
    assert float(alpha[40, 62]) < 0.15


def test_face_silhouette_rect_expands_beyond_bbox():
    """Silhouette rect is larger than the detector bbox (clamped to crop)."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # Front hugs right edge → more expand on right than left
    rect = ProfilePreprocessingPipeline._face_silhouette_rect(
        20, 20, 60, 80, 100, 100, expand_frac=0.18
    )
    assert rect is not None
    sx1, sy1, sx2, sy2 = rect
    assert sx1 < 20 and sx2 > 60
    assert sy2 > 80  # chin expand
    assert sx1 >= 0 and sy1 >= 0 and sx2 <= 100 and sy2 <= 100
    # Asymmetric: right expand (fw*0.18) > left expand (fw*0.18*0.5)
    assert (sx2 - 60) > (20 - sx1)


def test_face_silhouette_rect_asymmetric_toward_left_edge():
    """When face is closer to left crop edge, expand more leftward."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # gap_left=10, gap_right=30 → front is left
    rect = ProfilePreprocessingPipeline._face_silhouette_rect(
        10, 20, 70, 80, 100, 100, expand_frac=0.18
    )
    assert rect is not None
    sx1, _, sx2, _ = rect
    assert (10 - sx1) > (sx2 - 70)