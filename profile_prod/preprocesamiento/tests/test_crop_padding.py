"""Unit tests for asymmetric crop padding and rotation angle clamp."""

import numpy as np
import pytest

from app.utils.rotation_policy import MAX_ABS_ROTATION_DEG, should_skip_rotation


def test_rotation_policy_skips_extreme_angles():
    assert should_skip_rotation(90.0) is True
    assert should_skip_rotation(-90.0) is True
    assert should_skip_rotation(MAX_ABS_ROTATION_DEG + 0.1) is True
    assert should_skip_rotation(MAX_ABS_ROTATION_DEG) is False
    assert should_skip_rotation(10.0) is False
    assert should_skip_rotation(-12.5) is False


def test_asymmetric_pad_math_coeffs():
    """Locked pad coeffs: sides×1.35, top×1.65, bottom×1.20 of base pad."""
    box_w, box_h, pf = 200.0, 200.0, 0.40
    pad_side = box_w * pf * 1.35
    pad_h = box_h * pf
    pad_top = pad_h * 1.65
    pad_bottom = pad_h * 1.20
    assert pad_side == pytest.approx(108.0)
    assert pad_top == pytest.approx(132.0)
    assert pad_bottom == pytest.approx(96.0)
    assert pad_side > box_w * pf
    assert pad_top > pad_bottom


def test_rembg_margin_formula():
    crop_h, crop_w = 400, 300
    frac, min_px = 0.08, 12
    margin = max(min_px, int(min(crop_h, crop_w) * frac))
    assert margin == 24  # 8% of 300


def test_default_padding_factor_is_generous():
    pytest.importorskip("torch")
    # Prefer reading class defaults from source attrs after __new__ without loading models.
    # ProfilePreprocessingPipeline.__init__ loads heavy weights; mirror locked constants.
    from app.models import profile_preprocessing_pipeline as mod

    src = open(mod.__file__, encoding="utf-8").read()
    assert "self.default_padding_factor = 0.40" in src
    assert "self.rembg_edge_margin_frac = 0.16" in src
    assert "self.rembg_edge_margin_min_px = 20" in src
    assert "self.face_protect_core_frac = 0.25" in src
    assert "self._rembg_sessions" in src


def test_crop_adds_white_ring_before_rembg():
    """White ring expands the tensor rembg sees; protect_rect is offset by exact margin."""
    pytest.importorskip("torch")
    from unittest.mock import MagicMock

    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline
    from app.utils.image_processing import ImageProcessor

    pipe = ProfilePreprocessingPipeline.__new__(ProfilePreprocessingPipeline)
    pipe.default_target_size = (600, 600)
    pipe.default_padding_factor = 0.40
    pipe.rembg_edge_margin_frac = 0.08
    pipe.rembg_edge_margin_min_px = 12
    pipe.face_protect_core_frac = 0.25
    pipe.rembg_model_name = "isnet-general-use"
    pipe._face_protect_rect = MagicMock(return_value=(10, 10, 50, 50))
    pipe._face_silhouette_rect = MagicMock(return_value=(5, 5, 55, 55))
    pipe.apply_white_background = MagicMock(
        side_effect=lambda img, protect_rect=None, rembg_model=None,
        silhouette_rect=None: (img, True)
    )
    ImageProcessor.maybe_enhance_dark = staticmethod(lambda img: (img, False))

    image = np.full((800, 800, 3), 120, dtype=np.uint8)
    bbox = [300.0, 300.0, 500.0, 500.0]
    box_w = box_h = 200.0
    pf = 0.40
    expected_side = box_w * pf * 1.35
    expected_top = box_h * pf * 1.65
    expected_bottom = box_h * pf * 1.20

    out, applied, _ = pipe.crop_face_with_padding(image, bbox, (600, 600), pf)
    assert out.shape == (600, 600, 3)
    assert applied is True

    call_img = pipe.apply_white_background.call_args[0][0]
    raw_h = int(expected_top + box_h + expected_bottom)
    raw_w = int(expected_side + box_w + expected_side)
    margin = max(12, int(min(raw_h, raw_w) * 0.08))
    assert call_img.shape[0] == raw_h + 2 * margin
    assert call_img.shape[1] == raw_w + 2 * margin

    protect = pipe.apply_white_background.call_args.kwargs.get("protect_rect")
    if protect is None and len(pipe.apply_white_background.call_args) > 1:
        protect = pipe.apply_white_background.call_args[1].get("protect_rect")
    assert protect == (10 + margin, 10 + margin, 50 + margin, 50 + margin)
    silhouette = pipe.apply_white_background.call_args.kwargs.get("silhouette_rect")
    assert silhouette == (5 + margin, 5 + margin, 55 + margin, 55 + margin)
