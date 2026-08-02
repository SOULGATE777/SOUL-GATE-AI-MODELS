"""Unit tests for white-BG Form overrides and effective_pipeline echo."""

from unittest.mock import MagicMock

import numpy as np
import pytest


def test_face_protect_rect_core_frac_scales_guard():
    """Larger core_frac yields a larger protect rect; default 0.25 matches prior math."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    # Face bbox (10,10)-(50,60), fw=40 fh=50, center (30,35).
    default = ProfilePreprocessingPipeline._face_protect_rect(10, 10, 50, 60, 100, 100)
    assert default == (20, 22, 40, 48)

    half = ProfilePreprocessingPipeline._face_protect_rect(
        10, 10, 50, 60, 100, 100, core_frac=0.5
    )
    # half_w=20, half_h=25 → (10, 10, 50, 60)
    assert half == (10, 10, 50, 60)

    tiny = ProfilePreprocessingPipeline._face_protect_rect(
        10, 10, 50, 60, 100, 100, core_frac=0.1
    )
    # half_w=4, half_h=5 → (26, 30, 34, 40)
    assert tiny == (26, 30, 34, 40)
    assert (tiny[2] - tiny[0]) < (default[2] - default[0])


def test_face_protect_rect_default_core_frac_keeps_legacy_calls():
    """Calls without core_frac still use 0.25 (old hardcoded half-extent)."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    rect = ProfilePreprocessingPipeline._face_protect_rect(4, 4, 196, 196, 200, 200)
    assert rect is not None
    px1, py1, px2, py2 = rect
    assert px1 > 0 and py1 > 0 and px2 < 200 and py2 < 200


def test_apply_white_bg_false_skips_rembg():
    """apply_white_bg=False must not call rembg / apply_white_background."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline
    from app.utils.image_processing import ImageProcessor

    pipe = ProfilePreprocessingPipeline.__new__(ProfilePreprocessingPipeline)
    pipe.default_target_size = (600, 600)
    pipe.default_padding_factor = 0.40
    pipe.rembg_edge_margin_frac = 0.08
    pipe.rembg_edge_margin_min_px = 12
    pipe.face_protect_core_frac = 0.25
    pipe.rembg_model_name = "isnet-general-use"
    pipe._rembg_sessions = {}
    pipe.apply_white_background = MagicMock(
        side_effect=AssertionError("rembg must not be called when apply_white_bg=False")
    )
    ImageProcessor.maybe_enhance_dark = staticmethod(lambda img: (img, False))

    image = np.full((800, 800, 3), 120, dtype=np.uint8)
    bbox = [300.0, 300.0, 500.0, 500.0]
    out, applied, illum = pipe.crop_face_with_padding(
        image, bbox, (600, 600), 0.40, apply_white_bg=False
    )

    assert out.shape == (600, 600, 3)
    assert applied is False
    assert illum is False
    pipe.apply_white_background.assert_not_called()


def test_effective_pipeline_echo_defaults():
    """process_image returns effective_pipeline with resolved defaults (no detections)."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import (
        PADDING_ASYMMETRIC,
        ProfilePreprocessingPipeline,
    )

    pipe = ProfilePreprocessingPipeline.__new__(ProfilePreprocessingPipeline)
    pipe.default_confidence_threshold = 0.5
    pipe.default_target_size = (600, 600)
    pipe.default_padding_factor = 0.40
    pipe.rembg_edge_margin_frac = 0.08
    pipe.rembg_edge_margin_min_px = 12
    pipe.face_protect_core_frac = 0.25
    pipe.rembg_model_name = "isnet-general-use"
    pipe.rotation_aligner = None
    pipe.detect_faces = MagicMock(return_value=[])

    image = np.full((64, 64, 3), 80, dtype=np.uint8)
    result = pipe.process_image(image, apply_rotation=False)

    ep = result["effective_pipeline"]
    assert ep["rembg_model"] == "isnet-general-use"
    assert ep["apply_white_bg"] is True
    assert ep["rembg_edge_margin_frac"] == 0.08
    assert ep["rembg_edge_margin_min_px"] == 12
    assert ep["face_protect_core_frac"] == 0.25
    assert ep["padding_asymmetric"] == PADDING_ASYMMETRIC
    assert ep["default_padding_factor"] == 0.40
    assert result["processing_parameters"]["effective_pipeline"] == ep


def test_effective_pipeline_echo_overrides():
    """Request kwargs resolve into effective_pipeline without mutating instance attrs."""
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    pipe = ProfilePreprocessingPipeline.__new__(ProfilePreprocessingPipeline)
    pipe.default_confidence_threshold = 0.5
    pipe.default_target_size = (600, 600)
    pipe.default_padding_factor = 0.40
    pipe.rembg_edge_margin_frac = 0.08
    pipe.rembg_edge_margin_min_px = 12
    pipe.face_protect_core_frac = 0.25
    pipe.rembg_model_name = "isnet-general-use"
    pipe.rotation_aligner = None
    pipe.detect_faces = MagicMock(return_value=[])

    image = np.full((64, 64, 3), 80, dtype=np.uint8)
    result = pipe.process_image(
        image,
        apply_rotation=False,
        apply_white_bg=False,
        rembg_model="u2net",
        rembg_edge_margin_frac=0.12,
        rembg_edge_margin_min_px=20,
        face_protect_core_frac=0.4,
    )

    ep = result["effective_pipeline"]
    assert ep["apply_white_bg"] is False
    assert ep["rembg_model"] == "u2net"
    assert ep["rembg_edge_margin_frac"] == 0.12
    assert ep["rembg_edge_margin_min_px"] == 20
    assert ep["face_protect_core_frac"] == 0.4
    # Instance defaults unchanged (no request-local mutation leak).
    assert pipe.rembg_model_name == "isnet-general-use"
    assert pipe.rembg_edge_margin_frac == 0.08
    assert pipe.face_protect_core_frac == 0.25


def test_get_rembg_session_caches_by_model_name(monkeypatch):
    """Sessions are cached per model name; None uses rembg_model_name."""
    pytest.importorskip("torch")
    import app.models.profile_preprocessing_pipeline as mod
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    created = []

    def fake_new_session(name):
        created.append(name)
        return f"session:{name}"

    monkeypatch.setattr(mod, "rembg_new_session", fake_new_session)

    pipe = ProfilePreprocessingPipeline.__new__(ProfilePreprocessingPipeline)
    pipe.rembg_model_name = "isnet-general-use"
    pipe._rembg_sessions = {}

    s1 = pipe._get_rembg_session()
    s2 = pipe._get_rembg_session()
    s3 = pipe._get_rembg_session("u2net")
    s4 = pipe._get_rembg_session("u2net")

    assert s1 == "session:isnet-general-use"
    assert s2 is s1
    assert s3 == "session:u2net"
    assert s4 is s3
    assert created == ["isnet-general-use", "u2net"]


def test_check_white_bg_overrides_accepts_valid():
    from app.utils.white_bg_overrides import check_white_bg_overrides

    assert check_white_bg_overrides(None, None, None, None) is None
    assert check_white_bg_overrides("u2net", 0.1, 12, 0.25) is None
    assert check_white_bg_overrides("isnet-general-use", 0.0, 0, 0.1) is None
    assert check_white_bg_overrides("u2net", 0.25, 64, 0.8) is None


def test_check_white_bg_overrides_rejects_bad_model():
    from app.utils.white_bg_overrides import check_white_bg_overrides

    detail = check_white_bg_overrides("bad-model", None, None, None)
    assert detail is not None
    assert "rembg_model" in detail


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rembg_edge_margin_frac": -0.01},
        {"rembg_edge_margin_frac": 0.26},
        {"rembg_edge_margin_min_px": -1},
        {"rembg_edge_margin_min_px": 65},
        {"face_protect_core_frac": 0.05},
        {"face_protect_core_frac": 0.9},
    ],
)
def test_check_white_bg_overrides_rejects_ranges(kwargs):
    from app.utils.white_bg_overrides import check_white_bg_overrides

    base = {
        "rembg_model": None,
        "rembg_edge_margin_frac": None,
        "rembg_edge_margin_min_px": None,
        "face_protect_core_frac": None,
    }
    base.update(kwargs)
    assert check_white_bg_overrides(**base) is not None


def test_get_model_info_includes_white_bg_knobs():
    pytest.importorskip("torch")
    from app.models.profile_preprocessing_pipeline import ProfilePreprocessingPipeline

    pipe = ProfilePreprocessingPipeline.__new__(ProfilePreprocessingPipeline)
    pipe.device = "cpu"
    pipe.model_path = "/tmp/model.pth"
    pipe.num_classes = 2
    pipe.all_classes = ["profile"]
    pipe.default_confidence_threshold = 0.5
    pipe.default_target_size = (600, 600)
    pipe.default_padding_factor = 0.40
    pipe.rembg_model_name = "isnet-general-use"
    pipe.rembg_edge_margin_frac = 0.08
    pipe.rembg_edge_margin_min_px = 12
    pipe.face_protect_core_frac = 0.25

    info = pipe.get_model_info()
    assert info["rembg_model_name"] == "isnet-general-use"
    assert info["rembg_edge_margin_frac"] == 0.08
    assert info["rembg_edge_margin_min_px"] == 12
    assert info["face_protect_core_frac"] == 0.25
    assert info["padding_asymmetric"] == {"side": 1.35, "top": 1.65, "bottom": 1.20}
