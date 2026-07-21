"""CLAHE ownership: preprocess only — morph/antro must not re-apply."""

from pathlib import Path

import numpy as np

from app.utils.image_processing import maybe_enhance_dark

_REPO = Path(__file__).resolve().parents[3]

_DOWNSTREAM_NO_CLAHE = (
    "profile_prod/morfologico/app/utils/image_processing.py",
    "profile_prod/antropometrico/app/utils/image_processing.py",
    "frontal_prod/morfologico/app/utils/image_processing.py",
    "frontal_prod/antropometrico/app/utils/image_processing.py",
)


def test_preprocess_owns_clahe_on_dark_images():
    dark = np.full((64, 64, 3), 20, dtype=np.uint8)
    out, enhanced = maybe_enhance_dark(dark)
    assert enhanced is True
    assert not np.array_equal(out, dark)


def test_downstream_analysis_helpers_have_no_createCLAHE():
    """Morph/antro must not contain createCLAHE (pass-through ownership)."""
    for relative in _DOWNSTREAM_NO_CLAHE:
        path = _REPO / relative
        text = path.read_text(encoding="utf-8")
        assert "createCLAHE" not in text, f"{relative} still contains createCLAHE"


def test_preprocess_still_contains_createCLAHE():
    """Sanity: ownership lives in this service's enhance path."""
    path = Path(__file__).resolve().parents[1] / "app/utils/image_processing.py"
    assert "createCLAHE" in path.read_text(encoding="utf-8")
