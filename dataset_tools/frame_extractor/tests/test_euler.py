"""Unit tests for pitch normalization in ``app.euler``."""

from __future__ import annotations

import pytest

from app.euler import FaceEulerEstimator


class TestNormalizePitch:
    """Frontal-neutral raw ±180° must map to ~0; real tilts stay in (-90, 90]."""

    @pytest.mark.parametrize(
        ("raw_pitch", "expected"),
        [
            (-180.0, 0.0),
            (180.0, 0.0),
            (-175.0, 5.0),
            (175.0, -5.0),
            (0.0, 0.0),
            (30.0, 30.0),
            (-30.0, -30.0),
            (89.0, 89.0),
            (91.0, -89.0),
        ],
    )
    def test_mapping(self, raw_pitch: float, expected: float) -> None:
        assert FaceEulerEstimator._normalize_pitch(raw_pitch) == expected

    @pytest.mark.parametrize(
        "raw_pitch",
        [
            -180.0,
            180.0,
            -175.0,
            175.0,
            0.0,
            30.0,
            -30.0,
            89.0,
            91.0,
            -91.0,
            360.0,
            -360.0,
        ],
    )
    def test_result_within_ninety_degrees(self, raw_pitch: float) -> None:
        result = FaceEulerEstimator._normalize_pitch(raw_pitch)
        assert abs(result) <= 90.0
