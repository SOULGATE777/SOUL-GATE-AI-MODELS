"""Unit tests for angle-based subfolder routing in ``app.naming``."""

from __future__ import annotations

import pytest

from app.naming import (
    SUBRANGE_BOUNDARY_DEG,
    _horizontal_subfolder,
    _vertical_subfolder,
    resolve_category_subfolder,
)


class TestHorizontalSubfolder:
    @pytest.mark.parametrize(
        ("yaw", "expected"),
        [
            (0.0, "rh_1D"),
            (44.0, "rh_1D"),
            (44.9, "rh_1D"),
            (45.0, "rh_2D"),
            (89.0, "rh_2D"),
            (-0.1, "rh_1I"),
            (-44.0, "rh_1I"),
            (-44.9, "rh_1I"),
            (-45.0, "rh_1I"),
            (-45.1, "rh_2I"),
            (-89.0, "rh_2I"),
        ],
    )
    def test_yaw_bands(self, yaw: float, expected: str) -> None:
        assert _horizontal_subfolder(yaw) == expected

    def test_boundary_constant(self) -> None:
        assert SUBRANGE_BOUNDARY_DEG == 45


class TestVerticalSubfolder:
    @pytest.mark.parametrize(
        ("pitch", "expected"),
        [
            (0.0, "rv_1A"),
            (44.0, "rv_1A"),
            (-0.1, "rv_1B"),
            (-44.0, "rv_1B"),
        ],
    )
    def test_pitch_sign(self, pitch: float, expected: str) -> None:
        assert _vertical_subfolder(pitch) == expected


class TestResolveCategorySubfolder:
    @pytest.mark.parametrize(
        ("rotation_type", "yaw", "expected_subfolder"),
        [
            ("horizontal_0_45", 0.0, "rh_1D"),
            ("horizontal_0_45", 44.0, "rh_1D"),
            ("horizontal_0_45", 45.0, "rh_2D"),
            ("horizontal_45_90", 60.0, "rh_2D"),
            ("horizontal_0_neg45", -10.0, "rh_1I"),
            ("horizontal_0_neg45", -45.0, "rh_1I"),
            ("horizontal_0_neg45", -45.1, "rh_2I"),
            ("horizontal_neg45_neg90", -60.0, "rh_2I"),
        ],
    )
    def test_horizontal_routes_by_yaw_not_label(
        self,
        rotation_type: str,
        yaw: float,
        expected_subfolder: str,
    ) -> None:
        category, subfolder = resolve_category_subfolder(
            rotation_type, yaw, 0.0, 0.0
        )
        assert category == "rotacion_horizontal"
        assert subfolder == expected_subfolder

    @pytest.mark.parametrize(
        ("rotation_type", "pitch", "expected_subfolder"),
        [
            ("vertical_0_45", 10.0, "rv_1A"),
            ("vertical_0_45", 0.0, "rv_1A"),
            ("vertical_0_neg45", -5.0, "rv_1B"),
        ],
    )
    def test_vertical_routes_by_pitch_not_label(
        self,
        rotation_type: str,
        pitch: float,
        expected_subfolder: str,
    ) -> None:
        category, subfolder = resolve_category_subfolder(
            rotation_type, 0.0, pitch, 0.0
        )
        assert category == "rotacion_vertical"
        assert subfolder == expected_subfolder

    def test_circular_quadrant_unchanged(self) -> None:
        category, subfolder = resolve_category_subfolder(
            "circular", 45.0, 20.0, 0.0
        )
        assert category == "rotacion_circular"
        assert subfolder == "rc_Q1"
