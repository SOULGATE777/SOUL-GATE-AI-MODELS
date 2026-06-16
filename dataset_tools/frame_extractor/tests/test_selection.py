"""Unit tests for pose-bin selection / deduplication helpers."""

from __future__ import annotations

from app.selection import (
    is_better_candidate,
    pose_bin_signature,
)


class TestPoseBinSignature:
    """Binning axis and quantisation behaviour per rotation type."""

    def test_horizontal_bins_on_yaw(self):
        sig = pose_bin_signature("horizontal_0_45", yaw=12.4, pitch=3.0, roll=1.0, step=2)
        assert sig == ("yaw", 6)  # floor(12.4 / 2)

    def test_horizontal_ignores_pitch_and_roll(self):
        a = pose_bin_signature("horizontal_0_45", yaw=10.0, pitch=2.0, roll=0.0, step=2)
        b = pose_bin_signature("horizontal_0_45", yaw=10.9, pitch=40.0, roll=30.0, step=2)
        assert a == b == ("yaw", 5)

    def test_vertical_bins_on_pitch(self):
        sig = pose_bin_signature("vertical_0_45", yaw=80.0, pitch=21.0, roll=2.0, step=3)
        assert sig == ("pitch", 7)  # floor(21 / 3)

    def test_circular_bins_on_yaw_only(self):
        sig = pose_bin_signature("circular", yaw=30.0, pitch=12.0, roll=5.0, step=5)
        assert sig == ("yaw", 6)  # floor(30 / 5); pitch ignored

    def test_circular_pitch_jitter_same_yaw_same_bin(self):
        a = pose_bin_signature("circular", yaw=15.0, pitch=10.0, roll=1.0, step=2)
        b = pose_bin_signature("circular", yaw=15.0, pitch=40.0, roll=8.0, step=2)
        assert a == b == ("yaw", 7)

    def test_negative_angles_floor_correctly(self):
        # floor(-3/2) == -2, not -1 → adjacent negatives still separate cleanly
        sig = pose_bin_signature("horizontal_0_neg45", yaw=-3.0, pitch=0.0, roll=0.0, step=2)
        assert sig == ("yaw", -2)

    def test_step_clamped_to_minimum_one(self):
        sig = pose_bin_signature("horizontal_0_45", yaw=10.0, pitch=0.0, roll=0.0, step=0)
        assert sig == ("yaw", 10)

    def test_returns_none_when_dominant_angle_missing(self):
        assert (
            pose_bin_signature("horizontal_0_45", yaw=None, pitch=1.0, roll=1.0, step=2)
            is None
        )
        assert (
            pose_bin_signature("vertical_0_45", yaw=1.0, pitch=None, roll=1.0, step=2)
            is None
        )
        assert (
            pose_bin_signature("circular", yaw=None, pitch=1.0, roll=1.0, step=2)
            is None
        )
        assert pose_bin_signature(
            "circular", yaw=1.0, pitch=None, roll=1.0, step=2
        ) == ("yaw", 0)

    def test_legacy_type_bins_on_dominant_axis(self):
        # legacy "up"/"down" are pitch-dominant; "left"/"right" yaw-dominant
        yaw_sig = pose_bin_signature("left", yaw=20.0, pitch=2.0, roll=0.0, step=4)
        assert yaw_sig == ("yaw", 5)


class TestSameBinDedup:
    """Frames close in pose collapse to one bin; distinct poses stay separate."""

    def test_roll_jitter_collapses_to_same_bin(self):
        # The core duplication source: same yaw/pitch, differing roll.
        s1 = pose_bin_signature("circular", yaw=15.0, pitch=10.0, roll=1.0, step=2)
        s2 = pose_bin_signature("circular", yaw=15.0, pitch=10.0, roll=8.0, step=2)
        assert s1 == s2

    def test_distinct_yaw_separate_bins(self):
        s1 = pose_bin_signature("horizontal_0_45", yaw=10.0, pitch=0.0, roll=0.0, step=2)
        s2 = pose_bin_signature("horizontal_0_45", yaw=14.0, pitch=0.0, roll=0.0, step=2)
        assert s1 != s2


class TestIsBetterCandidate:
    def test_first_candidate_always_wins(self):
        assert is_better_candidate(0.30, None) is True

    def test_higher_quality_wins(self):
        assert is_better_candidate(0.45, 0.40) is True

    def test_lower_or_equal_quality_loses(self):
        assert is_better_candidate(0.40, 0.40) is False
        assert is_better_candidate(0.39, 0.40) is False
