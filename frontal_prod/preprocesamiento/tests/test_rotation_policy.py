"""Frontal rotation policy unit tests (no MediaPipe required)."""
from app.utils.rotation_policy import MAX_ABS_ROTATION_DEG, should_skip_rotation


def test_should_skip_catastrophic_angles():
    assert should_skip_rotation(90.0) is True
    assert should_skip_rotation(-90.0) is True
    assert should_skip_rotation(45.0) is True
    assert should_skip_rotation(MAX_ABS_ROTATION_DEG + 0.1) is True


def test_should_not_skip_mild_tilt():
    assert should_skip_rotation(MAX_ABS_ROTATION_DEG) is False
    assert should_skip_rotation(10.0) is False
    assert should_skip_rotation(-12.5) is False
    assert should_skip_rotation(2.5) is False
