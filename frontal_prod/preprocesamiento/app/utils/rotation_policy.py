"""Rotation alignment policy (pure; no heavy deps).

Mirrored from profile_prod (separate Docker image — no shared package).
"""

# Skip catastrophic mis-rotations from bad eye landmarks (QA: ~45–90° flips).
MAX_ABS_ROTATION_DEG = 35.0


def should_skip_rotation(
    angle_deg: float, max_abs: float = MAX_ABS_ROTATION_DEG
) -> bool:
    """Return True when alignment should keep the original image."""
    return abs(angle_deg) > max_abs
