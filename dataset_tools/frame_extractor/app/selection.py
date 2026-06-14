"""
Pose-bin frame selection (deduplication + best-quality keep).

The raw extractor samples every frame (``frame_interval=1``) so no rotation
angle is missed when a sweep is fast. That density produces many near-duplicate
frames: when the subject moves slowly, dozens of consecutive frames share the
same measured pose and differ only by roll/pitch jitter, inflating the dataset
with redundant images.

This module groups detected frames into **pose bins** (a quantised signature on
the meaningful Euler axes for the rotation type) and lets the caller keep only
the single highest-quality frame per bin. The goal is even angular coverage with
one representative image per pose, instead of a temporal burst of look-alikes.

Binning axes per rotation type:

* ``horizontal_*`` → yaw            (left-right turn; pitch/roll are jitter)
* ``vertical_*``   → pitch          (up-down; yaw/roll are jitter)
* ``circular``     → (yaw, pitch)   (2-D grid; the sweep moves on both axes)
* legacy types     → dominant axis  (see :func:`app.euler.dominant_axis`)

Roll is intentionally excluded from the signature: incidental head tilt is the
main source of redundant keys, so it must not create extra bins. Among frames in
the same bin the highest ``quality_score`` wins.
"""

from __future__ import annotations

import math
import os
from typing import Optional

from .euler import dominant_axis

# ---------------------------------------------------------------------------
# Configuration (env-driven, overridable per request in main.py)
# ---------------------------------------------------------------------------

# Bin width in degrees. One representative frame is kept per bin, so smaller
# values preserve finer angular resolution at the cost of more frames.
DEFAULT_POSE_BIN_STEP_DEG: int = max(
    1, int(os.environ.get("POSE_BIN_STEP_DEG", "2"))
)

# Minimum composite quality score for a frame to be eligible. Scores below this
# are dropped. Default 0.0 keeps every detected frame and relies purely on
# per-bin best selection (quality scores are tightly clustered in practice).
DEFAULT_MIN_QUALITY_SCORE: float = float(
    os.environ.get("MIN_QUALITY_SCORE", "0.0")
)

# Master switch. When disabled the extractor keeps the legacy behaviour of
# uploading every sampled frame.
DEFAULT_DEDUP_ENABLED: bool = os.environ.get(
    "FRAME_DEDUP_ENABLED", "true"
).strip().lower() in ("1", "true", "yes", "on")

# Rotation types whose sweep moves on both yaw and pitch and therefore need a
# 2-D bin signature to preserve the trajectory.
_TWO_AXIS_ROTATION_TYPES: frozenset[str] = frozenset({"circular"})


def pose_bin_signature(
    rotation_type: str,
    yaw: Optional[float],
    pitch: Optional[float],
    roll: Optional[float],
    step: int = DEFAULT_POSE_BIN_STEP_DEG,
) -> Optional[tuple]:
    """
    Return a hashable pose-bin signature for a frame, or ``None`` when it cannot
    be binned (the relevant angle is missing, e.g. no face detected).

    The signature is a tuple ``(axis_label, *bin_indices)`` where each bin index
    is ``floor(angle / step)``. Two frames with the same signature occupy the
    same pose bin and are considered duplicates of one pose.

    Args:
        rotation_type: Rotation type string (case-insensitive).
        yaw:   Yaw in degrees, or None.
        pitch: Pitch in degrees, or None.
        roll:  Roll in degrees, or None (never used for binning).
        step:  Bin width in degrees (clamped to >= 1).

    Returns:
        Tuple signature, or ``None`` when the dominant angle is unavailable.
    """
    effective_step = max(1, int(step))
    rtype = rotation_type.lower()

    def _bin(value: float) -> int:
        return int(math.floor(value / effective_step))

    if rtype.startswith("horizontal"):
        if yaw is None:
            return None
        return ("yaw", _bin(yaw))

    if rtype.startswith("vertical"):
        if pitch is None:
            return None
        return ("pitch", _bin(pitch))

    if rtype in _TWO_AXIS_ROTATION_TYPES:
        if yaw is None or pitch is None:
            return None
        return ("yaw_pitch", _bin(yaw), _bin(pitch))

    # Legacy / unknown types: bin on the dominant axis only.
    axis = dominant_axis(rtype)
    angle_by_axis: dict[str, Optional[float]] = {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }
    angle = angle_by_axis.get(axis)
    if angle is None:
        return None
    return (axis, _bin(angle))


def is_better_candidate(
    new_quality: float,
    existing_quality: Optional[float],
) -> bool:
    """Return True when ``new_quality`` should replace the current bin winner."""
    return existing_quality is None or new_quality > existing_quality
