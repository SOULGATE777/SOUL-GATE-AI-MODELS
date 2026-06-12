"""
Dataset S3 key layout and filename conventions.

See SOUL-GATE-FRONTEND-WEB docs/contribute/dataset-naming-convention.md for the
authoritative folder hierarchy and neutral-posture criteria.
"""

from __future__ import annotations

from typing import Optional

from .euler import EulerResult, compute_angle_range

# rotation_type (lowercase) → (category folder, subfolder or None for circular)
CATEGORY_MAP: dict[str, tuple[str, Optional[str]]] = {
    "horizontal_0_45": ("rotacion_horizontal", "rh_1D"),
    "horizontal_45_90": ("rotacion_horizontal", "rh_2D"),
    "horizontal_0_neg45": ("rotacion_horizontal", "rh_1I"),
    "horizontal_neg45_neg90": ("rotacion_horizontal", "rh_2I"),
    "vertical_0_45": ("rotacion_vertical", "rv_1A"),
    "vertical_0_neg45": ("rotacion_vertical", "rv_1B"),
    "circular": ("rotacion_circular", None),
}

CIRCULAR_SUBFOLDER_MAP: dict[str, str] = {
    "yaw_0_90": "rc_Q1",
    "yaw_90_180": "rc_Q2",
    "yaw_-180_-90": "rc_Q3",
    "yaw_-90_0": "rc_Q4",
}

# Sub-range split for horizontal/vertical (dataset-naming-convention.md)
SUBRANGE_BOUNDARY_DEG = 45

# Neutral posture thresholds (dataset-naming-convention.md)
NEUTRAL_FRONTAL_YAW_MAX = 5
NEUTRAL_FRONTAL_PITCH_MAX = 5
NEUTRAL_PROFILE_YAW_MIN = 80
NEUTRAL_PROFILE_YAW_MAX = -80
NEUTRAL_PROFILE_PITCH_MAX = 15

# Legacy rotation types keep the pre-redesign S3 layout (unchanged keys).
_LEGACY_ROTATION_TYPES: frozenset[str] = frozenset(
    {
        "frontal",
        "left",
        "right",
        "left_profile",
        "right_profile",
        "up",
        "down",
        "roll",
    }
)


def uses_new_dataset_layout(rotation_type: str) -> bool:
    """Return True when ``rotation_type`` maps to the new category tree."""
    return rotation_type.lower() in CATEGORY_MAP


def uses_new_naming(rotation_type: str) -> bool:
    """Alias for :func:`uses_new_dataset_layout` (used by ``main.py``)."""
    return uses_new_dataset_layout(rotation_type)


def is_legacy_rotation_type(rotation_type: str) -> bool:
    """Return True for pre-redesign rotation type strings."""
    return rotation_type.lower() in _LEGACY_ROTATION_TYPES


def resolve_capture_source_prefix(capture_source: Optional[str]) -> str:
    """
    Map capture origin to a single-letter filename prefix.

    ``web`` → ``W``, ``mobile`` → ``M``. Unknown or missing → empty (legacy).
    """
    if not capture_source:
        return ""
    normalized = capture_source.strip().lower()
    if normalized in ("web", "w"):
        return "W"
    if normalized in ("mobile", "m"):
        return "M"
    return ""


def format_profile_filename_token(
    profile_id: str,
    capture_source: Optional[str] = None,
) -> str:
    """``{W|M}{profileId}`` when source is known, else bare ``profileId``."""
    prefix = resolve_capture_source_prefix(capture_source)
    return f"{prefix}{profile_id}" if prefix else profile_id


def _format_euler_filename(
    profile_id: str,
    yaw: float,
    pitch: float,
    roll: float,
    capture_source: Optional[str] = None,
) -> str:
    """Build ``{W|M}{profileId}_y{yaw}_p{pitch}_r{roll}.jpg`` with integer angles."""
    token = format_profile_filename_token(profile_id, capture_source)
    y = int(round(yaw))
    p = int(round(pitch))
    r = int(round(roll))
    return f"{token}_y{y}_p{p}_r{r}.jpg"


def _horizontal_subfolder(yaw: float) -> str:
    """Route horizontal frames by measured yaw band (not source video label)."""
    if yaw >= SUBRANGE_BOUNDARY_DEG:
        return "rh_2D"
    if yaw >= 0:
        return "rh_1D"
    if yaw >= -SUBRANGE_BOUNDARY_DEG:
        return "rh_1I"
    return "rh_2I"


def _vertical_subfolder(pitch: float) -> str:
    """Route vertical frames by measured pitch sign (not source video label)."""
    return "rv_1A" if pitch >= 0 else "rv_1B"


def resolve_category_subfolder(
    rotation_type: str,
    yaw: Optional[float],
    pitch: Optional[float],
    roll: Optional[float],
) -> tuple[str, str]:
    """
    Map API rotation type + Euler angles to ``(category, subfolder)``.

    Category comes from ``CATEGORY_MAP``. Horizontal and vertical subfolders
    are derived from measured yaw/pitch bands so combined capture blobs (e.g.
    ``HORIZONTAL_0_45`` spanning 0°→90°) split correctly at extraction time.
    For ``circular``, subfolder is derived from yaw quadrant via
    :func:`compute_angle_range`.
    """
    rtype = rotation_type.lower()
    if rtype not in CATEGORY_MAP:
        raise ValueError(f"Unknown rotation type for new layout: {rotation_type}")

    category, mapped_subfolder = CATEGORY_MAP[rtype]
    if mapped_subfolder is None:
        angle_range = compute_angle_range(rtype, yaw, pitch, roll)
        subfolder = CIRCULAR_SUBFOLDER_MAP.get(angle_range, "rc_Q1")
    elif category == "rotacion_horizontal":
        if yaw is None:
            raise ValueError("yaw required for horizontal subfolder routing")
        subfolder = _horizontal_subfolder(yaw)
    elif category == "rotacion_vertical":
        if pitch is None:
            raise ValueError("pitch required for vertical subfolder routing")
        subfolder = _vertical_subfolder(pitch)
    else:
        subfolder = mapped_subfolder
    return category, subfolder


def build_frame_key(
    dataset_prefix: str,
    rotation_type: str,
    yaw: Optional[float],
    pitch: Optional[float],
    roll: Optional[float],
    profile_id: str,
    *,
    session_id: str = "",
    sampled_idx: int = 0,
    capture_source: Optional[str] = None,
) -> str:
    """
    Build the full S3 key for a dataset frame JPEG.

    New redesign types use ``{category}/{subfolder}/{profileId}_y*_p*_r*.jpg``.
    Legacy types keep ``{rotationType}/{angleRange}/{sessionId}_...``.
    """
    rtype = rotation_type.lower()

    if rtype in _LEGACY_ROTATION_TYPES or rtype not in CATEGORY_MAP:
        angle_range = compute_angle_range(rotation_type, yaw, pitch, roll)
        frame_id = f"{session_id}_{rotation_type}_{sampled_idx:04d}"
        return f"{dataset_prefix}/{rotation_type}/{angle_range}/{frame_id}.jpg"

    if yaw is None or pitch is None or roll is None:
        angle_range = compute_angle_range(rotation_type, yaw, pitch, roll)
        frame_id = f"{session_id}_{rotation_type}_{sampled_idx:04d}"
        return f"{dataset_prefix}/{rotation_type}/{angle_range}/{frame_id}.jpg"

    category, subfolder = resolve_category_subfolder(
        rotation_type, yaw, pitch, roll
    )
    filename = _format_euler_filename(
        profile_id, yaw, pitch, roll, capture_source=capture_source
    )
    return f"{dataset_prefix}/{category}/{subfolder}/{filename}"


def classify_neutral_posture(
    yaw: Optional[float],
    pitch: Optional[float],
) -> Optional[str]:
    """
    Classify a frame into a neutral posture subfolder, if it qualifies.

    Returns:
        ``pn_frontal``, ``pn_perfil_D``, ``pn_perfil_I``, or ``None``.
    """
    if yaw is None or pitch is None:
        return None

    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_yaw < NEUTRAL_FRONTAL_YAW_MAX and abs_pitch < NEUTRAL_FRONTAL_PITCH_MAX:
        return "pn_frontal"
    if yaw > NEUTRAL_PROFILE_YAW_MIN and abs_pitch < NEUTRAL_PROFILE_PITCH_MAX:
        return "pn_perfil_D"
    if yaw < NEUTRAL_PROFILE_YAW_MAX and abs_pitch < NEUTRAL_PROFILE_PITCH_MAX:
        return "pn_perfil_I"
    return None


def build_neutral_key(
    dataset_prefix: str,
    neutral_category: str,
    profile_id: str,
    result: EulerResult,
    *,
    capture_source: Optional[str] = None,
) -> str:
    """S3 key under ``posturas_neutrales/{neutral_category}/``."""
    yaw = result.yaw if result.yaw is not None else 0.0
    pitch = result.pitch if result.pitch is not None else 0.0
    roll = result.roll if result.roll is not None else 0.0
    filename = _format_euler_filename(
        profile_id, yaw, pitch, roll, capture_source=capture_source
    )
    return f"{dataset_prefix}/posturas_neutrales/{neutral_category}/{filename}"


def build_manifest_key(
    dataset_prefix: str,
    rotation_type: str,
    session_id: str,
) -> str:
    """
    S3 key for per-video extraction manifest JSON.

    New layout: ``{prefix}/{category}/{sessionId}_{rotationType}_manifest.json``.
    Legacy: ``{prefix}/{rotationType}/{sessionId}_manifest.json``.
    """
    rtype = rotation_type.lower()
    if uses_new_dataset_layout(rotation_type):
        category, _ = CATEGORY_MAP[rtype]
        return (
            f"{dataset_prefix}/{category}/"
            f"{session_id}_{rotation_type}_manifest.json"
        )
    return f"{dataset_prefix}/{rotation_type}/{session_id}_manifest.json"
