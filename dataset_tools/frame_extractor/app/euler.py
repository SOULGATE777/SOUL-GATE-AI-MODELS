"""
Euler-angle estimation from BGR frames via MediaPipe Face Mesh + cv2.solvePnP.

Initialisation mirrors the pattern used in
``frontal_prod/preprocesamiento/app/models/frontal_preprocessing_pipeline.py``:

* ``mp.solutions.face_mesh`` and ``mp.solutions.drawing_utils`` are stored as
  instance attributes.
* The FaceMesh detector is **lazily initialised** on first use (see
  ``_initialize_face_mesh``), exactly as in the pipeline template.
* ``static_image_mode=True``, ``max_num_faces=1``, ``refine_landmarks=True``.

Euler-angle convention
----------------------
Angles are extracted with :func:`cv2.RQDecomp3x3` applied to the rotation
matrix obtained from :func:`cv2.Rodrigues`.

* **Yaw**   (Y-axis): positive = head turned right, negative = left.
* **Pitch** (X-axis): positive = head up, negative = down.
* **Roll**  (Z-axis): positive = CCW tilt, negative = CW tilt.

Quality score
-------------
Composite of three normalised sub-scores, each in [0, 1]:

* **Sharpness** (40 %): Laplacian variance of the greyscale frame, capped at
  ``BLUR_NORM_CAP`` (200).
* **Face size** (30 %): face bounding-box area as a fraction of the frame,
  normalised by ``FACE_RATIO_CAP`` (25 %).
* **Detection confidence** (30 %): value reported by MediaPipe (or the
  configured threshold when not available).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical 6-point 3-D face model (mm, nose-tip as world origin).
# Points chosen for geometric spread and pose-estimation stability.
# ---------------------------------------------------------------------------
_CANONICAL_3D_FACE: np.ndarray = np.array(
    [
        [  0.0,    0.0,    0.0],   # Nose tip            → mesh landmark 4
        [  0.0, -330.0,  -65.0],   # Chin                → mesh landmark 152
        [-225.0,  170.0, -135.0],  # Left eye outer      → mesh landmark 33
        [ 225.0,  170.0, -135.0],  # Right eye outer     → mesh landmark 263
        [-150.0, -150.0, -125.0],  # Left mouth corner   → mesh landmark 61
        [ 150.0, -150.0, -125.0],  # Right mouth corner  → mesh landmark 291
    ],
    dtype=np.float32,
)

# MediaPipe Face Mesh landmark indices that correspond to the 3-D model above
_MESH_LM_INDICES: tuple[int, ...] = (4, 152, 33, 263, 61, 291)

# Normalisation caps for sub-scores
_BLUR_NORM_CAP: float = 200.0   # Laplacian variance considered "sharp enough"
_FACE_RATIO_CAP: float = 0.25   # Face occupying 25 % of frame = full size score


class EulerResult:
    """Result of Euler-angle estimation for a single frame.

    Attributes:
        yaw:   Yaw in degrees, or None when no face was detected.
        pitch: Pitch in degrees, or None when no face was detected.
        roll:  Roll in degrees, or None when no face was detected.
        quality_score: Composite quality metric in [0, 1].
        face_detected: Whether MediaPipe found a face in the frame.
        detection_confidence: Raw detection confidence in [0, 1].
    """

    __slots__ = (
        "yaw",
        "pitch",
        "roll",
        "quality_score",
        "face_detected",
        "detection_confidence",
    )

    def __init__(
        self,
        yaw: Optional[float],
        pitch: Optional[float],
        roll: Optional[float],
        quality_score: float,
        face_detected: bool,
        detection_confidence: float,
    ) -> None:
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.quality_score = quality_score
        self.face_detected = face_detected
        self.detection_confidence = detection_confidence

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON output."""
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "quality_score": self.quality_score,
            "face_detected": self.face_detected,
            "detection_confidence": self.detection_confidence,
        }


class FaceEulerEstimator:
    """
    Estimates yaw / pitch / roll Euler angles from a BGR frame using
    MediaPipe Face Mesh and :func:`cv2.solvePnP`.

    Mirrors the MediaPipe initialisation pattern from
    ``frontal_prod/preprocesamiento/app/models/frontal_preprocessing_pipeline.py``.
    The FaceMesh detector is lazily created on first call to :meth:`estimate`.

    Usage::

        estimator = FaceEulerEstimator()
        result = estimator.estimate(frame_bgr)
        print(result.yaw, result.pitch, result.roll, result.quality_score)
        estimator.close()
    """

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        """
        Args:
            min_detection_confidence: Minimum confidence for MediaPipe
                FaceMesh detection (default 0.5).
        """
        # Store mp namespaces as instance attributes (mirrors pipeline.py)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self._min_detection_confidence = min_detection_confidence
        # Lazy initialisation — matches the pattern in frontal_preprocessing_pipeline.py
        self.face_mesh_detector: Optional[mp.solutions.face_mesh.FaceMesh] = None

        logger.info(
            "FaceEulerEstimator created "
            f"(min_detection_confidence={min_detection_confidence})"
        )

    # ------------------------------------------------------------------
    # Lazy init — exact pattern from frontal_preprocessing_pipeline.py
    # ------------------------------------------------------------------

    def _initialize_face_mesh(self) -> None:
        """Lazily initialise the Face Mesh detector (first call only)."""
        if self.face_mesh_detector is None:
            self.face_mesh_detector = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_detection_confidence,
            )
            logger.info("MediaPipe FaceMesh initialised (lazy).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame_bgr: np.ndarray) -> EulerResult:
        """
        Estimate Euler angles and quality score for one BGR frame.

        Args:
            frame_bgr: OpenCV BGR image (numpy uint8 array, shape HxWx3).

        Returns:
            :class:`EulerResult` with angles in degrees and quality_score.
        """
        h, w = frame_bgr.shape[:2]
        blur_score = self._blur_score(frame_bgr)

        self._initialize_face_mesh()

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh_detector.process(rgb)
        except Exception as exc:
            logger.warning("MediaPipe processing failed: %s", exc)
            return EulerResult(
                yaw=None,
                pitch=None,
                roll=None,
                quality_score=0.0,
                face_detected=False,
                detection_confidence=0.0,
            )

        if not results.multi_face_landmarks:
            return EulerResult(
                yaw=None,
                pitch=None,
                roll=None,
                quality_score=0.0,
                face_detected=False,
                detection_confidence=0.0,
            )

        landmarks = results.multi_face_landmarks[0]

        # Detection confidence — use attribute when available (API varies by version)
        detection_confidence: float = self._min_detection_confidence
        if (
            hasattr(results, "multi_face_detection_confidences")
            and results.multi_face_detection_confidences
        ):
            detection_confidence = float(results.multi_face_detection_confidences[0])

        # 2-D pixel coordinates for the 6 anchor landmarks
        image_2d = self._extract_2d_points(landmarks, w, h)

        # Face bounding-box size fraction
        face_size_score = self._face_size_score(landmarks, w, h)

        # Head-pose via solvePnP → Euler angles
        yaw, pitch, roll = self._solve_pose(image_2d, w, h)

        quality_score = self._composite_quality(
            blur_score, face_size_score, detection_confidence
        )

        return EulerResult(
            yaw=round(float(yaw), 2),
            pitch=round(float(pitch), 2),
            roll=round(float(roll), 2),
            quality_score=round(quality_score, 4),
            face_detected=True,
            detection_confidence=round(detection_confidence, 4),
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.face_mesh_detector is not None:
            self.face_mesh_detector.close()
            self.face_mesh_detector = None
            logger.info("FaceEulerEstimator closed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_2d_points(
        landmarks: object,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Convert normalised landmark coords to pixel coords for the 6 anchors.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList for a single face.
            width:  Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            Float32 array of shape (6, 2).
        """
        pts: list[list[float]] = []
        for idx in _MESH_LM_INDICES:
            lm = landmarks.landmark[idx]
            pts.append([lm.x * width, lm.y * height])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _normalize_pitch(pitch_deg: float) -> float:
        """Shift raw pitch +180 so frontal-neutral reads 0, wrapped to (-180, 180]."""
        return ((pitch_deg + 360.0) % 360.0) - 180.0

    @staticmethod
    def _solve_pose(
        image_2d: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[float, float, float]:
        """
        Run :func:`cv2.solvePnP` and extract yaw / pitch / roll (degrees).

        Args:
            image_2d: Pixel coordinates for the 6 anchor points (shape 6×2).
            width:    Frame width (used to build the camera intrinsic matrix).
            height:   Frame height.

        Returns:
            ``(yaw_deg, pitch_deg, roll_deg)`` — all zero when solvePnP fails.
        """
        focal_length = float(max(width, height))
        camera_matrix = np.array(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        success, rvec, _ = cv2.solvePnP(
            _CANONICAL_3D_FACE,
            image_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.debug("solvePnP returned success=False; defaulting angles to 0.")
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)

        # RQDecomp3x3 returns [rx, ry, rz] in degrees (Euler XYZ convention).
        angles, _rx, _ry, _rz, _qr, _qt = cv2.RQDecomp3x3(rmat)

        pitch_deg = FaceEulerEstimator._normalize_pitch(float(angles[0]))  # X rotation = pitch (frontal → 0)
        yaw_deg   = float(angles[1])  # Y rotation = yaw
        roll_deg  = float(angles[2])  # Z rotation = roll

        return yaw_deg, pitch_deg, roll_deg

    @staticmethod
    def _blur_score(frame_bgr: np.ndarray) -> float:
        """
        Compute Laplacian variance of the greyscale frame as a sharpness proxy.

        Args:
            frame_bgr: BGR frame (numpy array).

        Returns:
            Non-negative float; higher = sharper.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _face_size_score(
        landmarks: object,
        width: int,
        height: int,
    ) -> float:
        """
        Fraction of the frame covered by the face bounding box, normalised to
        [0, 1] where 1.0 means the face occupies ``_FACE_RATIO_CAP`` (25 %) or
        more of the total frame area.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList.
            width:     Frame width in pixels.
            height:    Frame height in pixels.

        Returns:
            Float in [0, 1].
        """
        xs = np.array([lm.x for lm in landmarks.landmark], dtype=np.float32)
        ys = np.array([lm.y for lm in landmarks.landmark], dtype=np.float32)
        face_w = float((xs.max() - xs.min()) * width)
        face_h = float((ys.max() - ys.min()) * height)
        face_area = face_w * face_h
        total_area = float(width * height)
        if total_area == 0.0:
            return 0.0
        ratio = face_area / total_area
        return min(ratio / _FACE_RATIO_CAP, 1.0)

    @staticmethod
    def _composite_quality(
        blur_score: float,
        face_size_score: float,
        detection_confidence: float,
    ) -> float:
        """
        Weighted composite quality in [0, 1].

        Weights:
            40 % sharpness, 30 % face size, 30 % detection confidence.

        Args:
            blur_score:           Raw Laplacian variance.
            face_size_score:      Normalised face-area fraction.
            detection_confidence: MediaPipe confidence value.

        Returns:
            Float in [0, 1].
        """
        sharpness = min(blur_score / _BLUR_NORM_CAP, 1.0)
        return (
            0.4 * sharpness
            + 0.3 * face_size_score
            + 0.3 * float(detection_confidence)
        )


# ---------------------------------------------------------------------------
# Angle-range bucketing (public utility)
# ---------------------------------------------------------------------------

# Map from rotation_type → dominant Euler axis used for bucketing.
#
# Pipeline rotation-type strings are the lowercase form of the Prisma
# ``FaceRotationType`` enum (e.g. ``HORIZONTAL_0_45`` → ``horizontal_0_45``):
#
#   * horizontal_* → yaw   (left-right head turn)
#   * vertical_*   → pitch (up-down head movement)
#   * circular     → yaw   (continuous rotation; split into yaw quadrants)
#
# Legacy values (frontal / left / right / left_profile / right_profile /
# up / down / roll) are kept for backward compatibility with sessions captured
# before the redesign.
_DOMINANT_AXIS: dict[str, str] = {
    # --- New rotation types (Face Rotation Capture Redesign, 2026) ---
    "horizontal_0_45":        "yaw",
    "horizontal_45_90":       "yaw",
    "horizontal_0_neg45":     "yaw",
    "horizontal_neg45_neg90": "yaw",
    "vertical_0_45":          "pitch",
    "vertical_0_neg45":       "pitch",
    "circular":               "yaw",
    # --- Legacy rotation types (backward compatibility) ---
    "frontal":       "yaw",
    "left":          "yaw",
    "right":         "yaw",
    "left_profile":  "yaw",
    "right_profile": "yaw",
    "up":            "pitch",
    "down":          "pitch",
    "roll":          "roll",
}

# Rotation types whose frames are split into 90-degree yaw quadrants instead of
# the default fine-grained bins. The ``circular`` phase is one continuous head
# rotation; quadrant assignment happens here at extraction time.
_CIRCULAR_ROTATION_TYPES: frozenset[str] = frozenset({"circular"})

# Bin width (degrees) used for circular quadrant splitting. With floor-based
# bucketing this yields exactly four signed-yaw quadrant labels:
#   yaw in [0, 90)     → yaw_0_90      (Q1)
#   yaw in [90, 180)   → yaw_90_180    (Q2)
#   yaw in [-180, -90) → yaw_-180_-90  (Q3)
#   yaw in [-90, 0)    → yaw_-90_0     (Q4)
_CIRCULAR_QUADRANT_BIN_WIDTH: int = 90


def dominant_axis(rotation_type: str) -> str:
    """Return the dominant Euler axis (``yaw`` / ``pitch`` / ``roll``) for a rotation type.

    Falls back to ``yaw`` for unknown rotation types so legacy or unexpected
    values still bucket deterministically.

    Args:
        rotation_type: Video rotation type string (case-insensitive).

    Returns:
        ``"yaw"``, ``"pitch"``, or ``"roll"``.
    """
    return _DOMINANT_AXIS.get(rotation_type.lower(), "yaw")


def compute_angle_range(
    rotation_type: str,
    yaw: Optional[float],
    pitch: Optional[float],
    roll: Optional[float],
    bin_width: int = 5,
) -> str:
    """
    Bucket the dominant Euler angle into a degree bin.

    Bucketing scheme
    ~~~~~~~~~~~~~~~~
    The dominant axis depends on ``rotation_type`` (see :data:`_DOMINANT_AXIS`)::

        horizontal_*  → yaw   (left-right head turn)
        vertical_*    → pitch (up-down head movement)
        circular      → yaw   (split into 90-degree quadrants)
        frontal       → yaw   (legacy)
        left / right  → yaw   (legacy)
        up / down     → pitch (legacy)
        roll          → roll  (legacy)
        *other*       → yaw   (fallback)

    The bin label format is ``{axis}_{lo}_{hi}`` where ``lo`` and ``hi`` are
    integers (degrees).  Example: ``yaw_-10_-5``, ``pitch_5_10``.

    **Circular quadrant splitting.** For ``circular`` recordings the yaw angle
    is bucketed with a 90-degree width, producing four quadrant labels that stay
    consistent with the ``{axis}_{lo}_{hi}`` convention::

        yaw_0_90      (Q1:  0°   to  90°)
        yaw_90_180    (Q2:  90°  to 180°)
        yaw_-180_-90  (Q3: -180° to -90°)
        yaw_-90_0     (Q4: -90°  to   0°)

    When no face was detected (all angles ``None``) the label is ``no_face``.

    Args:
        rotation_type: Video rotation type string (case-insensitive).
        yaw:   Yaw in degrees, or None.
        pitch: Pitch in degrees, or None.
        roll:  Roll in degrees, or None.
        bin_width: Bucket width in degrees for non-circular types (default 5).

    Returns:
        Bucket label string, e.g. ``"yaw_-10_-5"``, ``"yaw_0_90"`` or
        ``"no_face"``.
    """
    angle_by_axis: dict[str, Optional[float]] = {
        "yaw":   yaw,
        "pitch": pitch,
        "roll":  roll,
    }
    rtype = rotation_type.lower()
    axis = dominant_axis(rtype)
    angle = angle_by_axis[axis]

    if angle is None:
        return "no_face"

    effective_bin_width = (
        _CIRCULAR_QUADRANT_BIN_WIDTH
        if rtype in _CIRCULAR_ROTATION_TYPES
        else bin_width
    )
    lo = int(math.floor(angle / effective_bin_width)) * effective_bin_width
    hi = lo + effective_bin_width
    return f"{axis}_{lo}_{hi}"
