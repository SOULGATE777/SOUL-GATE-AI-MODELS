"""
Frame Extractor — FastAPI microservice (port 8020).

Downloads face-rotation videos from AWS S3, extracts frames with OpenCV,
annotates each with Euler angles (yaw / pitch / roll) via MediaPipe Face Mesh,
computes a quality score, and writes the frame JPEGs + a JSON manifest back to
S3 under the ``face-rotation-dataset/`` prefix.

Endpoints
---------
GET  /health
POST /extract          — process a single video from an S3 key
POST /extract-session  — convenience: process all rotation types for a session

Pattern references
------------------
App structure, CORS, and lazy-loading follow:
  frontal_prod/preprocesamiento/app/main.py
  frontal_prod/rotacion/app/main.py
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .euler import EulerResult, FaceEulerEstimator, dominant_axis
from .naming import (
    build_frame_key,
    build_manifest_key,
    build_neutral_key,
    classify_neutral_posture,
)
from .selection import (
    DEFAULT_DEDUP_ENABLED,
    DEFAULT_MIN_QUALITY_SCORE,
    DEFAULT_POSE_BIN_STEP_DEG,
    is_better_candidate,
    pose_bin_signature,
)
from .profiles import get_or_create_profile_id
from .s3_client import (
    S3ClientError,
    download_video,
    upload_frame,
    upload_manifest,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default rotation types processed by /extract-session
#
# Prisma ``FaceRotationType`` enum values (must match S3 keys: {TYPE}.mp4).
# 4 horizontal + 2 vertical + 1 circular = 7 videos per session.
# Legacy lowercase names are still accepted when passed explicitly in the body.
# ---------------------------------------------------------------------------
DEFAULT_ROTATION_TYPES: list[str] = [
    "HORIZONTAL_0_45",
    "HORIZONTAL_45_90",
    "HORIZONTAL_0_NEG45",
    "HORIZONTAL_NEG45_NEG90",
    "VERTICAL_0_45",
    "VERTICAL_0_NEG45",
    "CIRCULAR",
]

# Default frame interval (extract every Nth frame from the video).
# Interval 1 captures every frame so fast sweeps still cover sub-range starts (0°/45°).
DEFAULT_FRAME_INTERVAL: int = max(
    1, int(os.environ.get("FRAME_INTERVAL", "1"))
)

# Dataset output prefix in S3
DATASET_PREFIX: str = "face-rotation-dataset"
SOURCE_PREFIX: str = "face-rotation-samples"

# ---------------------------------------------------------------------------
# Shared model — lazy singleton (mirrors MultiModelLoader pattern in repo)
# ---------------------------------------------------------------------------
_euler_estimator: Optional[FaceEulerEstimator] = None


def _get_estimator() -> FaceEulerEstimator:
    """Return the singleton :class:`FaceEulerEstimator`, creating it on first call."""
    global _euler_estimator
    if _euler_estimator is None:
        logger.info("Lazy-initialising FaceEulerEstimator…")
        _euler_estimator = FaceEulerEstimator(min_detection_confidence=0.5)
    return _euler_estimator


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Release MediaPipe resources when the server shuts down."""
    yield
    global _euler_estimator
    if _euler_estimator is not None:
        logger.info("Shutting down: releasing FaceEulerEstimator.")
        _euler_estimator.close()
        _euler_estimator = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Frame Extractor",
    description="Extract annotated frames from face-rotation videos stored in S3.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    """Body for POST /extract."""

    s3_key: str = Field(
        ...,
        alias="s3Key",
        description=(
            "Full S3 key for the source video, e.g. "
            "'face-rotation-samples/{userId}/{sessionId}/frontal.mp4'."
        ),
        examples=["face-rotation-samples/user42/sess01/frontal.mp4"],
    )
    s3_bucket: Optional[str] = Field(
        None,
        alias="s3Bucket",
        description="S3 bucket name; defaults to the AWS_BUCKET_NAME env var.",
    )
    rotation_type: str = Field(
        ...,
        alias="rotationType",
        description=(
            "Rotation type (lowercase Prisma enum). New: horizontal_0_45 | "
            "horizontal_45_90 | horizontal_0_neg45 | horizontal_neg45_neg90 | "
            "vertical_0_45 | vertical_0_neg45 | circular. "
            "Legacy still accepted: frontal | left | right | up | down | roll."
        ),
        examples=["horizontal_0_45"],
    )
    session_id: str = Field(
        ...,
        alias="sessionId",
        description="Session identifier used in the manifest and output keys.",
        examples=["sess01"],
    )
    user_id: Optional[str] = Field(
        None,
        alias="userId",
        description=(
            "User identifier for profile assignment in the dataset manifest. "
            "Node API sends snake_case ``user_id``; both forms are accepted."
        ),
        examples=["6ba7b810-9dad-11d1-80b4-00c04fd430c8"],
    )
    frame_interval: Optional[int] = Field(
        None,
        alias="frameInterval",
        description=f"Extract every Nth frame (default {DEFAULT_FRAME_INTERVAL}).",
        ge=1,
    )
    capture_source: Optional[str] = Field(
        None,
        alias="captureSource",
        description=(
            "Capture client origin: ``web`` (prefix ``W``) or ``mobile`` (prefix ``M``) "
            "in dataset frame filenames."
        ),
        examples=["web"],
    )
    dedup: Optional[bool] = Field(
        None,
        description=(
            "Enable pose-bin deduplication (keep best frame per angle bin). "
            f"Defaults to env FRAME_DEDUP_ENABLED ({DEFAULT_DEDUP_ENABLED})."
        ),
    )
    pose_bin_step: Optional[int] = Field(
        None,
        alias="poseBinStep",
        description=(
            "Angle bin width in degrees for deduplication. Defaults to env "
            f"POSE_BIN_STEP_DEG ({DEFAULT_POSE_BIN_STEP_DEG})."
        ),
        ge=1,
    )
    min_quality_score: Optional[float] = Field(
        None,
        alias="minQualityScore",
        description=(
            "Drop frames below this composite quality score. Defaults to env "
            f"MIN_QUALITY_SCORE ({DEFAULT_MIN_QUALITY_SCORE})."
        ),
        ge=0.0,
    )

    model_config = {"populate_by_name": True}


class ExtractSessionRequest(BaseModel):
    """Body for POST /extract-session."""

    user_id: str = Field(..., alias="userId", examples=["user42"])
    session_id: str = Field(..., alias="sessionId", examples=["sess01"])
    rotation_types: Optional[list[str]] = Field(
        None,
        alias="rotationTypes",
        description=(
            f"List of rotation types to process; defaults to "
            f"{DEFAULT_ROTATION_TYPES}."
        ),
    )
    frame_interval: Optional[int] = Field(
        None,
        alias="frameInterval",
        description=f"Extract every Nth frame (default {DEFAULT_FRAME_INTERVAL}).",
        ge=1,
    )
    capture_source: Optional[str] = Field(
        None,
        alias="captureSource",
        description=(
            "Capture client origin: ``web`` (prefix ``W``) or ``mobile`` (prefix ``M``) "
            "in dataset frame filenames."
        ),
        examples=["mobile"],
    )
    dedup: Optional[bool] = Field(
        None,
        description=(
            "Enable pose-bin deduplication (keep best frame per angle bin). "
            f"Defaults to env FRAME_DEDUP_ENABLED ({DEFAULT_DEDUP_ENABLED})."
        ),
    )
    pose_bin_step: Optional[int] = Field(
        None,
        alias="poseBinStep",
        description=(
            "Angle bin width in degrees for deduplication. Defaults to env "
            f"POSE_BIN_STEP_DEG ({DEFAULT_POSE_BIN_STEP_DEG})."
        ),
        ge=1,
    )
    min_quality_score: Optional[float] = Field(
        None,
        alias="minQualityScore",
        description=(
            "Drop frames below this composite quality score. Defaults to env "
            f"MIN_QUALITY_SCORE ({DEFAULT_MIN_QUALITY_SCORE})."
        ),
        ge=0.0,
    )

    model_config = {"populate_by_name": True}


class FrameAnnotation(BaseModel):
    """Annotation for a single extracted frame (entry in the manifest)."""

    frame_id: str
    source_video_key: str
    rotation_type: str
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    quality_score: float
    face_detected: bool
    session_id: str
    profile_id: str
    timestamp_ms: float
    s3_frame_key: str


class ExtractResponse(BaseModel):
    """Response body for POST /extract."""

    frames_extracted: int
    manifest_key: str
    session_id: str
    profile_id: str
    rotation_type: str
    per_frame: list[FrameAnnotation]


class ExtractSessionResponse(BaseModel):
    """Response body for POST /extract-session."""

    session_id: str
    results: list[ExtractResponse]
    errors: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Health-check endpoint.

    Returns:
        JSON with ``status`` and ``service`` fields.
    """
    return {"status": "ok", "service": "frame_extractor"}


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    """
    Download a face-rotation video from S3, extract annotated frames, and
    upload the frames + manifest back to S3.

    Args:
        req: :class:`ExtractRequest` with S3 key, rotation type, session ID,
             and optional frame interval.

    Returns:
        :class:`ExtractResponse` with frame count, manifest key, and per-frame
        annotations.

    Raises:
        HTTPException 400: When the request contains invalid parameters.
        HTTPException 500: On S3 or OpenCV processing errors.
    """
    interval = req.frame_interval or DEFAULT_FRAME_INTERVAL
    dedup_enabled = (
        req.dedup if req.dedup is not None else DEFAULT_DEDUP_ENABLED
    )
    pose_bin_step = req.pose_bin_step or DEFAULT_POSE_BIN_STEP_DEG
    min_quality_score = (
        req.min_quality_score
        if req.min_quality_score is not None
        else DEFAULT_MIN_QUALITY_SCORE
    )
    logger.info(
        "extract: s3_key=%s rotation_type=%s session_id=%s interval=%d "
        "dedup=%s bin_step=%d min_quality=%.3f",
        req.s3_key,
        req.rotation_type,
        req.session_id,
        interval,
        dedup_enabled,
        pose_bin_step,
        min_quality_score,
    )

    try:
        result = _process_video(
            s3_key=req.s3_key,
            s3_bucket=req.s3_bucket,
            rotation_type=req.rotation_type,
            session_id=req.session_id,
            frame_interval=interval,
            user_id=req.user_id,
            capture_source=req.capture_source,
            dedup_enabled=dedup_enabled,
            pose_bin_step=pose_bin_step,
            min_quality_score=min_quality_score,
        )
    except S3ClientError as exc:
        logger.error("S3 error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during extraction.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.post("/extract-session", response_model=ExtractSessionResponse)
async def extract_session(req: ExtractSessionRequest) -> ExtractSessionResponse:
    """
    Convenience endpoint — process all rotation-type videos for a session.

    Constructs the S3 key for each rotation type from the ``userId`` and
    ``sessionId`` following the convention::

        face-rotation-samples/{userId}/{sessionId}/{rotationType}.mp4

    Args:
        req: :class:`ExtractSessionRequest` with user ID, session ID, and
             optional list of rotation types.

    Returns:
        :class:`ExtractSessionResponse` with per-rotation results and errors.
    """
    rotation_types = req.rotation_types or DEFAULT_ROTATION_TYPES
    interval = req.frame_interval or DEFAULT_FRAME_INTERVAL
    dedup_enabled = (
        req.dedup if req.dedup is not None else DEFAULT_DEDUP_ENABLED
    )
    pose_bin_step = req.pose_bin_step or DEFAULT_POSE_BIN_STEP_DEG
    min_quality_score = (
        req.min_quality_score
        if req.min_quality_score is not None
        else DEFAULT_MIN_QUALITY_SCORE
    )
    logger.info(
        "extract-session: user_id=%s session_id=%s types=%s interval=%d "
        "dedup=%s bin_step=%d min_quality=%.3f",
        req.user_id,
        req.session_id,
        rotation_types,
        interval,
        dedup_enabled,
        pose_bin_step,
        min_quality_score,
    )

    successes: list[ExtractResponse] = []
    errors: list[dict] = []

    for rtype in rotation_types:
        key = f"{SOURCE_PREFIX}/{req.user_id}/{req.session_id}/{rtype}.mp4"
        try:
            result = _process_video(
                s3_key=key,
                s3_bucket=None,
                rotation_type=rtype,
                session_id=req.session_id,
                frame_interval=interval,
                user_id=req.user_id,
                capture_source=req.capture_source,
                dedup_enabled=dedup_enabled,
                pose_bin_step=pose_bin_step,
                min_quality_score=min_quality_score,
            )
            successes.append(result)
        except (S3ClientError, Exception) as exc:
            logger.error("Error processing %s: %s", rtype, exc)
            errors.append({"rotation_type": rtype, "s3_key": key, "error": str(exc)})

    return ExtractSessionResponse(
        session_id=req.session_id,
        results=successes,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Video orientation (phone clips may decode sideways without metadata)
# ---------------------------------------------------------------------------

_ROTATION_CANDIDATES: tuple[int, ...] = (0, 90, 180, 270)
_ORIENTATION_SAMPLE_COUNT = 15


def _apply_rotation(frame: np.ndarray, code: int) -> np.ndarray:
    """Rotate a BGR frame by ``code`` degrees clockwise (0/90/180/270)."""
    if code == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if code == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if code == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _detect_video_rotation(path: str, estimator: FaceEulerEstimator) -> int:
    """
    Pick the upright orientation that yields the most face detections.

    Samples up to ~15 evenly spaced frames, tries rotations
    ``{0, 90, 180, 270}``, and returns the code with the highest count.
    Ties prefer 0 (no rotation).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            return 0

        sample_count = min(_ORIENTATION_SAMPLE_COUNT, total_frames)
        if sample_count == 1:
            indices = [0]
        else:
            indices = [
                int(round(i * (total_frames - 1) / (sample_count - 1)))
                for i in range(sample_count)
            ]

        scores = {code: 0 for code in _ROTATION_CANDIDATES}
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            for code in _ROTATION_CANDIDATES:
                rotated = _apply_rotation(frame_bgr, code)
                result = estimator.estimate(rotated)
                if result.face_detected:
                    scores[code] += 1

        best_score = max(scores.values())
        if best_score == 0:
            return 0
        for code in _ROTATION_CANDIDATES:
            if scores[code] == best_score:
                return code
        return 0
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def _process_video(
    s3_key: str,
    s3_bucket: Optional[str],
    rotation_type: str,
    session_id: str,
    frame_interval: int,
    user_id: Optional[str] = None,
    capture_source: Optional[str] = None,
    dedup_enabled: bool = DEFAULT_DEDUP_ENABLED,
    pose_bin_step: int = DEFAULT_POSE_BIN_STEP_DEG,
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
) -> ExtractResponse:
    """
    Full pipeline: download → extract frames → annotate → upload → manifest.

    Temp file is always cleaned up in the ``finally`` block.

    Args:
        s3_key:         Source video S3 key.
        s3_bucket:      S3 bucket (None → env default).
        rotation_type:  Rotation type string (used for bucketing and keys).
        session_id:     Session identifier.
        frame_interval: Extract every Nth frame.

    Returns:
        :class:`ExtractResponse` with annotations and manifest key.

    Raises:
        S3ClientError: On S3 download/upload failures.
        RuntimeError:  When OpenCV cannot open the video.
    """
    local_path: Optional[str] = None
    profile_id = (
        get_or_create_profile_id(user_id, session_id, s3_bucket)
        if user_id
        else "00000"
    )
    try:
        # 1. Download video
        local_path = download_video(s3_key, s3_bucket)

        # 2. Extract and annotate frames
        annotations, per_frame_data, selection_stats = _extract_and_annotate(
            local_path=local_path,
            s3_key=s3_key,
            s3_bucket=s3_bucket,
            rotation_type=rotation_type,
            session_id=session_id,
            frame_interval=frame_interval,
            profile_id=profile_id,
            capture_source=capture_source,
            dedup_enabled=dedup_enabled,
            pose_bin_step=pose_bin_step,
            min_quality_score=min_quality_score,
        )

        # 3. Build and upload manifest
        manifest_key = _build_manifest_key(rotation_type, session_id)
        manifest = _build_manifest(
            annotations=per_frame_data,
            session_id=session_id,
            rotation_type=rotation_type,
            source_key=s3_key,
            profile_id=profile_id,
            selection_stats=selection_stats,
        )
        upload_manifest(manifest, manifest_key, s3_bucket)

        return ExtractResponse(
            frames_extracted=len(annotations),
            manifest_key=manifest_key,
            session_id=session_id,
            profile_id=profile_id,
            rotation_type=rotation_type,
            per_frame=annotations,
        )

    finally:
        if local_path and os.path.exists(local_path):
            try:
                os.unlink(local_path)
                logger.debug("Cleaned up temp file: %s", local_path)
            except OSError as exc:
                logger.warning("Could not delete temp file %s: %s", local_path, exc)


def _expected_zones_for_rotation_type(rotation_type: str) -> tuple[str, ...]:
    """Return dataset subfolders expected for a rotation type family."""
    rtype = rotation_type.lower()
    if rtype.startswith("horizontal"):
        return ("rh_1D", "rh_2D", "rh_1I", "rh_2I")
    if rtype.startswith("vertical"):
        return ("rv_1A", "rv_1B")
    if rtype == "circular":
        return ("rc_Q1", "rc_Q2", "rc_Q3", "rc_Q4")
    return ()


def _subfolder_from_s3_key(s3_key: str) -> Optional[str]:
    """Extract the subfolder segment from a dataset frame S3 key."""
    parts = s3_key.split("/")
    if len(parts) < 2:
        return None
    return parts[-2]


def _compute_coverage_by_zone(s3_keys: list[str]) -> dict[str, int]:
    """Count kept frames per rh_/rv_/rc_ subfolder (excludes neutral postures)."""
    coverage: dict[str, int] = {}
    for key in s3_keys:
        if "/posturas_neutrales/" in key:
            continue
        zone = _subfolder_from_s3_key(key)
        if zone is None or not (
            zone.startswith("rh_")
            or zone.startswith("rv_")
            or zone.startswith("rc_")
        ):
            continue
        coverage[zone] = coverage.get(zone, 0) + 1
    return coverage


def _log_coverage_gaps(
    rotation_type: str,
    session_id: str,
    coverage_by_zone: dict[str, int],
) -> None:
    """Warn when an expected angular zone has no representative frame."""
    for zone in _expected_zones_for_rotation_type(rotation_type):
        if coverage_by_zone.get(zone, 0) == 0:
            logger.warning(
                "Coverage gap: zone %s has 0 frames for rotation %s session %s",
                zone,
                rotation_type,
                session_id,
            )


def _track_neutral_candidate(
    neutral_candidates: dict[str, dict],
    category: str,
    jpeg_bytes: bytes,
    neutral_key: str,
    result: EulerResult,
    annotation: FrameAnnotation,
) -> None:
    """Keep the highest-quality frame per neutral posture category."""
    existing = neutral_candidates.get(category)
    if existing is None or result.quality_score > existing["quality_score"]:
        neutral_candidates[category] = {
            "jpeg": jpeg_bytes,
            "key": neutral_key,
            "quality_score": result.quality_score,
            "annotation": annotation.model_dump(),
        }


def _extract_and_annotate(
    local_path: str,
    s3_key: str,
    s3_bucket: Optional[str],
    rotation_type: str,
    session_id: str,
    frame_interval: int,
    profile_id: str,
    capture_source: Optional[str] = None,
    dedup_enabled: bool = DEFAULT_DEDUP_ENABLED,
    pose_bin_step: int = DEFAULT_POSE_BIN_STEP_DEG,
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
) -> tuple[list[FrameAnnotation], list[dict], dict]:
    """
    Open the video, extract every ``frame_interval``-th frame, run Euler
    estimation, and either upload every frame (legacy) or keep only the best
    frame per pose bin (deduplication).

    When ``dedup_enabled`` is True, each detected frame is grouped into a pose
    bin (see :func:`app.selection.pose_bin_signature`); only the highest-quality
    frame of each bin is uploaded. Frames without a detected face, or below
    ``min_quality_score``, are dropped. This collapses temporal bursts of
    near-identical poses (and roll jitter) into one representative image while
    preserving angular coverage. Neutral-posture selection is unchanged and runs
    independently.

    Args:
        local_path:        Path to the downloaded video file.
        s3_key:            Original S3 key (stored in annotation for provenance).
        s3_bucket:         Target S3 bucket.
        rotation_type:     Rotation type string.
        session_id:        Session identifier.
        frame_interval:    Sampling stride.
        profile_id:        Assigned profile identifier.
        capture_source:    ``web`` / ``mobile`` origin for filename prefix.
        dedup_enabled:     Keep only the best frame per pose bin when True.
        pose_bin_step:     Bin width in degrees for deduplication.
        min_quality_score: Drop frames below this composite quality score.

    Returns:
        Tuple of (kept :class:`FrameAnnotation` list, manifest dicts, selection
        stats dict).

    Raises:
        RuntimeError: When OpenCV cannot open the video.
    """
    estimator = _get_estimator()
    rotation_code = _detect_video_rotation(local_path, estimator)
    if rotation_code != 0:
        logger.info(
            "Detected video orientation correction: rotate %d° clockwise for %s",
            rotation_code,
            local_path,
        )

    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {local_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    annotations: list[FrameAnnotation] = []
    per_frame_data: list[dict] = []
    neutral_candidates: dict[str, dict] = {}
    # Pose-bin winners: signature → best candidate (dedup mode only).
    pose_candidates: dict[tuple, dict] = {}
    rtype_lower = rotation_type.lower()
    frame_idx = 0
    sampled_idx = 0
    processed = 0
    skipped_no_face = 0
    skipped_low_quality = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_ms = (frame_idx / fps) * 1000.0

                frame_bgr = _apply_rotation(frame_bgr, rotation_code)

                # Euler + quality
                result = estimator.estimate(frame_bgr)
                processed += 1

                frame_s3_key = build_frame_key(
                    DATASET_PREFIX,
                    rotation_type,
                    result.yaw,
                    result.pitch,
                    result.roll,
                    profile_id,
                    session_id=session_id,
                    sampled_idx=sampled_idx,
                    capture_source=capture_source,
                )
                frame_id = frame_s3_key.rsplit("/", 1)[-1].removesuffix(".jpg")

                # Encode JPEG in-memory (vectorised numpy already inside cv2)
                ok, jpeg_bytes = cv2.imencode(
                    ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                jpeg_payload: Optional[bytes] = bytes(jpeg_bytes) if ok else None
                if not ok:
                    logger.warning("JPEG encoding failed for frame %d.", frame_idx)

                annotation = FrameAnnotation(
                    frame_id=frame_id,
                    source_video_key=s3_key,
                    rotation_type=rotation_type,
                    yaw=result.yaw,
                    pitch=result.pitch,
                    roll=result.roll,
                    quality_score=result.quality_score,
                    face_detected=result.face_detected,
                    session_id=session_id,
                    profile_id=profile_id,
                    timestamp_ms=round(timestamp_ms, 2),
                    s3_frame_key=frame_s3_key,
                )

                if not dedup_enabled:
                    # Legacy behaviour: upload every frame immediately.
                    if jpeg_payload is not None:
                        upload_frame(jpeg_payload, frame_s3_key, s3_bucket)
                    annotations.append(annotation)
                    per_frame_data.append(annotation.model_dump())
                elif jpeg_payload is None:
                    pass  # nothing to keep when encoding failed
                elif not result.face_detected:
                    skipped_no_face += 1
                elif result.quality_score < min_quality_score:
                    skipped_low_quality += 1
                else:
                    # Group by pose bin; keep the highest-quality frame per bin.
                    signature = pose_bin_signature(
                        rotation_type,
                        result.yaw,
                        result.pitch,
                        result.roll,
                        pose_bin_step,
                    )
                    if signature is None:
                        signature = ("__idx__", sampled_idx)
                    existing = pose_candidates.get(signature)
                    if is_better_candidate(
                        result.quality_score,
                        existing["quality_score"] if existing else None,
                    ):
                        pose_candidates[signature] = {
                            "jpeg": jpeg_payload,
                            "key": frame_s3_key,
                            "quality_score": result.quality_score,
                            "annotation": annotation.model_dump(),
                        }

                # Neutral posture selection (independent of dedup).
                if (
                    jpeg_payload is not None
                    and rtype_lower.startswith("horizontal_")
                    and result.face_detected
                ):
                    neutral_cat = classify_neutral_posture(result.yaw, result.pitch)
                    if neutral_cat:
                        neutral_key = build_neutral_key(
                            DATASET_PREFIX,
                            neutral_cat,
                            profile_id,
                            result,
                            capture_source=capture_source,
                        )
                        neutral_frame_id = neutral_key.rsplit("/", 1)[-1].removesuffix(
                            ".jpg"
                        )
                        neutral_annotation = FrameAnnotation(
                            frame_id=neutral_frame_id,
                            source_video_key=s3_key,
                            rotation_type=rotation_type,
                            yaw=result.yaw,
                            pitch=result.pitch,
                            roll=result.roll,
                            quality_score=result.quality_score,
                            face_detected=result.face_detected,
                            session_id=session_id,
                            profile_id=profile_id,
                            timestamp_ms=round(timestamp_ms, 2),
                            s3_frame_key=neutral_key,
                        )
                        _track_neutral_candidate(
                            neutral_candidates,
                            neutral_cat,
                            jpeg_payload,
                            neutral_key,
                            result,
                            neutral_annotation,
                        )

                sampled_idx += 1

            frame_idx += 1

    finally:
        cap.release()

    # Upload pose-bin winners (dedup mode).
    if dedup_enabled:
        for candidate in pose_candidates.values():
            upload_frame(candidate["jpeg"], candidate["key"], s3_bucket)
            annotations.append(FrameAnnotation(**candidate["annotation"]))
            per_frame_data.append(candidate["annotation"])
        logger.info(
            "Pose-bin dedup: %d processed → %d kept "
            "(skipped %d no-face, %d low-quality) step=%d° for %s",
            processed,
            len(pose_candidates),
            skipped_no_face,
            skipped_low_quality,
            pose_bin_step,
            rotation_type,
        )

    for cat, candidate in neutral_candidates.items():
        upload_frame(candidate["jpeg"], candidate["key"], s3_bucket)
        annotations.append(FrameAnnotation(**candidate["annotation"]))
        per_frame_data.append(candidate["annotation"])
        logger.info(
            "Uploaded best neutral frame for %s → %s (quality=%.4f)",
            cat,
            candidate["key"],
            candidate["quality_score"],
        )

    selection_stats = {
        "dedup_enabled": dedup_enabled,
        "pose_bin_step_deg": pose_bin_step,
        "min_quality_score": min_quality_score,
        "frames_processed": processed,
        "frames_selected": len(pose_candidates) if dedup_enabled else processed,
        "frames_skipped_no_face": skipped_no_face,
        "frames_skipped_low_quality": skipped_low_quality,
        "neutral_frames": len(neutral_candidates),
    }

    dataset_keys = [
        a.s3_frame_key
        for a in annotations
        if "/posturas_neutrales/" not in a.s3_frame_key
    ]
    coverage_by_zone = _compute_coverage_by_zone(dataset_keys)
    _log_coverage_gaps(rotation_type, session_id, coverage_by_zone)
    selection_stats["coverage_by_zone"] = coverage_by_zone

    logger.info(
        "Extracted %d frames from %s (every %d-th of %d total, dedup=%s).",
        len(annotations),
        local_path,
        frame_interval,
        frame_idx,
        dedup_enabled,
    )
    return annotations, per_frame_data, selection_stats


def _build_manifest_key(rotation_type: str, session_id: str) -> str:
    """Build the S3 key for the manifest JSON file."""
    return build_manifest_key(DATASET_PREFIX, rotation_type, session_id)


def _build_manifest(
    annotations: list[dict],
    session_id: str,
    rotation_type: str,
    source_key: str,
    profile_id: str,
    selection_stats: Optional[dict] = None,
) -> dict:
    """
    Construct the manifest dictionary.

    Schema::

        {
          "session_id": str,
          "rotation_type": str,
          "source_video_key": str,
          "created_at": float,          # Unix timestamp
          "summary": {
            "session_id": str,
            "rotation_type": str,
            "total_frames": int,
            "frames_with_face": int,
            "angle_min": float | None,
            "angle_max": float | None,
            "quality_score_mean": float | None
          },
          "frames": [ { ...FrameAnnotation fields... }, ... ]
        }

    The ``angle_min`` / ``angle_max`` refer to the dominant axis for the
    rotation type: yaw for horizontal / circular / frontal / left / right,
    pitch for vertical / up / down, and roll for the legacy roll type.

    Args:
        annotations:   List of per-frame annotation dicts.
        session_id:    Session identifier.
        rotation_type: Rotation type string.
        source_key:    Original source video S3 key.

    Returns:
        Dict ready for ``json.dumps``.
    """
    # Dominant angle key for summary stats — derived from the same axis map
    # used for bucketing (yaw for horizontal/circular/frontal, pitch for
    # vertical/up/down, roll for legacy roll).
    angle_key = dominant_axis(rotation_type)

    detected = [a for a in annotations if a.get("face_detected")]
    angles = [
        a[angle_key] for a in detected if a.get(angle_key) is not None
    ]
    qualities = [a["quality_score"] for a in detected]

    angle_arr = np.array(angles, dtype=np.float64) if angles else None
    quality_arr = np.array(qualities, dtype=np.float64) if qualities else None

    summary = {
        "session_id": session_id,
        "rotation_type": rotation_type,
        "total_frames": len(annotations),
        "frames_with_face": len(detected),
        "angle_min": float(angle_arr.min()) if angle_arr is not None else None,
        "angle_max": float(angle_arr.max()) if angle_arr is not None else None,
        "quality_score_mean": (
            float(quality_arr.mean()) if quality_arr is not None else None
        ),
    }
    if selection_stats is not None:
        summary["selection"] = selection_stats

    return {
        "session_id": session_id,
        "profile_id": profile_id,
        "rotation_type": rotation_type,
        "source_video_key": source_key,
        "created_at": time.time(),
        "summary": summary,
        "frames": annotations,
    }
