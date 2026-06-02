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

from .euler import FaceEulerEstimator, compute_angle_range, dominant_axis
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
# Lowercase form of the Prisma ``FaceRotationType`` enum (post-redesign):
# 4 horizontal + 2 vertical + 1 circular = 7 videos per session.
# Legacy types (frontal / left / right) are still accepted when passed
# explicitly via the request body for processing older sessions.
# ---------------------------------------------------------------------------
DEFAULT_ROTATION_TYPES: list[str] = [
    "horizontal_0_45",
    "horizontal_45_90",
    "horizontal_0_neg45",
    "horizontal_neg45_neg90",
    "vertical_0_45",
    "vertical_0_neg45",
    "circular",
]

# Default frame interval (extract every Nth frame from the video)
DEFAULT_FRAME_INTERVAL: int = 5

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
    frame_interval: Optional[int] = Field(
        None,
        alias="frameInterval",
        description=f"Extract every Nth frame (default {DEFAULT_FRAME_INTERVAL}).",
        ge=1,
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
    timestamp_ms: float
    s3_frame_key: str


class ExtractResponse(BaseModel):
    """Response body for POST /extract."""

    frames_extracted: int
    manifest_key: str
    session_id: str
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
    logger.info(
        "extract: s3_key=%s rotation_type=%s session_id=%s interval=%d",
        req.s3_key,
        req.rotation_type,
        req.session_id,
        interval,
    )

    try:
        result = _process_video(
            s3_key=req.s3_key,
            s3_bucket=req.s3_bucket,
            rotation_type=req.rotation_type,
            session_id=req.session_id,
            frame_interval=interval,
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
    logger.info(
        "extract-session: user_id=%s session_id=%s types=%s",
        req.user_id,
        req.session_id,
        rotation_types,
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
                frame_interval=DEFAULT_FRAME_INTERVAL,
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
# Core processing logic
# ---------------------------------------------------------------------------

def _process_video(
    s3_key: str,
    s3_bucket: Optional[str],
    rotation_type: str,
    session_id: str,
    frame_interval: int,
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
    try:
        # 1. Download video
        local_path = download_video(s3_key, s3_bucket)

        # 2. Extract and annotate frames
        annotations, per_frame_data = _extract_and_annotate(
            local_path=local_path,
            s3_key=s3_key,
            s3_bucket=s3_bucket,
            rotation_type=rotation_type,
            session_id=session_id,
            frame_interval=frame_interval,
        )

        # 3. Build and upload manifest
        manifest_key = _build_manifest_key(rotation_type, session_id)
        manifest = _build_manifest(
            annotations=per_frame_data,
            session_id=session_id,
            rotation_type=rotation_type,
            source_key=s3_key,
        )
        upload_manifest(manifest, manifest_key, s3_bucket)

        return ExtractResponse(
            frames_extracted=len(annotations),
            manifest_key=manifest_key,
            session_id=session_id,
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


def _extract_and_annotate(
    local_path: str,
    s3_key: str,
    s3_bucket: Optional[str],
    rotation_type: str,
    session_id: str,
    frame_interval: int,
) -> tuple[list[FrameAnnotation], list[dict]]:
    """
    Open the video, extract every ``frame_interval``-th frame, run Euler
    estimation, upload the JPEG, and return annotation objects.

    Args:
        local_path:     Path to the downloaded video file.
        s3_key:         Original S3 key (stored in annotation for provenance).
        s3_bucket:      Target S3 bucket.
        rotation_type:  Rotation type string.
        session_id:     Session identifier.
        frame_interval: Sampling stride.

    Returns:
        Tuple of (list of :class:`FrameAnnotation`, list of raw dicts for the
        manifest).

    Raises:
        RuntimeError: When OpenCV cannot open the video.
    """
    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {local_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    estimator = _get_estimator()

    annotations: list[FrameAnnotation] = []
    per_frame_data: list[dict] = []
    frame_idx = 0
    sampled_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_ms = (frame_idx / fps) * 1000.0

                # Euler + quality
                result = estimator.estimate(frame_bgr)

                angle_range = compute_angle_range(
                    rotation_type,
                    result.yaw,
                    result.pitch,
                    result.roll,
                )

                frame_id = f"{session_id}_{rotation_type}_{sampled_idx:04d}"
                frame_s3_key = (
                    f"{DATASET_PREFIX}/{rotation_type}/{angle_range}"
                    f"/{frame_id}.jpg"
                )

                # Encode JPEG in-memory (vectorised numpy already inside cv2)
                ok, jpeg_bytes = cv2.imencode(
                    ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if ok:
                    upload_frame(
                        bytes(jpeg_bytes),
                        frame_s3_key,
                        s3_bucket,
                    )
                else:
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
                    timestamp_ms=round(timestamp_ms, 2),
                    s3_frame_key=frame_s3_key,
                )
                annotations.append(annotation)
                per_frame_data.append(annotation.model_dump())
                sampled_idx += 1

            frame_idx += 1

    finally:
        cap.release()

    logger.info(
        "Extracted %d frames from %s (every %d-th of %d total).",
        len(annotations),
        local_path,
        frame_interval,
        frame_idx,
    )
    return annotations, per_frame_data


def _build_manifest_key(rotation_type: str, session_id: str) -> str:
    """
    Build the S3 key for the manifest JSON file.

    Format::

        face-rotation-dataset/{rotationType}/{sessionId}_manifest.json

    Args:
        rotation_type: Rotation type string.
        session_id:    Session identifier.

    Returns:
        S3 key string.
    """
    return f"{DATASET_PREFIX}/{rotation_type}/{session_id}_manifest.json"


def _build_manifest(
    annotations: list[dict],
    session_id: str,
    rotation_type: str,
    source_key: str,
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

    return {
        "session_id": session_id,
        "rotation_type": rotation_type,
        "source_video_key": source_key,
        "created_at": time.time(),
        "summary": {
            "session_id": session_id,
            "rotation_type": rotation_type,
            "total_frames": len(annotations),
            "frames_with_face": len(detected),
            "angle_min": float(angle_arr.min()) if angle_arr is not None else None,
            "angle_max": float(angle_arr.max()) if angle_arr is not None else None,
            "quality_score_mean": (
                float(quality_arr.mean()) if quality_arr is not None else None
            ),
        },
        "frames": annotations,
    }
