"""
Profile ID assignment backed by ``face-rotation-dataset/profiles_manifest.json`` on S3.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .s3_client import download_json, upload_manifest

logger = logging.getLogger(__name__)

PROFILES_KEY = "face-rotation-dataset/profiles_manifest.json"


def _empty_manifest() -> dict:
    return {"profiles": []}


def _load(bucket: Optional[str] = None) -> dict:
    data = download_json(PROFILES_KEY, bucket)
    if not data:
        return _empty_manifest()
    if "profiles" not in data or not isinstance(data["profiles"], list):
        return _empty_manifest()
    return data


def _save(manifest: dict, bucket: Optional[str] = None) -> None:
    upload_manifest(manifest, PROFILES_KEY, bucket)


def get_or_create_profile_id(
    user_id: str,
    session_id: str,
    bucket: Optional[str] = None,
) -> str:
    """
    Return a stable 5-digit ``profile_id`` for ``user_id``, recording ``session_id``.

    Sequential IDs are assigned from ``00001`` upward without reuse.
    """
    manifest = _load(bucket)
    profiles: list[dict] = manifest["profiles"]

    for entry in profiles:
        if entry.get("user_id") == user_id:
            sessions = entry.setdefault("sessions", [])
            if session_id not in sessions:
                sessions.append(session_id)
                _save(manifest, bucket)
            return str(entry["profile_id"])

    next_num = len(profiles) + 1
    profile_id = f"{next_num:05d}"
    profiles.append(
        {
            "profile_id": profile_id,
            "user_id": user_id,
            "sessions": [session_id],
            "assigned_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save(manifest, bucket)
    logger.info(
        "Assigned profile_id=%s for user_id=%s session_id=%s",
        profile_id,
        user_id,
        session_id,
    )
    return profile_id
