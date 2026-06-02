"""
AWS S3 client for the frame-extractor service.

All credentials and region are read from environment variables::

    AWS_REGION           — AWS region (default "us-east-1")
    AWS_ACCESS_KEY_ID    — IAM access key
    AWS_SECRET_ACCESS_KEY — IAM secret key
    AWS_BUCKET_NAME      — default S3 bucket (can be overridden per call)

All public functions wrap boto3 calls in ``try/except`` and raise
:class:`S3ClientError` on failure.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Union

import boto3
import botocore.exceptions

logger = logging.getLogger(__name__)


class S3ClientError(RuntimeError):
    """Raised when an S3 operation fails."""


def _get_client() -> "boto3.client":
    """
    Build and return a boto3 S3 client from environment variables.

    Returns:
        Configured boto3 S3 client.

    Raises:
        S3ClientError: When required credentials are missing.
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        raise S3ClientError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set."
        )

    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _default_bucket() -> str:
    """
    Return the default bucket name from the environment.

    Raises:
        S3ClientError: When AWS_BUCKET_NAME is not set.
    """
    bucket = os.environ.get("AWS_BUCKET_NAME")
    if not bucket:
        raise S3ClientError("AWS_BUCKET_NAME environment variable is not set.")
    return bucket


def download_video(key: str, bucket: str | None = None) -> str:
    """
    Download an S3 video object to a local temporary file.

    The caller **must** delete the file when finished (``finally`` block).
    File extension is preserved from the S3 key so that ``cv2.VideoCapture``
    can detect the codec automatically.

    Args:
        key:    S3 object key, e.g. ``face-rotation-samples/uid/sid/frontal.mp4``.
        bucket: S3 bucket name; falls back to ``AWS_BUCKET_NAME`` env var.

    Returns:
        Absolute path to the downloaded temporary file.

    Raises:
        S3ClientError: On any download failure.
    """
    bucket = bucket or _default_bucket()
    suffix = Path(key).suffix or ".mp4"

    fd, local_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # Close the OS-level fd; boto3 will open it separately.

    try:
        logger.info("Downloading s3://%s/%s → %s", bucket, key, local_path)
        client = _get_client()
        client.download_file(bucket, key, local_path)
        logger.info("Download complete (%s).", local_path)
        return local_path
    except botocore.exceptions.ClientError as exc:
        # Clean up orphaned temp file on error
        try:
            os.unlink(local_path)
        except OSError:
            pass
        code = exc.response["Error"]["Code"]
        raise S3ClientError(
            f"Failed to download s3://{bucket}/{key}: {code} — {exc}"
        ) from exc


def upload_frame(
    data: Union[bytes, str, Path],
    key: str,
    bucket: str | None = None,
    content_type: str = "image/jpeg",
) -> None:
    """
    Upload a JPEG frame to S3.

    Args:
        data:         Either raw bytes (JPEG-encoded) or a local file path.
        key:          Destination S3 key.
        bucket:       S3 bucket; falls back to ``AWS_BUCKET_NAME``.
        content_type: MIME type for the uploaded object.

    Raises:
        S3ClientError: On any upload failure.
    """
    bucket = bucket or _default_bucket()

    try:
        client = _get_client()

        if isinstance(data, (str, Path)):
            with open(data, "rb") as fh:
                body = fh.read()
        else:
            body = data

        logger.debug("Uploading frame → s3://%s/%s (%d bytes)", bucket, key, len(body))
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
    except botocore.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise S3ClientError(
            f"Failed to upload frame to s3://{bucket}/{key}: {code} — {exc}"
        ) from exc


def upload_manifest(data: dict, key: str, bucket: str | None = None) -> None:
    """
    Serialise ``data`` to JSON and upload to S3.

    Args:
        data:   Python dict that will be serialised with ``json.dumps``.
        key:    Destination S3 key, e.g. ``face-rotation-dataset/.../manifest.json``.
        bucket: S3 bucket; falls back to ``AWS_BUCKET_NAME``.

    Raises:
        S3ClientError: On any upload failure.
    """
    bucket = bucket or _default_bucket()

    try:
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        body = payload.encode("utf-8")
        client = _get_client()
        logger.info("Uploading manifest → s3://%s/%s", bucket, key)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except botocore.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise S3ClientError(
            f"Failed to upload manifest to s3://{bucket}/{key}: {code} — {exc}"
        ) from exc


def object_exists(key: str, bucket: str | None = None) -> bool:
    """
    Check whether an S3 object key exists without downloading it.

    Args:
        key:    S3 object key.
        bucket: S3 bucket; falls back to ``AWS_BUCKET_NAME``.

    Returns:
        ``True`` if the object exists, ``False`` otherwise.
    """
    bucket = bucket or _default_bucket()
    try:
        client = _get_client()
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise S3ClientError(
            f"Failed to check s3://{bucket}/{key}: {exc}"
        ) from exc
