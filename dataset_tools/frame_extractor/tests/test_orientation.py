"""Unit tests for video orientation helpers in ``app.main``."""

from __future__ import annotations

import cv2
import numpy as np

from app.main import _apply_rotation


def test_apply_rotation_identity() -> None:
    frame = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(_apply_rotation(frame, 0), frame)


def test_apply_rotation_90_clockwise() -> None:
    frame = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    assert np.array_equal(_apply_rotation(frame, 90), expected)


def test_apply_rotation_180() -> None:
    frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    expected = cv2.rotate(frame, cv2.ROTATE_180)
    assert np.array_equal(_apply_rotation(frame, 180), expected)


def test_apply_rotation_270_counterclockwise() -> None:
    frame = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    expected = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert np.array_equal(_apply_rotation(frame, 270), expected)
