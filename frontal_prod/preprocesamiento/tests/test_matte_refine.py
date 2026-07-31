"""Frontal selfie-matte refine + alignment clamp (no MediaPipe required)."""
from unittest.mock import MagicMock

import numpy as np

from app.models.frontal_preprocessing_pipeline import FrontalPreprocessingPipeline


def test_refine_selfie_matte_drops_detached_speck():
    mask = np.zeros((60, 60), dtype=np.float32)
    mask[10:50, 10:40] = 1.0
    mask[2:6, 52:56] = 1.0
    alpha = FrontalPreprocessingPipeline._refine_selfie_matte(mask)
    assert float(alpha[30, 25]) > 0.8
    assert float(alpha[4, 54]) < 0.1


def test_refine_selfie_matte_fills_interior_hole():
    mask = np.zeros((60, 60), dtype=np.float32)
    mask[10:50, 10:50] = 1.0
    mask[25:35, 25:35] = 0.0
    alpha = FrontalPreprocessingPipeline._refine_selfie_matte(mask)
    assert float(alpha[30, 30]) > 0.8


def test_align_face_skips_catastrophic_angle():
    pipeline = object.__new__(FrontalPreprocessingPipeline)
    pipeline.alignment_threshold = 2.0
    pipeline.MAX_ABS_ROTATION_DEG = 35.0
    pipeline.detect_eye_landmarks = MagicMock(return_value=((10, 10), (50, 50)))
    # dy/dx = 40/40 → 45°
    image = np.full((80, 80, 3), 128, dtype=np.uint8)
    out, meta = pipeline.align_face(image)
    assert meta.get("alignment_skipped_max_angle") is True
    assert meta.get("alignment_applied") is False
    np.testing.assert_array_equal(out, image)
