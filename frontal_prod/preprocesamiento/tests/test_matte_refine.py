"""Frontal alignment clamp tests (no MediaPipe required)."""
from unittest.mock import MagicMock

import numpy as np

from app.models.frontal_preprocessing_pipeline import FrontalPreprocessingPipeline


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
