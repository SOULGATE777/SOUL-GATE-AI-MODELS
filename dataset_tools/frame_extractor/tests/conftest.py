"""Pytest configuration — ensure ``app`` package is importable from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

_FRAME_EXTRACTOR_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAME_EXTRACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAME_EXTRACTOR_ROOT))
