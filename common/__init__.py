# ============================================================================
# NewFeature.md: Common validation modules
# ============================================================================
# Centralized threshold validation, rule engine, and diagnosis resolution
# for all ML modules according to NewFeature.md specifications.
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
# ============================================================================

"""
Common validation and diagnosis modules for SOUL-GATE ML services.

This package provides centralized validation, rule engine, and diagnosis
resolution functionality to ensure consistent application of NewFeature.md
requirements across all ML modules.
"""

from .threshold_validator import ThresholdValidator, ModuleType, ThresholdRule
from .rule_engine import RuleEngine
from .proportion_analyzer import ProportionAnalyzer
from .diagnosis_resolver import DiagnosisResolver
from .colorimetry_analyzer import ColorimetryAnalyzer
from .colorimetry_config import (
    EYE_COLOR_RANGES,
    PALM_COLOR_RANGES,
    SKIN_TONE_CATEGORIES
)
from .threshold_validation_logger import ThresholdValidationLogger, ValidationEntry, CategoryValidation
from .newfeature_references import (
    get_newfeature_reference,
    get_module_references,
    get_category_references,
    NEWFEATURE_REFERENCES
)

__all__ = [
    'ThresholdValidator',
    'ModuleType',
    'ThresholdRule',
    'RuleEngine',
    'ProportionAnalyzer',
    'DiagnosisResolver',
    'ColorimetryAnalyzer',
    'EYE_COLOR_RANGES',
    'PALM_COLOR_RANGES',
    'SKIN_TONE_CATEGORIES',
    'ThresholdValidationLogger',
    'ValidationEntry',
    'CategoryValidation',
    'get_newfeature_reference',
    'get_module_references',
    'get_category_references',
    'NEWFEATURE_REFERENCES',
]
