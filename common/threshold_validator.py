# ============================================================================
# NewFeature.md: Threshold Validator - Centralized validation (líneas 13-270)
# ============================================================================
# Single Source of Truth for all threshold validations across ML modules.
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
#
# Purpose:
# - Validate predictions against confidence thresholds
# - Apply module-specific rules (Espejo, Morfológico, Antropométrico, etc.)
# - Handle exceptions (Venus 40%, Plutón 7%)
# ============================================================================

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ModuleType(Enum):
    """
    NewFeature.md: Tipos de módulos ML
    Cada módulo tiene sus propios umbrales y reglas específicas.
    """
    # Espejo (líneas 13-73)
    ESPEJO_FRENTE = "espejo_frente"
    ESPEJO_ROSTRO = "espejo_rostro"
    
    # Frontal (líneas 75-270)
    FRONTAL_MORFOLOGICO = "frontal_morfologico"
    FRONTAL_ANTROPOMETRICO = "frontal_antropometrico"
    
    # Perfil (líneas 271-503)
    PROFILE_MORFOLOGICO = "profile_morfologico"
    PROFILE_ANTROPOMETRICO = "profile_antropometrico"


@dataclass
class ThresholdRule:
    """
    NewFeature.md: Regla de umbral para un diagnóstico específico
    
    Attributes:
        diagnosis_name: Nombre del diagnóstico
        threshold: Umbral de confianza (0.0 - 1.0)
        rule_type: Tipo de regla ('minimum', 'solo', 'exclusion')
        is_exception: Si es una excepción a la regla general
        description: Descripción con referencia a NewFeature.md
    """
    diagnosis_name: str
    threshold: float
    rule_type: str  # 'minimum', 'solo', 'exclusion'
    is_exception: bool = False
    description: str = ""
    
    def __post_init__(self):
        """Validate rule configuration"""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")
        if self.rule_type not in ['minimum', 'solo', 'exclusion']:
            raise ValueError(f"Invalid rule_type: {self.rule_type}")


class ThresholdValidator:
    """
    NewFeature.md: Validador centralizado de umbrales
    
    Single Source of Truth para todos los umbrales especificados en NewFeature.md.
    Maneja validación de predicciones, reglas de solo diagnosis, y umbrales generales.
    
    Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 13-270
    """
    
    def __init__(self):
        """
        Initialize validator and load threshold rules from configuration.
        """
        self.rules = self._load_threshold_rules()
        self._validate_rules()
    
    def _load_threshold_rules(self) -> Dict[ModuleType, Dict[str, ThresholdRule]]:
        """
        NewFeature.md: Carga TODAS las reglas de umbrales desde configuración.
        
        Returns:
            Dict mapping ModuleType -> {diagnosis_name: ThresholdRule}
        """
        from .threshold_config import THRESHOLD_CONFIG
        return THRESHOLD_CONFIG
    
    def validate_predictions(
        self,
        module_type: ModuleType,
        predictions: Dict[str, float],
        additional_context: Optional[Dict] = None
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        NewFeature.md: Valida predicciones contra umbrales del módulo.
        
        Args:
            module_type: Tipo de módulo (Espejo, Morfológico, etc.)
            predictions: Dict {diagnosis_name: confidence}
            additional_context: Contexto adicional (e.g., proporción facial)
        
        Returns:
            Tuple[valid_predictions, applied_rules]
            
        Example:
            # NewFeature.md línea 17-19: Espejo rostro umbral 18%
            validator = ThresholdValidator()
            preds = {'venus_corazon': 0.45, 'pluton_hexagonal': 0.08}
            valid, rules = validator.validate_predictions(
                ModuleType.ESPEJO_ROSTRO, preds
            )
            # valid = {'venus_corazon': 0.45, 'pluton_hexagonal': 0.08}
            # (Venus > 40%, Plutón > 7%)
        """
        if module_type not in self.rules:
            raise ValueError(f"Unknown module type: {module_type}")
        
        module_rules = self.rules[module_type]
        valid_predictions = {}
        applied_rules = []
        
        # Get general threshold for this module
        general_threshold = self.get_general_threshold(module_type)
        
        # 1. Aplicar exclusion rules primero
        for diag_name, confidence in predictions.items():
            rule = module_rules.get(diag_name)
            
            if rule and rule.rule_type == 'exclusion':
                if confidence < rule.threshold:
                    applied_rules.append(
                        f"Excluded {diag_name} (confidence {confidence:.1%} < {rule.threshold:.1%})"
                    )
                    continue
            
            valid_predictions[diag_name] = confidence
        
        # 2. Aplicar minimum thresholds (individual o general)
        final_predictions = {}
        for diag_name, confidence in valid_predictions.items():
            rule = module_rules.get(diag_name)
            
            if rule and rule.rule_type in ['minimum', 'solo']:
                # NewFeature.md: Usar umbral específico del diagnóstico
                if confidence >= rule.threshold:
                    final_predictions[diag_name] = confidence
                    applied_rules.append(
                        f"Accepted {diag_name} (confidence {confidence:.1%} >= {rule.threshold:.1%})"
                    )
                else:
                    applied_rules.append(
                        f"Rejected {diag_name} (confidence {confidence:.1%} < {rule.threshold:.1%})"
                    )
            elif general_threshold > 0:
                # NewFeature.md: Aplicar umbral general si no hay regla específica
                if confidence >= general_threshold:
                    final_predictions[diag_name] = confidence
                    applied_rules.append(
                        f"Accepted {diag_name} (confidence {confidence:.1%} >= general {general_threshold:.1%})"
                    )
                else:
                    applied_rules.append(
                        f"Rejected {diag_name} (confidence {confidence:.1%} < general {general_threshold:.1%})"
                    )
            else:
                # No rule found and no general threshold, keep prediction
                final_predictions[diag_name] = confidence
        
        return final_predictions, applied_rules
    
    def check_solo_diagnosis(
        self,
        module_type: ModuleType,
        predictions: Dict[str, float]
    ) -> Optional[str]:
        """
        NewFeature.md: Verifica si alguna predicción cumple con solo diagnosis rule.
        
        Solo diagnosis rule significa que si un diagnóstico tiene confidence muy alta,
        es el único diagnóstico válido (excluye todos los demás).
        
        Args:
            module_type: Tipo de módulo
            predictions: Dict {diagnosis_name: confidence}
        
        Returns:
            Diagnosis name si cumple solo diagnosis, None otherwise
            
        Example:
            # NewFeature.md línea 682-690: Solo diagnosis rules para rostro
            preds = {'venus_corazon': 0.70}  # > 65% (old threshold)
            solo = validator.check_solo_diagnosis(ModuleType.ESPEJO_ROSTRO, preds)
            # solo = 'venus_corazon'
        """
        if module_type not in self.rules:
            return None
        
        module_rules = self.rules[module_type]
        
        for diag_name, confidence in predictions.items():
            rule = module_rules.get(diag_name)
            if rule and rule.rule_type == 'solo' and confidence >= rule.threshold:
                return diag_name
        
        return None
    
    def get_general_threshold(self, module_type: ModuleType) -> float:
        """
        NewFeature.md: Obtiene umbral general para un módulo.
        
        Umbrales generales definidos:
        - ESPEJO_FRENTE: 18% (línea 19)
        - ESPEJO_ROSTRO: 18% (línea 17)
        - FRONTAL_MORFOLOGICO: 15 puntos de diferencia (línea 73)
        
        Args:
            module_type: Tipo de módulo
        
        Returns:
            Threshold value (e.g., 0.18 for Espejo)
        """
        # ============================================================================
        # NewFeature.md: Umbrales generales por módulo
        # ============================================================================
        GENERAL_THRESHOLDS = {
            # Línea 19: "Si predicción de frente es mayor al 18%"
            ModuleType.ESPEJO_FRENTE: 0.18,
            
            # Línea 17: "Si predicción de rostro es mayor al 18%"
            ModuleType.ESPEJO_ROSTRO: 0.18,
            
            # Línea 73: "tenga mínimo 15 puntos de porcentaje mayor"
            ModuleType.FRONTAL_MORFOLOGICO: 0.15,  # Regla de 15 puntos diferencia
        }
        
        return GENERAL_THRESHOLDS.get(module_type, 0.0)
    
    def _validate_rules(self):
        """
        Valida que las reglas cargadas son consistentes.
        Verifica que existan reglas para módulos requeridos.
        """
        # NewFeature.md líneas 13-73: Espejo es obligatorio
        required_modules = [
            ModuleType.ESPEJO_FRENTE,
            ModuleType.ESPEJO_ROSTRO
        ]
        
        for module in required_modules:
            if module not in self.rules:
                raise ValueError(f"Missing threshold rules for {module}")
        
        # Validate that exception rules are properly marked
        for module_type, rules in self.rules.items():
            for diag_name, rule in rules.items():
                if rule.is_exception and rule.rule_type not in ['minimum', 'solo']:
                    print(f"Warning: Exception rule {diag_name} has rule_type {rule.rule_type}, expected 'minimum' or 'solo'")

