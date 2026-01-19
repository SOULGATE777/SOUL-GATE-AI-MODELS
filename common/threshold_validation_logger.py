# ============================================================================
# Threshold Validation Logger - Sistema de Logging de Validaciones de Umbrales
# ============================================================================
# Genera logs Markdown detallados de cada validación de umbral con:
# - Nombre de característica evaluada
# - Valor obtenido del modelo
# - Umbral aplicado (y su origen en NewFeature.md)
# - Decisión tomada (APROBADO/RECHAZADO/OMITIDO)
# - Texto exacto de NewFeature.md que sustenta la validación
#
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
# ============================================================================

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ValidationEntry:
    """
    Entrada individual de validación de umbral.
    
    Attributes:
        characteristic: Nombre de la característica (ej: "ceja_curva", "venus_corazon")
        value: Valor/confidence obtenido del modelo (0.0 - 1.0)
        threshold: Umbral aplicado (0.0 - 1.0, None si N/A)
        decision: "APROBADO", "RECHAZADO", "OMITIDO"
        reason: Razón de la decisión (ej: "Confidence 62% >= 50% threshold")
        newfeature_lines: Líneas de NewFeature.md (ej: "123-132")
        newfeature_text: Texto relevante de NewFeature.md
        is_exception: Si es una excepción a la regla general
        is_final: Si este diagnóstico fue seleccionado como final
    """
    characteristic: str
    value: float
    threshold: Optional[float]
    decision: str  # "APROBADO", "RECHAZADO", "OMITIDO"
    reason: str
    newfeature_lines: str = ""
    newfeature_text: str = ""
    is_exception: bool = False
    is_final: bool = False


@dataclass
class CategoryValidation:
    """
    Validaciones de una categoría completa (ej: "Cejas Derecha", "Entrecejo").
    
    Attributes:
        category_name: Nombre de la categoría
        module_name: Nombre del módulo (ej: "frontal_morfologico")
        validations: Lista de ValidationEntry
        newfeature_reference_lines: Líneas de referencia general de la categoría
        newfeature_reference_text: Texto de referencia general
        final_diagnoses: Lista de diagnósticos finales seleccionados
    """
    category_name: str
    module_name: str
    validations: List[ValidationEntry] = field(default_factory=list)
    newfeature_reference_lines: str = ""
    newfeature_reference_text: str = ""
    final_diagnoses: List[Dict[str, Any]] = field(default_factory=list)


class ThresholdValidationLogger:
    """
    Logger de validaciones de umbrales para servicios ML.
    Genera logs Markdown detallados del proceso de validación.
    
    Usage:
        logger = ThresholdValidationLogger(service_name="espejo")
        analysis_id = logger.start_analysis()
        
        logger.start_category("rostro_menton", "espejo", 
                             newfeature_lines="17-23",
                             newfeature_text="Si predicción de rostro...")
        
        logger.log_validation(
            characteristic="venus_corazon",
            value=0.45,
            threshold=0.40,
            decision="APROBADO",
            reason="Confidence 45% >= 40% threshold",
            is_exception=True
        )
        
        logger.set_final_diagnoses(["venus_corazon"])
        logger.end_category()
        
        logger.save_markdown()
    """
    
    def __init__(self, service_name: str, base_path: str = "/app/analysis_logs", log_dir: Optional[Path] = None):
        """
        Inicializa el validation logger.
        
        Args:
            service_name: Nombre del servicio ML (ej: "espejo", "morfologico")
            base_path: Ruta base para guardar logs (default: /app/analysis_logs)
            log_dir: Directorio específico donde guardar logs (para consolidar con MLAuditLogger)
        """
        self.service_name = service_name
        self.base_path = Path(base_path)
        self._explicit_log_dir = log_dir  # If set, use this instead of generating a new one
        self._current_analysis_id: Optional[str] = None
        self._current_timestamp: Optional[str] = None
        self._categories: List[CategoryValidation] = []
        self._current_category: Optional[CategoryValidation] = None
        self._metadata: Dict[str, Any] = {}
    
    def start_analysis(self, analysis_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Inicia un nuevo análisis generando UUID y timestamp.
        
        Args:
            analysis_id: ID del análisis (para reusar el mismo ID que MLAuditLogger)
            metadata: Metadata adicional del análisis
            
        Returns:
            analysis_id generado o reutilizado
        """
        # Use provided analysis_id or generate a new one
        self._current_analysis_id = analysis_id or str(uuid.uuid4())
        self._current_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        self._categories = []
        self._metadata = metadata or {}
        self._metadata["service_name"] = self.service_name
        self._metadata["analysis_id"] = self._current_analysis_id
        self._metadata["timestamp"] = datetime.utcnow().isoformat()
        
        return self._current_analysis_id
    
    def log_raw_predictions(self, module_name: str, predictions: Dict[str, float]):
        """
        Registra las predicciones crudas antes de aplicar umbrales.
        
        Args:
            module_name: Nombre del módulo (ej: "frontal_morfologico")
            predictions: Dict de {feature_name: confidence_score}
        """
        if not self._metadata:
            self._metadata = {}
        
        if "raw_predictions" not in self._metadata:
            self._metadata["raw_predictions"] = {}
        
        self._metadata["raw_predictions"][module_name] = {
            "total_predictions": len(predictions),
            "predictions": predictions
        }
    
    def start_category(
        self,
        category_name: str,
        module_name: str,
        newfeature_lines: str = "",
        newfeature_text: str = ""
    ):
        """
        Inicia una nueva categoría de validación.
        
        Args:
            category_name: Nombre de la categoría (ej: "Cejas Derecha (cj_d)")
            module_name: Nombre del módulo (ej: "frontal_morfologico")
            newfeature_lines: Líneas de NewFeature.md (ej: "123-132")
            newfeature_text: Texto de referencia de NewFeature.md
        """
        if self._current_category:
            # Guardar categoría anterior
            self._categories.append(self._current_category)
        
        self._current_category = CategoryValidation(
            category_name=category_name,
            module_name=module_name,
            newfeature_reference_lines=newfeature_lines,
            newfeature_reference_text=newfeature_text
        )
    
    def log_validation(
        self,
        characteristic: str,
        value: float,
        threshold: Optional[float],
        decision: str,
        reason: str,
        newfeature_lines: str = "",
        newfeature_text: str = "",
        is_exception: bool = False,
        is_final: bool = False
    ):
        """
        Registra una validación individual.
        
        Args:
            characteristic: Nombre de la característica
            value: Valor obtenido (0.0 - 1.0)
            threshold: Umbral aplicado (None si N/A)
            decision: "APROBADO", "RECHAZADO", "OMITIDO"
            reason: Razón de la decisión
            newfeature_lines: Líneas específicas de NewFeature.md
            newfeature_text: Texto específico de NewFeature.md
            is_exception: Si es una excepción a la regla general
            is_final: Si fue seleccionado como diagnóstico final
        """
        if not self._current_category:
            raise ValueError("Must call start_category() first")
        
        entry = ValidationEntry(
            characteristic=characteristic,
            value=value,
            threshold=threshold,
            decision=decision,
            reason=reason,
            newfeature_lines=newfeature_lines,
            newfeature_text=newfeature_text,
            is_exception=is_exception,
            is_final=is_final
        )
        
        self._current_category.validations.append(entry)
    
    def set_final_diagnoses(self, diagnoses: List[Dict[str, Any]]):
        """
        Establece los diagnósticos finales para la categoría actual.
        
        Args:
            diagnoses: Lista de diagnósticos finales 
                      [{"name": "venus_corazon", "confidence": 0.45}, ...]
        """
        if not self._current_category:
            raise ValueError("Must call start_category() first")
        
        self._current_category.final_diagnoses = diagnoses
        
        # Marcar validaciones como finales
        final_names = [d.get("name") if isinstance(d, dict) else d 
                      for d in diagnoses]
        for validation in self._current_category.validations:
            if validation.characteristic in final_names:
                validation.is_final = True
    
    def end_category(self):
        """
        Finaliza la categoría actual y la guarda.
        """
        if self._current_category:
            self._categories.append(self._current_category)
            self._current_category = None
    
    def get_analysis_dir(self) -> Path:
        """
        Retorna el directorio para el análisis actual.
        
        Returns:
            Path del directorio del análisis
        """
        # If explicit log_dir was provided (for consolidation with MLAuditLogger), use it
        if self._explicit_log_dir:
            return Path(self._explicit_log_dir)
        
        if not self._current_analysis_id:
            raise ValueError("Must call start_analysis() first")
        
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        dir_name = f"{self._current_analysis_id}_{self._current_timestamp}"
        return self.base_path / date_str / dir_name
    
    def save_markdown(self) -> str:
        """
        Guarda el log de validaciones en formato Markdown.
        
        Returns:
            Path del archivo guardado (vacío si falla)
        """
        try:
            # Finalizar categoría actual si existe
            if self._current_category:
                self.end_category()
            
            if not self._categories:
                print("⚠️ No categories to save")
                return ""
            
            analysis_dir = self.get_analysis_dir()
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = analysis_dir / "validation_log.md"
            
            markdown_content = self._generate_markdown()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"✅ Validation log saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"⚠️ Validation logging failed: {e}")
            return ""
    
    def _generate_markdown(self) -> str:
        """
        Genera el contenido Markdown del log de validaciones.
        
        Returns:
            Contenido Markdown completo
        """
        lines = []
        
        # Header
        lines.append("# Threshold Validation Report")
        lines.append("")
        lines.append(f"**Analysis ID**: `{self._current_analysis_id}`")
        lines.append(f"**Service**: {self.service_name}")
        lines.append(f"**Timestamp**: {self._metadata.get('timestamp', 'N/A')}")
        lines.append("")
        
        # Metadata adicional
        if len(self._metadata) > 3:  # Más que service_name, analysis_id, timestamp
            lines.append("## Analysis Metadata")
            lines.append("")
            for key, value in self._metadata.items():
                if key not in ['service_name', 'analysis_id', 'timestamp']:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Cada categoría
        for category in self._categories:
            lines.extend(self._generate_category_markdown(category))
        
        # Summary al final
        lines.append("---")
        lines.append("")
        lines.append("## Validation Summary")
        lines.append("")
        
        total_validations = sum(len(cat.validations) for cat in self._categories)
        total_approved = sum(
            sum(1 for v in cat.validations if v.decision == "APROBADO")
            for cat in self._categories
        )
        total_rejected = sum(
            sum(1 for v in cat.validations if v.decision == "RECHAZADO")
            for cat in self._categories
        )
        total_omitted = sum(
            sum(1 for v in cat.validations if v.decision == "OMITIDO")
            for cat in self._categories
        )
        
        lines.append(f"- **Total Categories**: {len(self._categories)}")
        lines.append(f"- **Total Validations**: {total_validations}")
        lines.append(f"- **Approved**: {total_approved} ✅")
        lines.append(f"- **Rejected**: {total_rejected} ❌")
        lines.append(f"- **Omitted**: {total_omitted} ⚪")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_category_markdown(self, category: CategoryValidation) -> List[str]:
        """
        Genera el Markdown para una categoría.
        
        Args:
            category: CategoryValidation a renderizar
            
        Returns:
            Lista de líneas de Markdown
        """
        lines = []
        
        lines.append(f"## Module: {category.module_name} - {category.category_name}")
        lines.append("")
        
        # NewFeature.md reference
        if category.newfeature_reference_lines or category.newfeature_reference_text:
            lines.append(f"### NewFeature.md Reference (Lines {category.newfeature_reference_lines})")
            lines.append("")
            if category.newfeature_reference_text:
                # Quote the text
                for line in category.newfeature_reference_text.strip().split('\n'):
                    lines.append(f"> {line}")
                lines.append("")
        
        # Validations table
        if category.validations:
            lines.append("### Validations")
            lines.append("")
            lines.append("| Característica | Valor | Umbral | Decisión | Razón |")
            lines.append("|---|---|---|---|---|")
            
            for validation in category.validations:
                char_display = validation.characteristic
                if validation.is_final:
                    char_display = f"**{char_display}**"
                if validation.is_exception:
                    char_display = f"{char_display} ⚠️"
                
                value_display = f"{validation.value:.2%}"
                threshold_display = f"{validation.threshold:.2%}" if validation.threshold is not None else "N/A"
                
                decision_icon = {
                    "APROBADO": "✅",
                    "RECHAZADO": "❌",
                    "OMITIDO": "⚪"
                }.get(validation.decision, "")
                
                decision_display = f"{decision_icon} {validation.decision}"
                
                lines.append(f"| {char_display} | {value_display} | {threshold_display} | {decision_display} | {validation.reason} |")
            
            lines.append("")
        
        # Final diagnoses
        if category.final_diagnoses:
            lines.append("**Final Diagnosis**:")
            for diag in category.final_diagnoses:
                if isinstance(diag, dict):
                    name = diag.get("name", "Unknown")
                    confidence = diag.get("confidence", 0.0)
                    lines.append(f"- {name} (Confidence: {confidence:.2%})")
                else:
                    lines.append(f"- {diag}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def get_log_path(self) -> Path:
        """
        Retorna el path del directorio de logs actual.
        
        Returns:
            Path del directorio de análisis
        """
        return self.get_analysis_dir()
    
    def save_json(self) -> str:
        """
        Guarda el log de validaciones en formato JSON (adicional al Markdown).
        
        Returns:
            Path del archivo guardado (vacío si falla)
        """
        try:
            # Finalizar categoría actual si existe
            if self._current_category:
                self.end_category()
            
            analysis_dir = self.get_analysis_dir()
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = analysis_dir / "validation_log.json"
            
            data = {
                "analysis_id": self._current_analysis_id,
                "service_name": self.service_name,
                "timestamp": self._metadata.get('timestamp'),
                "metadata": self._metadata,
                "categories": [
                    {
                        "category_name": cat.category_name,
                        "module_name": cat.module_name,
                        "newfeature_reference_lines": cat.newfeature_reference_lines,
                        "newfeature_reference_text": cat.newfeature_reference_text,
                        "validations": [
                            {
                                "characteristic": v.characteristic,
                                "value": v.value,
                                "threshold": v.threshold,
                                "decision": v.decision,
                                "reason": v.reason,
                                "newfeature_lines": v.newfeature_lines,
                                "newfeature_text": v.newfeature_text,
                                "is_exception": v.is_exception,
                                "is_final": v.is_final
                            }
                            for v in cat.validations
                        ],
                        "final_diagnoses": cat.final_diagnoses
                    }
                    for cat in self._categories
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Validation JSON saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"⚠️ Validation JSON logging failed: {e}")
            return ""

