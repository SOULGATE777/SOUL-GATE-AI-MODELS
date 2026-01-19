# ============================================================================
# ML Audit Logger - Sistema de Auditoría para Servicios ML
# ============================================================================
# Guarda resultados RAW (antes de umbrales) y PROCESSED (después de umbrales)
# de cada análisis ML con UUID único para correlación posterior.
#
# Uso:
#   audit_logger = MLAuditLogger()
#   analysis_id = audit_logger.start_analysis()
#   audit_logger.log_raw("espejo", raw_data)
#   audit_logger.log_processed("espejo", processed_data)
#   audit_logger.save_metadata({"confidence": 0.5})
# ============================================================================

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class MLAuditLogger:
    """
    Sistema de auditoría para servicios ML.
    Guarda resultados RAW y PROCESSED de cada análisis con UUID único.
    
    Estructura de archivos:
        /analysis_logs/
        └── YYYY-MM-DD/
            └── {uuid}_{timestamp}/
                ├── {module}_raw.json
                ├── {module}_processed.json
                └── metadata.json
    """
    
    def __init__(self, base_path: str = "/app/analysis_logs"):
        """
        Inicializa el audit logger.
        
        Args:
            base_path: Ruta base para guardar logs (default: /app/analysis_logs)
        """
        self.base_path = Path(base_path)
        self._current_analysis_id: Optional[str] = None
        self._current_timestamp: Optional[str] = None
    
    def start_analysis(self) -> str:
        """
        Inicia un nuevo análisis generando UUID y timestamp.
        Debe llamarse al inicio de cada request.
        
        Returns:
            analysis_id generado (UUID)
        """
        self._current_analysis_id = str(uuid.uuid4())
        self._current_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return self._current_analysis_id
    
    def get_analysis_dir(self) -> Path:
        """
        Retorna el directorio para el análisis actual.
        
        Returns:
            Path del directorio del análisis
            
        Raises:
            ValueError: Si no se ha llamado start_analysis() primero
        """
        if not self._current_analysis_id:
            raise ValueError("Must call start_analysis() first")
        
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        dir_name = f"{self._current_analysis_id}_{self._current_timestamp}"
        return self.base_path / date_str / dir_name
    
    def log_raw(self, module_name: str, data: Dict) -> str:
        """
        Guarda resultado RAW (antes de umbrales).
        
        Args:
            module_name: Nombre del módulo (ej: "espejo", "morfologico")
            data: Datos a guardar
            
        Returns:
            Path del archivo guardado (vacío si falla)
        """
        return self._save_json(module_name, "raw", data)
    
    def log_processed(self, module_name: str, data: Dict) -> str:
        """
        Guarda resultado PROCESSED (después de umbrales).
        
        Args:
            module_name: Nombre del módulo (ej: "espejo", "morfologico")
            data: Datos a guardar
            
        Returns:
            Path del archivo guardado (vacío si falla)
        """
        return self._save_json(module_name, "processed", data)
    
    def _save_json(self, module_name: str, stage: str, data: Dict) -> str:
        """
        Guarda JSON en el directorio del análisis.
        
        Args:
            module_name: Nombre del módulo
            stage: "raw" o "processed"
            data: Datos a guardar
            
        Returns:
            Path del archivo guardado (vacío si falla)
        """
        try:
            analysis_dir = self.get_analysis_dir()
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{module_name}_{stage}.json"
            filepath = analysis_dir / filename
            
            payload = {
                "analysis_id": self._current_analysis_id,
                "timestamp": datetime.utcnow().isoformat(),
                "module": module_name,
                "stage": stage,
                "data": data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Audit log saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            # No fallar el análisis si el logging falla
            print(f"⚠️ Audit logging failed for {module_name}_{stage}: {e}")
            return ""
    
    def save_metadata(self, extra_info: Dict = None) -> str:
        """
        Guarda metadata del análisis.
        
        Args:
            extra_info: Información adicional a guardar (opcional)
            
        Returns:
            Path del archivo guardado (vacío si falla)
        """
        try:
            analysis_dir = self.get_analysis_dir()
            filepath = analysis_dir / "metadata.json"
            
            metadata = {
                "analysis_id": self._current_analysis_id,
                "created_at": datetime.utcnow().isoformat(),
                "timestamp_id": self._current_timestamp,
                **(extra_info or {})
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Metadata saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"⚠️ Metadata logging failed: {e}")
            return ""

