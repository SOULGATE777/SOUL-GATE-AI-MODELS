# Plan: Implementación de Threshold Validator Centralizado

**Fecha**: 2025-12-12  
**Objetivo**: Implementar validación centralizada de umbrales para TODOS los módulos ML según `NewFeature.md`  
**Documento Fuente**: `/home/mitza/proyectos/SOUL-GATE/NewFeature.md`

---

## 📋 RESUMEN EJECUTIVO

Crear un módulo centralizado (`ThresholdValidator`) que maneje TODAS las validaciones de umbrales de TODOS los módulos ML (Espejo, Morfológico, Antropométrico, etc.), evitando duplicación de lógica y facilitando mantenimiento.

**Beneficios**:
- ✅ **Single Source of Truth**: Un solo lugar para todos los umbrales
- ✅ **Mantenibilidad**: Cambios en 1 archivo, no en 15 servicios
- ✅ **Testabilidad**: Tests centralizados
- ✅ **Auditoría**: Fácil verificar compliance con NewFeature.md
- ✅ **Extensibilidad**: Agregar nuevos módulos fácilmente

---

## 🔍 ANÁLISIS DE SITUACIÓN ACTUAL

### Módulo Espejo (Personalidad) - Estado Actual

**Archivo**: `frontal_prod/espejo/app/models/espejo_pipeline.py`

#### Umbrales FRENTE (líneas 642-647)
```python
exclusion_rules = {
    'solar_lunar_combined': 0.19,                   # Actual: 19%
    'neptuno_combined': 0.20,                       # Actual: 20%
    'jupiter_aplio_base_ancha': 0.20,              # Actual: 20%
    'venus_corazon_o_trapezoide_angosto': 0.50     # Actual: 50%
}
```

#### Umbrales ROSTRO (líneas 682-711)
```python
# Solo diagnosis rules
solo_diagnosis_rules = {
    'saturno_trapezoide_base_angosta': 0.60,   # 60%
    'venus_corazon': 0.65,                      # 65%
    'luna_jupiter_combined': 0.10,              # 10%
    'mercurio_triangular': 0.35,                # 35%
    'pluton_hexagonal': 0.45,                   # 45%
    'marte_tierra_rectangulo': 0.88,            # 88%
    'sol_neptuno_combined': 0.10                # 10%
}

# Exclusion rules
exclusion_rules = {
    'venus_corazon': 0.35,                      # 35%
    'pluton_hexagonal': 0.15,                   # 15%
    'luna_jupiter_combined': 0.03,              # 3%
    'saturno_trapezoide_base_angosta': 0.23,    # 23%
    'mercurio_triangular': 0.17,                # 17%
    'sol_neptuno_combined': 0.04,               # 4%
    'marte_tierra_rectangulo': 0.30             # 30%
}
```

---

## 📊 UMBRALES REQUERIDOS (NewFeature.md)

### Módulo Espejo

| Región | Diagnóstico | Umbral Actual | Umbral Requerido | Cambio |
|--------|-------------|---------------|------------------|--------|
| **FRENTE** | General | 19-20% | **18%** | ✅ Simplificar |
| **FRENTE** | Venus Corazón/Trap Angosto | 50% | No especificado | ⚠️ Verificar |
| **ROSTRO** | General | Variable | **18%** | ✅ Nuevo umbral general |
| **ROSTRO** | Venus Corazón | 35% (excl) / 65% (solo) | **40%** | 🔄 Cambiar |
| **ROSTRO** | Plutón Hexagonal | 15% (excl) / 45% (solo) | **7%** | 🔄 Cambiar |

**NewFeature.md (líneas 15-23)**:
```
Si predicción de rostro es mayor al 18% se toma como diagnostico certero, de lo contario, se omite.
Si predicción de frente es mayor al 18% se toma como diagnostico certero, de lo contario, se omite.

Excepciones:
Pluton Hexagonal (cuyo umbral es mayor a 7%) y Venus Corazon (mayor a 40%).
```

---

### Otros Módulos con Umbrales (NewFeature.md)

#### Frontal Antropométrico (líneas 75-118)
- Cejas: Se omite siempre
- Tercios rostro: Se omite si "objeto tapando frente" o "cabello tapando"
- **No umbrales de confidence** explícitos (toma "exactamente como está")

#### Frontal Morfológico (líneas 119-270)

| Categoría | Diagnósticos | Umbrales |
|-----------|--------------|----------|
| **Cejas (cj_d, cj_i)** | cV, el, rc | 50% cada uno |
| **Entrecejo** | uniceja, lineas_verticales | 60% cada uno |
| **Párpado (d/i)** | ptosis, pliegue | 60% cada uno |
| **Ojo (d/i)** | al, fr, md, md_a | 22%, 35%, 70%, 22% |
| **Oído (d/i)** | sp_sl, sl, pg, pm | 30%, 33%, 25%, 25% |
| **Nariz (grosor)** | nrml, grueso, delgada | 65%, 22%, 14% |
| **Nariz (punta)** | rd, pn | 50%, 60% |
| **Pómulo (d/i)** | pm, pl | 60% cada uno |
| **Cachete (d/i)** | ll, pl, hn, lineas_sonriza | 60%, 80%, 60%, 80% |
| **Boca (forma)** | lunar, solar, mercurial, pursed | 60%, 30%, 50%, 30% |
| **Arco cupido (d/i)** | nd, on, pc | 70%, 40%, 10% |

**Regla General** (línea 73):
> "si no existe umbral, se toma el de mayor porcentaje tenga mínimo 15 puntos de porcentaje mayor a los demás"

#### Perfil Morfológico (líneas 271-373)

| Categoría | Diagnósticos | Umbrales |
|-----------|--------------|----------|
| **Lóbulo** | lobulo_pegado, lobulo_despegado | 60% cada uno |
| **Submenton** | visible | 45% |
| **Frente morfo** | f_rd_incl, f_pl_incl, fr_vert, ab_t_inf | 27%, 27%, 20%/19%, 25%/21% |

#### Perfil Antropométrico (líneas 374-503)
- Distancia trago-antitrago: **Se omite**
- Nariz largo: Requiere coincidencia entre ambos perfiles
- **No umbrales de confidence** explícitos (diagnóstico basado en mediciones)

---

## 🏗️ ARQUITECTURA PROPUESTA

### 1. Módulo Centralizado: `ThresholdValidator`

**Ubicación sugerida**: `common/threshold_validator.py` (nuevo directorio)

```
SOUL-GATE-AI-MODELS/
├── common/                          # ✅ NUEVO
│   ├── __init__.py
│   ├── threshold_validator.py      # Validador centralizado
│   ├── threshold_config.py         # Configuración de umbrales
│   └── tests/
│       └── test_threshold_validator.py
├── frontal_prod/
│   ├── espejo/
│   │   └── app/models/
│   │       └── espejo_pipeline.py  # Usa ThresholdValidator
│   ├── morfologico/
│   └── antropometrico/
└── profile_prod/
    └── ...
```

---

### 2. Diseño de `ThresholdValidator`

```python
# common/threshold_validator.py

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ModuleType(Enum):
    """Tipos de módulos ML"""
    ESPEJO_FRENTE = "espejo_frente"
    ESPEJO_ROSTRO = "espejo_rostro"
    FRONTAL_MORFOLOGICO = "frontal_morfologico"
    FRONTAL_ANTROPOMETRICO = "frontal_antropometrico"
    PROFILE_MORFOLOGICO = "profile_morfologico"
    PROFILE_ANTROPOMETRICO = "profile_antropometrico"

@dataclass
class ThresholdRule:
    """Regla de umbral para un diagnóstico específico"""
    diagnosis_name: str
    threshold: float
    rule_type: str  # 'minimum', 'solo', 'exclusion'
    is_exception: bool = False
    description: str = ""

class ThresholdValidator:
    """
    Validador centralizado de umbrales para TODOS los módulos ML.
    Single Source of Truth para NewFeature.md thresholds.
    """
    
    def __init__(self):
        self.rules = self._load_threshold_rules()
        self._validate_rules()
    
    def _load_threshold_rules(self) -> Dict[ModuleType, Dict[str, ThresholdRule]]:
        """
        Carga TODAS las reglas de umbrales desde configuración.
        
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
        Valida predicciones contra umbrales del módulo.
        
        Args:
            module_type: Tipo de módulo (Espejo, Morfológico, etc.)
            predictions: Dict {diagnosis_name: confidence}
            additional_context: Contexto adicional (e.g., proporción facial)
        
        Returns:
            Tuple[valid_predictions, applied_rules]
        """
        if module_type not in self.rules:
            raise ValueError(f"Unknown module type: {module_type}")
        
        module_rules = self.rules[module_type]
        valid_predictions = {}
        applied_rules = []
        
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
        
        # 2. Aplicar minimum thresholds
        final_predictions = {}
        for diag_name, confidence in valid_predictions.items():
            rule = module_rules.get(diag_name)
            
            if rule and rule.rule_type in ['minimum', 'solo']:
                if confidence >= rule.threshold:
                    final_predictions[diag_name] = confidence
                    applied_rules.append(
                        f"Accepted {diag_name} (confidence {confidence:.1%} >= {rule.threshold:.1%})"
                    )
                else:
                    applied_rules.append(
                        f"Rejected {diag_name} (confidence {confidence:.1%} < {rule.threshold:.1%})"
                    )
            else:
                # No rule found, keep prediction
                final_predictions[diag_name] = confidence
        
        return final_predictions, applied_rules
    
    def check_solo_diagnosis(
        self,
        module_type: ModuleType,
        predictions: Dict[str, float]
    ) -> Optional[str]:
        """
        Verifica si alguna predicción cumple con solo diagnosis rule.
        
        Returns:
            Diagnosis name si cumple solo diagnosis, None otherwise
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
        Obtiene umbral general para un módulo.
        
        Returns:
            Threshold value (e.g., 0.18 for Espejo)
        """
        # Umbrales generales por módulo
        GENERAL_THRESHOLDS = {
            ModuleType.ESPEJO_FRENTE: 0.18,
            ModuleType.ESPEJO_ROSTRO: 0.18,
            ModuleType.FRONTAL_MORFOLOGICO: 0.15,  # Regla de 15 puntos
        }
        
        return GENERAL_THRESHOLDS.get(module_type, 0.0)
    
    def _validate_rules(self):
        """Valida que las reglas cargadas son consistentes"""
        # Verificar que existan reglas para módulos requeridos
        required_modules = [
            ModuleType.ESPEJO_FRENTE,
            ModuleType.ESPEJO_ROSTRO
        ]
        
        for module in required_modules:
            if module not in self.rules:
                raise ValueError(f"Missing threshold rules for {module}")
```

---

### 3. Configuración de Umbrales: `threshold_config.py`

```python
# common/threshold_config.py

from .threshold_validator import ModuleType, ThresholdRule

# =====================================================
# ESPEJO MODULE - NewFeature.md lines 13-23
# =====================================================

ESPEJO_FRENTE_RULES = {
    # Umbral general: 18% (NewFeature.md línea 19)
    # "Si predicción de frente es mayor al 18% se toma como diagnostico certero"
    
    'solar_lunar_combined': ThresholdRule(
        diagnosis_name='solar_lunar_combined',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
    'neptuno_combined': ThresholdRule(
        diagnosis_name='neptuno_combined',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
    'jupiter_aplio_base_ancha': ThresholdRule(
        diagnosis_name='jupiter_aplio_base_ancha',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
    'marte_rectangular': ThresholdRule(
        diagnosis_name='marte_rectangular',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
    'venus_corazon_o_trapezoide_angosto': ThresholdRule(
        diagnosis_name='venus_corazon_o_trapezoide_angosto',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum (no exception for FRENTE)'
    ),
    'mercurio_triangulo': ThresholdRule(
        diagnosis_name='mercurio_triangulo',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
}

ESPEJO_ROSTRO_RULES = {
    # Umbral general: 18% (NewFeature.md línea 17)
    # "Si predicción de rostro es mayor al 18% se toma como diagnostico certero"
    
    # EXCEPCIONES (NewFeature.md líneas 22-23)
    'venus_corazon': ThresholdRule(
        diagnosis_name='venus_corazon',
        threshold=0.40,  # 40% exception
        rule_type='minimum',
        is_exception=True,
        description='NewFeature.md: Venus Corazon exception >40%'
    ),
    'pluton_hexagonal': ThresholdRule(
        diagnosis_name='pluton_hexagonal',
        threshold=0.07,  # 7% exception
        rule_type='minimum',
        is_exception=True,
        description='NewFeature.md: Pluton Hexagonal exception >7%'
    ),
    
    # Resto con umbral general 18%
    'saturno_trapezoide_base_angosta': ThresholdRule(
        diagnosis_name='saturno_trapezoide_base_angosta',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
    'luna_jupiter_combined': ThresholdRule(
        diagnosis_name='luna_jupiter_combined',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
    'mercurio_triangular': ThresholdRule(
        diagnosis_name='mercurio_triangular',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
    'marte_tierra_rectangulo': ThresholdRule(
        diagnosis_name='marte_tierra_rectangulo',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
    'sol_neptuno_combined': ThresholdRule(
        diagnosis_name='sol_neptuno_combined',
        threshold=0.18,
        rule_type='minimum',
        description='NewFeature.md: 18% minimum'
    ),
}

# =====================================================
# FRONTAL MORFOLÓGICO - NewFeature.md lines 119-270
# =====================================================

FRONTAL_MORFOLOGICO_RULES = {
    # CEJAS (líneas 123-131)
    'cj_d_cv': ThresholdRule(
        diagnosis_name='cj_d_cv',
        threshold=0.50,
        rule_type='minimum',
        description='NewFeature.md: Ceja curva 50%'
    ),
    'cj_d_el': ThresholdRule(
        diagnosis_name='cj_d_el',
        threshold=0.50,
        rule_type='minimum',
        description='NewFeature.md: Ceja inclinada 50%'
    ),
    'cj_d_rc': ThresholdRule(
        diagnosis_name='cj_d_rc',
        threshold=0.50,
        rule_type='minimum',
        description='NewFeature.md: Ceja recta 50%'
    ),
    # ... (similar para cj_i)
    
    # ENTRECEJO (líneas 133-141)
    'entrecejo_uniceja': ThresholdRule(
        diagnosis_name='entrecejo_uniceja',
        threshold=0.60,
        rule_type='minimum',
        description='NewFeature.md: Uniceja 60%'
    ),
    'entrecejo_lineas_verticales': ThresholdRule(
        diagnosis_name='entrecejo_lineas_verticales',
        threshold=0.60,
        rule_type='minimum',
        description='NewFeature.md: Líneas verticales 60%'
    ),
    
    # OJO (líneas 151-164)
    'oj_d_al': ThresholdRule(
        diagnosis_name='oj_d_al',
        threshold=0.22,
        rule_type='minimum',
        description='NewFeature.md: Ojo almendrado 22%'
    ),
    'oj_d_fr': ThresholdRule(
        diagnosis_name='oj_d_fr',
        threshold=0.35,
        rule_type='minimum',
        description='NewFeature.md: Ojo fruncido 35%'
    ),
    'oj_d_md': ThresholdRule(
        diagnosis_name='oj_d_md',
        threshold=0.70,
        rule_type='minimum',
        description='NewFeature.md: Ojo media luna arriba 70%'
    ),
    
    # ... (continuar con todos los umbrales)
}

# =====================================================
# CONFIGURATION DICT - Single Source of Truth
# =====================================================

THRESHOLD_CONFIG = {
    ModuleType.ESPEJO_FRENTE: ESPEJO_FRENTE_RULES,
    ModuleType.ESPEJO_ROSTRO: ESPEJO_ROSTRO_RULES,
    ModuleType.FRONTAL_MORFOLOGICO: FRONTAL_MORFOLOGICO_RULES,
    # ... agregar otros módulos según se implementen
}
```

---

## 📝 PLAN DE IMPLEMENTACIÓN

### FASE 1: Setup Inicial (2-3 horas)

**Objetivo**: Crear estructura base del `ThresholdValidator`

1. **Crear directorio `common/`**
   ```bash
   mkdir -p common/tests
   touch common/__init__.py
   touch common/threshold_validator.py
   touch common/threshold_config.py
   touch common/tests/__init__.py
   touch common/tests/test_threshold_validator.py
   ```

2. **Implementar clases base**
   - `ModuleType` enum
   - `ThresholdRule` dataclass
   - `ThresholdValidator` class (métodos básicos)

3. **Configurar umbrales Espejo**
   - `ESPEJO_FRENTE_RULES`
   - `ESPEJO_ROSTRO_RULES`
   - `THRESHOLD_CONFIG` dict

4. **Tests unitarios iniciales**
   - Test de carga de configuración
   - Test de validación básica
   - Test de excepciones (Venus, Plutón)

**Archivos Modificados**: ✅ Ninguno (solo nuevos)

**Deliverables**:
- [ ] `common/threshold_validator.py` creado
- [ ] `common/threshold_config.py` creado
- [ ] Tests básicos pasando

---

### FASE 2: Integración con Módulo Espejo (3-4 horas)

**Objetivo**: Refactorizar `espejo_pipeline.py` para usar `ThresholdValidator`

1. **Import ThresholdValidator en espejo_pipeline.py**
   ```python
   from common.threshold_validator import ThresholdValidator, ModuleType
   ```

2. **Modificar `_apply_frente_decision_tree()`**
   - **ANTES**: Hardcoded `exclusion_rules` dict
   - **DESPUÉS**: Usar `validator.validate_predictions(ModuleType.ESPEJO_FRENTE, ...)`
   
   ```python
   def _apply_frente_decision_tree(self, predictions, probabilities):
       # Old code (DELETE):
       # exclusion_rules = {
       #     'solar_lunar_combined': 0.19,
       #     ...
       # }
       
       # New code (ADD):
       pred_dict = {pred: prob for pred, prob in zip(predictions, probabilities)}
       
       valid_preds, applied_rules = self.threshold_validator.validate_predictions(
           module_type=ModuleType.ESPEJO_FRENTE,
           predictions=pred_dict
       )
       
       # Continue with decision tree logic using valid_preds...
   ```

3. **Modificar `_apply_rostro_menton_decision_tree()`**
   - **ANTES**: Hardcoded `solo_diagnosis_rules` y `exclusion_rules`
   - **DESPUÉS**: Usar `validator.validate_predictions(ModuleType.ESPEJO_ROSTRO, ...)`
   
   ```python
   def _apply_rostro_menton_decision_tree(self, predictions, probabilities, face_proportion):
       # Old code (DELETE):
       # solo_diagnosis_rules = {...}
       # exclusion_rules = {...}
       
       # New code (ADD):
       pred_dict = {pred: prob for pred, prob in zip(predictions, probabilities)}
       
       # Check solo diagnosis first
       solo_diag = self.threshold_validator.check_solo_diagnosis(
           module_type=ModuleType.ESPEJO_ROSTRO,
           predictions=pred_dict
       )
       
       if solo_diag:
           # Handle solo diagnosis with proportion splitting
           return self._apply_proportion_based_splitting(solo_diag, face_proportion, ...)
       
       # Validate remaining predictions
       valid_preds, applied_rules = self.threshold_validator.validate_predictions(
           module_type=ModuleType.ESPEJO_ROSTRO,
           predictions=pred_dict
       )
       
       # Continue with decision tree logic...
   ```

4. **Inicializar validator en `__init__`**
   ```python
   def __init__(self):
       # ... existing code ...
       self.threshold_validator = ThresholdValidator()
   ```

5. **Testing exhaustivo**
   - Test con imágenes reales
   - Verificar que umbrales 18%, 40%, 7% se aplican correctamente
   - Comparar resultados ANTES vs DESPUÉS

**Archivos Modificados**:
- ✅ `frontal_prod/espejo/app/models/espejo_pipeline.py` (refactor)
- ✅ `frontal_prod/espejo/tests/test_espejo_thresholds.py` (nuevo)

**Deliverables**:
- [ ] Espejo usando ThresholdValidator
- [ ] Tests de integración pasando
- [ ] Validación con imágenes reales
- [ ] Documented changes in `@docs/newfeature-implementation.md`

---

### FASE 3: Extensión a Otros Módulos (6-8 horas)

**Objetivo**: Agregar umbrales para Morfológico y Antropométrico

1. **Completar `threshold_config.py`**
   - `FRONTAL_MORFOLOGICO_RULES` (todos los umbrales de líneas 119-270)
   - `PROFILE_MORFOLOGICO_RULES` (líneas 271-373)
   - Implementar lógica de "15 puntos de diferencia" para morfológico

2. **Refactorizar módulos uno por uno**
   - `frontal_prod/morfologico/app/models/facial_analysis_pipeline.py`
   - `frontal_prod/antropometrico/app/models/anthropometric_pipeline.py`
   - `profile_prod/morfologico/app/models/profile_analysis_pipeline.py`
   - `profile_prod/antropometrico/app/models/profile_anthropometric_pipeline.py`

3. **Testing por módulo**
   - Tests unitarios
   - Integration tests
   - Validación con imágenes reales

**Archivos Modificados**:
- ✅ `common/threshold_config.py` (extensión)
- ✅ 4-5 archivos `*_pipeline.py` (refactor)
- ✅ Tests correspondientes

**Deliverables**:
- [ ] Todos los módulos usando ThresholdValidator
- [ ] Tests pasando
- [ ] Documentation completa

---

### FASE 4: Testing & Validación (2-3 horas)

**Objetivo**: Validar que TODAS las reglas de NewFeature.md están implementadas

1. **Test suite completo**
   ```python
   # common/tests/test_newfeature_compliance.py
   
   def test_espejo_frente_18_percent():
       """Verify FRENTE threshold is 18%"""
       validator = ThresholdValidator()
       threshold = validator.get_general_threshold(ModuleType.ESPEJO_FRENTE)
       assert threshold == 0.18
   
   def test_espejo_venus_40_percent():
       """Verify Venus Corazon exception is 40%"""
       validator = ThresholdValidator()
       rule = validator.rules[ModuleType.ESPEJO_ROSTRO]['venus_corazon']
       assert rule.threshold == 0.40
       assert rule.is_exception is True
   
   def test_espejo_pluton_7_percent():
       """Verify Pluton Hexagonal exception is 7%"""
       validator = ThresholdValidator()
       rule = validator.rules[ModuleType.ESPEJO_ROSTRO]['pluton_hexagonal']
       assert rule.threshold == 0.07
       assert rule.is_exception is True
   ```

2. **Validación con imágenes reales**
   - Tomar 10-20 imágenes de prueba
   - Comparar resultados ANTES vs DESPUÉS
   - Verificar que cambios son los esperados

3. **Documentation final**
   - Actualizar `@docs/newfeature-implementation.md`
   - Actualizar `@docs/architecture.md` con ThresholdValidator
   - Crear `common/README.md` explicando uso

**Deliverables**:
- [ ] 100% tests passing
- [ ] Validación con imágenes reales completa
- [ ] Documentation actualizada

---

## ⚠️ CONSIDERACIONES IMPORTANTES

### 1. Backwards Compatibility

**⚠️ ROMPE COMPATIBILIDAD**: Los resultados cambiarán por umbrales diferentes

**Solución**:
- Feature flag para habilitar/deshabilitar nuevo validador
- Logging detallado de cambios
- A/B testing en producción

```python
# Opción para rollback
USE_NEW_THRESHOLDS = os.getenv('USE_NEW_THRESHOLDS', 'false').lower() == 'true'

if USE_NEW_THRESHOLDS:
    valid_preds, rules = self.threshold_validator.validate_predictions(...)
else:
    # Old logic (mantener temporalmente)
    valid_preds = self._old_validate_logic(...)
```

### 2. Performance

**Impacto**: Mínimo (validación es O(n) donde n = número de predicciones)

**Optimización**:
- Cache de reglas cargadas
- Evitar re-crear `ThresholdValidator` por request

### 3. Testing

**Crítico**: Validación exhaustiva antes de producción

**Strategy**:
1. Unit tests (cada threshold rule)
2. Integration tests (pipelines completos)
3. End-to-end tests (imágenes reales)
4. A/B testing en producción

### 4. Documentation

**Obligatorio**: Documentar cada cambio

**Archivos**:
- `@docs/newfeature-implementation.md` (tracking)
- `@docs/architecture.md` (diseño)
- `common/README.md` (uso del validador)
- `CHANGELOG.md` (historial de cambios)

---

## 📊 ESTIMACIONES DE TIEMPO

| Fase | Descripción | Tiempo Estimado |
|------|-------------|-----------------|
| **FASE 1** | Setup inicial ThresholdValidator | 2-3 horas |
| **FASE 2** | Integración Módulo Espejo | 3-4 horas |
| **FASE 3** | Extensión otros módulos | 6-8 horas |
| **FASE 4** | Testing & Validación | 2-3 horas |
| **TOTAL** | | **13-18 horas** |

---

## ✅ CHECKLIST DE VALIDACIÓN

### Pre-Implementation
- [ ] NewFeature.md leído y entendido completo
- [ ] Umbrales actuales identificados en todos los módulos
- [ ] Arquitectura ThresholdValidator diseñada
- [ ] Plan aprobado por equipo

### FASE 1 - Setup
- [ ] Directorio `common/` creado
- [ ] `threshold_validator.py` implementado
- [ ] `threshold_config.py` con umbrales Espejo
- [ ] Tests unitarios básicos pasando

### FASE 2 - Espejo
- [ ] `espejo_pipeline.py` refactorizado
- [ ] Umbrales 18%, 40%, 7% aplicándose correctamente
- [ ] Tests de integración pasando
- [ ] Validación con imágenes reales OK

### FASE 3 - Otros Módulos
- [ ] `threshold_config.py` completo (todos los módulos)
- [ ] Morfológico refactorizado
- [ ] Antropométrico refactorizado
- [ ] Profile módulos refactorizados
- [ ] Tests pasando para todos

### FASE 4 - Final
- [ ] 100% compliance con NewFeature.md
- [ ] A/B testing en producción
- [ ] Documentation completa
- [ ] Feature flag implementado (rollback capability)

---

## 📚 REFERENCIAS

- **NewFeature.md**: `/home/mitza/proyectos/SOUL-GATE/NewFeature.md`
- **Espejo Pipeline**: `frontal_prod/espejo/app/models/espejo_pipeline.py`
- **Cursor Rule**: `.cursor/rules/frontal-specific.mdc`
- **Tracking Doc**: `@docs/newfeature-implementation.md`

---

## 🎯 SIGUIENTE PASO

**¿Proceder con implementación?**

Opciones:
1. ✅ **"sí, comenzar FASE 1"** - Crear estructura base de ThresholdValidator
2. ✅ **"sí, implementar todo"** - Ejecutar FASES 1-4 completas
3. ❌ **"ajustar plan primero"** - Modificar approach antes de implementar

**Esperando tu confirmación para proceder...**

