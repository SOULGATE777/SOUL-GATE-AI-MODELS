# Common Validation Modules - NewFeature.md Implementation

## Descripción

Este módulo contiene la implementación centralizada de validación de umbrales, reglas de negocio y resolución de diagnósticos para todos los servicios ML del proyecto SOUL-GATE-AI-MODELS.

**Referencia**: `/home/mitza/proyectos/SOUL-GATE/NewFeature.md`

## Arquitectura

```
common/
├── __init__.py                      # Exports principales
├── threshold_validator.py           # Validador de umbrales centralizado
├── threshold_config.py              # Configuración de TODOS los umbrales
├── rule_engine.py                   # Motor de reglas (omisión, coincidencia)
├── proportion_analyzer.py           # Análisis de proporción facial/frente
├── diagnosis_resolver.py            # Resolución de diagnósticos complejos
├── README.md                        # Este archivo
└── tests/
    ├── __init__.py
    ├── test_threshold_validator.py
    └── test_*.py                    # Tests adicionales
```

## Módulos

### 1. ThresholdValidator (`threshold_validator.py`)

**NewFeature.md**: Líneas 13-270

Validador centralizado de umbrales para todos los módulos ML.

#### Características:

- **Single Source of Truth** para todos los umbrales
- Maneja excepciones (Venus 40%, Plutón 7%)
- Umbrales generales por módulo (Espejo 18%, etc.)
- Soporte para reglas de "solo diagnosis"

#### Uso:

```python
from common.threshold_validator import ThresholdValidator, ModuleType

validator = ThresholdValidator()

# Validar predicciones de ESPEJO ROSTRO
predictions = {
    'venus_corazon': 0.45,          # > 40% (excepción)
    'pluton_hexagonal': 0.08,       # > 7% (excepción)
    'saturno_trapezoide': 0.15      # < 18% (rechazar)
}

valid_preds, rules = validator.validate_predictions(
    module_type=ModuleType.ESPEJO_ROSTRO,
    predictions=predictions
)

# valid_preds = {'venus_corazon': 0.45, 'pluton_hexagonal': 0.08}
# saturno rechazado por < 18%
```

#### Módulos soportados:

- `ESPEJO_FRENTE`: Umbral general 18% (NewFeature.md línea 19)
- `ESPEJO_ROSTRO`: Umbral general 18%, excepciones Venus/Plutón (líneas 17, 22-23)
- `FRONTAL_MORFOLOGICO`: Regla de 15 puntos de diferencia (línea 73)
- `PROFILE_MORFOLOGICO`: Umbrales específicos por categoría (líneas 271-373)
- `PROFILE_ANTROPOMETRICO`: Reglas de coincidencia (líneas 374-428)

### 2. ThresholdConfig (`threshold_config.py`)

**NewFeature.md**: Líneas 13-428

Configuración centralizada de TODOS los umbrales del sistema.

#### Estructura:

```python
THRESHOLD_CONFIG = {
    ModuleType.ESPEJO_FRENTE: ESPEJO_FRENTE_RULES,
    ModuleType.ESPEJO_ROSTRO: ESPEJO_ROSTRO_RULES,
    ModuleType.FRONTAL_MORFOLOGICO: FRONTAL_MORFOLOGICO_RULES,
    ModuleType.PROFILE_MORFOLOGICO: PROFILE_MORFOLOGICO_RULES,
    ModuleType.PROFILE_ANTROPOMETRICO: PROFILE_ANTROPOMETRICO_RULES,
}
```

#### Umbrales implementados:

##### Espejo (líneas 13-73):
- FRENTE: 18% general
- ROSTRO: 18% general, Venus 40%, Plutón 7%

##### Frontal Morfológico (líneas 119-270):
- Cejas: 50% (cV, el, rc)
- Entrecejo: 60% (uniceja, líneas verticales)
- Párpado: 60% (ptosis, pliegue)
- Ojo: 22-70% (almendrado, fruncido, media luna)
- Oído: 25-33% (salido, pegado, promedio)
- Nariz: 14-65% (normal, grueso, delgada)
- Pómulo: 60% (promedio, plano)
- Cachete: 60-80% (lleno, plano, hundido, líneas)
- Boca: 30-60% (lunar, solar, mercurial, pursed)
- Arco cupido: 10-70% (no definido, marcado, triangular)

##### Perfil Morfológico (líneas 271-373):
- Lóbulo: 60% (pegado, despegado), 20% (hacia adelante)
- Frente: 19-55% (redondeada, plana, vertical, abultamiento)

### 3. RuleEngine (`rule_engine.py`)

**NewFeature.md**: Líneas 81-503

Motor de reglas para omisiones, coincidencias y lateralización.

#### Funcionalidades:

##### Reglas de Omisión:
- **Cejas antropométrico**: Siempre se omiten (línea 81)
- **Tercios rostro**: Omitir si hay obstáculos (línea 83)
- **Distancia trago-antitrago**: Siempre se omite (línea 377)

##### Reglas de Coincidencia:
- **Nariz largo**: Ambos perfiles deben coincidir (línea 379)
- **Protrusión ocular**: Ambos deben estar presentes y coincidir (línea 479)
- **Oreja largo**: Ambos deben coincidir (línea 491)

#### Uso:

```python
from common.rule_engine import RuleEngine

engine = RuleEngine()

# Verificar si cejas se omiten
if engine.should_omit_cejas_antropometrico():
    # Omitir análisis de cejas
    pass

# Verificar coincidencia de nariz entre perfiles
nariz_final = engine.check_nariz_coincidence(
    left_profile_nariz='nariz_larga',
    right_profile_nariz='nariz_larga'
)
# nariz_final = 'nariz_larga' (coinciden)
```

### 4. ProportionAnalyzer (`proportion_analyzer.py`)

**NewFeature.md**: Líneas 29-69

Analizador de proporciones para splitting de diagnósticos.

#### Proporciones analizadas:

##### Proporción Facial (líneas 29-54):
- **< 0.99**: Categoría "bajo"
- **0.99 - 1.17**: Categoría "medio"
- **>= 1.17**: Categoría "alto"

##### Proporción Frente (líneas 55-69):
- **> 0.35**: solar_lunar_emotivo
- **<= 0.35**: solar_lunar_creativo

#### Uso:

```python
from common.proportion_analyzer import ProportionAnalyzer

analyzer = ProportionAnalyzer()

# Analizar proporción facial
adjusted_diag, narrative, metadata = analyzer.analyze_facial_proportion(
    diagnosis='marte_tierra_rectangulo',
    proportion=1.05,
    confidence=0.85
)
# adjusted_diag = 'marte_tierra_accion' (proporción > 0.99)

# Analizar proporción frente
adjusted_diag, narrative, metadata = analyzer.analyze_frente_proportion(
    diagnosis='solar_lunar_combined',
    proportion=0.40,
    confidence=0.75
)
# adjusted_diag = 'solar_lunar_emotivo' (proporción > 0.35)
```

### 5. DiagnosisResolver (`diagnosis_resolver.py`)

**NewFeature.md**: Líneas 323-428

Resolución de diagnósticos complejos basados en combinaciones.

#### Funcionalidades:

##### Frente Perfil - 2 Criterios (líneas 323-330):
- **Escenario 1**: Solo perfil izquierdo → antropométrico
- **Escenario 2**: Solo perfil derecho → morfológico
- **Escenario 3**: Ambos → coincidir o priorizar derecho

##### Combinaciones (líneas 396-428):
- **Nariz corta + punta arriba**: persona impaciente
- **Mandíbula + Mentón**: temperamento combinado

#### Uso:

```python
from common.diagnosis_resolver import DiagnosisResolver

resolver = DiagnosisResolver()

# Resolver frente perfil con 2 criterios
frente_final = resolver.resolve_frente_perfil(
    left_profile={'exists': True},
    right_profile={'exists': True},
    antropometrico_result='frente_vertical',
    morfologico_result='frente_redondeada'
)
# frente_final = 'frente_redondeada' (prioriza derecho si no coinciden)

# Resolver nariz combinada
nariz_result = resolver.resolve_nariz_combined(
    nariz_largo='nariz_corta',
    nariz_angulo='punta_hacia_arriba'
)
# nariz_result = 'persona_impaciente'
```

## Integración con Servicios ML

### Espejo Pipeline (Ejemplo)

```python
# frontal_prod/espejo/app/models/espejo_pipeline.py

from common.threshold_validator import ThresholdValidator, ModuleType
from common.proportion_analyzer import ProportionAnalyzer
from common.rule_engine import RuleEngine

class EspejoAnalyzer:
    def __init__(self):
        # Initialize validation modules
        self.threshold_validator = ThresholdValidator()
        self.proportion_analyzer = ProportionAnalyzer()
        self.rule_engine = RuleEngine()
    
    def _apply_rostro_menton_decision_tree(self, predictions, probabilities, face_proportion):
        pred_dict = {pred: prob for pred, prob in zip(predictions, probabilities)}
        
        # NewFeature.md líneas 17, 22-23: Validar con umbrales
        valid_preds, rules = self.threshold_validator.validate_predictions(
            module_type=ModuleType.ESPEJO_ROSTRO,
            predictions=pred_dict
        )
        
        # Get highest confidence
        top_pred, top_prob = max(valid_preds.items(), key=lambda x: x[1])
        
        # NewFeature.md líneas 29-54: Aplicar proporción
        adjusted_diag, narrative, metadata = self.proportion_analyzer.analyze_facial_proportion(
            diagnosis=top_pred,
            proportion=face_proportion,
            confidence=top_prob
        )
        
        return adjusted_diag, rules
```

## Testing

### Ejecutar tests:

```bash
cd /home/mitza/proyectos/SOUL-GATE-AI-MODELS
python3 -m pytest common/tests/ -v
```

### Tests de compliance:

Todos los tests verifican compliance exacto con NewFeature.md:

```python
# common/tests/test_threshold_validator.py

def test_venus_corazon_40_percent_exception(self):
    """NewFeature.md línea 23: Venus Corazón excepción >40%"""
    rule = self.validator.rules[ModuleType.ESPEJO_ROSTRO]['venus_corazon']
    assert rule.threshold == 0.40, "NewFeature.md línea 23 requiere 40%"
```

## Feature Flag para Rollback

Todos los pipelines integrados incluyen feature flag para habilitar/deshabilitar:

```python
import os

USE_NEWFEATURE_THRESHOLDS = os.getenv('USE_NEWFEATURE_THRESHOLDS', 'true').lower() == 'true'

if USE_NEWFEATURE_THRESHOLDS:
    # Nueva lógica con ThresholdValidator
    results = self._analyze_with_newfeature_logic(image)
else:
    # Lógica antigua (rollback)
    results = self._analyze_legacy(image)
```

## Referencias

- **NewFeature.md**: `/home/mitza/proyectos/SOUL-GATE/NewFeature.md`
- **Plan de Implementación**: `/home/mitza/proyectos/SOUL-GATE-AI-MODELS/docs/PLAN_THRESHOLD_VALIDATOR.md`
- **Cursor Rules**: `.cursor/rules/*.mdc`

## Convención de Comentarios

**CRÍTICO**: Todos los comentarios en código DEBEN referenciar NewFeature.md con líneas:

```python
# ============================================================================
# NewFeature.md: [Título Sección] (líneas X-Y)
# ============================================================================
# [Descripción de la regla]
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
#
# Requisito:
# "[Cita literal del documento]"
# ============================================================================
```

## Próximos Pasos

**NO implementados en esta iteración (para posterior)**:

- Sistema de Temperamento complejo (NewFeature.md líneas 429-477)
- Colorimetría Ojos (líneas 505-608)
- Colorimetría Palmas (líneas 609-669)

## Contacto

Para preguntas sobre implementación, consultar:
- **Documento fuente**: `SOUL-GATE/NewFeature.md`
- **Plan detallado**: `docs/PLAN_THRESHOLD_VALIDATOR.md`

