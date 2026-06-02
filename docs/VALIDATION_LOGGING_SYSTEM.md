# Threshold Validation Logging System

## Overview

This system provides detailed Markdown logs of **every** threshold validation performed by the ML services, allowing auditing against [`NewFeature.md`](../../SOUL-GATE/NewFeature.md) requirements.

## What It Does

For each analysis, the system logs:
- **Characteristic name** being validated (e.g., "venus_corazon", "ceja_curva")
- **Value** obtained from the ML model (confidence 0.0-1.0)
- **Threshold** applied from NewFeature.md
- **Decision** made (APROBADO ✅ / RECHAZADO ❌ / OMITIDO ⚪)
- **Reason** for the decision
- **NewFeature.md text** that justifies the threshold

## Output Example

### Log File Structure
```
/app/analysis_logs/
└── 2025-12-23/
    └── {uuid}_{timestamp}/
        ├── metadata.json
        ├── espejo_raw.json
        ├── espejo_processed.json
        └── validation_log.md  ⭐ NEW
```

### Sample validation_log.md

```markdown
# Threshold Validation Report

**Analysis ID**: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
**Service**: espejo
**Timestamp**: 2025-12-23T23:45:12Z

---

## Module: espejo - Frente (ESPEJO)

### NewFeature.md Reference (Lines 19-19)
> Si predicción de frente es mayor al 18% se toma como diagnostico certero, de lo contario, se omite.

### Validations

| Característica | Valor | Umbral | Decisión | Razón |
|---|---|---|---|---|
| solar_lunar_combined | 22.50% | 18.00% | ✅ APROBADO | Confidence 22.5% >= 18% threshold |
| mercurio_triangulo | 15.00% | 18.00% | ❌ RECHAZADO | Confidence 15% < 18% threshold |
| neptuno_combined | 62.30% | 18.00% | ✅ APROBADO | Confidence 62.3% >= 18% threshold |

**Final Diagnosis**: neptuno_combined (Confidence: 62.30%)

---

## Module: espejo - Rostro Menton

### NewFeature.md Reference (Lines 17-23)
> Si predicción de rostro es mayor al 18% se toma como diagnostico certero, de lo contario, se omite.
> Excepciones:
> Pluton Hexagonal (cuyo umbral es mayor a 7%) y Venus Corazon (mayor a 40%).

### Validations

| Característica | Valor | Umbral | Decisión | Razón |
|---|---|---|---|---|
| **venus_corazon** ⚠️ | 45.00% | 40.00% | ✅ APROBADO | EXCEPCIÓN: Confidence 45% >= 40% threshold |
| pluton_hexagonal ⚠️ | 8.00% | 7.00% | ✅ APROBADO | EXCEPCIÓN: Confidence 8% >= 7% threshold |
| saturno_trapezoide | 15.00% | 18.00% | ❌ RECHAZADO | Confidence 15% < 18% general threshold |
| mercurio_triangular | 22.00% | 18.00% | ✅ APROBADO | Confidence 22% >= 18% general threshold |

**Final Diagnosis**: 
- venus_corazon (Confidence: 45.00%)
- pluton_hexagonal (Confidence: 8.00%)
- mercurio_triangular (Confidence: 22.00%)

---

## Validation Summary

- **Total Categories**: 2
- **Total Validations**: 7
- **Approved**: 4 ✅
- **Rejected**: 3 ❌
- **Omitted**: 0 ⚪
```

## How to Use

### In Your ML Service

```python
from common.threshold_validation_logger import ThresholdValidationLogger
from common.newfeature_references import get_newfeature_reference, ESPEJO_ROSTRO_REFERENCES

# 1. Start analysis
validation_logger = ThresholdValidationLogger(service_name="espejo")
analysis_id = validation_logger.start_analysis(metadata={"user_id": "abc123"})

# 2. Start a category
validation_logger.start_category(
    category_name="Rostro Menton",
    module_name="espejo",
    newfeature_lines="17-23",
    newfeature_text="Si predicción de rostro es mayor al 18%..."
)

# 3. Log each validation
for diag_name, confidence in predictions.items():
    is_valid = confidence >= threshold
    decision = "APROBADO" if is_valid else "RECHAZADO"
    
    # Get NewFeature.md reference
    diag_ref = get_newfeature_reference(diag_name, module="espejo")
    
    validation_logger.log_validation(
        characteristic=diag_name,
        value=confidence,
        threshold=threshold,
        decision=decision,
        reason=f"Confidence {confidence:.1%} {'>='' <'} {threshold:.1%}",
        newfeature_lines=diag_ref.get("lines", "N/A"),
        newfeature_text=diag_ref.get("text", "")[:200],
        is_exception=diag_ref.get("is_exception", False),
        is_final=(diag_name in final_diagnoses)
    )

# 4. Set final diagnoses
validation_logger.set_final_diagnoses([
    {"name": "venus_corazon", "confidence": 0.45}
])

# 5. End category
validation_logger.end_category()

# 6. Save logs
validation_logger.save_markdown()  # Saves to /app/analysis_logs/.../validation_log.md
validation_logger.save_json()      # Optional: also save as JSON
```

## Integration Status

### ✅ Completed
- Core `ThresholdValidationLogger` class
- Comprehensive NewFeature.md mapping (`newfeature_references.py`)
- Unit tests
- Common module exports
- Integration pattern established in Espejo service

### 🔄 Integration Pattern Established

The Espejo service demonstrates the integration pattern:
- **File**: `frontal_prod/espejo/app/models/espejo_pipeline.py`
- **Methods**: `_apply_frente_decision_tree()`, `_apply_rostro_menton_decision_tree()`
- **Pattern**: Accept optional `validation_logger` parameter, log each validation, set final diagnoses

### 📋 Services to Integrate (Following Same Pattern)

Apply the same pattern as Espejo to:
1. **Frontal Morfológico** - 15+ categories (cejas, entrecejo, párpado, ojo, oído, nariz, pómulo, cachete, boca, arco cupido, tercios)
2. **Frontal Antropométrico** - 3 categories (tamaño ojo, tamaño boca, área facial)
3. **Perfil Morfológico** - 6 categories (dorso nariz, lóbulo, mandíbula, submentón, frente)
4. **Perfil Antropométrico** - 6 categories (nariz largo/ángulo, mentón, mandíbula, protrusión ocular, oreja)
5. **Body Morfológico** - (TBD based on model structure)
6. **Body Antropométrico** - (TBD based on model structure)

## Validation Against NewFeature.md

To validate compliance:

1. **Run analysis** with any ML service that has validation logging integrated
2. **Find the log**: `/app/analysis_logs/YYYY-MM-DD/{uuid}_{timestamp}/validation_log.md`
3. **Open side-by-side**:
   - Left: `validation_log.md`
   - Right: [`NewFeature.md`](../../SOUL-GATE/NewFeature.md)
4. **Check each validation**:
   - Does the threshold match NewFeature.md?
   - Is the NewFeature.md line reference correct?
   - Is the quoted text accurate?
   - Is the decision (APROBADO/RECHAZADO) correct given the threshold?

## Troubleshooting

### No validation_log.md generated
- Check that `validation_logger.save_markdown()` is being called
- Verify `/app/analysis_logs` directory is writable
- Check logs for errors during save

### Missing validations
- Ensure `validation_logger.start_category()` is called before logging validations
- Verify `validation_logger.log_validation()` is called for **every** prediction, not just valid ones
- Check that `validation_logger.end_category()` is called

### Wrong NewFeature.md references
- Update `common/newfeature_references.py` with correct line numbers and text
- Use `get_newfeature_reference(diagnosis_name, module)` to retrieve refs
- Check that diagnosis names match between model output and `newfeature_references.py` keys

## Architecture

```
common/
├── threshold_validation_logger.py  # Core logger class
├── newfeature_references.py        # NewFeature.md mapping (COMPREHENSIVE)
├── threshold_validator.py          # Validates against thresholds
├── threshold_config.py             # Threshold definitions
├── __init__.py                     # Exports logger and references
└── tests/
    └── test_threshold_validation_logger.py  # Unit tests

{service}/
├── app/
│   ├── main.py                     # FastAPI endpoint - start logger, pass to analyzer
│   └── models/
│       └── {analyzer}.py           # Analyzer - accept logger, log validations
└── analysis_logs/                  # Logs output here
    └── YYYY-MM-DD/
        └── {uuid}_{timestamp}/
            └── validation_log.md   # ⭐ Generated Markdown log
```

## Benefits

1. **Full Traceability**: Every single threshold check is logged with its NewFeature.md justification
2. **Easy Auditing**: Open Markdown files side-by-side with NewFeature.md to verify compliance
3. **Debugging**: See exactly which predictions were accepted/rejected and why
4. **Documentation**: Logs serve as proof of NewFeature.md compliance for stakeholders
5. **Maintenance**: When NewFeature.md changes, logs show what needs updating in models

## Next Steps

1. **Complete integrations**: Apply the Espejo pattern to remaining services
2. **CI/CD Check**: Add automated validation that checks logs against NewFeature.md
3. **Dashboard**: Create web UI to browse and search validation logs
4. **Alerts**: Set up monitoring for unexpected threshold validation patterns

---

**Author**: AI Assistant  
**Date**: 2025-12-23  
**Reference**: [`NewFeature.md`](../../SOUL-GATE/NewFeature.md)

