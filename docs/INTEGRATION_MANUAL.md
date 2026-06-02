# Manual de Integración: Threshold Validation Logger

## Estado Actual de Implementación

### ✅ COMPLETADO - Infraestructura Core
- ✅ `ThresholdValidationLogger` - Logger funcional completo
- ✅ `newfeature_references.py` - Mapeo COMPLETO de todos los módulos de NewFeature.md
- ✅ Tests unitarios - 20+ tests comprehensivos
- ✅ Documentación - README y guías de uso

### 🔄 EN PROGRESO - Integraciones
- 🔄 **Espejo**: 50% (método frente completo, falta rostro + main.py)
- ⏳ **Frontal Morfológico**: 0%
- ⏳ **Frontal Antropométrico**: 0%
- ⏳ **Perfil Morfológico**: 0%
- ⏳ **Perfil Antropométrico**: 0%

---

## Patrón de Integración (Aplicar a TODOS los servicios)

### Paso 1: Modificar el archivo de pipeline/analyzer

#### 1.1 Agregar Imports

```python
# Al inicio del archivo, después de imports existentes de common
from common.threshold_validation_logger import ThresholdValidationLogger
from common.newfeature_references import get_newfeature_reference, get_module_references
```

#### 1.2 Modificar firma del método de análisis

```python
# ANTES:
def analyze_category(self, predictions, probabilities):
    
# DESPUÉS:
def analyze_category(self, predictions, probabilities, validation_logger=None):
```

#### 1.3 Agregar logging en el método

```python
def analyze_category(self, predictions, probabilities, validation_logger=None):
    """Analiza una categoría con logging de validaciones"""
    
    # 1. Iniciar categoría
    if validation_logger:
        validation_logger.start_category(
            category_name="Nombre de la Categoría",
            module_name="nombre_modulo",
            newfeature_lines="XX-YY",  # Líneas de NewFeature.md
            newfeature_text="Texto de referencia de NewFeature.md..."
        )
    
    # 2. Validar predicciones (tu lógica existente)
    pred_dict = {pred: prob for pred, prob in zip(predictions, probabilities)}
    valid_preds = validate_with_threshold(pred_dict)  # Tu validación
    
    # 3. Loggear CADA validación
    if validation_logger:
        for diag_name, confidence in pred_dict.items():
            is_valid = diag_name in valid_preds
            decision = "APROBADO" if is_valid else "RECHAZADO"
            
            # Obtener referencia de NewFeature.md
            diag_ref = get_newfeature_reference(diag_name, module="tu_modulo")
            threshold = diag_ref.get("threshold_value", 0.18)
            
            validation_logger.log_validation(
                characteristic=diag_name,
                value=confidence,
                threshold=threshold,
                decision=decision,
                reason=f"Confidence {confidence:.1%} {'>='' <'} {threshold:.1%}",
                newfeature_lines=diag_ref.get("lines", "N/A"),
                newfeature_text=diag_ref.get("text", "")[:200],
                is_exception=diag_ref.get("is_exception", False),
                is_final=False  # Se marca después
            )
    
    # 4. Seleccionar diagnóstico final (tu lógica)
    final_diagnosis = select_final(valid_preds)
    
    # 5. Marcar diagnóstico final y cerrar categoría
    if validation_logger:
        validation_logger.set_final_diagnoses([
            {"name": final_diagnosis, "confidence": pred_dict[final_diagnosis]}
        ])
        validation_logger.end_category()
    
    return final_diagnosis
```

### Paso 2: Modificar main.py del servicio

```python
from fastapi import FastAPI, UploadFile, File
from common.threshold_validation_logger import ThresholdValidationLogger

app = FastAPI()
analyzer = YourAnalyzer()  # Tu analyzer

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # 1. Crear logger
        validation_logger = ThresholdValidationLogger(service_name="nombre_servicio")
        analysis_id = validation_logger.start_analysis(metadata={
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        })
        
        # 2. Procesar imagen
        image = await process_image(file)
        
        # 3. Ejecutar análisis (pasando el logger)
        results = analyzer.analyze(image, validation_logger=validation_logger)
        
        # 4. Guardar logs
        validation_logger.save_markdown()
        validation_logger.save_json()  # Opcional
        
        # 5. Retornar resultados
        return {
            "success": True,
            "results": results,
            "_audit_id": analysis_id
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## Servicios a Integrar (Según NewFeature.md)

### 1. Espejo (frontal_prod/espejo)

**NewFeature.md**: Líneas 13-73  
**Archivo**: `app/models/espejo_pipeline.py`  
**Métodos**:
- ✅ `_apply_frente_decision_tree()` - YA INTEGRADO
- ⏳ `_apply_rostro_menton_decision_tree()` - FALTA COMPLETAR

**Categorías**:
1. Frente (ESPEJO) - Umbral 18%
2. Rostro Menton (ESPEJO) - Umbral 18% + Excepciones (Venus 40%, Plutón 7%)

**Acción**: Completar método rostro + modificar `main.py`

---

### 2. Frontal Morfológico (frontal_prod/morfologico)

**NewFeature.md**: Líneas 119-270  
**Archivo**: `app/models/morfologico_pipeline.py`  
**Categorías** (15 total):

1. **Cejas** (cj_d, cj_i) - Umbral 50%
   - Posibles: ceja_curva, ceja_inclinada, ceja_recta
   
2. **Entrecejo** - Umbrales variables
   - Posibles: normal, uniceja (60%), lineas_verticales (60%)
   
3. **Párpado** (d/i) - Umbral 60%
   - Posibles: ptosis, pliegue
   
4. **Ojo** (d/i) - Umbrales variables
   - circular (15% diferencia), almendrado (22%), fruncido (35%), media_luna_arriba (70%), media_luna_abajo (22%)
   
5. **Oído** (d/i) - Umbrales: 30%, 33%, 25%, 25%
   
6. **Nariz Grosor** - Umbrales: normal (65%), grueso (22%), delgada (14%)
   
7. **Punta Nariz** - Umbrales: redondeada (50%), puntiaguda (60%)
   
8. **Pómulo** (d/i) - Umbrales: promedio (60%), plano (60%)
   
9. **Cachete** (d/i) - Umbrales: lleno (60%), plano (80%), hundido (60%), lineas_sonrisa (80%)
   
10. **Forma Boca** - Umbrales: lunar (60%), solar (30%), mercurial (50%), pursed (30%)
    
11. **Arco Cupido** (d/i) - Umbrales: no_definido (70%), marcado (40%), triangular (10%)
    
12. **Tercios Faciales** - Umbral 15% diferencia

**Acción**: Agregar logging a cada método de análisis de categoría

---

### 3. Frontal Antropométrico (frontal_prod/antropometrico)

**NewFeature.md**: Líneas 75-118  
**Archivo**: `app/models/anthropometric_analyzer.py`  
**Categorías** (3 total):

1. **Tamaño Ojo** - Sin umbral (se toma exactamente)
   - ojo_grande, ojo_mediano, ojo_pequeño
   
2. **Tamaño Boca** - Sin umbral
   - boca_grande, boca_promedio, boca_pequeña
   
3. **Área Facial** - Sin umbral
   - cara_interna_promedio, cara_interna_pequeña, cara_interna_grande

**Nota**: Antropométrico NO tiene umbrales de confidence, se toma "exactamente como está"

**Acción**: Agregar logging que muestre los valores medidos sin filtrado por umbral

---

### 4. Perfil Morfológico (profile_prod/morfologico)

**NewFeature.md**: Líneas 271-373  
**Archivo**: `app/models/profile_analysis_pipeline.py`  
**Categorías** (6 total):

1. **Dorso Nariz** - "Mayor porcentaje de certeza"
   - convexa, concava, recta
   
2. **Lóbulo** (d/i) - Umbrales: pegado (60%), despegado (60%), hacia_adelante (20%)
   
3. **Mandíbula** (d/i) - "Mayor certeza"
   - nerviosa, biliosa, sanguinea, linfatica
   
4. **Submentón** - Umbral 45%
   - no_visible, visible
   
5. **Frente** (d/i) - Umbrales variables
   - redondeada_inclinada (27%), plana_inclinada (27%), vertical (20% derecha, 19% izquierda), abultamiento_tercio_inferior (25% derecha, 21% izquierda)

6. **Frente Antropométrica** (combinación con morfológico)

**Acción**: Agregar logging en cada categoría

---

### 5. Perfil Antropométrico (profile_prod/antropometrico)

**NewFeature.md**: Líneas 375-503  
**Archivo**: `app/models/profile_anthropometric_analyzer.py`  
**Categorías** (6 total):

1. **Nariz Largo** - "Ambos perfiles deben coincidir"
   - nariz_corta, nariz_protruyente, nariz_normal
   
2. **Nariz Ángulo** - Combinación con largo
   - promedio, hacia_arriba, hacia_abajo
   
3. **Mentón** (d/i)
   - sanguineo, biloso/linfatico, nervioso
   
4. **Mandíbula** (d/i)
   - bilosa, sanguinea, intermedia
   
5. **Protrusión Ocular** (d/i) - "Ambos deben coincidir"
   - positiva, nula, negativa
   
6. **Oreja** - "Ambos deben coincidir"
   - normal, corta, larga

**Acción**: Agregar logging con validación de coincidencia bilateral

---

### 6. Validación (frontal_prod/validacion)

**NewFeature.md**: Líneas 83 (excepción)  
**Nota**: "tercios de rostro se omite si diagnostico en modulo de validacion da objeto tapando frente o cabello tapando"

**Archivo**: YOLOv8 detector  
**Acción**: Posiblemente NO necesita logging detallado, solo registrar detecciones

---

## Orden Recomendado de Integración

1. ✅ **Espejo** (completar) - Ya está 50% hecho
2. ⏳ **Frontal Antropométrico** - Solo 3 categorías, sin umbrales
3. ⏳ **Frontal Morfológico** - 15 categorías, más complejo
4. ⏳ **Perfil Antropométrico** - 6 categorías
5. ⏳ **Perfil Morfológico** - 6 categorías

---

## Testing Post-Integración

Para cada servicio integrado:

```bash
# 1. Rebuild del servicio
cd /home/mitza/proyectos/SOUL-GATE-AI-MODELS
docker-compose build nombre_servicio
docker-compose restart nombre_servicio

# 2. Enviar imagen de prueba
curl -X POST -F "file=@test_image.jpg" \
  http://localhost:PUERTO/analyze

# 3. Verificar log generado
cat ./nombre_servicio/analysis_logs/YYYY-MM-DD/{uuid}_{timestamp}/validation_log.md

# 4. Comparar contra NewFeature.md
# Abrir side-by-side:
# - Left: validation_log.md
# - Right: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
```

---

## Resumen

- **Infraestructura**: 100% completa ✅
- **Mapeos NewFeature.md**: 100% completo ✅  
- **Integraciones**: 5% completo (1 de 5 servicios, parcial) 🔄
- **Documentación**: 100% completa ✅

**Total de Categorías a Integrar**: ~36 categorías across 5 servicios

**Esfuerzo Estimado**: 2-4 horas de trabajo manual siguiendo el patrón establecido

---

**Última Actualización**: 2025-12-23  
**Estado**: Core completo, integraciones pendientes

