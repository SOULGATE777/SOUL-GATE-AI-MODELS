# 05 — Servicios de análisis IA (wrappers y ML)

**Repositorios:** `API-GATAWAY` (wrappers Node), `SOUL-GATE-AI-MODELS` (ML Python)  
**Última actualización:** 2026-05-21

---

## Arquitectura en dos capas

```
Cliente (API negocio)
    → API Gateway :4051
        → souldate-ia-* :4301-4306  (Express proxy)
            → FastAPI SOUL-GATE-AI-MODELS :8000-8014  (GPU)
```

Excepción: **temperamentos (4306)** ejecuta lógica Node en el repo gateway, no Python.

---

## Wrappers Node (`API-GATAWAY`)

### `souldate-ia-frontal` — puerto **4301**

| Método | Ruta wrapper | Backend ML típico |
|--------|--------------|-------------------|
| `GET` | `/health`, `/api/v1/health` | — |
| `POST` | `/api/v1/analyze-face` | `:8000/analyze-face` |
| `POST` | `/api/v1/analyze-anthropometric` | `:8001/analyze-anthropometric` |
| `POST` | `/api/v1/analyze-validation` | `:8002/analyze-validation` |
| `POST` | `/api/v1/analyze-espejo` | `:8008/analyze-espejo` |

### `souldate-ia-perfil` — **4302**

| Ruta | ML |
|------|-----|
| `/api/v1/analyze-profile-morphological` | 8003 |
| `/api/v1/analyze-profile-anthropometric` | 8004 |
| `/api/v1/analyze-profile-validation` | 8005 |
| `/api/v1/preprocess-profile` | 8010 |

### `souldate-ia-cuerpo` — **4303**

| Ruta | ML |
|------|-----|
| `/api/v1/analyze-body-anthropometry` | 8006–8007 (según despliegue) |
| `/api/v1/analyze-body-morphology` | idem |

**Gateway consolidado:** llamadas a cuerpo **comentadas/deshabilitadas**.

### `souldate-ia-palmas` — **4304**

| Ruta | ML |
|------|-----|
| `/api/v1/analyze-hand-comprehensive` | 8009 |

### `souldate-ia-ojos` — **4305**

| Ruta | ML |
|------|-----|
| `/api/v1/analyze-eye-colorimetry` | 8001 (colorimetría) |

### `souldate-ia-temperamentos` — **4306**

| Ruta | Descripción |
|------|-------------|
| `POST /api/v1/calculate-temperament` | Algoritmo 5 etapas L/S/B/N |
| `GET /api/v1/temperament-info` | Metadatos |

Entrada: tags agregados desde resultados ML (gateway).

---

## Microservicios Python (`SOUL-GATE-AI-MODELS`)

Ver inventario completo en [07-ai-models-pipeline.md](./07-ai-models-pipeline.md).

| Puerto | Módulo | Servicio |
|--------|--------|----------|
| 8000 | frontal | Morfológico |
| 8001 | frontal | Antropométrico (+ colorimetría ojos) |
| 8002 | frontal | Validación YOLO |
| 8008 | frontal | Espejo / personalidad |
| 8012 | frontal | Rotación |
| 8014 | frontal | Preprocesamiento |
| 8003 | perfil | Morfológico perfil |
| 8004 | perfil | Antropométrico perfil |
| 8005 | perfil | Validación perfil |
| 8010 | perfil | Preprocesamiento perfil |
| 8009 | body | Manos |
| 8013 | age | Estimación edad (no siempre en pipeline producto) |

Patrón API: `GET /health`, `POST /analyze-*` con `UploadFile` + `Form` (`confidence_threshold`, `include_visualization`).

---

## Stubs y placeholders

| Servicio | Puerto | Estado |
|----------|--------|--------|
| `ia-orchestrator` | 4100 | Solo `/health`, `/frontal` proxy |
| `faceid-service` | 4200 | Solo `/health` |

---

## Reglas de negocio (umbrales)

- Fuente: `NewFeature.md` + `common/newfeature_references.py`
- Auditoría: `ThresholdValidationLogger` → `analysis_logs/`
- Estado integración: `docs/INTEGRATION_STATUS.md` (repo AI-MODELS)

---

## Referencias

- Gateway: [04-api-gateway-overview.md](./04-api-gateway-overview.md)
- Catálogo gateway: `API-GATAWAY/docs/03-ai-services-catalog.md`
- Espejo / ADRs: `SOUL-GATE-AI-MODELS/docs/architecture.md`
