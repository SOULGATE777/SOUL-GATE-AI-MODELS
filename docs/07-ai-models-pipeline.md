# 07 — Pipeline de modelos IA (SOUL-GATE-AI-MODELS)

**Repositorio:** `/home/mitza/proyectos/SOUL-GATE-AI-MODELS`  
**Última actualización:** 2026-05-21

---

## Propósito

~15 microservicios **Python 3.9+ / FastAPI / PyTorch** que exponen inferencia ML por HTTP. Cada servicio es autocontenido (Dockerfile, `docker-compose`, modelos en `models/`, `/health`).

---

## Estructura del repositorio

```
SOUL-GATE-AI-MODELS/
├── common/              # ThresholdValidationLogger, newfeature_references
├── frontal_prod/        # 8000, 8001, 8002, 8008, 8012, 8014
├── profile_prod/        # 8003, 8004, 8005, 8010
├── body_prod/           # 8009 (+ morfo/antropo TBD)
├── age_prod/            # 8013
├── scripts/
└── docs/
```

---

## Inventario de puertos

| Puerto | Carpeta | Función |
|--------|---------|---------|
| 8000 | `frontal_prod/morfologico` | Tags morfológicos frontal |
| 8001 | `frontal_prod/antropometrico` | Antropometría, landmarks, colorimetría ojos |
| 8002 | `frontal_prod/validacion` | YOLOv8 validación |
| 8008 | `frontal_prod/espejo` | Espejo + árbol decisión |
| 8012 | `frontal_prod/rotacion` | Viabilidad rotación frontal |
| 8014 | `frontal_prod/preprocesamiento` | MediaPipe → base64 |
| 8003 | `profile_prod/morfologico` | Morfo perfil |
| 8004 | `profile_prod/antropometrico` | Antropo perfil |
| 8005 | `profile_prod/validacion` | Validación perfil |
| 8010 | `profile_prod/preprocesamiento` | Preproceso perfil |
| 8009 | `body_prod/manos` | Manos dorso/palma |
| 8013 | `age_prod` | InsightFace edad |

---

## Patrón de API (FastAPI)

```python
GET  /health
POST /analyze-<dominio>   # multipart: file + Form(confidence_threshold, include_visualization)
```

- CORS abierto para integración backend.
- Carga lazy de modelos al arranque o primera petición.
- Salida: JSON estructurado + opcional visualización.

---

## Pipeline lógico (producto)

```
Imagen
  → [8014|8010] Preprocesamiento (base64 opcional)
  → Paralelo: morfo, antropo, validación, espejo, ojos, perfil, manos
  → Gateway: tags + temperamento (4306) + personalidad + Gemini
  → API negocio: persistencia Analysis
```

---

## NewFeature.md y umbrales

| Módulo | Regla destacada |
|--------|-----------------|
| Espejo | 18% general; Venus Corazón ≥40%; Plutón ≥7% |
| Morfo frontal | Umbrales por categoría de rasgo |
| Validación | Exclusión tercios (objeto/cabello) |
| Cuerpo | Resultados **excluidos** de producto (L671) |

**Auditoría:** `common/threshold_validation_logger.py` → `analysis_logs/{fecha}/{uuid}/validation_log.md`.

**Estado integración logger (dic. 2025):** Espejo, morfo/antropo frontal y perfil ✅; body omitido; validación N/A.

---

## Stack

| Componente | Versión / nota |
|------------|----------------|
| Python | 3.9+ |
| FastAPI | 0.104.x |
| PyTorch | 2.0–2.1 + CUDA 11.8/12.1 |
| Contenedor | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |
| Tests | pytest en `common/` y por servicio |

Detalle: [tech-stack.md](./tech-stack.md), [deployment-guide.md](./deployment-guide.md).

---

## Operaciones

```bash
cd frontal_prod/espejo
docker-compose build && docker-compose up -d
curl -s http://localhost:8008/health
```

Health masivo: ver [README.md](./README.md) o [SISTEMA-COMPLETO.md](./SISTEMA-COMPLETO.md) §9.

---

## Integración con gateway

| Modo | Quién llama puerto 800x |
|------|-------------------------|
| Producción típica | Wrapper 430x → host ML (`172.31.10.78`) |
| `USE_LOCAL_ML=true` | Gateway directo a 800x |
| Reportes masivos | Cliente Python → `13.58.240.149:800x` (batch) |

---

## Documentos especializados (este repo)

| Doc | Tema |
|-----|------|
| [SISTEMA-COMPLETO.md](./SISTEMA-COMPLETO.md) | Consolidado ML |
| [architecture.md](./architecture.md) | ADRs, Espejo |
| [VALIDATION_LOGGING_SYSTEM.md](./VALIDATION_LOGGING_SYSTEM.md) | Logger |
| [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md) | Integración por servicio |

---

## Referencias

- Gateway: [04-api-gateway-overview.md](./04-api-gateway-overview.md)
- Servicios wrapper: [05-ai-analysis-services.md](./05-ai-analysis-services.md)
- Reportes: [08-facial-analysis-reports.md](./08-facial-analysis-reports.md)
