# SOUL-GATE-AI-MODELS — Sistema completo

**Última actualización:** 2026-05-21  
**Consolidado desde:** documentos en `/docs` (dic. 2025 – may. 2026)  
**Índice navegable:** [README.md](./README.md)

---

## Tabla de contenidos

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Propósito y alcance](#2-propósito-y-alcance)
3. [Arquitectura](#3-arquitectura)
4. [Inventario de servicios](#4-inventario-de-servicios)
5. [Stack tecnológico](#5-stack-tecnológico)
6. [Flujo de datos e integración](#6-flujo-de-datos-e-integración)
7. [NewFeature.md y umbrales de negocio](#7-newfeaturemd-y-umbrales-de-negocio)
8. [Sistema de logging de validación de umbrales](#8-sistema-de-logging-de-validación-de-umbrales)
9. [Despliegue y operaciones](#9-despliegue-y-operaciones)
10. [Pruebas](#10-pruebas)
11. [Resolución de problemas](#11-resolución-de-problemas)
12. [Estado del proyecto y roadmap](#12-estado-del-proyecto-y-roadmap)
13. [Temas pendientes (stubs)](#13-temas-pendientes-stubs)
14. [Referencias cruzadas](#14-referencias-cruzadas)

---

## 1. Resumen ejecutivo

SOUL-GATE-AI-MODELS agrupa **~15 microservicios ML** independientes que analizan rostro frontal, perfil, manos y edad estimada. Cada servicio expone una API REST (FastAPI), corre en Docker con soporte CUDA y devuelve JSON estructurado para que el **backend Soul Gate** (y el API Gateway) consoliden resultados, apliquen reglas de `NewFeature.md` y generen narrativas con Gemini.

**Principios de diseño:**

- Un servicio = un propósito ML acotado
- Sin estado compartido entre servicios (solo modelos en disco)
- Preprocesadores devuelven **base64** para encadenar pipelines sin almacenamiento compartido
- Health checks en todos los servicios (`/health`)
- Trazabilidad de umbrales vía `ThresholdValidationLogger` (auditoría vs. `NewFeature.md`)

---

## 2. Propósito y alcance

### Qué hace el ecosistema

| Dominio | Capacidades |
|---------|-------------|
| **Frontal** | Morfología (tags), antropometría, validación YOLO, espejo/personalidad, rotación, preprocesamiento |
| **Perfil** | Morfología, antropometría, validación, preprocesamiento, rotación |
| **Cuerpo** | Manos (dorso/palma, colorimetría) |
| **Edad** | Estimación con InsightFace (~±3–5 años) |

### Qué queda fuera de este repo

- Persistencia de usuarios, pagos, suscripciones → **user-service** / backend en SOUL-GATE-FRONTEND-WEB
- Orquestación HTTP entre servicios en producción → **API-GATAWAY**
- Generación de narrativa en lenguaje natural → backend + **Gemini**
- Reportes PDF masivos → **Proyecto-Reportes-Masivos**

### Estructura de directorios

```
SOUL-GATE-AI-MODELS/
├── common/                 # Logger de umbrales, referencias NewFeature, tests
├── frontal_prod/           # 6 servicios (8000–8002, 8008, 8012, 8014)
├── profile_prod/           # 5 servicios (8003–8005, 8010, rotación)
├── body_prod/              # manos (8009); morfo/antropo TBD
├── age_prod/               # edad (8013)
├── scripts/                # utilidades de despliegue/integración
└── docs/                   # esta documentación
```

---

## 3. Arquitectura

### Estilo

**Microservicios** — despliegue, escalado y fallos aislados por servicio.

### Estructura estándar de un servicio

```
{servicio}/
├── Dockerfile
├── docker-compose.yml      # GPU: nvidia runtime
├── requirements.txt
├── README.md
├── app/
│   ├── main.py             # FastAPI
│   ├── models/*_pipeline.py
│   └── utils/
├── models/                 # pesos .pth, .pt, .dat (no en Git)
└── results/                # salidas opcionales
```

### ADRs registrados

| ID | Decisión | Razón |
|----|----------|-------|
| ADR-001 | Microservicios vs monolito | Escalado independiente, aislamiento GPU y de fallos |
| ADR-002 | Base64 desde preprocesadores | Pipeline sin filesystem compartido, menos I/O |
| ADR-003 | FastAPI vs Flask | Async, OpenAPI, validación con tipos, rendimiento |

### Módulo Espejo (árbol de decisión)

```
Imagen facial
  → landmarks (dlib 68) + detección (Faster R-CNN 13) + clasificador CNN
  → generación espejo izquierda/derecha
  → regiones FRENTE (7) y rostro_menton (8)
  → árbol de decisión con umbrales NewFeature.md
  → posible split por proporción facial
  → diagnósticos de personalidad
```

**Umbrales clave (NewFeature.md):**

- General frente/rostro: **18%** mínimo para diagnóstico certero
- Excepciones rostro: **Venus Corazón ≥ 40%**, **Plutón Hexagonal ≥ 7%**

### Arquitectura de despliegue

```
AWS EC2 (instancias GPU)
  → contenedores Docker + NVIDIA Container Toolkit
  → health checks cada ~30s, restart unless-stopped
  → logs por servicio (docker-compose logs)
```

Detalle: [architecture.md](./architecture.md), [deployment-guide.md](./deployment-guide.md).

---

## 4. Inventario de servicios

### Frontal (`frontal_prod/`)

| Puerto | Servicio | Modelo / técnica | Notas |
|--------|----------|------------------|-------|
| 8000 | Morfológico | Ensemble Faster R-CNN + CNNs (~45 tags) | Umbrales por categoría (NewFeature L119–270) |
| 8001 | Antropométrico | dlib 68 + 13 puntos custom | Sin umbral de confidence; valor medido |
| 8002 | Validación | YOLOv8 (~17 features) | Exclusiones tercios (objeto/cabello) |
| 8008 | Espejo | dlib + Faster R-CNN + CNN + árbol | Logger integrado |
| 8012 | Rotación | EfficientNet-B0 | Viabilidad de rotación |
| 8014 | Preprocesamiento | MediaPipe | Detección + alineación → base64 |

### Perfil (`profile_prod/`)

| Puerto | Servicio | Notas |
|--------|----------|-------|
| 8003 | Morfológico | Ensemble perfil |
| 8004 | Antropométrico | Mediciones perfil |
| 8005 | Validación | YOLOv8 perfil |
| 8010 | Preprocesamiento | Faster R-CNN + rotación |
| — | Rotación | EfficientNet (sin puerto fijo en overview) |

### Cuerpo (`body_prod/`)

| Puerto | Servicio | Estado |
|--------|----------|--------|
| 8009 | Manos | ResNet50 + K-means colorimetría |
| — | Morfológico | **TBD** |
| — | Antropométrico | **TBD** |

NewFeature.md indica **excluir resultados de cuerpo** de momento (L671).

### Edad (`age_prod/`)

| Puerto | Servicio | Modelo |
|--------|----------|--------|
| 8013 | Age estimation | InsightFace + ONNX |

### Comandos de despliegue por servicio

Patrón universal:

```bash
cd <ruta_servicio>
docker-compose build && docker-compose up -d
curl -s http://localhost:<PUERTO>/health
```

Tabla de rutas: [deployment-guide.md](./deployment-guide.md).

---

## 5. Stack tecnológico

| Capa | Tecnología |
|------|------------|
| Lenguaje | Python 3.9+ |
| API | FastAPI 0.104.1, uvicorn 0.24.0 |
| ML | PyTorch 2.0.1–2.1.0, torchvision, CUDA 11.8 / 12.1 |
| Visión | OpenCV 4.8.1, Pillow 10.x, numpy 1.24.3 |
| Especializado | dlib, ultralytics (YOLOv8), insightface, timm, mediapipe, scikit-learn |
| Tests | pytest 7.4.3, pytest-asyncio, pytest-cov |
| Contenedores | Docker 24.x, imagen base `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |

**GPU:** RTX 20xx+, VRAM ≥ 4 GB (8 GB recomendado), NVIDIA Container Toolkit.

Detalle: [tech-stack.md](./tech-stack.md).

---

## 6. Flujo de datos e integración

### Pipeline lógico

```
1. Imagen de entrada
2. [Preprocesamiento] → rostro/cuerpo recortado (base64)
3. [Análisis paralelo o secuencial] → JSON por servicio
4. [Backend / API Gateway] → consolidación + reglas NewFeature
5. [Gemini] → narrativa legible
6. [DB] → persistencia del análisis
```

### Comunicación entre servicios

- **Protocolo:** HTTP REST
- **Formato:** JSON; uploads multipart para imágenes
- **CORS:** habilitado para integración con backend
- **Async:** operaciones I/O en FastAPI

### Integración con el resto de Soul Gate

| Componente | Función |
|------------|---------|
| API-GATAWAY | Enruta análisis consolidado, FaceID, servicios 4301–4305 |
| SOUL-GATE-FRONTEND-WEB `api/` | Reglas de negocio, narrativas, compatibilidad |
| Proyecto-Reportes-Masivos | Orquesta microservicios Python legacy (SG/) y genera PDF |

> **Gap:** ver §13.2 — falta un documento único de contratos HTTP entre gateway y estos puertos.

---

## 7. NewFeature.md y umbrales de negocio

**Fuente:** `/home/mitza/proyectos/SOUL-GATE/NewFeature.md`

### Resumen por módulo

| Módulo | Líneas (aprox.) | Regla principal |
|--------|-----------------|-----------------|
| Espejo | 13–73 | 18% general; Venus 40%, Plutón 7% |
| Frontal antropométrico | 75–118 | Tomar diagnóstico tal cual (sin % confidence) |
| Frontal morfológico | 119–270 | Umbrales por categoría (cejas 50%, ojos variables, etc.) |
| Perfil morfológico | 271–373 | Mayor certeza / umbrales por rasgo |
| Perfil antropométrico | 375–503 | Coincidencia bilateral en varios rasgos |
| Validación | ~83 | Omite tercios si objeto/cabello tapa frente |
| Cuerpo | ~671 | Excluir resultados por ahora |

### Espejo — estado de código vs documento

El tracking histórico en [newfeature-implementation.md](./newfeature-implementation.md) (2025-12-12) marcaba umbrales Espejo como pendientes. Paralelamente:

- [PLAN_THRESHOLD_VALIDATOR.md](./PLAN_THRESHOLD_VALIDATOR.md) describe un **`ThresholdValidator` centralizado** (plan 2025-12-12).
- La implementación operativa actual usa **`ThresholdValidationLogger`** + reglas en pipelines (dic. 2025).

**Acción recomendada:** verificar en código los valores en `espejo_pipeline.py` y comparar con `validation_log.md` tras un análisis de prueba.

### Validador centralizado (plan)

`ThresholdValidator` en `common/` unificaría umbrales de todos los módulos (single source of truth). Beneficios: un solo archivo de configuración, tests centralizados, auditoría. Estado del plan: ver [PLAN_THRESHOLD_VALIDATOR.md](./PLAN_THRESHOLD_VALIDATOR.md) (puede coexistir con el logger ya desplegado).

---

## 8. Sistema de logging de validación de umbrales

### Objetivo

Generar por cada análisis un informe **Markdown** (y opcionalmente JSON) con **cada** comparación valor vs umbral, decisión (APROBADO / RECHAZADO / OMITIDO) y cita textual de `NewFeature.md`.

### Componentes (`common/`)

| Archivo | Rol |
|---------|-----|
| `threshold_validation_logger.py` | Clase `ThresholdValidationLogger`, export MD/JSON |
| `newfeature_references.py` | Mapeo diagnóstico → líneas y texto NewFeature |
| `tests/test_threshold_validation_logger.py` | Tests unitarios |

### Salida en disco

```
analysis_logs/{YYYY-MM-DD}/{uuid}_{timestamp}/
├── validation_log.md    # informe principal
├── validation_log.json  # opcional
└── … otros artefactos del servicio
```

### Patrón de integración en pipeline

1. En `main.py`: crear logger, `start_analysis()`, pasar a analyzer.
2. En pipeline: `start_category()` → `log_validation()` por predicción → `set_final_diagnoses()` → `end_category()`.
3. Al final: `save_markdown()` / `save_json()`.

Guía detallada: [VALIDATION_LOGGING_SYSTEM.md](./VALIDATION_LOGGING_SYSTEM.md), [INTEGRATION_MANUAL.md](./INTEGRATION_MANUAL.md).

### Estado de integración por servicio (dic. 2025)

Según [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md):

| Servicio | Estado |
|----------|--------|
| Espejo | ✅ |
| Frontal morfológico | ✅ |
| Frontal antropométrico | ✅ |
| Perfil morfológico | ✅ |
| Perfil antropométrico | ✅ |
| Body | ⏭ Excluido (NewFeature) |
| Validación | ⏭ No aplica (solo exclusiones) |

> **Nota:** [INTEGRATION_MANUAL.md](./INTEGRATION_MANUAL.md) refleja un estado intermedio (integraciones al ~5%). Para decisiones operativas, priorizar **INTEGRATION_STATUS** e **IMPLEMENTATION_SUMMARY**.

### QA con logs

```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8008/<endpoint-analyze>
cat ./analysis_logs/$(date +%Y-%m-%d)/*/validation_log.md
```

Comparar lado a lado con `NewFeature.md`.

---

## 9. Despliegue y operaciones

### Prerrequisitos

- Ubuntu 20.04+, Docker, Compose, GPU NVIDIA, drivers, Container Toolkit
- Modelos descargados a cada `models/` (S3/storage interno — no en Git)
- Permisos: `chmod 644 models/*.pth`

### Variables de entorno típicas

```bash
CUDA_VISIBLE_DEVICES=0   # o -1 para CPU
PYTHONPATH=/app
```

### Health check masivo

```bash
for port in 8000 8001 8002 8008 8012 8014 8003 8004 8005 8010 8009 8013; do
  curl -sf "http://localhost:${port}/health" || echo "FALLO $port"
done
```

### Producción AWS

- Tipo de instancia: **p3.2xlarge** o superior
- Storage: ≥ 100 GB para pesos
- Security group: puertos 8000–8014 según necesidad
- Script de setup: ver [deployment-guide.md](./deployment-guide.md)

### Monitoreo actual

- `docker stats`, `nvidia-smi`, logs por contenedor
- **Futuro:** Prometheus + Grafana, Sentry, ELK (ver §13.4)

### Backup

```bash
tar -czf models_backup_$(date +%Y%m%d).tar.gz */models/
```

---

## 10. Pruebas

### Framework

- **pytest** + **pytest-asyncio**
- Cobertura objetivo: **80%+** en código nuevo (`pytest-cov`)

### Comandos

```bash
cd <servicio>
pytest -v
pytest --cov=app --cov-report=html
pytest -m "not slow"
```

### Tests críticos

1. `GET /health` → 200, `status: healthy`
2. Carga de modelo al iniciar pipeline
3. Inferencia mínima con imagen dummy/fixture
4. `common/tests/test_threshold_validation_logger.py` para auditoría de umbrales

Guía: [testing-guide.md](./testing-guide.md).

---

## 11. Resolución de problemas

| Problema | Acciones rápidas |
|----------|------------------|
| Servicio no arranca | `docker-compose logs -f`, verificar `models/`, rebuild `--no-cache` |
| CUDA OOM | `CUDA_VISIBLE_DEVICES=-1`, reducir batch, `torch.cuda.empty_cache()` |
| Puerto en uso | `lsof -i :PORT`, matar PID o cambiar mapping en compose |
| Modelo no encontrado | Verificar ruta `models/`, permisos, descarga desde storage |
| Health unhealthy | `curl localhost:PORT/health` dentro del contenedor |
| GPU no detectada | `nvidia-smi`, toolkit, bloque `deploy.resources` en compose |
| Inferencia lenta | Confirmar `cuda` en logs, `model.eval()` |

Detalle: [troubleshooting.md](./troubleshooting.md).

---

## 12. Estado del proyecto y roadmap

### Completado ✅

- ~15 servicios ML operativos con Docker + GPU
- Health checks y READMEs por servicio
- Infraestructura `ThresholdValidationLogger` + integración en 5 servicios de análisis (dic. 2025)
- Reglas Cursor en `.cursor/rules/`
- Documentación en `/docs`

### En progreso ⏳

- Cobertura pytest ampliada en todos los servicios
- Sincronización total de umbrales Espejo vs `NewFeature.md` (verificar con logs)
- `ThresholdValidator` centralizado (si se adopta el plan)

### Pendiente 📋

- Monitoring (Prometheus/Grafana)
- CI/CD automatizado
- API Gateway consolidado en este repo (hoy vive en API-GATAWAY)
- Benchmarks de rendimiento
- Servicios body morfo/antropo
- Dashboard web para logs de validación

---

## 13. Temas pendientes (stubs)

Estado tras la serie documental **00–18** (may. 2026). Los ítems cubiertos enlazan al doc dedicado; lo que sigue 🔲 requiere trabajo operativo o OpenAPI formal.

### 13.1 Catálogo de API REST

**Estado:** 🟡 Parcial — [17-api-endpoints-master.md](./17-api-endpoints-master.md)

**Cubierto:** montaje Express, gateway `:4051`, soporte, validación, TTS, tabla ML por puerto.

**Pendiente:** OpenAPI, schemas request/response campo a campo, `curl` por cada handler admin/stripe, versionado.

**Referencias ML:** README por servicio en `frontal_prod/*/README.md`.

---

### 13.2 Integración API Gateway y backend

**Estado:** 🟡 Parcial — [04-api-gateway-overview.md](./04-api-gateway-overview.md), [03-frontend-api-backend.md](./03-frontend-api-backend.md), [13-end-to-end-flows.md](./13-end-to-end-flows.md)

**Cubierto:** `API_GATEWAY_URL`, `analyze-consolidated`, wrappers 4301–4306, ML 8000–8014, `USE_LOCAL_ML`.

**Pendiente:** circuit breaker documentado, matriz timeouts/reintentos formal, contrato JSON consolidado exportado.

**Repos:** `API-GATAWAY`, `SOUL-GATE-FRONTEND-WEB/api`

---

### 13.3 CI/CD y release

**Estado:** 🔲 Pendiente

**Debe incluir:** pipeline matriz por servicio, tags Docker, gates post-deploy.

**Borrador:** [deployment-guide.md](./deployment-guide.md), [12-infrastructure-deployment.md](./12-infrastructure-deployment.md).

---

### 13.4 Observabilidad y alertas

**Estado:** 🔲 Pendiente

**Parcial:** health agregado gateway `/health`, `/api/health/all`; dashboard legacy :3000 obsoleto — ver [14-auxiliary-services.md](./14-auxiliary-services.md).

**Pendiente:** Prometheus/Grafana, alertas OOM, Sentry en ML.

---

### 13.5 Seguridad y cumplimiento

**Estado:** 🟡 Parcial — [09-auth-security.md](./09-auth-security.md)

**Cubierto:** JWT, contexto empresa, boundary gateway, Stripe webhook.

**Pendiente:** política retención imágenes biométricas, hardening EC2/TLS, auditoría CORS producción.

---

### 13.6 Benchmarks y capacidad

**Estado:** 🔲 Pendiente

**Referencia económica:** [soul-cost](file:///home/mitza/proyectos/soul-cost) (no benchmarks de latencia ML).

**Pendiente:** p95 por servicio, VRAM, guía instancias GPU.

---

### 13.7 Módulo cuerpo (morfológico / antropométrico)

**Estado:** 🔲 Pendiente — **TBD en código**

**Parcial:** manos :8009 — [05-ai-analysis-services.md](./05-ai-analysis-services.md), [07-ai-models-pipeline.md](./07-ai-models-pipeline.md). Cuerpo deshabilitado en consolidado gateway.

**Pendiente:** cuando NewFeature habilite morfo/antropo corporal.

---

### 13.8 Changelog y versionado

**Estado:** 🔲 Pendiente

**Pendiente:** CHANGELOG.md ecosistema, alineación releases gateway/backend/ML, vínculo commits `newfeature_references.py`.

---

## 14. Referencias cruzadas

### Serie documental 00–18 (ecosistema completo)

| Archivo | Uso principal |
|---------|----------------|
| [00-documentation-index.md](./00-documentation-index.md) | Índice maestro 6 repos |
| [01-frontend-architecture.md](./01-frontend-architecture.md) | Web Vite/React |
| [02-frontend-ux-menus.md](./02-frontend-ux-menus.md) | UX y rutas |
| [03-frontend-api-backend.md](./03-frontend-api-backend.md) | API negocio |
| [04-api-gateway-overview.md](./04-api-gateway-overview.md) | Gateway :4051 |
| [05-ai-analysis-services.md](./05-ai-analysis-services.md) | Wrappers IA |
| [06-user-billing-subscriptions.md](./06-user-billing-subscriptions.md) | Stripe/planes |
| [07-ai-models-pipeline.md](./07-ai-models-pipeline.md) | Pipeline ML |
| [08-facial-analysis-reports.md](./08-facial-analysis-reports.md) | Reportes/PDF |
| [09-auth-security.md](./09-auth-security.md) | Auth y seguridad |
| [10-compatibility-narrative-ia.md](./10-compatibility-narrative-ia.md) | Compatibilidad Gemini |
| [11-mobile-app.md](./11-mobile-app.md) | Expo móvil |
| [12-infrastructure-deployment.md](./12-infrastructure-deployment.md) | Infra y deploy |
| [13-end-to-end-flows.md](./13-end-to-end-flows.md) | Flujos E2E |
| [14-auxiliary-services.md](./14-auxiliary-services.md) | TTS, soul-cost |
| [15-business-domain-glossary.md](./15-business-domain-glossary.md) | Glosario |
| [16-data-model.md](./16-data-model.md) | Prisma |
| [17-api-endpoints-master.md](./17-api-endpoints-master.md) | Endpoints |
| [18-customer-support-ia.md](./18-customer-support-ia.md) | Soporte IA |

### Documentos en `/docs` (ML y operaciones)

| Archivo | Uso principal |
|---------|----------------|
| [README.md](./README.md) | Índice en español |
| [project-overview.md](./project-overview.md) | Entrada rápida |
| [architecture.md](./architecture.md) | ADRs, Espejo |
| [tech-stack.md](./tech-stack.md) | Dependencias |
| [deployment-guide.md](./deployment-guide.md) | Ops Docker/AWS |
| [testing-guide.md](./testing-guide.md) | pytest |
| [troubleshooting.md](./troubleshooting.md) | Incidencias |
| [newfeature-implementation.md](./newfeature-implementation.md) | Tracking umbrales |
| [PLAN_THRESHOLD_VALIDATOR.md](./PLAN_THRESHOLD_VALIDATOR.md) | Plan validador central |
| [VALIDATION_LOGGING_SYSTEM.md](./VALIDATION_LOGGING_SYSTEM.md) | Uso del logger |
| [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) | Resumen implementación logger |
| [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md) | Estado integraciones |
| [INTEGRATION_MANUAL.md](./INTEGRATION_MANUAL.md) | Patrón integración manual |

### Código compartido relevante

```
common/threshold_validation_logger.py
common/newfeature_references.py
common/threshold_validator.py      # si existe según plan
common/threshold_config.py
scripts/integrate_validation_logger.py
```

### Reglas Cursor

- `.cursor/rules/99-documentation-practices.mdc` — obligación de consultar/actualizar docs
- `.cursor/rules/frontal-specific.mdc` — convenciones módulo frontal (si aplica)

---

**Mantenimiento:** al añadir servicios, puertos o reglas de `NewFeature.md`, actualizar §4, §7, §8 y el [README.md](./README.md). Para cambios arquitectónicos mayores, actualizar también [architecture.md](./architecture.md) y la fecha de este documento.
