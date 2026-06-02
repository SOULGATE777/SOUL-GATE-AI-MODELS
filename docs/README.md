# Documentación — SOUL-GATE-AI-MODELS

**Última actualización:** 2026-05-21  
**Versión del ecosistema:** 1.0.0  
**Estado:** Producción activa

---

## Visión general

**SOUL-GATE-AI-MODELS** es un ecosistema de **microservicios ML** (Python + FastAPI + PyTorch) para análisis **frontal**, de **perfil**, **corporal** (manos) y **estimación de edad**. Cada servicio es autocontenido: Dockerfile, `docker-compose`, modelos, health check y soporte GPU NVIDIA.

La documentación del **producto completo** (web, API negocio, gateway, móvil, billing, soporte IA) vive en la **serie 00–18** y en **[SISTEMA-COMPLETO.md](./SISTEMA-COMPLETO.md)**.

### Flujo típico en producción

```
Imagen → Preprocesamiento (base64) → Servicios ML (8000–8014)
       → API Gateway :4051 → API negocio (Prisma) → Gemini (narrativa)
       → Cliente web / móvil
```

### Repositorios del ecosistema

| Repositorio | Rol |
|-------------|-----|
| [SOUL-GATE-AI-MODELS](../) (este repo) | Microservicios ML + `docs/` central |
| [SOUL-GATE-FRONTEND-WEB](file:///home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB) | Web, API Express, Prisma, Gemini producto |
| [API-GATAWAY](file:///home/mitza/proyectos/API-GATAWAY) | Gateway :4051, wrappers 4301–4306 |
| [Proyecto-Reportes-Masivos](file:///home/mitza/proyectos/Proyecto-Reportes-Masivos) | Analizador unificado + PDF |
| [TEXT-TO-VOICE](file:///home/mitza/proyectos/TEXT-TO-VOICE) | TTS Edge :5032 |
| [soul-cost](file:///home/mitza/proyectos/soul-cost) | Unit economics (interno) |
| `/home/mitza/proyectos/SOUL-GATE/NewFeature.md` | Umbrales y reglas de negocio |

---

## Índice maestro — serie 00–18

| # | Documento | Contenido |
|---|-----------|-----------|
| 00 | [00-documentation-index.md](./00-documentation-index.md) | **Índice maestro** — catálogo de docs en los 6 repos |
| 01 | [01-frontend-architecture.md](./01-frontend-architecture.md) | Arquitectura web (Vite, React, Redux) |
| 02 | [02-frontend-ux-menus.md](./02-frontend-ux-menus.md) | Menús, rutas, flujos UX |
| 03 | [03-frontend-api-backend.md](./03-frontend-api-backend.md) | API de negocio Express |
| 04 | [04-api-gateway-overview.md](./04-api-gateway-overview.md) | API Gateway Soul AI |
| 05 | [05-ai-analysis-services.md](./05-ai-analysis-services.md) | Servicios de análisis (wrappers) |
| 06 | [06-user-billing-subscriptions.md](./06-user-billing-subscriptions.md) | Usuarios, Stripe, planes, créditos |
| 07 | [07-ai-models-pipeline.md](./07-ai-models-pipeline.md) | Pipeline ML y puertos |
| 08 | [08-facial-analysis-reports.md](./08-facial-analysis-reports.md) | Reportes faciales y PDF |
| 09 | [09-auth-security.md](./09-auth-security.md) | Autenticación y seguridad |
| 10 | [10-compatibility-narrative-ia.md](./10-compatibility-narrative-ia.md) | Compatibilidad + narrativa Gemini |
| 11 | [11-mobile-app.md](./11-mobile-app.md) | App móvil Expo |
| 12 | [12-infrastructure-deployment.md](./12-infrastructure-deployment.md) | Infraestructura y despliegue |
| 13 | [13-end-to-end-flows.md](./13-end-to-end-flows.md) | Flujos de extremo a extremo |
| 14 | [14-auxiliary-services.md](./14-auxiliary-services.md) | TTS, soul-cost, monitoring legacy |
| 15 | [15-business-domain-glossary.md](./15-business-domain-glossary.md) | Glosario de dominio |
| 16 | [16-data-model.md](./16-data-model.md) | Modelo de datos Prisma |
| 17 | [17-api-endpoints-master.md](./17-api-endpoints-master.md) | Catálogo de endpoints |
| 18 | [18-customer-support-ia.md](./18-customer-support-ia.md) | Atención al cliente con IA |

---

## Documento consolidado ML

| Documento | Contenido |
|-----------|-----------|
| [SISTEMA-COMPLETO.md](./SISTEMA-COMPLETO.md) | Visión ML, arquitectura, umbrales, ops, **§13 gaps** actualizado |

---

## Fundamentos (este repositorio)

| Documento | Contenido |
|-----------|-----------|
| [project-overview.md](./project-overview.md) | Qué es el proyecto, módulos, puertos |
| [architecture.md](./architecture.md) | Microservicios, flujo, ADRs |
| [tech-stack.md](./tech-stack.md) | Python, FastAPI, PyTorch, CUDA |

---

## Operaciones

| Documento | Contenido |
|-----------|-----------|
| [deployment-guide.md](./deployment-guide.md) | Docker, GPU, AWS EC2 |
| [troubleshooting.md](./troubleshooting.md) | CUDA OOM, puertos, health |
| [testing-guide.md](./testing-guide.md) | pytest, cobertura |

---

## NewFeature.md y validación de umbrales

| Documento | Contenido |
|-----------|-----------|
| [newfeature-implementation.md](./newfeature-implementation.md) | Seguimiento de umbrales |
| [PLAN_THRESHOLD_VALIDATOR.md](./PLAN_THRESHOLD_VALIDATOR.md) | Plan `ThresholdValidator` |
| [VALIDATION_LOGGING_SYSTEM.md](./VALIDATION_LOGGING_SYSTEM.md) | `ThresholdValidationLogger` |
| [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) | Resumen logger (dic. 2025) |
| [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md) | Estado por servicio |
| [INTEGRATION_MANUAL.md](./INTEGRATION_MANUAL.md) | Guía de integración |

**Fuente externa:** [NewFeature.md](file:///home/mitza/proyectos/SOUL-GATE/NewFeature.md)

---

## Documentación en otros repos (referencia)

| Repo | Ruta destacada |
|------|----------------|
| API-GATAWAY | `docs/00-glossary-domain.md`, `docs/01-api-gateway-endpoints.md` |
| SOUL-GATE-FRONTEND-WEB | `docs/00-INDICE-DOCUMENTACION-SISTEMA.md`, `docs/SISTEMA_IA_ATENCION_CLIENTE.md` |

---

## Inicio rápido (ML)

### Requisitos

- Ubuntu 20.04+, Docker 24+, Docker Compose 2.x
- GPU NVIDIA (recomendado), NVIDIA Container Toolkit
- Modelos en cada `*/models/` (no versionados)

### Health check puertos habituales

```bash
for port in 8000 8001 8002 8008 8012 8014 8003 8004 8005 8010 8009 8013; do
  echo -n "Puerto $port: "
  curl -sf "http://localhost:${port}/health" && echo OK || echo FALLO
done
```

### Tests logger (`common/`)

```bash
cd /home/mitza/proyectos/SOUL-GATE-AI-MODELS/common
pytest tests/test_threshold_validation_logger.py -v
```

---

## Mapa de servicios ML (resumen)

| Puerto | Módulo | Servicio |
|--------|--------|----------|
| 8000 | Frontal | Morfológico |
| 8001 | Frontal | Antropométrico |
| 8002 | Frontal | Validación (YOLOv8) |
| 8008 | Frontal | Espejo |
| 8012 | Frontal | Rotación |
| 8014 | Frontal | Preprocesamiento |
| 8003–8005 | Perfil | Morfo / antropo / validación |
| 8010 | Perfil | Preprocesamiento |
| 8009 | Cuerpo | Manos |
| 8013 | Edad | Age estimation |

Detalle: [07-ai-models-pipeline.md](./07-ai-models-pipeline.md).

---

## Gaps documentales (resumen)

Ver **[SISTEMA-COMPLETO.md §13](./SISTEMA-COMPLETO.md#13-temas-pendientes-stubs)**:

| Tema | Estado |
|------|--------|
| Catálogo API completo (OpenAPI) | 🟡 [17](./17-api-endpoints-master.md) |
| Integración gateway ↔ backend | 🟡 [04](./04-api-gateway-overview.md), [13](./13-end-to-end-flows.md) |
| CI/CD automatizado | 🔲 |
| Observabilidad (Prometheus/Grafana) | 🔲 |
| Seguridad / cumplimiento biométrico | 🟡 [09](./09-auth-security.md) |
| Benchmarks latencia GPU | 🔲 |
| Cuerpo morfo/antropo | 🔲 TBD código |
| Changelog versionado ecosistema | 🔲 |

---

## Mantenimiento

1. Cambio transversal → actualizar doc **00–18** correspondiente + [SISTEMA-COMPLETO.md](./SISTEMA-COMPLETO.md).
2. Umbrales → `NewFeature.md`, `common/newfeature_references.py`, [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md).
3. Nuevo endpoint producto → [17-api-endpoints-master.md](./17-api-endpoints-master.md).

**README del repo (carpetas):** [../README.md](../README.md)
