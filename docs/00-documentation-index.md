# Índice maestro de documentación — Ecosistema Soul Gate

**Ubicación canónica (serie 00–18):** `/home/mitza/proyectos/SOUL-GATE-AI-MODELS/docs/`  
**Última actualización:** 2026-05-21  
**Idioma:** Español

---

## Propósito

Este índice cataloga la documentación técnica del ecosistema **Soul Gate** en **seis repositorios** del workspace. La serie numerada `00–18` en este directorio es la **fuente consolidada** para arquitectura transversal; cada repo mantiene documentación especializada.

---

## Repositorios del ecosistema

| Repositorio | Ruta | Rol principal |
|-------------|------|---------------|
| **SOUL-GATE-FRONTEND-WEB** | `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB` | Web (Vite/React), API de negocio (Express/Prisma), admin, casting |
| **SOUL-GATE-APP-MOVIL** | `…/SOUL-GATE-FRONTEND-WEB/SOUL-GATE-APP-MOVIL` | App móvil Expo / React Native |
| **API-GATAWAY** | `/home/mitza/proyectos/API-GATAWAY` | Orquestación IA biométrica (gateway 4051, wrappers 4301–4306) |
| **SOUL-GATE-AI-MODELS** | `/home/mitza/proyectos/SOUL-GATE-AI-MODELS` | Microservicios ML Python/FastAPI (puertos 8000–8014) |
| **Proyecto-Reportes-Masivos** | `/home/mitza/proyectos/Proyecto-Reportes-Masivos` | Análisis batch + PDF (acceso directo a ML) |
| **TEXT-TO-VOICE** | `/home/mitza/proyectos/TEXT-TO-VOICE` | TTS Edge (puerto 5032), proxy vía API de negocio |
| **soul-cost** | `/home/mitza/proyectos/soul-cost` | Dashboard interno de unit economics (no producción) |

**Fuente externa de reglas ML:** `/home/mitza/proyectos/SOUL-GATE/NewFeature.md` (umbrales y exclusiones).

---

## Serie principal (este directorio)

| # | Archivo | Contenido |
|---|---------|-----------|
| 00 | [00-documentation-index.md](./00-documentation-index.md) | Este índice |
| 01 | [01-frontend-architecture.md](./01-frontend-architecture.md) | Stack web, Redux, servicios, build |
| 02 | [02-frontend-ux-menus.md](./02-frontend-ux-menus.md) | Rutas, navegación, flujos UX |
| 03 | [03-frontend-api-backend.md](./03-frontend-api-backend.md) | API Express, montaje de rutas, integraciones |
| 04 | [04-api-gateway-overview.md](./04-api-gateway-overview.md) | Gateway 4051, compose, orquestación |
| 05 | [05-ai-analysis-services.md](./05-ai-analysis-services.md) | Wrappers 4301–4306 y servicios ML |
| 06 | [06-user-billing-subscriptions.md](./06-user-billing-subscriptions.md) | Usuarios, Stripe, planes, créditos |
| 07 | [07-ai-models-pipeline.md](./07-ai-models-pipeline.md) | Pipeline ML Python, puertos, umbrales |
| 08 | [08-facial-analysis-reports.md](./08-facial-analysis-reports.md) | Reportes masivos y PDF |
| 09 | [09-auth-security.md](./09-auth-security.md) | JWT, sesiones Redis, seguridad |
| 10 | [10-compatibility-narrative-ia.md](./10-compatibility-narrative-ia.md) | Compatibilidad + Gemini narrativa |
| 11 | [11-mobile-app.md](./11-mobile-app.md) | Expo, API, auth móvil |
| 12 | [12-infrastructure-deployment.md](./12-infrastructure-deployment.md) | Docker, puertos, despliegue |
| 13 | [13-end-to-end-flows.md](./13-end-to-end-flows.md) | Flujos E2E producto → ML → DB |
| 14 | [14-auxiliary-services.md](./14-auxiliary-services.md) | TTS, soul-cost, monitoreo legacy |
| 15 | [15-business-domain-glossary.md](./15-business-domain-glossary.md) | Glosario de dominio unificado |
| 16 | [16-data-model.md](./16-data-model.md) | Prisma / PostgreSQL |
| 17 | [17-api-endpoints-master.md](./17-api-endpoints-master.md) | Catálogo maestro de endpoints |
| 18 | [18-customer-support-ia.md](./18-customer-support-ia.md) | Soporte con Gemini + tickets |

**Documento transversal ML:** [SISTEMA-COMPLETO.md](./SISTEMA-COMPLETO.md) — visión del repo AI-MODELS + §13 gaps.

**Entrada rápida ML:** [README.md](./README.md).

---

## Documentación por repositorio

### SOUL-GATE-FRONTEND-WEB

| Ruta | Contenido |
|------|-----------|
| `docs/00-INDICE-DOCUMENTACION-SISTEMA.md` | Índice producto (serie 01–19 planificada en web repo) |
| `docs/ARQUITECTURA_GENERAL.md` | Arquitectura ecosistema (legado) |
| `docs/SISTEMA_IA_ATENCION_CLIENTE.md` | Soporte IA (detalle) |
| `docs/flujo-analisis-sistema.md` | Pipeline de análisis |
| `api/docs/` | Sesiones, soporte, admin, function calling |
| `CREDENTIALS_AND_DEPLOYMENT_GUIDE.md` | Credenciales y despliegue |

### API-GATAWAY

| Ruta | Contenido |
|------|-----------|
| `docs/README.md` | Índice gateway |
| `docs/01-api-gateway-*.md` | Endpoints y orquestación |
| `docs/03-ai-*.md` | Catálogo servicios IA |
| `docs/07-ai-models-overview.md` | Mapa puertos ML |
| `docs/15-end-to-end-data-flows.md` | Flujos de datos |
| `docs/00-glossary-domain.md` | Glosario (copia ampliada en `15` de AI-MODELS) |

### SOUL-GATE-AI-MODELS (operaciones ML)

| Ruta | Contenido |
|------|-----------|
| `docs/project-overview.md`, `architecture.md`, `tech-stack.md` | Fundamentos |
| `docs/deployment-guide.md`, `troubleshooting.md`, `testing-guide.md` | Ops |
| `docs/VALIDATION_LOGGING_SYSTEM.md`, `INTEGRATION_STATUS.md` | Umbrales y auditoría |

### Proyecto-Reportes-Masivos

| Ruta | Contenido |
|------|-----------|
| `CLAUDE.md`, `README` implícito en scripts | `unified_analyzer.py`, puertos 800x |

### TEXT-TO-VOICE / soul-cost

| Repo | Doc clave |
|------|-----------|
| TEXT-TO-VOICE | `README`, `app/main.py` — puerto **5032** |
| soul-cost | `*.md` costos; ref. `API-GATAWAY/docs/10-soul-cost.md` |

---

## Convenciones unificadas (verificadas en código)

| Tema | Valor canónico |
|------|----------------|
| API de negocio | Express en `SOUL-GATE-FRONTEND-WEB/api`, prefijo mayoría `/api/v1` |
| Puerto API negocio (local) | `5001` (`PORT`); Docker compose API → **80** |
| API Gateway IA | **4051** (`API_GATEWAY_URL`) |
| Wrappers IA Docker | **4301–4306** |
| ML FastAPI (AWS/host) | **8000–8014** (ver `07-ai-models-pipeline.md`) |
| TTS | **5032** vía `TTS_SERVER_URL` |
| `user-service` en API-GATAWAY | **Eliminado** — usuarios en API de negocio |
| Web dev | Vite **5173** (HTTPS) |
| Admin dev | **5174** |

Ante contradicción: **código vigente** → serie `00–18` aquí → docs históricos por repo.

---

## Lectura recomendada por rol

| Rol | Orden |
|-----|-------|
| **Onboarding técnico** | 00 → 13 → 04 → 03 → 07 |
| **Frontend** | 01 → 02 → 03 → 17 |
| **Backend producto** | 03 → 06 → 16 → 09 → 10 |
| **IA / ML ops** | 07 → 05 → 04 → SISTEMA-COMPLETO |
| **DevOps** | 12 → 07 → deployment-guide (AI-MODELS) |
| **Soporte / CX** | 18 → 03 (rutas support) |
| **Producto** | 15 → 13 → 06 → 10 |

---

## Mantenimiento

1. Al crear un doc numerado nuevo, actualizar la tabla de la serie y [README.md](./README.md).
2. Cambios de puertos o contratos: actualizar **17**, **04**, **07** y `SISTEMA-COMPLETO.md` §4.
3. Umbrales ML: sincronizar con `NewFeature.md` y `common/newfeature_references.py`.

---

*Índice maestro Soul Gate — mayo 2026.*
