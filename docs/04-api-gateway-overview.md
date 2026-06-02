# 04 — API Gateway (orquestación IA)

**Repositorio:** `/home/mitza/proyectos/API-GATAWAY`  
**Código principal:** `api-gateway/src/index.ts` (no existe `app.ts`)  
**Última actualización:** 2026-05-21

---

## Rol en el ecosistema

El **API Gateway** es el punto de entrada HTTP para el **análisis biométrico consolidado**: recibe fotos (multipart), orquesta llamadas a wrappers Node (**4301–4306**) o directamente a ML FastAPI (**8000–8014** según `USE_LOCAL_ML`), calcula temperamento y personalidad, y genera **narrativa Gemini**.

**No** gestiona usuarios, Stripe ni persistencia de perfiles — eso vive en `SOUL-GATE-FRONTEND-WEB/api`.

---

## Docker Compose (servicios actuales)

| Servicio Compose | Puerto | Función |
|------------------|--------|---------|
| `api-gateway` | **4051** | Orquestación principal |
| `ia-orchestrator` | 4100 | Stub health + proxy frontal |
| `faceid-service` | 4200 | Placeholder FaceID |
| `souldate-ia-frontal` | 4301 | Proxy frontal |
| `souldate-ia-perfil` | 4302 | Proxy perfil |
| `souldate-ia-cuerpo` | 4303 | Proxy cuerpo |
| `souldate-ia-palmas` | 4304 | Proxy manos |
| `souldate-ia-ojos` | 4305 | Proxy colorimetría ojos |
| `souldate-ia-temperamentos` | 4306 | Cálculo temperamento (Node) |

**No incluidos hoy:** `user-service` (4001, eliminado), PostgreSQL, `monitoring-dashboard` (3000).

**Red externa:** `age_prod_default` (api-gateway).

**Volúmenes:** `./llm_logs`, `./analysis_logs`.

---

## Endpoints del gateway

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health agregado (200/503) |
| `GET` | `/api/health/all` | Health detallado por servicio |
| `POST` | `/api/ia/process` | Proxy a orchestrator `/process` (**no implementado** en orchestrator) |
| `POST` | `/api/v1/analyze-consolidated` | **Pipeline principal** |
| `POST` | `/api/v1/validate-images` | Validación frontal + perfil |

Timeout servidor: ~**5 minutos** (análisis largo).

---

## `POST /api/v1/analyze-consolidated`

### Entrada (multipart)

| Campo | Obligatorio | Descripción |
|-------|-------------|-------------|
| `file` | Sí | Foto frontal |
| `perfil_izquierdo`, `perfil_derecho` | No | Perfiles laterales |
| `cuerpo` | No | Cuerpo (análisis corporal **deshabilitado** en gateway actual) |
| `palma` | No | Manos / palmas |
| `confidence_threshold` | No | Umbral modelos |
| `include_visualization` | No | Salidas visuales |
| `subject_sex` / `sex` | No | Sexo para temperamento |

### Fases (resumen)

1. Determinar `analysisType`: `partial_frontal` | `partial_profile` | `complete`.
2. Preprocesar perfiles si aplica (`preprocess-profile`).
3. **Paralelo** (`Promise.allSettled`): morfo frontal, antropo, validación, espejo, colorimetría ojos; por lado perfil morfo/antropo/validación; palmas si hay foto.
4. Extracción de tags, filtros, diagnósticos (CPU en gateway).
5. `POST` temperamento → **4306** `/api/v1/calculate-temperament`.
6. Scoring personalidad (`scorePersonalityTraits` / V2).
7. Narrativa → **Gemini** (`GEMINI_API_KEY`).
8. Respuesta JSON consolidada.

Detalle: `API-GATAWAY/docs/01-api-gateway-orchestration.md`.

---

## Modo `USE_LOCAL_ML`

| Valor | Destino |
|-------|---------|
| `false` (default Docker) | `http://souldate-ia-*:430x/api/v1/...` |
| `true` | `http://172.31.10.78:800x/...` (paths cortos sin `/api/v1`) |

Los wrappers **430x** suelen reenviar de todos modos al host ML AWS configurado en su código.

---

## Variables críticas

| Variable | Uso |
|----------|-----|
| `PORT` | 4051 |
| `GEMINI_API_KEY` | Narrativa personalidad |
| `USE_LOCAL_ML` | Selección Docker vs ML directo |
| Host ML | Hardcoded `172.31.10.78` en varios servicios |

---

## Consumidores

| Cliente | Cómo llama |
|---------|------------|
| API de negocio | `API_GATEWAY_URL` desde Express |
| Scripts test | `test-consolidated-curl.sh`, etc. |
| Reportes masivos | **No** — llama ML directo (800x) |

---

## Monitoreo legacy

`monitoring-dashboard/` (React, puerto 3000) **no** está en compose; apunta puertos obsoletos (4000, 4001). Preferir `GET :4051/health` o `/api/health/all`.

---

## Referencias

- Servicios IA: [05-ai-analysis-services.md](./05-ai-analysis-services.md)
- ML puertos: [07-ai-models-pipeline.md](./07-ai-models-pipeline.md)
- Endpoints: [17-api-endpoints-master.md](./17-api-endpoints-master.md)
- `API-GATAWAY/docs/04-docker-compose-reference.md`
