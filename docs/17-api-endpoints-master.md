# 17 — Catálogo maestro de endpoints API

**Última actualización:** 2026-05-21  
**Fuentes:** `SOUL-GATE-FRONTEND-WEB/api/src/app.ts`, rutas en `api/src/routes/`, `API-GATAWAY/api-gateway/src/index.ts`

---

## API de negocio (Express)

**Base:** `http(s)://<host>:<PORT>` — puerto típico **5001** (dev) o **80** detrás de proxy.  
**Prefijo principal:** `/api/v1` (excepto auth en `/` y webhooks Stripe).

### Montaje de routers (`app.ts`)

| Prefijo | Router |
|---------|--------|
| `/` | `authRoutes` |
| `/api/v1/stripe` | `webhookRouter` (raw body) |
| `/api/v1` | `stripeRouter` |
| `/api/v1/setting` | settings |
| `/api/v1/redux` | redux |
| `/api/v1/support` | soporte IA |
| `/api/v1/user-profiles` | perfiles y análisis |
| `/api/v1/user` | usuario |
| `/api/v1/notifications` | notificaciones |
| `/api/v1/plans` | planes |
| `/api/v1/admin` | admin |
| `/api/v1/sessions` | sesiones |
| `/api/v1/surveys` | encuestas |
| `/api/v1` | validación imágenes (sin subpath extra) |
| `/api/v1/free-analysis` | análisis gratuito |
| `/api/v1/tts` | proxy TTS |
| `/api/v1/audio-share` | tarjetas audio |
| `/card` | OG / share público |
| `/api/v1/invoices` | facturas |
| `/api/v1/analysis-pdf` | PDF análisis |
| `/api/v1/compatibility-pdf` | PDF compatibilidad |
| `/api/v1/casting-applications` | casting |

---

### Autenticación (`auth.routes.ts` — raíz `/`)

| Método | Ruta | Auth | Notas |
|--------|------|------|-------|
| `POST` | `/auth/pre-register` | — | Pre-registro |
| `POST` | `/auth/login` | — | Login |
| `POST` | `/auth/validate-credentials` | — | |
| `POST` | `/auth/verify-password` | JWT | |
| `POST` | `/auth/register` | JWT | |
| `PUT` | `/auth/update/:id` | — | |
| `POST` | `/auth/simple-register` | — | |
| `POST` | `/auth/reset-password` | — | |
| `GET` | `/users` | — | Por email |
| `POST` | `/auth/google-login` | — | OAuth |
| `POST` | `/auth/facebook-login` | — | |
| `POST` | `/auth/apple-login` | — | |
| `GET` | `/auth/facebook/callback` | — | |
| `GET` | `/users/managed-users` | JWT + context | Empresa |
| `POST` | `/users/managed-users` | JWT | |
| `DELETE` | `/users/managed-users/:userId` | JWT | |
| `PUT` | `/users/managed-users/:userId/permissions` | JWT | |
| `GET` | `/user/users/managed-users/get-permissions` | JWT + context | |
| `GET` | `/auth/available-contexts` | JWT | |
| `GET` | `/auth/context-profile` | JWT + context | |
| `POST` | `/developer/generate-token` | — | Dev API |
| `POST` | `/developer/validate-token` | — | |

---

### Validación de imágenes (`/api/v1`)

| Método | Ruta | Auth | Notas |
|--------|------|------|-------|
| `GET` | `/api/health/validation-service` | — | Health proxy gateway |
| `POST` | `/api/validate-images` | — | Público (free analysis) |
| `POST` | `/api/validate-images-detailed` | — | |
| `POST` | `/api/analyze-profile-rotation` | — | |
| `POST` | `/api/estimate-age` | — | |

*Nota:* montado en `app.use("/api/v1", imageValidationRoutes)` — rutas definidas sin repetir prefijo en el router.

---

### Soporte IA (`/api/v1/support`)

| Método | Ruta | Auth |
|--------|------|------|
| `GET` | `/faqs` | JWT |
| `GET` | `/user-info` | JWT |
| `POST` | `/support` | JWT |
| `POST` | `/chat` | JWT |
| `GET` | `/chat/history` | JWT |
| `POST` | `/chat/new` | JWT |
| `GET` | `/tickets` | JWT |
| `GET` | `/tickets/:id` | JWT |
| `POST` | `/tickets/message` | JWT |
| `PATCH` | `/tickets/:ticketId/toggle-ai` | JWT |

Detalle: [18-customer-support-ia.md](./18-customer-support-ia.md).

---

### TTS (`/api/v1/tts`)

| Método | Ruta | Auth |
|--------|------|------|
| `POST` | `/stream` | — |
| `GET` | `/health` | — |

Proxy → `TTS_SERVER_URL` (TEXT-TO-VOICE :5032).

---

### Perfiles y análisis

Rutas en `userProfile.routes.ts` — consultar archivo para lista completa. Incluye típicamente:

- CRUD `UserProfile`
- Subida fotos / disparo análisis
- `POST` compatibilidad entre perfiles
- Consulta `Analysis`

**Variable:** `API_GATEWAY_URL` → gateway `POST /api/v1/analyze-consolidated`.

---

### Stripe, planes, admin, encuestas, casting

Ver routers dedicados bajo `api/src/routes/`:

- `stripe.routes.ts`, `plan.routes.ts`, `admin.routes.ts`
- `survey.routes.ts`, `castingApplication.routes.ts`
- `freeAnalysis.routes.ts`, `analysisPdf.routes.ts`, `compatibilityPdf.routes.ts`

---

## API Gateway (Soul AI)

**Base:** `http://<host>:4051`  
**Código:** `API-GATAWAY/api-gateway/src/index.ts`

| Método | Ruta | Content-Type |
|--------|------|--------------|
| `GET` | `/health` | — |
| `GET` | `/api/health/all` | — |
| `POST` | `/api/ia/process` | JSON → orchestrator :4100 |
| `POST` | `/api/v1/analyze-consolidated` | `multipart/form-data` |
| `POST` | `/api/v1/validate-images` | `multipart/form-data` |

**Downstream wrappers (Docker):** 4301 frontal, 4302 perfil, 4303 cuerpo, 4304 palmas, 4305 ojos, 4306 temperamentos.  
**ML remoto:** puertos **8000–8014** según `USE_LOCAL_ML` / host EC2.

Documentación extendida: `API-GATAWAY/docs/01-api-gateway-endpoints.md`, [04-api-gateway-overview.md](./04-api-gateway-overview.md).

---

## Microservicios ML (directo)

| Puerto | Servicio | Health |
|--------|----------|--------|
| 8000 | Morfo frontal | `/health` |
| 8001 | Antropo frontal | `/health` |
| 8002 | Validación frontal | `/health` |
| 8008 | Espejo | `/health` |
| 8012 | Rotación frontal | `/health` |
| 8014 | Preproceso frontal | `/health` |
| 8003–8010 | Perfil (análogo) | `/health` |
| 8009 | Manos | `/health` |
| 8013 | Edad | `/health` |

Endpoints por módulo: README en cada carpeta bajo `SOUL-GATE-AI-MODELS/*/`.

---

## TEXT-TO-VOICE

| Puerto | Rutas |
|--------|-------|
| 5032 | `GET /health`, `GET /voices`, `POST /tts`, `POST /tts/stream` |

---

## Gaps de este catálogo

- **No exhaustivo** en rutas `admin`, `stripe`, `user-profiles` (decenas de handlers) — derivar con `rg 'router\\.(get|post)' api/src/routes`.
- **OpenAPI** formal no generado en repo.
- Contratos request/response campo a campo: ver controladores y tests `test-*.sh` en API-GATAWAY.

---

## Referencias

- [13-end-to-end-flows.md](./13-end-to-end-flows.md)
- [03-frontend-api-backend.md](./03-frontend-api-backend.md)
