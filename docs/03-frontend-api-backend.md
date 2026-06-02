# 03 — API de negocio (backend Express)

**Repositorio:** `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB/api`  
**Entrada:** `api/src/app.ts`  
**Última actualización:** 2026-05-21

---

## Resumen

API **Express 4** con **Prisma** (PostgreSQL), **Redis** (sesiones), **Socket.IO**, integración **Stripe**, proxy al **API Gateway** (`API_GATEWAY_URL`) y **Gemini** (OAuth2 service account + narrativas).

**Puerto por defecto:** `5001` (`PORT` en `config.ts`). En Docker (`api/docker-compose.yml`): host **80**.

---

## Montaje de rutas (`app.ts`)

| Prefijo | Archivo | Notas |
|---------|---------|-------|
| `/` | `auth.routes.ts` | `/auth/*`, `/users/*` — **sin** `/api/v1` |
| `/api/v1` | `stripe.routes.ts` | Pagos y suscripciones |
| `/api/v1/stripe` | `webhook.routes.ts` | Webhooks Stripe (body raw, antes de JSON parser) |
| `/api/v1/setting` | `setting.routes.ts` | Cuenta, email, teléfono |
| `/api/v1/redux` | `redux.routes.ts` | Persistencia estado cliente |
| `/api/v1/support` | `support.routes.ts` | FAQs, chat, tickets |
| `/api/v1/user-profiles` | `userProfile.routes.ts` | Perfiles, análisis, compatibilidad |
| `/api/v1/user` | `user.routes.ts` | Perfil usuario, facturación |
| `/api/v1/notifications` | `notification.routes.ts` | Notificaciones |
| `/api/v1/plans` | `plan.routes.ts` | Planes y límites |
| `/api/v1/admin` | `admin.routes.ts` | Tras `authMiddleware` + `adminMiddleware` |
| `/api/v1/sessions` | `session.routes.ts` | Sesiones dispositivo |
| `/api/v1/surveys` | `survey.routes.ts` | Encuestas |
| `/api/v1` | `imageValidation.routes.ts` | `/validate-images`, health validación |
| `/api/v1/free-analysis` | `freeAnalysis.routes.ts` | **Público** |
| `/api/v1/tts` | `tts.routes.ts` | Proxy TTS → `TTS_SERVER_URL` |
| `/api/v1/audio-share` | `audioShare.routes.ts` | Tarjetas audio |
| `/card` | `audioShareCard.routes.ts` | OG HTML redes sociales |
| `/api/v1/invoices` | `invoice.routes.ts` | Facturas |
| `/api/v1/analysis-pdf` | `analysisPdf.routes.ts` | PDF análisis |
| `/api/v1/compatibility-pdf` | `compatibilityPdf.routes.ts` | PDF compatibilidad |
| `/api/v1/casting-applications` | `castingApplication.routes.ts` | Casting |
| `GET /health` | — | DB + Redis |
| `GET /googleoauth2/callback` | — | OAuth YouTube admin |

**Nota:** `castingApplicant.routes.ts` existe pero **no** está montado en `app.ts`; usar `casting-applications`.

---

## Autenticación y contexto

| Middleware | Archivo | Función |
|------------|---------|---------|
| `authMiddleware` | `api/src/config/middleware/auth.ts` | JWT Bearer + `sessionId` en Redis |
| `optionalAuthMiddleware` | mismo | Rutas opcionales |
| `contextMiddleware` | `api/src/config/middleware/context.ts` | Header `X-Active-Context` (empresa) |
| `adminMiddleware` | `api/src/middlewares/admin.middleware.ts` | Rol admin / permisos |

---

## Servicios críticos

| Servicio | Ruta | Rol |
|----------|------|-----|
| `userProfile.controller` | controllers | Subida fotos, proxy gateway, persistencia `Analysis` |
| `compatibilityNarrative.service` | services | Narrativa compatibilidad Gemini |
| `support.controller` + `geminiWithFunctions.ts` | support | Chat soporte con function calling |
| `redis.service` / `session.service` | services | Sesiones y tokens |
| `notification.service` | services | Push / socket notificaciones |

---

## Variables de entorno (selección)

| Variable | Uso |
|----------|-----|
| `DATABASE_URL` | PostgreSQL Prisma |
| `JWT_SECRET` | Tokens |
| `REDIS_URL` | Sesiones |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | Pagos |
| `API_GATEWAY_URL` | IA consolidada (default `http://soul-gate-api-ai-api-gateway-1:4051`) |
| `PROFILE_ROTATION_URL` | Validación rotación perfil |
| `TTS_SERVER_URL` | ej. `http://18.220.61.87:5032` |
| `GEMINI_API_KEY` | Referencia; narrativas usan OAuth2 + `service-account.json` |
| `AWS_*`, buckets | S3 fotos y audio cards |
| `CLIENT_URL`, `FRONTEND_URL`, `APP_URL` | URLs producto |
| `LLM_LOGS_DIR` | Logs narrativa compatibilidad |

Listado ampliado: [12-infrastructure-deployment.md](./12-infrastructure-deployment.md).

---

## Socket.IO

- Mismo servidor HTTP que Express.
- Clientes web/móvil: `io(API_URL)`.
- Eventos: soporte (`ticket:*`), notificaciones, progreso análisis (según implementación).

Documentación: `api/docs/socket-session-notifications.md`, `session-management.md`.

---

## Integración IA

1. **Validación:** `imageValidation.routes` → puede llamar servicios de validación / rotación.
2. **Análisis completo:** `userProfile` → multipart al gateway `POST /api/v1/analyze-consolidated`.
3. **Narrativa producto:** generada en gateway (Gemini) o enriquecida en API de negocio para compatibilidad.

---

## Documentación adicional en repo

| Ruta | Tema |
|------|------|
| `api/docs/support-api-documentation.md` | Soporte |
| `api/docs/function-calling-architecture.md` | Gemini tools |
| `api/docs/session-management.md` | Sesiones |
| `api/docs/system-admin-endpoints.md` | Admin |
| `docs/SISTEMA_IA_ATENCION_CLIENTE.md` | Soporte IA (vista sistema) |

---

## Referencias

- Endpoints maestro: [17-api-endpoints-master.md](./17-api-endpoints-master.md)
- Modelo datos: [16-data-model.md](./16-data-model.md)
- Auth: [09-auth-security.md](./09-auth-security.md)
- Gateway: [04-api-gateway-overview.md](./04-api-gateway-overview.md)
