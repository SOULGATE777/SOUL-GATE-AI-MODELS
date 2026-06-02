# 09 — Autenticación y seguridad

**Alcance:** API de negocio, gateway, ML, frontend  
**Última actualización:** 2026-05-21

---

## API de negocio — autenticación

### JWT + sesión Redis

| Componente | Ubicación |
|------------|-----------|
| Middleware | `api/src/config/middleware/auth.ts` |
| Validación | `Authorization: Bearer <token>` |
| Sesión | Query/body `sessionId` validado en Redis |
| Usuario en request | `req.user` tras validación |

**`optionalAuthMiddleware`:** rutas que aceptan usuario anónimo o autenticado.

### OAuth social

| Proveedor | Ruta |
|-----------|------|
| Google | `POST /auth/google-login` |
| Facebook | `POST /auth/facebook-login`, callback |
| Apple | `POST /auth/apple-login` |

Campos en `User`: `googleId`, `facebookId`, `appleId`, `pictureUrl`.

### Multi-contexto empresa

- Header: **`X-Active-Context`**
- Middleware: `contextMiddleware`
- Filtra perfiles y operaciones por empresa activa.

### Admin

- `adminMiddleware`, `superAdminMiddleware`, `permissionMiddleware`
- Rutas bajo `/api/v1/admin/*` tras `authMiddleware`.

### Sesiones dispositivo

- Modelos: `DeviceSession`, `SessionToken`
- Rutas: `/api/v1/sessions/*`
- Doc: `api/docs/session-management.md`, `mobile-single-login-documentation.md`

---

## Rutas públicas (sin JWT)

| Ruta | Motivo |
|------|--------|
| `/api/v1/free-analysis/*` | Trial / funnel |
| `POST /api/v1/casting-applications` | Formulario casting |
| `GET /card/*`, OG audio share | Tarjetas públicas |
| `GET /health` | Health check |
| Webhooks Stripe | Firma Stripe, no JWT |

---

## API Gateway — seguridad

| Tema | Estado actual |
|------|---------------|
| Autenticación en gateway | **No** — confía en red interna / quien expone 4051 |
| CORS | Habilitado en servicios |
| Secretos | `GEMINI_API_KEY` en env gateway |
| Imágenes biométricas | Tránsito HTTP; retención en API negocio/S3 |

**Recomendación producción:** TLS terminación, red privada Docker, no exponer 800x públicamente.

---

## Microservicios ML

- Sin auth en `/health` y `/analyze-*` típicamente.
- Riesgo: exposición de puertos en EC2 — restringir security groups.
- No almacenar PII en logs de validación más allá de metadatos de análisis.

---

## Datos sensibles

| Dato | Tratamiento |
|------|-------------|
| Fotos faciales | S3 / almacenamiento API; política privacidad web |
| Tokens JWT | Cliente `localStorage` / secure store móvil |
| Stripe | PCI delegado a Stripe |
| Logs Gemini | `LLM_LOGS_DIR`, `llm_logs/` gateway — revisar retención |

---

## Checklist seguridad (cambios)

1. No commitear `.env`, `service-account.json`, claves Stripe.
2. Validar permisos en rutas admin y contexto empresa.
3. Webhook Stripe: verificar firma con `STRIPE_WEBHOOK_SECRET`.
4. Server Actions / uploads: validar tipo MIME y tamaño (API negocio).
5. Semgrep en superficies auth/pagos cuando aplique.

---

## Gaps documentados

Ver [SISTEMA-COMPLETO.md](./SISTEMA-COMPLETO.md) §13.5 — política formal de retención biométrica, mTLS gateway↔ML, rotación secretos.

---

## Referencias

- Usuarios y Stripe: [06-user-billing-subscriptions.md](./06-user-billing-subscriptions.md)
- `API-GATAWAY/docs/11-security-overview.md`
- `SOUL-GATE-FRONTEND-WEB/docs` (anti-abuso free analysis)
