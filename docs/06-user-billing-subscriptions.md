# 06 — Usuarios, facturación y suscripciones

**Repositorio canónico:** `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB/api`  
**Persistencia:** PostgreSQL + Prisma  
**Pagos:** Stripe  
**Última actualización:** 2026-05-21

---

## Nota histórica API-GATAWAY

El microservicio **`user-service` (puerto 4001)** fue **eliminado** del árbol de `API-GATAWAY`. Toda la lógica de usuarios, planes y Stripe vive en la **API de negocio** del monorepo frontend-web.

Documentación histórica: `API-GATAWAY/docs/02-user-service-*.md`.

---

## Tipos de usuario (`UserType`)

| Tipo | Descripción |
|------|-------------|
| `INDIVIDUAL` | Usuario final estándar |
| `COMPANY` | Cuenta empresa con usuarios gestionados |
| `MANAGED` | Usuario bajo una empresa (`parentCompanyId`, `role`, `permissions`) |
| `SYSTEM_ADMIN` | Administración de sistema |

**Roles:** `ADMIN`, `USER`, `SUPER_ADMIN` (usuarios gestionados / admin).

---

## Autenticación (resumen)

- Email/password, Google, Facebook, Apple (`auth.routes.ts`).
- JWT en `Authorization: Bearer`.
- Sesiones dispositivo en **Redis** (`DeviceSession`, `SessionToken`).
- Contexto multi-empresa: header `X-Active-Context`.

Detalle: [09-auth-security.md](./09-auth-security.md).

---

## Stripe

| Concepto | Implementación |
|----------|----------------|
| Cliente | `User.stripeCustomerId` |
| Suscripciones | `Subscription` + `Plan.stripePriceId` |
| Pagos únicos | `Payment` |
| Créditos comprados | `CreditInvoice` |
| Webhooks | `webhook.routes.ts` bajo `/api/v1/stripe` (raw body) |
| Reembolsos | `Refund` |

**Variables:** `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_FREE_PLAN_PRICE_ID`.

Documentación ampliada: `SOUL-GATE-FRONTEND-WEB/docs/INTEGRACION_STRIPE_PAGOS_SUSCRIPCIONES.md`.

---

## Planes y límites

| Modelo | Uso |
|--------|-----|
| `Plan` | Catálogo (tier, tipo, precios Stripe) |
| `PlanFeature` | Features por plan |
| `PlanUsage` | Consumo por lectura / compatibilidad |
| `Credits` | Saldo de créditos |
| `usedTrialForPlanTypes` | Un trial por usuario (array) |

**Validación:** `plan.routes.ts` + middleware en rutas de análisis.

---

## Perfiles y análisis (negocio)

| Modelo | Relación |
|--------|----------|
| `UserProfile` | Persona analizada bajo `User` |
| `Analysis` | Resultado IA persistido (JSON carácter, temperamento, narrativa, …) |
| `CompatibilityAnalysis` | Par de perfiles + score + detalles |
| `FreeAnalysis` | Análisis gratuito (anti-abuso) |

---

## Rutas API relevantes

| Área | Prefijo |
|------|---------|
| Auth | `/auth/*`, `/users/*` |
| Perfiles | `/api/v1/user-profiles` |
| Planes | `/api/v1/plans` |
| Stripe | `/api/v1/stripe` |
| Facturas | `/api/v1/invoices` |
| Admin planes/usuarios | `/api/v1/admin` |

---

## Colaboración y transferencia

- `UserCollaboration`: colaboradores en empresas.
- `OwnershipTransferRequest`: transferencia de propiedad de cuenta empresa.

---

## Referencias

- Modelo datos: [16-data-model.md](./16-data-model.md)
- Endpoints: [17-api-endpoints-master.md](./17-api-endpoints-master.md)
- Glosario: [15-business-domain-glossary.md](./15-business-domain-glossary.md)
