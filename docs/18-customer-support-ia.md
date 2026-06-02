# 18 — Atención al cliente con IA

**Última actualización:** 2026-05-21  
**Fuente canónica extendida:** `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB/docs/SISTEMA_IA_ATENCION_CLIENTE.md`  
**Código:** `api/src/controllers/support.controller.ts`, `api/src/routes/support.routes.ts`, `api/src/controllers/support.utils/`

---

## Resumen

Chatbot de soporte basado en **Google Gemini** (function calling) integrado en web, móvil y panel admin. Gestiona FAQs, consulta de plan/créditos, compra de créditos vía chat, escalamiento a humano y tickets con historial.

---

## Arquitectura

```
[Web supportService | Mobile supportApi | Admin support.ts]
        │ HTTP + Socket.IO
        ▼
[Express /api/v1/support/*]
        ▼
[support.controller]
        ├── callGeminiAI.ts
        ├── geminiWithFunctions.ts  ← herramientas (plan, créditos, ticket)
        ├── detectHumanIntention.ts
        ├── detectPlanInquiry.ts
        └── format*Context.ts
        ▼
[Gemini 2.0-flash / 2.5-flash]
        ▼
[Prisma → PostgreSQL]
```

---

## Endpoints (prefijo `/api/v1/support`)

| Método | Ruta | Función |
|--------|------|---------|
| `GET` | `/faqs` | Lista FAQs |
| `GET` | `/user-info` | Plan, créditos, suscripción |
| `POST` | `/support` | Crear ticket / pregunta |
| `POST` | `/chat` | Mensaje chat IA |
| `GET` | `/chat/history` | Historial conversación |
| `POST` | `/chat/new` | Nueva conversación |
| `GET` | `/tickets` | Tickets del usuario |
| `GET` | `/tickets/:id` | Detalle ticket |
| `POST` | `/tickets/message` | Mensaje en ticket |
| `PATCH` | `/tickets/:ticketId/toggle-ai` | Activar/desactivar IA en ticket |

Todas las rutas usan `authMiddleware` salvo que el controlador indique lo contrario.

---

## Flujo de conversación

1. Usuario envía mensaje → `POST /chat`.
2. Detección de intención humana / consulta de plan.
3. Contexto formateado (plan, créditos, FAQs).
4. `geminiWithFunctions` ejecuta tools si el modelo las invoca.
5. Respuesta persistida en `ChatMessage` / `SupportMessage`.
6. Si escalamiento: estado ticket → agente humano vía admin.

**Socket.IO:** eventos en tiempo real para tickets y mensajes (mismo servidor API que análisis).

---

## Function calling (Gemini)

Herramientas típicas (ver `geminiWithFunctions.ts`):

- Consultar información de usuario y suscripción
- Información de créditos y consumo
- Crear o actualizar tickets de soporte
- Escalar a soporte humano

**Autenticación Gemini:** OAuth2 / service account según variables en `api` (no exponer claves en cliente).

---

## Modelos de datos

| Modelo | Uso |
|--------|-----|
| `SupportQuestion` | Ticket principal |
| `SupportMessage` | Mensajes por ticket |
| `SupportFile` | Adjuntos |
| `ChatMessage` | Historial chat IA |

Relaciones con `User`. Detalle: [16-data-model.md](./16-data-model.md).

---

## Frontend

| Cliente | Ubicación aproximada |
|---------|---------------------|
| Web | `src/services/supportService.ts`, componentes soporte |
| Móvil | `supportApi` en SOUL-GATE-APP-MOVIL |
| Admin | gestión tickets, `toggle-ai` |

---

## Seguridad

- JWT obligatorio en rutas de soporte.
- No enviar PII innecesaria al prompt; contexto acotado a datos del usuario autenticado.
- Admin puede desactivar IA por ticket (`toggle-ai`).

Ver también: [09-auth-security.md](./09-auth-security.md).

---

## Operación y debugging

- Logs en controlador y utilidades Gemini.
- Fallos de quota/API: mensaje genérico al usuario + retry en cliente.
- Probar con usuario de prueba y plan activo para tools de billing.

---

## Gaps

- Catálogo exacto de **declaraciones de functions** (nombres y schemas JSON) — leer `geminiWithFunctions.ts` al cambiar tools.
- Métricas de satisfacción / CSAT no documentadas en repo.
- Runbook de escalamiento humano: proceso operativo fuera de código.

---

## Referencias

- Doc extendida: `SOUL-GATE-FRONTEND-WEB/docs/SISTEMA_IA_ATENCION_CLIENTE.md`
- Endpoints: [17-api-endpoints-master.md](./17-api-endpoints-master.md)
- Flujo E2E: [13-end-to-end-flows.md](./13-end-to-end-flows.md) §5
