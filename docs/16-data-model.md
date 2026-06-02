# 16 — Modelo de datos (PostgreSQL / Prisma)

**Schema:** `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB/api/prisma/schema.prisma`  
**ORM:** Prisma  
**Última actualización:** 2026-05-21

---

## Resumen

La persistencia de producto vive en la **API de negocio**. El API-GATAWAY **no** tiene base de datos de usuarios en el árbol actual.

---

## Modelos principales

### Identidad y cuenta

| Modelo | Descripción |
|--------|-------------|
| `User` | Cuenta: auth, Stripe, tipo (`UserType`), empresa, OAuth ids |
| `PreRegister` | Lista espera pre-registro |
| `DeviceSession` | Sesión por dispositivo |
| `SessionToken` | Tokens de sesión |
| `UserCollaboration` | Colaboradores en empresas |
| `OwnershipTransferRequest` | Transferencia propiedad empresa |

**Enums:** `UserType` (INDIVIDUAL, COMPANY, MANAGED, SYSTEM_ADMIN), `UserRole` (ADMIN, USER, SUPER_ADMIN).

### Pagos

| Modelo | Descripción |
|--------|-------------|
| `Payment` | Pago puntual Stripe |
| `Subscription` | Suscripción recurrente |
| `Plan` | Catálogo plan |
| `PlanFeature` | Features por plan |
| `Credits` | Saldo créditos |
| `CreditInvoice` | Compra de créditos |
| `Refund` | Reembolsos |
| `PlanUsage` | Consumo por lectura/compatibilidad |

### Análisis

| Modelo | Descripción |
|--------|-------------|
| `UserProfile` | Persona analizada (nombre, sexo, país, …) |
| `Analysis` | JSON: character, temperament, narrative, strategies, photos refs, … |
| `CompatibilityAnalysis` | Par de perfiles, score, details |
| `FreeAnalysis` | Análisis gratuito / funnel |

### Soporte y comunicación

| Modelo | Descripción |
|--------|-------------|
| `SupportQuestion` | Ticket soporte |
| `SupportMessage` | Mensajes ticket |
| `SupportFile` | Adjuntos |
| `ChatMessage` | Historial chat IA |
| `ResourceGuide` | Guías recurso |
| `Notification` | Notificaciones usuario |

### Encuestas y casting

| Modelo | Descripción |
|--------|-------------|
| `Survey`, `SurveyQuestion`, `SurveyResponse`, `SurveyAnswer` | Encuestas post-análisis |
| `CastingApplication` | Postulación casting |
| `CastingApplicationComment` | Comentarios internos |
| `CastingPageEvent` | Analytics página casting |

### Otros

| Modelo | Descripción |
|--------|-------------|
| `AudioShareCard` | Tarjetas audio compartibles (S3) |
| `SystemSetting` | Configuración sistema |

---

## Relaciones clave

```
User 1──* UserProfile
UserProfile 1──* Analysis
User 1──* CompatibilityAnalysis
User 1──* Subscription
User 1──1 Credits (por contexto de diseño)
User *──* Plan (vía Subscription)
```

---

## Campos JSON importantes (`Analysis`)

Typical payloads (nombres verificados en uso, no exhaustivo):

- `character` — puntuaciones rasgos
- Datos temperamento (estructura según gateway)
- Narrativa y estrategias
- Referencias fotos / S3
- Metadatos para compatibilidad

**Migraciones:** `npx prisma db push` / migrate según política del equipo.

---

## Índices

Varios modelos incluyen índices en `status`, `createdAt`, `userId` para listados admin y facturación.

---

## Histórico API-GATAWAY

`API-GATAWAY/docs/02-user-service-database.md` describe esquema **eliminado** del microservicio `user-service` — útil solo como referencia histórica; **canónico = schema actual en api/prisma**.

---

## Referencias

- Billing: [06-user-billing-subscriptions.md](./06-user-billing-subscriptions.md)
- Glosario: [15-business-domain-glossary.md](./15-business-domain-glossary.md)
