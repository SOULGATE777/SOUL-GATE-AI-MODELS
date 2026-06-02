# 10 — Compatibilidad y narrativa IA

**Repositorio:** `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB/api`  
**Servicio clave:** `api/src/services/compatibilityNarrative.service.ts`  
**Última actualización:** 2026-05-21

---

## Resumen

La **compatibilidad** compara dos perfiles con análisis completos, calcula métricas de afinidad y genera una **narrativa en lenguaje natural** vía **Google Gemini** (OAuth2 service account), usando reglas en `api/src/config/compatibility-rules/`.

El análisis biométrico base ya existió en cada `Analysis`; este módulo **no** re-ejecuta el pipeline ML completo salvo que el producto dispare nuevo análisis.

---

## Modelo de datos

**`CompatibilityAnalysis`** (Prisma):

- Referencias a dos perfiles / análisis
- `score` y JSON `details`
- Narrativa y metadatos persistidos según implementación del controlador

---

## API (negocio)

| Método | Ruta |
|--------|------|
| `POST` | `/api/v1/user-profiles/compatibility` |
| `GET` | `/api/v1/user-profiles/compatibility/my-analyses` |
| `GET` | `/api/v1/user-profiles/compatibility/:analysisId` |
| PDF | `/api/v1/compatibility-pdf` (generación descarga) |

**Controlador:** `userProfile.controller.ts` — invoca `generateCompatibilityNarrative`.

---

## Servicio de narrativa

**Archivo:** `compatibilityNarrative.service.ts`

| Aspecto | Detalle |
|---------|---------|
| Modelos | `gemini-2.5-flash-lite` con fallback |
| Auth | OAuth2 (service account JSON) |
| Entrada | Rasgos, temperamentos, pares duales, reglas de compatibilidad |
| Salida | Texto legible sin jerga técnica planetaria |
| Logs | `LLM_LOGS_DIR` (auditoría prompts/respuestas) |

**Reglas:** archivos bajo `api/src/config/compatibility-rules/` (umbrales, descripciones de personalidad, pares opuestos).

---

## Flujo UX

```
/compatibility o /app
  → usuario selecciona 2 perfiles con análisis
  → POST compatibility
  → backend calcula score + llama Gemini
  → UI muestra narrativa + métricas
  → opcional: descarga PDF
```

---

## Gateway vs API negocio

| Capa | Narrativa personalidad individual | Compatibilidad |
|------|-----------------------------------|----------------|
| API Gateway | Gemini en `analyze-consolidated` | No |
| API negocio | Puede enriquecer / mostrar | **Gemini dedicado** |

---

## i18n

Narrativa puede generarse o traducirse según idioma del usuario (ver docs traducción automática en monorepo web).

---

## Referencias

- Glosario: [15-business-domain-glossary.md](./15-business-domain-glossary.md) (temperamento, pares duales)
- Endpoints: [17-api-endpoints-master.md](./17-api-endpoints-master.md)
- `API-GATAWAY/docs/06-compatibility-narrative.md`
- Frontend: [02-frontend-ux-menus.md](./02-frontend-ux-menus.md)
