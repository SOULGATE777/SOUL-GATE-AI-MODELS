# 14 — Servicios auxiliares

**Última actualización:** 2026-05-21

---

## TEXT-TO-VOICE (TTS)

**Repositorio:** `/home/mitza/proyectos/TEXT-TO-VOICE`  
**Puerto:** **5032** (Docker / producción)

### Propósito

Síntesis de voz para narraciones de análisis, estrategias y compatibilidad usando **Microsoft Edge TTS** (online).

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado + RAM |
| `GET` | `/voices` | Catálogo voces |
| `POST` | `/tts` | MP3 completo |
| `POST` | `/tts/stream` | Stream multipart (usado por frontend) |

### Integración

```
Cliente → API negocio POST /api/v1/tts/stream
        → TTS_SERVER_URL (ej. http://18.220.61.87:5032)
        → Edge TTS
```

**Componentes cliente:** `AudioReader.tsx` (web), móvil `ttsService`, admin pruebas TTS.

**Nota:** README legacy menciona puertos 8000/8010 (Kokoro); implementación vigente es **v2 Edge en 5032**.

Doc: `API-GATAWAY/docs/09-text-to-voice.md`.

---

## soul-cost (unit economics)

**Repositorio:** `/home/mitza/proyectos/soul-cost`  
**Puerto dev:** **5173**

### Propósito

Dashboard **interno** (no producción) para:

- Costo unitario por lectura
- Proyecciones de volumen
- Margen planes vs infra (USD compute / MXN precios)

### Stack

Vite + React + TypeScript + Tailwind. Lógica en `src/soulgate-cost-analysis.tsx` con constantes embebidas; `src/data/aws-infra.json` como snapshot.

### Integración ecosistema

- **No** llama APIs de producto.
- Referencia costos de ML, TTS, Gemini en documentación markdown del repo.
- Billing real: API negocio + Stripe.

Doc: `API-GATAWAY/docs/10-soul-cost.md`.

---

## Monitoring dashboard (legacy)

**Repositorio:** `API-GATAWAY/monitoring-dashboard/`  
**Puerto:** 3000 (CRA, **no** en docker-compose actual)

- Poll `GET /health` cada 10s.
- Solo 4 servicios con puertos **obsoletos** (4000, 4001).
- **No usar** para operación actual.

**Alternativa:** `GET http://localhost:4051/api/health/all`.

Doc: `API-GATAWAY/docs/14-monitoring-dashboard.md`.

---

## ElasticMQ (desarrollo)

**Repo:** `SOUL-GATE-FRONTEND-WEB/docker-compose.dev.yml`

- Puertos **9324** (API), **9325** (UI) — colas SQS-compatibles en dev local.

---

## Referencias

- Infra puertos: [12-infrastructure-deployment.md](./12-infrastructure-deployment.md)
- Índice web TTS: serie `19-TTS` en `SOUL-GATE-FRONTEND-WEB/docs/00-INDICE`
