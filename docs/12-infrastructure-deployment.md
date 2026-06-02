# 12 — Infraestructura y despliegue

**Ecosistema Soul Gate** — resumen operativo  
**Última actualización:** 2026-05-21

---

## Mapa de puertos

| Componente | Puerto(s) | Repo |
|------------|-----------|------|
| Web Vite dev | 5173 (HTTPS) | SOUL-GATE-FRONTEND-WEB |
| Admin Vite dev | 5174 | admin-panel |
| API negocio local | 5001 | api/ |
| API negocio Docker | 80 | api/docker-compose.yml |
| API Gateway | **4051** | API-GATAWAY |
| IA wrappers | 4301–4306 | API-GATAWAY |
| ML FastAPI | 8000–8014 | SOUL-GATE-AI-MODELS |
| TTS | **5032** | TEXT-TO-VOICE |
| Redis dev | 6379 | docker-compose.dev.yml |
| soul-cost dev | 5173 | soul-cost (local only) |

**Producción API pública:** `https://api.soulgate.com` (típicamente nginx 443 → backend).

---

## Docker — API-GATAWAY

```bash
cd /home/mitza/proyectos/API-GATAWAY
docker compose up -d
curl -s http://localhost:4051/health
```

Servicios: ver [04-api-gateway-overview.md](./04-api-gateway-overview.md).

---

## Docker — API negocio

```bash
cd /home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB/api
docker compose up -d
# API en puerto 80 del host según compose
```

Requiere `DATABASE_URL`, `REDIS_URL`, secretos Stripe, `API_GATEWAY_URL`.

---

## Docker — ML (por servicio)

```bash
cd /home/mitza/proyectos/SOUL-GATE-AI-MODELS/frontal_prod/espejo
docker-compose build && docker-compose up -d
```

**GPU:** NVIDIA Container Toolkit, `deploy.resources` en compose.

Guía: `SOUL-GATE-AI-MODELS/docs/deployment-guide.md` (EC2 p3.*, backup modelos).

---

## Docker — Web / Admin

`SOUL-GATE-FRONTEND-WEB/docker-compose.yaml` — imágenes frontend y admin-panel.

---

## TEXT-TO-VOICE

```bash
# Puerto 5032
docker compose up -d   # en repo TEXT-TO-VOICE
```

Variable en API negocio: `TTS_SERVER_URL=http://<host>:5032`.

---

## Variables críticas (API negocio)

| Variable | Ejemplo / nota |
|----------|----------------|
| `DATABASE_URL` | PostgreSQL |
| `REDIS_URL` | `redis://redis:6379` |
| `JWT_SECRET` | Obligatorio |
| `API_GATEWAY_URL` | `http://api-gateway:4051` en red Docker |
| `TTS_SERVER_URL` | Host:5032 |
| `STRIPE_*` | Pagos |
| `AWS_*` | S3 |

Listado ampliado: `SOUL-GATE-FRONTEND-WEB/docs/00-INDICE` → doc 16 planificado; `API-GATAWAY/docs/18-environment-configuration.md`.

---

## CI/CD

**Estado:** pipelines formales **pendientes** (stubs en SISTEMA-COMPLETO §13.3).

Práctica actual: build Docker manual, scripts `start-services.sh` / `deploy_all.sh` en repos, EAS para móvil.

---

## Nginx / producción

- Casting aplicantes: `docs/APLICANTES_PROXY_NGINX_PRODUCCION.md`
- Credenciales: `CREDENTIALS_AND_DEPLOYMENT_GUIDE.md` (monorepo web)

---

## Monitoreo actual

| Herramienta | Alcance |
|-------------|---------|
| `GET :4051/health` | Gateway + dependencias |
| `docker stats`, `nvidia-smi` | ML GPU |
| `GET /health` API negocio | DB + Redis |
| monitoring-dashboard (legacy) | Puertos obsoletos — no usar en prod |

---

## Referencias

- ML ops: [07-ai-models-pipeline.md](./07-ai-models-pipeline.md)
- Gateway compose: `API-GATAWAY/docs/04-docker-compose-reference.md`
- Troubleshooting ML: `SOUL-GATE-AI-MODELS/docs/troubleshooting.md`
