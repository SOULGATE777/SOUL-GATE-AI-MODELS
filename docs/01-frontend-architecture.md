# 01 — Arquitectura del frontend web (Soul Gate)

**Repositorio:** `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB`  
**Última actualización:** 2026-05-21

---

## Resumen

La aplicación web de Soul Gate es una SPA **React 18 + TypeScript + Vite 5**, con enrutamiento **React Router v6**, estado global **Redux Toolkit**, UI **MUI 6** y comunicación HTTP vía **axios**. No usa Next.js.

---

## Estructura del monorepo (frontend-relevante)

```
SOUL-GATE-FRONTEND-WEB/
├── src/                 # Aplicación web principal
├── aplicantes/          # Entrada Vite secundaria (casting)
├── admin-panel/         # Panel administración (React + Vite)
├── public/
├── index.html
├── vite.config.ts
├── package.json
└── docker-compose.yaml  # Contenedores web + admin
```

---

## Stack técnico

| Capa | Tecnología |
|------|------------|
| Build | Vite 5, TypeScript |
| UI | MUI 6, Emotion |
| Estado | Redux Toolkit |
| Routing | `createBrowserRouter` en `src/main.tsx` |
| HTTP | axios + interceptores (`src/interceptors/axiosInterceptor.ts`) |
| Pagos | `@stripe/react-stripe-js` |
| i18n | i18next (20+ idiomas documentados en `docs/ARQUITECTURA_GENERAL.md`) |
| Tiempo real | `socket.io-client` (soporte, notificaciones) |

---

## Configuración de desarrollo

| Parámetro | Valor |
|-----------|-------|
| Puerto dev | **5173** (HTTPS en `vite.config.ts`) |
| Variable API | `VITE_API_URL` (ej. `https://api.soulgate.com`, local `http://localhost:80`) |
| Stripe público | `VITE_APP_STRIPE_PUBLIC_KEY` |

**Build:** `npm run build` → salida en `dist/`. Entrada dual: `index.html` + `aplicantes/index.html`.

---

## Capas de la aplicación (`src/`)

| Área | Ubicación típica | Responsabilidad |
|------|------------------|-----------------|
| Páginas | `src/pages/` | Pantallas por ruta (`Home`, `BasicInfoPage`, `Analysis`, `Compatibility`, …) |
| Componentes | `src/components/` | UI reutilizable (análisis, pagos, soporte, audio) |
| Servicios | `src/services/` | Llamadas REST (`userProfile.service.ts`, `authService.ts`, …) |
| Store | `src/store/` o slices Redux | Estado de usuario, contexto empresa, UI |
| Hooks | `src/hooks/` | `useSupportChat`, notificaciones, etc. |
| Constantes API | `src/services/const.ts` | `API_URL` desde `import.meta.env.VITE_API_URL` |

---

## Integración con backend

- **Base:** `${VITE_API_URL}` sin `baseURL` global en axios.
- **Prefijo mayoría:** `/api/v1/...`
- **Auth sin v1:** `/auth/login`, `/auth/register`, OAuth (`/auth/google-login`, etc.)
- **Headers:**
  - `Authorization: Bearer <token>` desde `localStorage.token`
  - `X-Active-Context: <companyId>` para cuentas multi-empresa (interceptor)
- **Excepciones sin token:** `/api/v1/free-analysis/*`, algunas rutas casting

Ver [03-frontend-api-backend.md](./03-frontend-api-backend.md) y [17-api-endpoints-master.md](./17-api-endpoints-master.md).

---

## Integración con motor IA

El frontend **no** llama directamente a puertos 800x ni 4051 en la mayoría de flujos producto:

1. Sube fotos al **API de negocio** (`POST /api/v1/user-profiles/:id/analysis/photos`).
2. El backend proxya al **API Gateway** (`API_GATEWAY_URL`, puerto **4051**).
3. Progreso y resultado vía respuesta HTTP + **Socket.IO** en el mismo origen que la API.

---

## Admin panel (`admin-panel/`)

- Stack análogo: React + Vite + Redux.
- Dev: puerto **5174** (documentado en `admin-panel/CLAUDE.md`).
- Token: `localStorage.admin_token`.
- Funciones: usuarios, planes, soporte, monitoreo servicios, pruebas TTS.

Detalle operativo: [11-mobile-app.md](./11-mobile-app.md) (móvil) y docs web `11-ADMIN` en índice FRONTEND-WEB.

---

## Docker (web)

`docker-compose.yaml` en raíz del monorepo:

- Frontend producto: mapeo host **3016→3017** (según despliegue documentado).
- Admin: puerto **81**.

Desarrollo local habitual: `npm run dev` en raíz, sin Docker.

---

## Referencias

- Rutas y menús: [02-frontend-ux-menus.md](./02-frontend-ux-menus.md)
- Flujo E2E: [13-end-to-end-flows.md](./13-end-to-end-flows.md)
- Índice web: `SOUL-GATE-FRONTEND-WEB/docs/00-INDICE-DOCUMENTACION-SISTEMA.md`
