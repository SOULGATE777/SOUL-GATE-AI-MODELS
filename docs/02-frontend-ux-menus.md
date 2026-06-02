# 02 — Frontend web: rutas, menús y UX

**Repositorio:** `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB`  
**Fuente de rutas:** `src/main.tsx`  
**Última actualización:** 2026-05-21

---

## Enrutamiento principal

Router: **React Router v6** (`createBrowserRouter`).

| Ruta | Componente | Protección |
|------|------------|------------|
| `/` | `App` (landing / home) | Pública |
| `/privacy-policy` | `PrivacyPolicyPage` | Pública |
| `/registration` | `RegistrationPage` | `ProtectedRoute` |
| `/confirmPayment` | `ConfirmPayment` | Protegida |
| `/bankTransfer` | `BankTransfers` | Protegida |
| `/app` | `BasicInfoPage` | Protegida — **hub principal post-login** |
| `/analysis` | `Analysis` | Pública |
| `/free-analysis` | `FreeAnalysisPage` | Pública |
| `/mobile-analysis` | `MobileAnalysisFlow` | Pública |
| `/compatibility` | `Compatibility` | Pública |
| `/accept-transfer` | `OwnershipAcceptTransfer` | Pública |
| `/socket-test` | `SocketTestComponent` | Pública (dev) |
| `/aplicantes` | `AplicantesPage` | Pública (nav oculta) |
| `/aplicantes/formulario` | Formulario casting | Pública |
| `/aplicantes/confirmacion` | Confirmación casting | Pública |

**Router alternativo legado:** `src/mainRouter.tsx` (subconjunto; producción usa `main.tsx`).

---

## Navegación global

**Componente:** `src/pages/Home/NavSection/NavSection.tsx`

- Barra MUI: login/registro, selector de idioma, menú de usuario.
- Tras login: navegación a **`/app`**.
- **Nav oculta** en:
  - `/free-analysis`
  - `/mobile-analysis`
  - rutas bajo `/aplicantes*`

---

## Hub post-login (`/app`)

`BasicInfoPage` concentra:

- Gestión de **perfiles** (`UserProfile`)
- Lanzamiento de **análisis** (subida de fotos, progreso)
- **Compatibilidad** entre perfiles
- **Pagos**, facturas, suscripción
- Acceso a **soporte** (chat / tickets)
- Configuración de cuenta y contexto empresa

---

## Flujos UX clave

### Registro y pago

```
/ → registro (/registration) → confirmación pago (/confirmPayment | /bankTransfer) → /app
```

### Análisis de personalidad (usuario autenticado)

```
/app → crear/editar perfil → subir fotos (frontal, perfil, cuerpo, palmas según plan)
     → validación imágenes (API) → análisis consolidado (gateway vía API)
     → visualización resultados (temperamento, carácter, narrativa, estrategias)
```

### Análisis gratuito

```
/free-analysis o /mobile-analysis → fingerprint dispositivo → API pública /api/v1/free-analysis
```

### Compatibilidad

```
/compatibility o desde /app → selección 2 perfiles → POST /api/v1/user-profiles/compatibility
→ narrativa Gemini (backend) → PDF opcional
```

### Casting (aplicantes)

```
/aplicantes → formulario → POST /api/v1/casting-applications (público + tracking)
```

Proxy Nginx producción: `docs/APLICANTES_PROXY_NGINX_PRODUCCION.md`.

---

## Estados de UI (obligatorios en cambios)

Para pantallas con carga asíncrona o envío de formularios:

- Loading / skeleton durante análisis y pagos
- Error con mensaje recuperable (validación fotos, pago fallido)
- Empty (sin perfiles, sin análisis previo)
- Disabled durante submit

Reglas globales: `.cursor/rules/user-24-ui-state-and-accessibility.mdc`.

---

## i18n

- **i18next** en web; claves por módulo.
- Traducción automática documentada en `docs/INTEGRACION_TRADUCCION_AUTOMATICA_IDIOMAS.md`.
- Narrativas de compatibilidad generadas en idioma del usuario (backend).

---

## Accesibilidad

- Preferir HTML semántico, labels en formularios, foco visible.
- Skill de proyecto: `SOUL-GATE-FRONTEND-WEB/.agents/skills/accessibility/`.

---

## Referencias cruzadas

- Arquitectura: [01-frontend-architecture.md](./01-frontend-architecture.md)
- API consumida: [17-api-endpoints-master.md](./17-api-endpoints-master.md)
- Flujo análisis: [13-end-to-end-flows.md](./13-end-to-end-flows.md)
