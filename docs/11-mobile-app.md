# 11 — Aplicación móvil (Expo)

**Repositorio:** `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB/SOUL-GATE-APP-MOVIL`  
**Última actualización:** 2026-05-21

---

## Stack

| Tecnología | Versión / nota |
|------------|----------------|
| Expo | SDK ~54 |
| React Native | Con Expo Router |
| Estado | Redux Toolkit |
| HTTP | axios (`apiClient`) |
| Auth tokens | Almacenamiento seguro (secure store) |
| Tiempo real | Socket.IO (soporte, notificaciones) |

---

## API

- **Base producción:** `https://api.soulgate.com` (`app/config/api.ts` según `CLAUDE.md` móvil).
- Mismos contratos que web: `/auth/*`, `/api/v1/user-profiles`, Stripe, soporte.
- TTS: proxy `POST /api/v1/tts/stream` (igual que web).

---

## Funcionalidades alineadas con web

- Login (email + OAuth donde esté habilitado)
- Perfiles y análisis (subida de fotos)
- Compatibilidad
- Suscripción / pagos (Stripe móvil)
- Soporte chat + tickets
- Audio reader (narración)

---

## Autenticación móvil

Documentación backend:

- `api/docs/mobile-authentication-flow-diagram.md`
- `api/docs/react-native-authentication-implementation.md`
- `api/docs/mobile-single-login-documentation.md`

Patrón: JWT + `sessionId` + validación sesión única por dispositivo donde aplique.

---

## Build y despliegue

- Android / iOS vía EAS (ver `docs/expo-ios-production-build.md`, `VALIDACION_BUILD_GOOGLE_PLAY_SIGNIN.md` en monorepo web).
- Variables de entorno Expo para `API_URL` y Stripe publishable key.

---

## Referencias

- Backend: [03-frontend-api-backend.md](./03-frontend-api-backend.md)
- Auth: [09-auth-security.md](./09-auth-security.md)
- UX web (paridad): [02-frontend-ux-menus.md](./02-frontend-ux-menus.md)
- `API-GATAWAY/docs/13-mobile-application.md`
