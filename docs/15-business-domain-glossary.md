# 15 — Glosario de dominio de negocio

**Ecosistema Soul Gate** — términos unificados  
**Basado en:** `API-GATAWAY/docs/00-glossary-domain.md` + código producto  
**Última actualización:** 2026-05-21

---

## Marcas y producto

| Término | Definición |
|---------|------------|
| **Soul Gate** | Marca y aplicación usuario final (web, móvil, casting). |
| **Soul AI** | Ecosistema de microservicios de análisis biométrico (`API-GATAWAY`). |
| **Análisis morfológico** | Resultado completo: rasgos físicos, temperamento, carácter, narrativa, estrategias. |
| **Perfil** (`UserProfile`) | Persona registrada con datos demográficos y opcionalmente un `Analysis`. |
| **Análisis** (`Analysis`) | Resultado persistido del pipeline IA. |

---

## Modalidades de captura

| Término | Definición |
|---------|------------|
| **Análisis frontal** | Vista de frente: validación, morfo, antropo, espejo, rotación. |
| **Análisis de perfil** | Vista lateral: morfo, antropo, validación, preproceso. |
| **Análisis corporal** | Cuerpo completo (deshabilitado en gateway consolidado actual). |
| **Manos / palmas** | Dorso/palma; colorimetría para elemento e1 temperamento. |
| **Colorimetría ocular** | Colores iris (hasta 9 categorías). |
| **Validación de imágenes** | Calidad, rotación, rasgos detectables antes del análisis completo. |
| **Análisis consolidado** | Un request orquesta todos los servicios (`analyze-consolidated`). |

---

## Espejo

| Término | Definición |
|---------|------------|
| **Espejo** | Hemicaras reflejadas; regiones FRENTE y rostro_mentón. |
| **Diagnóstico final** | Etiquetas tras umbrales (ej. `solar`, `venus_corazon`). |
| **Umbrales** | 18% general; Venus Corazón ≥40%; Plutón ≥7%. |

Formas planetarias son **tags técnicos**, no astrología en UX.

---

## Morfología y antropometría

| Término | Definición |
|---------|------------|
| **Morfología** | Clasificación ML de rasgos (tags). |
| **Antropometría** | Medidas geométricas (landmarks, proporciones). |
| **Tag** | Etiqueta atómica del modelo. |
| **Preprocesamiento** | Detección, alineación, recorte. |

---

## Temperamento (L / S / B / N)

| Código | Nombre |
|--------|--------|
| **L** | Linfático |
| **S** | Sanguíneo |
| **B** | Bilioso |
| **N** | Nervioso |

**Elementos e1–e8:** fuentes ponderadas (tez, cuerpo, espejo, mandíbula, nariz, mentón, frente perfil, ojos).

---

## Personalidad y producto

| Término | Definición |
|---------|------------|
| **Carácter** | Rasgos con puntuación en `Analysis.character`. |
| **Par dual** | Dos rasgos opuestos normalizados a 100%. |
| **Estrategias** | Recomendaciones de interacción (`Analysis.strategies`). |
| **Narrativa** | Texto Gemini legible para el usuario. |
| **Compatibilidad** | Comparación dos perfiles + narrativa Gemini. |

---

## Monetización

| Término | Definición |
|---------|------------|
| **Plan** | Suscripción Stripe con límites (`Plan`, `PlanFeature`). |
| **Créditos** | Consumo pay-as-you-go (`Credits`). |
| **PlanUsage** | Registro de cada lectura/compatibilidad consumida. |
| **Trial** | Un trial por usuario (`usedTrialForPlanTypes`). |

---

## Organización

| Término | Definición |
|---------|------------|
| **COMPANY** | Cuenta empresa. |
| **MANAGED** | Usuario gestionado por empresa. |
| **X-Active-Context** | Header para operar en contexto empresa. |

---

## Referencias

- Reglas ML: `/home/mitza/proyectos/SOUL-GATE/NewFeature.md`
- Modelo datos: [16-data-model.md](./16-data-model.md)
