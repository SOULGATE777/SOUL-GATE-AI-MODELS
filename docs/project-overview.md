# SOUL-GATE-AI-MODELS - Project Overview

**Last Updated**: 2025-12-12  
**Status**: Production Active  
**Version**: 1.0.0

---

## Qué es SOUL-GATE-AI-MODELS

SOUL-GATE-AI-MODELS es un ecosistema de **15 microservicios ML independientes** para análisis facial, de perfil y corporal, desplegados con Docker + GPU NVIDIA support.

---

## Propósito

Proveer servicios ML especializados para:
- **Análisis Frontal**: Morfología, antropometría, validación, espejo (personalidad), rotación
- **Análisis de Perfil**: Morfología, antropometría, validación, rotación  
- **Análisis Corporal**: Manos (clasificación + colorimetría)
- **Estimación de Edad**: Age estimation usando InsightFace

---

## Arquitectura High-Level

```
SOUL-GATE-AI-MODELS/
├── frontal_prod/      # 6 servicios (puertos 8000-8002, 8008, 8012, 8014)
├── profile_prod/      # 5 servicios (puertos 8003-8005, 8010)
├── body_prod/         # 3 servicios (puerto 8009)
├── age_prod/          # 1 servicio (puerto 8013)
└── docs/              # Documentación del proyecto
```

Cada servicio es **autocontenido** con:
- Dockerfile + docker-compose.yml
- FastAPI REST API
- ML models (PyTorch)
- Health checks
- GPU support

---

## Servicios por Módulo

### Frontal (6 servicios)

| Puerto | Servicio | Propósito | Modelo ML |
|--------|----------|-----------|-----------|
| 8000 | Morfológico | 45 tags morfológicos | 3-model ensemble (Faster R-CNN + CNNs) |
| 8001 | Antropométrico | Mediciones faciales | dlib 68 + custom 13 points |
| 8002 | Validación | 17 features validation | YOLOv8 |
| 8008 | Espejo | Análisis personalidad | dlib + Faster R-CNN + CNN + decision tree |
| 8012 | Rotación | Viability assessment | EfficientNet-B0 |
| 8014 | Preprocesamiento | Face detection + alignment | MediaPipe |

### Profile (5 servicios)

| Puerto | Servicio | Propósito | Modelo ML |
|--------|----------|-----------|-----------|
| 8003 | Morfológico | Perfil morfológico | 3-model ensemble |
| 8004 | Antropométrico | Mediciones perfil | Custom models |
| 8005 | Validación | Validación perfil | YOLOv8 |
| 8010 | Preprocesamiento | Profile detection + rotation | Faster R-CNN |
| N/A | Rotación | Rotación assessment | EfficientNet |

### Body (3 servicios)

| Puerto | Servicio | Propósito | Modelo ML |
|--------|----------|-----------|-----------|
| 8009 | Manos | Dorso/palma + colorimetría | ResNet50 + K-means |
| N/A | Morfológico | Análisis corporal | TBD |
| N/A | Antropométrico | Mediciones corporales | TBD |

### Age (1 servicio)

| Puerto | Servicio | Propósito | Modelo ML |
|--------|----------|-----------|-----------|
| 8013 | Age Estimation | Estimación edad (±3-5 años) | InsightFace + ONNX |

---

## Stack Tecnológico

- **Language**: Python 3.9+
- **API Framework**: FastAPI 0.104.1
- **ML Framework**: PyTorch 2.0.1-2.1.0 + CUDA 11.8/12.1
- **Image Processing**: OpenCV 4.8.1, Pillow 10.x
- **Deployment**: Docker + Docker Compose
- **GPU**: NVIDIA CUDA support

Ver detalles completos en [`tech-stack.md`](./tech-stack.md).

---

## Integration Pattern

```
Input Image → [Preprocessing Service] → Base64 Image
                                          ↓
                              [Analysis Services] → JSON Results
                                          ↓
                              [Backend API] → Consolidate + Business Rules
                                          ↓
                              [Gemini AI] → Generate Narrative
                                          ↓
                              [Database] → Save Analysis
```

---

## Deployment

**Ambiente**: Production (AWS EC2 con GPUs NVIDIA)

**Health Monitoring**:
- Health checks cada 30s
- Auto-restart con `unless-stopped`
- Logging estructurado

**Scalability**:
- Microservicios independientes
- Load balancing posible
- Horizontal scaling por servicio

Ver detalles en [`deployment-guide.md`](./deployment-guide.md).

---

## Estado del Proyecto

### ✅ Completado
- 15 servicios ML operacionales
- Docker deployment con GPU
- Health checks configurados
- READMEs documentados
- Cursor Project Rules (.mdc)

### ⏳ En Progreso
- Implementación de NewFeature.md (umbrales)
- Testing coverage (pytest)
- CI/CD pipeline

### 📋 Pendiente
- Monitoring (Prometheus + Grafana)
- Automated testing en CI
- Performance benchmarking
- API Gateway consolidado

---

## Para Empezar

### Desarrollador Nuevo

1. Lee este documento
2. Revisa [`tech-stack.md`](./tech-stack.md)
3. Consulta [`architecture.md`](./architecture.md)
4. Sigue [`deployment-guide.md`](./deployment-guide.md)
5. Revisa Cursor Rules en `.cursor/rules/`

### AI Agent

1. **SIEMPRE** consulta `@docs/project-overview.md` antes de tareas
2. Lee `@docs/newfeature-implementation.md` si implementas features
3. Actualiza docs cuando modifiques código
4. Registra decisiones en `@docs/architecture.md`

---

## Links Útiles

- **README Principal**: `/README.md`
- **NewFeature.md**: `/home/mitza/proyectos/SOUL-GATE/NewFeature.md`
- **Backend API**: `/home/mitza/proyectos/SOUL-GATE-FRONTEND-WEB/api`
- **Cursor Rules**: `.cursor/rules/*.mdc`

---

## Contacto

**Project**: SOUL-GATE  
**Repository**: SOUL-GATE-AI-MODELS  
**Maintained by**: Development Team

---

**Este documento es el punto de entrada al proyecto. Mantenerlo actualizado es crítico.**

