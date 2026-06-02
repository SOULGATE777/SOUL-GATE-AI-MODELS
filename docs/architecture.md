# Architecture - SOUL-GATE-AI-MODELS

**Last Updated**: 2025-12-12

---

## Architectural Style

**Microservices** - Cada servicio ML es independiente, autocontenido y desplegable por separado.

---

## Design Principles

1. **Separation of Concerns**: Cada servicio tiene un propósito único
2. **Stateless Services**: No state compartido, solo modelos ML
3. **Base64 Integration**: Preprocesadores retornan base64 para pipeline seamless
4. **Health Monitoring**: Todos los servicios exponen `/health`
5. **GPU Isolation**: Cada servicio puede usar su propia GPU

---

## Service Structure (Standard)

```
{service}/
├── Dockerfile               # Container definition
├── docker-compose.yml       # Orchestration (GPU support)
├── requirements.txt         # Python dependencies
├── README.md               # Service documentation
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI entrypoint
│   ├── models/
│   │   └── *_pipeline.py  # Core ML pipeline
│   └── utils/
│       ├── image_processing.py
│       └── visualization.py
├── models/                 # ML model files
└── results/                # Generated outputs
```

---

## Integration Flow

### 1. Preprocessing Phase
```
Input Image → [Preprocessing Service] → Base64 Cropped Face/Body
```

### 2. Analysis Phase
```
Base64 → [Analysis Services] → JSON Results
```

### 3. Consolidation Phase
```
Multiple JSON Results → [Backend API] → Consolidated Response
```

### 4. Narrative Generation
```
Consolidated Data → [Gemini AI] → Human-Readable Narrative
```

---

## Communication Pattern

- **Protocol**: HTTP REST
- **Format**: JSON
- **CORS**: Enabled para backend integration
- **Async**: I/O operations

---

## Módulo Espejo (Personalidad)

### Decision Tree Architecture

```
Input: Facial Image
  ↓
[dlib 68 + Faster R-CNN 13 + CNN Classifier]
  ↓
[Mirror Generation: Left/Right]
  ↓
[Region Classification: FRENTE (7) + rostro_menton (8)]
  ↓
[Decision Tree con Umbrales]
  ↓
[Proporción Facial Splitting]
  ↓
Output: Personalidad Diagnoses
```

### Umbrales (según NewFeature.md)
- **General**: 18% mínimo
- **Venus Corazón**: 40% mínimo  
- **Plutón Hexagonal**: 7% mínimo

**Decision Tree Rules**:
1. Solo Diagnosis (alta confidence → diagnóstico único)
2. Exclusion Rules (baja confidence → excluir)
3. Proporción splitting (múltiples diagnósticos)

---

## Deployment Architecture

```
AWS EC2 (GPU Instances)
  ↓
[Docker Containers con GPU Support]
  ↓
[NVIDIA Runtime + CUDA]
  ↓
[Health Checks + Auto-Restart]
```

---

## Architectural Decisions (ADRs)

### ADR-001: Microservices vs Monolith
**Decision**: Microservices  
**Razón**: 
- Escalabilidad independiente
- Deployment aislado
- GPU isolation por servicio
- Failure isolation

### ADR-002: Base64 Integration Pattern
**Decision**: Preprocesadores retornan base64  
**Razón**:
- Seamless pipeline integration
- No necesidad de storage compartido
- Reduce latency (no file I/O)

### ADR-003: FastAPI sobre Flask
**Decision**: FastAPI  
**Razón**:
- Async support nativo
- Auto-documentation (Swagger/OpenAPI)
- Type hints + validation
- Performance superior

---

## Escalabilidad

- **Horizontal**: Múltiples instancias por servicio (load balancer)
- **Vertical**: GPU más potentes por servicio
- **Resource Isolation**: Docker limits por contenedor

---

## Monitoring (Futuro)

- Prometheus para métricas
- Grafana para dashboards
- Sentry para error tracking
- ELK stack para logs

---

**Registra decisiones arquitectónicas importantes en este documento.**

---

## Change Log

### 2025-12-12
- Initial architecture documentation
- Documented Espejo decision tree
- Defined ADRs 001-003

