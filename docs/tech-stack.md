# Tech Stack - SOUL-GATE-AI-MODELS

**Last Updated**: 2025-12-12

---

## Core Stack

### Languages & Frameworks
- **Python**: 3.9+
- **FastAPI**: 0.104.1 (REST APIs)
- **uvicorn**: 0.24.0 (ASGI server)

### ML Frameworks
- **PyTorch**: 2.0.1-2.1.0
- **torchvision**: 0.15.2-0.16.0
- **CUDA**: 11.8 / 12.1 (GPU acceleration)

### Image Processing
- **opencv-python**: 4.8.1.78
- **Pillow**: 10.0.1-10.1.0
- **numpy**: 1.24.3

### Specialized ML Libraries
- **dlib**: 19.22.1 (68-point facial landmarks)
- **ultralytics**: 8.0.200 (YOLOv8)
- **insightface**: 0.7.3 (Age estimation)
- **timm**: 0.9.7 (EfficientNet)
- **scikit-learn**: 1.3.0 (K-means, clustering)
- **mediapipe**: 0.10.x (Face detection/mesh)

### Testing
- **pytest**: 7.4.3
- **pytest-asyncio**: 0.21.1
- **pytest-cov**: 4.1.0

### Deployment
- **Docker**: 24.x
- **Docker Compose**: 3.8
- **Base Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

---

## Dependencies por Servicio

Ver `requirements.txt` en cada servicio para detalles específicos.

---

## GPU Requirements

- **NVIDIA GPU**: RTX 20XX+ o superior
- **CUDA**: 11.8 o 12.1
- **VRAM**: Mínimo 4GB, recomendado 8GB+
- **NVIDIA Container Toolkit**: Para Docker GPU support

---

## Deprecations & Updates

**Ninguna librería deprecada detectada** - Stack moderno (2023-2024).

---

**Consulta este documento al agregar nuevas dependencias.**

