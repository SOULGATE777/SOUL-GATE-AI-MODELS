# Deployment Guide - SOUL-GATE-AI-MODELS

**Last Updated**: 2025-12-12

---

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ (Linux)
- **Docker**: 24.x+
- **Docker Compose**: 2.x+
- **GPU**: NVIDIA GPU (RTX 20XX+ o superior)
- **NVIDIA Drivers**: 525.x+
- **NVIDIA Container Toolkit**: Latest

### Verificar GPU
```bash
nvidia-smi
```

### Verificar Docker GPU Support
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Deployment por Servicio

### Estructura General

Cada servicio sigue este pattern:

```bash
cd frontal_prod/espejo  # o cualquier servicio

# Build
docker-compose build

# Start
docker-compose up -d

# Check logs
docker-compose logs -f

# Health check
curl http://localhost:8008/health

# Stop
docker-compose down
```

---

## Servicios y Puertos

| Puerto | Servicio | Comando |
|--------|----------|---------|
| 8000 | Morfológico Frontal | `cd frontal_prod/morfologico && docker-compose up -d` |
| 8001 | Antropométrico Frontal | `cd frontal_prod/antropometrico && docker-compose up -d` |
| 8002 | Validación Frontal | `cd frontal_prod/validacion && docker-compose up -d` |
| 8008 | Espejo | `cd frontal_prod/espejo && docker-compose up -d` |
| 8012 | Rotación Frontal | `cd frontal_prod/rotacion && docker-compose up -d` |
| 8014 | Preprocesamiento Frontal | `cd frontal_prod/preprocesamiento && docker-compose up -d` |
| 8003 | Morfológico Profile | `cd profile_prod/morfologico && docker-compose up -d` |
| 8004 | Antropométrico Profile | `cd profile_prod/antropometrico && docker-compose up -d` |
| 8005 | Validación Profile | `cd profile_prod/validacion && docker-compose up -d` |
| 8010 | Preprocesamiento Profile | `cd profile_prod/preprocesamiento && docker-compose up -d` |
| 8009 | Manos | `cd body_prod/manos && docker-compose up -d` |
| 8013 | Age Estimation | `cd age_prod && docker-compose up -d` |

---

## Environment Variables

Cada servicio usa:

```bash
CUDA_VISIBLE_DEVICES=0  # GPU index (0, 1, 2...) o -1 para CPU
PYTHONPATH=/app
```

Para cambiar GPU:
```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=1  # Usar GPU 1 en lugar de GPU 0
```

---

## Model Files

Cada servicio requiere archivos de modelo en `models/`:

```bash
# Ejemplo: Espejo service
frontal_prod/espejo/models/
├── best_morphological_model.pth
├── best_eyebrow_size_model.pth
├── shape_predictor_68_face_landmarks.dat
└── improved_anthropometric_model.pth
```

**⚠️ IMPORTANTE**: Models NO están en Git (archivos grandes).
- Descargar de storage compartido o S3
- Colocar en carpeta `models/` correspondiente
- Verificar permisos: `chmod 644 models/*.pth`

---

## Health Checks

### Verificar todos los servicios

```bash
# Health check script
for port in 8000 8001 8002 8008 8012 8014 8003 8004 8005 8010 8009 8013; do
    echo "Checking port $port..."
    curl -f http://localhost:$port/health || echo "❌ Port $port not healthy"
done
```

---

## Troubleshooting

### Service no inicia

```bash
# Ver logs
docker-compose logs -f

# Verificar que modelo existe
ls -lh models/

# Verificar GPU disponible
nvidia-smi

# Reiniciar servicio
docker-compose restart
```

### CUDA Out of Memory

```bash
# Opción 1: Usar CPU
# En docker-compose.yml:
environment:
  - CUDA_VISIBLE_DEVICES=-1

# Opción 2: Limpiar cache
docker exec container_name python -c "import torch; torch.cuda.empty_cache()"
```

### Port Already in Use

```bash
# Ver qué proceso usa el puerto
lsof -i :8008

# Matar proceso
kill -9 <PID>
```

---

## Production Deployment (AWS EC2)

### Instance Requirements
- **Type**: p3.2xlarge o superior (con GPU)
- **OS**: Ubuntu 20.04+
- **Storage**: 100GB+ (modelos ML)
- **Security Group**: Puertos 8000-8014 abiertos

### Setup Script

```bash
#!/bin/bash
# setup_ml_services.sh

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Clone repository
git clone <repo_url> /opt/soul-gate-ai-models
cd /opt/soul-gate-ai-models

# Download models from S3 (example)
# aws s3 sync s3://bucket/models/ ./models/

# Deploy all services
./scripts/deploy_all.sh
```

---

## Monitoring

### Docker Stats

```bash
docker stats
```

### GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Logs Aggregation

```bash
# Todos los logs
docker-compose logs -f --tail=100

# Filtrar por servicio
docker-compose logs -f espejo-api
```

---

## Backup & Recovery

### Backup Models

```bash
tar -czf models_backup_$(date +%Y%m%d).tar.gz */models/
```

### Restore Service

```bash
cd service_directory
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## CI/CD (Futuro)

GitHub Actions workflow example:

```yaml
name: Deploy ML Services

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to EC2
        run: |
          ssh user@ec2-instance "cd /opt/soul-gate-ai-models && git pull && ./deploy_all.sh"
```

---

**Sigue esta guía para deployment consistente y seguro.**

