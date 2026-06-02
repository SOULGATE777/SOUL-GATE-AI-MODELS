# Troubleshooting - SOUL-GATE-AI-MODELS

**Last Updated**: 2025-12-12

---

## Common Issues & Solutions

### 1. Service Won't Start

#### Symptom
```
docker-compose up
ERROR: Cannot start service...
```

#### Solutions

```bash
# Check logs
docker-compose logs -f

# Verify model files exist
ls -lh models/

# Verify GPU
nvidia-smi

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

### 2. CUDA Out of Memory

#### Symptom
```
RuntimeError: CUDA out of memory
```

#### Solutions

**Opción 1**: Usar CPU
```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=-1
```

**Opción 2**: Reducir batch size (en código)

**Opción 3**: Limpiar cache
```python
torch.cuda.empty_cache()
```

**Opción 4**: Usar GPU más grande (AWS p3.xlarge → p3.2xlarge)

---

### 3. Port Already in Use

#### Symptom
```
Error: bind: address already in use
```

#### Solutions

```bash
# Ver qué usa el puerto
lsof -i :8008

# Matar proceso
kill -9 <PID>

# O cambiar puerto en docker-compose.yml
ports:
  - "8009:8008"  # External:Internal
```

---

### 4. Model Not Found

#### Symptom
```
FileNotFoundError: Model not found at /app/models/model.pth
```

#### Solutions

```bash
# Verificar que modelo existe
ls -lh models/

# Descargar modelo (ejemplo)
# aws s3 cp s3://bucket/models/model.pth models/

# Verificar permisos
chmod 644 models/*.pth
```

---

### 5. Health Check Failing

#### Symptom
```
service is unhealthy
```

#### Solutions

```bash
# Check logs
docker-compose logs -f

# Manual health check
curl http://localhost:8008/health

# Verificar que servicio responde
docker exec container_name curl localhost:8008/health

# Verificar puerto correcto en Dockerfile
```

---

### 6. Slow Inference

#### Symptom
Inference toma >10s por imagen

#### Solutions

1. **Verificar GPU usage**: `nvidia-smi` (debería mostrar uso)
2. **Verificar device**: Logs deben decir "Using device: cuda"
3. **Usar batch processing**: Más rápido que una por una
4. **Verificar que model.eval()** está llamado

---

### 7. Import Errors

#### Symptom
```
ModuleNotFoundError: No module named 'X'
```

#### Solutions

```bash
# Verificar requirements.txt tiene la librería
cat requirements.txt | grep X

# Rebuild container
docker-compose build --no-cache

# O instalar en container running
docker exec container_name pip install X
```

---

### 8. GPU Not Detected

#### Symptom
```
Using device: cpu
```

#### Solutions

```bash
# Verificar NVIDIA drivers
nvidia-smi

# Verificar Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Instalar NVIDIA Container Toolkit
# (ver deployment-guide.md)

# Verificar docker-compose.yml tiene:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          capabilities: [gpu]
```

---

### 9. Permission Denied

#### Symptom
```
PermissionError: [Errno 13] Permission denied: '/app/results/...'
```

#### Solutions

```bash
# Verificar permisos de carpetas
ls -la

# Arreglar permisos
chmod 755 results/
chmod 755 models/
```

---

### 10. Container Exits Immediately

#### Symptom
```
Container exits with code 1
```

#### Solutions

```bash
# Ver logs completos
docker-compose logs

# Ver error específico
docker-compose logs | grep -i error

# Common causes:
# - Model file missing
# - Import error
# - Syntax error in Python code
```

---

## Debugging Tools

### Docker Commands

```bash
# Logs en tiempo real
docker-compose logs -f

# Entrar al container
docker exec -it container_name bash

# Ver recursos usados
docker stats

# Reiniciar servicio específico
docker-compose restart service_name
```

### Python Debugging

```bash
# Dentro del container
docker exec -it container_name python

>>> import torch
>>> torch.cuda.is_available()  # Should be True
>>> torch.cuda.get_device_name(0)
```

---

## Performance Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# Docker stats
docker stats

# Logs con timestamp
docker-compose logs -f --timestamps
```

---

## Getting Help

1. **Check logs first**: `docker-compose logs -f`
2. **Verify health**: `curl http://localhost:PORT/health`
3. **Check GPU**: `nvidia-smi`
4. **Rebuild**: `docker-compose build --no-cache`
5. **Consult docs**: `@docs/deployment-guide.md`

---

**Este documento se actualiza con nuevos issues encontrados.**

