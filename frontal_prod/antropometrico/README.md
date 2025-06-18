# Antropometrico Analysis Module

Advanced anthropometric facial analysis with custom point detection using ensemble of dlib landmarks and trained Faster R-CNN model.

## Features

- **Hybrid Detection**: Combines dlib 68-point landmarks with custom trained model
- **Enhanced Measurements**: Precision facial proportion analysis
- **Model Integration**: Uses custom Faster R-CNN for key anthropometric points
- **Production Ready**: Docker containerization with GPU support
- **Independent Service**: Runs on port 8001, completely separate from morfologico

## Architecture

### Model Ensemble
1. **dlib Facial Landmarks**: 68 standard facial landmarks
2. **Custom Faster R-CNN**: Detects 3 key anthropometric points
   - Point 1: Specialized detection point
   - Point 2: Between eyebrows (replaces inferred point 68)
   - Point 3: Top of head (replaces inferred point 69)

### Extended Point System
- Points 0-67: Standard dlib landmarks
- Point 68: Between eyebrows (model-enhanced)
- Point 69: Top of head (model-enhanced)
- Point 70-71: Calculated pupil centers
- Point 72: Model point 1 (when detected)

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit for GPU support

### Required Model Files
Place these files in the `models/` directory:
- `facial_points_detection_model.pth` - Your trained Faster R-CNN model
- `shape_predictor_68_face_landmarks.dat` - dlib landmark predictor

### Deployment

```bash
# Navigate to antropometrico directory
cd /home/carlos/Documents/SG_prod/frontal_prod/antropometrico/

# Build and start the service
docker compose up --build -d

# Check service health
curl http://localhost:8001/health
```

### CPU-Only Deployment
If you don't have GPU support, modify `docker-compose.yml`:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=-1  # Force CPU usage
# Remove the entire deploy section
```

## API Endpoints

### Complete Analysis
```bash
curl -X POST "http://localhost:8001/analyze-anthropometric" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"
```

### Individual Components
- **Landmarks Only**: `POST /detect-landmarks`
- **Model Points Only**: `POST /detect-points`
- **Health Check**: `GET /health`

### API Documentation
Access interactive documentation:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Analysis Output Labels

### Facial Thirds Classification
- **tercio superior largo** - First third proportion > 0.38 (forehead too tall)
- **tercio superior corto** - First third proportion < 0.27 (forehead too short)
- **tercio superior standard** - First third proportion 0.27-0.38 (normal forehead)
- **tercio medio largo** - Second third proportion > 0.38 (midface too long)
- **tercio medio corto** - Second third proportion < 0.27 (midface too short)
- **tercio medio standard** - Second third proportion 0.27-0.38 (normal midface)
- **tercio inferior largo** - Third third proportion > 0.38 (lower face too long)
- **tercio inferior corto** - Third third proportion < 0.27 (lower face too short)
- **tercio inferior standard** - Third third proportion 0.27-0.38 (normal lower face)

### Eye Relationship Analysis
- **Cercanos** - Internal eye proportion < 0.3 (eyes too close together)
- **Standard** - Internal eye proportion 0.3-0.37 (normal eye spacing)
- **Lejanos** - Internal eye proportion > 0.37 (eyes too far apart)

### Mouth-Pupil Relationship
- **boca grande en relación a las pupilas** - Mouth/pupil ratio > 1.0 (mouth wide relative to eye spacing)
- **boca pequeña en relación a las pupilas** - Mouth/pupil ratio < 0.7 (mouth narrow relative to eye spacing)
- **relación boca-pupilas estándar** - Mouth/pupil ratio 0.7-1.0 (normal mouth-eye proportion)

### Eyebrow Slope Analysis (Portions 1 & 2)
- **portion_X - Ascendente** - Angle 5-75° (eyebrow section slopes upward)
- **portion_X - Recto** - Angle -1 to 5° (eyebrow section is straight/horizontal)
- **portion_X - Descendente** - Angle ≤ 0° (eyebrow section slopes downward)

### Eyebrow Slope Analysis (Portion 3 - Tail)
- **portion_3 - Descendente** - Angle > 75° (eyebrow tail drops sharply)
- **portion_3 - Normal** - Angle 10-75° (normal eyebrow tail curve)
- **portion_3 - Ascendente** - Angle < 10° (eyebrow tail curves upward)

### Model Integration Status
- **✓ Punto 2: Sí** - Custom model successfully detected between-eyebrows point
- **✓ Punto 3: Sí** - Custom model successfully detected top-of-head point
- **✓ Punto 1: Sí** - Custom model detected additional reference point
