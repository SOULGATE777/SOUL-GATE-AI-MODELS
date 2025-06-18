# IA-MODELS - SG Production Pipeline

Complete production-ready AI models pipeline for image analysis.

## Project Structure

```
SG_prod/
├── frontal_prod/           # Facial recognition API
│   ├── app/               # FastAPI application
│   ├── models/            # Trained model files (.pth)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── [future components]     # profile_prod, whole_body_prod
```

## Current Components

### Frontal Facial Recognition API

**Location:** `frontal_prod/`

**Features:**
- 3-Model ensemble (landmark detection + classification + anthropometric points)
- GPU acceleration (CUDA support)
- Beautiful visualizations with modern styling
- RESTful API with automatic documentation
- Production-ready Docker containerization

**Quick Start:**
```bash
cd frontal_prod
docker compose up --build
```

**API Endpoints:**
- Complete Analysis: `POST /analyze-face`
- Facial Landmarks: `POST /detect-landmarks`
- Anthropometric Points: `POST /detect-points`
- Documentation: http://localhost:8000/docs

## Model Files

**IMPORTANT:** Model files (.pth) are not included due to size constraints.

Required models for frontal_prod:
- `facial_landmarks_detection_model.pth`
- `best_facial_landmark_classifier.pth`
- `facial_points_detection_model.pth`

Place these files in `frontal_prod/models/` directory before deployment.

## Production Deployment

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers
- NVIDIA Container Toolkit

### Deploy Frontal Recognition API
```bash
cd frontal_prod
docker compose up --build -d
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Future Components

- **profile_prod**: Profile facial analysis
- **whole_body_prod**: Full body analysis
- **Master orchestration**: Multi-service deployment

## GPU Configuration

**Production (GPU enabled):**
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
```

**CPU fallback:**
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=-1
```

## Testing

### Test Frontal API
```bash
# Test with image file
curl -X POST "http://localhost:8000/analyze-face" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"
```

### View Results
Access visualization at: `http://localhost:8000/visualization/[generated_filename].jpg`

## Architecture

Each component runs as an independent microservice:
- **Scalable**: Individual scaling per service type
- **Maintainable**: Independent model updates
- **Resource efficient**: Shared GPU resources
- **Production ready**: Complete Docker orchestration

## Support

For deployment assistance or model file access, contact the development team.
