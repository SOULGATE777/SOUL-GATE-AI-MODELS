# SG - Facial Recognition Production Pipeline

Complete production-ready facial recognition system with morphological analysis capabilities.

## Project Structure

```
SG/
├── frontal_prod/                   # Frontal facial analysis module
│   └── morfologico/               # Morphological facial analysis
│       ├── app/                   # FastAPI application
│       │   ├── main.py           # API endpoints and startup
│       │   ├── models/           # Model implementations
│       │   │   ├── facial_analysis_pipeline.py
│       │   │   └── anthropometric_detection.py
│       │   └── utils/            # Utility functions
│       │       ├── visualization.py
│       │       └── image_processing.py
│       ├── models/               # Trained model weights
│       │   ├── facial_landmarks_detection_model.pth    (158MB)
│       │   ├── facial_points_detection_model.pth       (158MB)
│       │   └── best_facial_landmark_classifier.pth     (3.6MB)
│       ├── Dockerfile            # Container configuration
│       ├── docker-compose.yml    # Service orchestration
│       ├── requirements.txt      # Python dependencies
│       └── results/              # Generated visualizations
├── .gitattributes               # Git LFS configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Features

### Morphological Facial Analysis
- **3-Model Ensemble Architecture**:
  - Facial landmark detection (Faster R-CNN)
  - Characteristic classification (CNN)
  - Anthropometric point detection
- **GPU Acceleration**: Full CUDA support
- **Beautiful Visualizations**: Modern, clean annotations
- **Production Ready**: Docker containerization
- **RESTful API**: FastAPI with automatic documentation

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit (for GPU support)

### Deployment

```bash
# Clone the repository
git clone https://github.com/quantileMX/SG.git
cd SG

# Navigate to morphological analysis
cd frontal_prod/morfologico

# Deploy with GPU acceleration
docker compose up --build -d

# Check health
curl http://localhost:8000/health
```

### API Documentation
Once deployed, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Complete Facial Analysis
```bash
curl -X POST "http://localhost:8000/analyze-face" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"
```

### Individual Components
- **Facial Landmarks**: `POST /detect-landmarks`
- **Anthropometric Points**: `POST /detect-points`
- **Health Check**: `GET /health`

### Response Format
```json
{
  "facial_landmarks": {
    "count": 16,
    "detections": [...]
  },
  "anthropometric_points": {
    "count": 20,
    "detections": [...]
  },
  "summary": {
    "total_detections": 36,
    "confidence_threshold": 0.5
  },
  "visualization_path": "/app/results/analysis_xxx.jpg"
}
```

## Model Information

### Included Models
1. **Facial Landmarks Detection** (158MB)
   - Architecture: Faster R-CNN ResNet50 FPN
   - Classes: 18 facial regions (eyes, nose, mouth, etc.)

2. **Facial Points Detection** (158MB)
   - Architecture: Faster R-CNN ResNet50 FPN
   - Classes: 13 anthropometric measurement points

3. **Characteristic Classification** (3.6MB)
   - Architecture: Custom CNN
   - Classes: 50 facial characteristics and features

### Model Storage
Large model files (>100MB) are stored using Git LFS for efficient repository management.

## Configuration

### GPU Production Setup
```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### CPU Fallback
```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=-1
# Remove deploy section
```

## Development

### Local Development
```bash
cd frontal_prod/morfologico

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Test with sample image
curl -X POST "http://localhost:8000/analyze-face" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5"

# View generated visualization
# Check results/ directory or access via API
```

## Production Deployment

### System Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ for models and containers
- **CPU**: Multi-core processor for preprocessing

### Scaling
For high-throughput production:
- Deploy multiple container instances
- Use load balancer for request distribution
- Configure GPU memory optimization
- Implement request queuing for batch processing

### Monitoring
```bash
# Container health
docker compose ps

# GPU utilization
nvidia-smi

# API metrics
curl http://localhost:8000/health
```

## Architecture Roadmap

### Current: Frontal Analysis
- ✅ `frontal_prod/morfologico/` - Morphological facial analysis

### Planned Extensions
- 🔄 `frontal_prod/[other_analysis]/` - Additional frontal analysis types
- 🔄 `profile_prod/` - Profile facial analysis
- 🔄 `whole_body_prod/` - Full body analysis
- 🔄 Master orchestration for multi-service deployment

## Support

### Issues & Questions
- Repository: https://github.com/quantileMX/SG
- Documentation: See `/docs` endpoint when API is running

### Model Access
All trained model weights are included in this repository via Git LFS. No additional downloads required.

## License

[Add appropriate license information]

---

**quantileMX** - Advanced AI Solutions
