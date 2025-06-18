# SG - Facial Recognition Production Pipeline

Complete production-ready facial recognition system with morphological and anthropometric analysis capabilities.

## Project Structure

```
SG/
├── frontal_prod/                   # Frontal facial analysis modules
│   ├── morfologico/               # Morphological facial analysis
│   │   ├── app/                   # FastAPI application
│   │   │   ├── main.py           # API endpoints and startup
│   │   │   ├── models/           # Model implementations
│   │   │   │   ├── facial_analysis_pipeline.py
│   │   │   │   └── anthropometric_detection.py
│   │   │   └── utils/            # Utility functions
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/               # Trained model weights
│   │   │   ├── facial_landmarks_detection_model.pth    (158MB)
│   │   │   ├── facial_points_detection_model.pth       (158MB)
│   │   │   └── best_facial_landmark_classifier.pth     (3.6MB)
│   │   ├── Dockerfile            # Container configuration
│   │   ├── docker-compose.yml    # Service orchestration
│   │   ├── requirements.txt      # Python dependencies
│   │   └── results/              # Generated visualizations
│   └── antropometrico/           # Anthropometric facial analysis
│       ├── app/                  # FastAPI application
│       │   ├── main.py          # API endpoints and startup
│       │   ├── models/          # Model implementations
│       │   │   └── anthropometric_pipeline.py
│       │   └── utils/           # Utility functions
│       │       ├── visualization.py
│       │       └── image_processing.py
│       ├── models/              # Trained model weights
│       │   ├── facial_points_detection_model.pth        (158MB)
│       │   └── shape_predictor_68_face_landmarks.dat    (95MB)
│       ├── Dockerfile           # Container configuration
│       ├── docker-compose.yml   # Service orchestration
│       ├── requirements.txt     # Python dependencies
│       └── results/             # Generated visualizations
├── .gitattributes              # Git LFS configuration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Features

### Morphological Facial Analysis (Port 8000)
- **3-Model Ensemble Architecture**:
  - Facial landmark detection (Faster R-CNN)
  - Characteristic classification (CNN)
  - Anthropometric point detection
- **GPU Acceleration**: Full CUDA support
- **Beautiful Visualizations**: Modern, clean annotations
- **Production Ready**: Docker containerization
- **RESTful API**: FastAPI with automatic documentation

### Anthropometric Facial Analysis (Port 8001)
- **Hybrid Detection System**:
  - 68 standard dlib facial landmarks
  - Custom Faster R-CNN for 3 key anthropometric points
  - Enhanced facial proportion calculations
- **Advanced Measurements**:
  - Facial thirds analysis with model-enhanced precision
  - Eye relationship analysis
  - Mouth-pupil proportions
  - Eyebrow slope calculations
- **Model Integration**: Uses custom trained points to replace inferred measurements
- **Independent Service**: Completely separate from morfologico module

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit (for GPU support)

### Deploy Morfologico Module

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

### Deploy Antropometrico Module

```bash
# Navigate to anthropometric analysis
cd frontal_prod/antropometrico

# Deploy with GPU acceleration
docker compose up --build -d

# Check health
curl http://localhost:8001/health
```

### Deploy Both Modules

```bash
# Deploy both services independently
cd frontal_prod/morfologico
docker compose up --build -d

cd ../antropometrico
docker compose up --build -d

# Both services now running:
# - Morfologico: http://localhost:8000
# - Antropometrico: http://localhost:8001
```

## API Documentation

### Morfologico Module
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Antropometrico Module
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## API Endpoints

### Morfologico Module (Port 8000)

#### Complete Facial Analysis
```bash
curl -X POST "http://localhost:8000/analyze-face" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"
```

#### Individual Components
- **Facial Landmarks**: `POST /detect-landmarks`
- **Anthropometric Points**: `POST /detect-points`
- **Health Check**: `GET /health`

### Antropometrico Module (Port 8001)

#### Complete Anthropometric Analysis
```bash
curl -X POST "http://localhost:8001/analyze-anthropometric" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"
```

#### Individual Components
- **Facial Landmarks**: `POST /detect-landmarks`
- **Model Points**: `POST /detect-points`
- **Health Check**: `GET /health`

## Response Formats

### Morfologico Response
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

### Antropometrico Response
```json
{
  "facial_landmarks": {
    "count": 68,
    "extended_points": 73
  },
  "model_predictions": {
    "1": [243, 154],
    "2": [243, 131],
    "3": [149, 509]
  },
  "proportions": {
    "distance_69_68_proportion": 0.324,
    "distance_68_34_proportion": 0.331,
    "distance_34_9_proportion": 0.345,
    "eye_distance_proportion": 0.469,
    "mouth_to_eye_proportion": 0.615
  },
  "analysis_summary": {
    "facial_thirds": {
      "primer_tercio": "tercio superior standard",
      "segundo_tercio": "tercio medio standard",
      "tercer_tercio": "tercio inferior standard"
    },
    "model_integration": {
      "point_2_used": true,
      "point_3_used": true,
      "point_1_detected": true
    }
  },
  "visualization_path": "/app/results/anthropometric_xxx.jpg"
}
```

## Model Information

### Morfologico Models
1. **Facial Landmarks Detection** (158MB)
   - Architecture: Faster R-CNN ResNet50 FPN
   - Classes: 18 facial regions (eyes, nose, mouth, etc.)

2. **Facial Points Detection** (158MB)
   - Architecture: Faster R-CNN ResNet50 FPN
   - Classes: 13 anthropometric measurement points

3. **Characteristic Classification** (3.6MB)
   - Architecture: Custom CNN
   - Classes: 50 facial characteristics and features

### Antropometrico Models
1. **dlib Facial Landmarks** (95MB)
   - Pre-trained 68-point facial landmark detector
   - Standard facial feature detection

2. **Custom Facial Points Detection** (158MB)
   - Architecture: Faster R-CNN ResNet50 FPN
   - Classes: 3 key anthropometric points (between eyebrows, top of head, reference point)
   - Enhances dlib landmarks with model-predicted precision points

## Analysis Output Labels

### Antropometrico Facial Thirds Classification
- **tercio superior largo/corto/standard** - First third proportion analysis
- **tercio medio largo/corto/standard** - Second third proportion analysis  
- **tercio inferior largo/corto/standard** - Third third proportion analysis

### Eye Relationship Analysis
- **Cercanos/Standard/Lejanos** - Internal eye spacing classification

### Mouth-Pupil Relationship
- **boca grande/pequeña/estándar en relación a las pupilas** - Mouth size relative to eye spacing

## Configuration

### GPU Production Setup (Both Modules)
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

### Local Development - Morfologico
```bash
cd frontal_prod/morfologico
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Local Development - Antropometrico
```bash
cd frontal_prod/antropometrico
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## Production Deployment

### System Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **RAM**: 12GB+ system memory (for both modules)
- **Storage**: 4GB+ for models and containers
- **CPU**: Multi-core processor for preprocessing

### Independent Scaling
Each module can be scaled independently:
- Deploy multiple instances of either module
- Use load balancer for request distribution
- Configure GPU memory optimization per module
- Implement request queuing for batch processing

### Monitoring
```bash
# Morfologico health
curl http://localhost:8000/health

# Antropometrico health  
curl http://localhost:8001/health

# Container status
docker compose ps

# GPU utilization
nvidia-smi
```

## Architecture Roadmap

### Current: Frontal Analysis
- ✅ `frontal_prod/morfologico/` - Morphological facial analysis
- ✅ `frontal_prod/antropometrico/` - Anthropometric facial analysis

### Planned Extensions
- 🔄 `frontal_prod/[other_analysis]/` - Additional frontal analysis types
- 🔄 `profile_prod/` - Profile facial analysis
- 🔄 `whole_body_prod/` - Full body analysis
- 🔄 Master orchestration for multi-service deployment

## Support

### Issues & Questions
- Repository: https://github.com/quantileMX/SG
- Documentation: See `/docs` endpoint when APIs are running

### Model Access
All trained model weights are included in this repository. Large files use Git LFS for efficient repository management.

## License

[Add appropriate license information]

---

**quantileMX** - Advanced AI Solutions
