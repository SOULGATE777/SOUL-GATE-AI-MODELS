# SG - Facial Recognition Production Pipeline

Complete production-ready facial recognition system with morphological, anthropometric, and validation analysis capabilities.

## Project Structure

```
SG/
├── frontal_prod/                   
│   ├── validacion/                 
│   │   ├── app/                   
│   │   │   ├── main.py           
│   │   │   ├── models/           
│   │   │   │   └── facial_validation_pipeline.py
│   │   │   └── utils/            
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/               
│   │   │   └── best.pt                              
│   │   ├── Dockerfile            
│   │   ├── docker-compose.yml     
│   │   ├── requirements.txt      
│   │   └── results/              
│   ├── morfologico/               
│   │   ├── app/                   
│   │   │   ├── main.py           
│   │   │   ├── models/           
│   │   │   │   ├── facial_analysis_pipeline.py
│   │   │   │   └── anthropometric_detection.py
│   │   │   └── utils/            
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/               
│   │   │   ├── facial_landmarks_detection_model.pth    (158MB)
│   │   │   ├── facial_points_detection_model.pth       (158MB)
│   │   │   └── best_facial_landmark_classifier.pth     (3.6MB)
│   │   ├── Dockerfile            
│   │   ├── docker-compose.yml     
│   │   ├── requirements.txt      
│   │   └── results/              
│   └── antropometrico/           
│       ├── app/                  
│       │   ├── main.py          
│       │   ├── models/          
│       │   │   └── anthropometric_pipeline.py
│       │   └── utils/           
│       │       ├── visualization.py
│       │       └── image_processing.py
│       ├── models/              
│       │   ├── facial_points_detection_model.pth        (158MB)
│       │   └── shape_predictor_68_face_landmarks.dat    (95MB)
│       ├── Dockerfile           
│       ├── docker-compose.yml   
│       ├── requirements.txt     
│       └── results/             
├── .gitattributes              
├── .gitignore                  
└── README.md                   
```

## Features

### Facial Feature Validation (Port 8002) - NEW
- **YOLO-Based Detection**: Custom trained YOLOv8 model for 17 facial feature classes
- **Feature Categories**:
  - Hair Coverage (cabello_tapando_i, cabello_tapando_derecho, cabello_tapando_central)
  - Facial Hair (barba, bc_bigote)
  - Facial Expression (bc_abierta, bc_sonriendo)
  - Accessories (piercing, lentes, objeto_frente)
  - Body Modifications (tatuaje)
  - Head Characteristics (calvo)
  - Eye Features (l_ej_i, l_ej_d)
  - Facial Points (p_d_g_iz, p_d_g_d, p_d_v)
- **Image Quality Assessment**: Automatic evaluation of image suitability for analysis
- **Smart Recommendations**: AI-powered suggestions for better image quality
- **Beautiful Visualizations**: Color-coded detection boxes with category grouping
- **GPU Acceleration**: Full CUDA support with PyTorch backend
- **Production Ready**: Docker containerization with health monitoring

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

### Deploy All Modules

```bash
# Clone the repository
git clone https://github.com/quantileMX/SG.git
cd SG/frontal_prod

# Deploy Validacion Module (Port 8002)
cd validacion
docker compose up --build -d
cd ..

# Deploy Morfologico Module (Port 8000)
cd morfologico
docker compose up --build -d
cd ..

# Deploy Antropometrico Module (Port 8001)
cd antropometrico
docker compose up --build -d
cd ..

# Check all services
curl http://localhost:8000/health  # Morfologico
curl http://localhost:8001/health  # Antropometrico
curl http://localhost:8002/health  # Validacion
```

### Deploy Individual Modules

#### Validacion Module
```bash
cd frontal_prod/validacion
docker compose up --build -d
curl http://localhost:8002/health
```

#### Morfologico Module
```bash
cd frontal_prod/morfologico
docker compose up --build -d
curl http://localhost:8000/health
```

#### Antropometrico Module
```bash
cd frontal_prod/antropometrico
docker compose up --build -d
curl http://localhost:8001/health
```

## API Documentation

### Validacion Module (Port 8002)
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

### Morfologico Module (Port 8000)
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Antropometrico Module (Port 8001)
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## API Endpoints

### Validacion Module (Port 8002)

#### Complete Facial Feature Validation
```bash
curl -X POST "http://localhost:8002/analyze-validation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.20" \
  -F "include_visualization=true"
```

#### Feature Detection Only
```bash
curl -X POST "http://localhost:8002/detect-features" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.20"
```

#### Health Check
```bash
curl http://localhost:8002/health
```

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

### Validacion Response
```json
{
  "detection_results": {
    "total_detections": 5,
    "detections": [...],
    "class_counts": {"lentes": 1, "barba": 1},
    "average_confidence": 0.756,
    "high_confidence_count": 3
  },
  "feature_analysis": {
    "categorized_features": {...},
    "feature_summary": {...}
  },
  "validation_summary": {
    "image_suitable": true,
    "suitability_score": 85,
    "quality_issues": [],
    "recommendations": [...]
  },
  "visualization_path": "/app/results/validation_xxx.jpg",
  "visualization_url": "/visualization/validation_xxx.jpg"
}
```

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

### Validacion Models
1. **YOLO Facial Feature Detection** (Custom size)
   - Architecture: YOLOv8 Custom Trained
   - Classes: 17 facial features and characteristics
   - Features: piercing, cabello_tapando_*, tatuaje, barba, facial_points, lentes, etc.

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

### Validacion Feature Categories
- **Hair Coverage**: cabello_tapando_i, cabello_tapando_derecho, cabello_tapando_central
- **Facial Hair**: barba, bc_bigote  
- **Facial Expression**: bc_abierta, bc_sonriendo
- **Accessories**: piercing, lentes, objeto_frente
- **Body Modifications**: tatuaje
- **Head Characteristics**: calvo
- **Eye Features**: l_ej_i, l_ej_d
- **Facial Points**: p_d_g_iz, p_d_g_d, p_d_v

### Antropometrico Facial Thirds Classification
- **tercio superior largo/corto/standard** - First third proportion analysis
- **tercio medio largo/corto/standard** - Second third proportion analysis  
- **tercio inferior largo/corto/standard** - Third third proportion analysis

### Eye Relationship Analysis
- **Cercanos/Standard/Lejanos** - Internal eye spacing classification

### Mouth-Pupil Relationship
- **boca grande/pequeña/estándar en relación a las pupilas** - Mouth size relative to eye spacing

## Configuration

### GPU Production Setup (All Modules)
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

### Local Development - Validacion
```bash
cd frontal_prod/validacion
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
```

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
- **RAM**: 16GB+ system memory (for all three modules)
- **Storage**: 6GB+ for models and containers
- **CPU**: Multi-core processor for preprocessing

### Independent Scaling
Each module can be scaled independently:
- Deploy multiple instances of any module
- Use load balancer for request distribution
- Configure GPU memory optimization per module
- Implement request queuing for batch processing

### Monitoring
```bash
# All modules health
curl http://localhost:8000/health  # Morfologico
curl http://localhost:8001/health  # Antropometrico  
curl http://localhost:8002/health  # Validacion

# Container status
docker ps

# GPU utilization
nvidia-smi
```

## Architecture Roadmap

### Current: Frontal Analysis
- ✅ `frontal_prod/validacion/` - Facial feature validation and quality assessment
- ✅ `frontal_prod/morfologico/` - Morphological facial analysis
- ✅ `frontal_prod/antropometrico/` - Anthropometric facial analysis

### Planned Extensions
- 🔄 `frontal_prod/[other_analysis]/` - Additional frontal analysis types
- 🔄 `profile_prod/` - Profile facial analysis
- 🔄 `whole_body_prod/` - Full body analysis
- 🔄 Master orchestration for multi-service deployment

## Usage Workflow

### Recommended Analysis Pipeline
1. **Validacion** (Port 8002): First validate image quality and detect potential issues
2. **Morfologico** (Port 8000): Perform morphological facial analysis if image is suitable
3. **Antropometrico** (Port 8001): Conduct detailed anthropometric measurements

### Quality-First Approach
The **Validacion** module serves as a quality gate, identifying:
- Hair covering facial features
- Problematic accessories (glasses, objects)
- Poor lighting or image quality
- Unsuitable facial expressions
- Recommendations for better image capture

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
