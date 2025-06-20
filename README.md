# SG - Facial Recognition Production Pipeline
Complete production-ready facial recognition system with morphological, anthropometric, and validation analysis capabilities for both frontal and profile views.
## Project Structure
```
SG_prod/
├── frontal_prod/                   
│   ├── validacion/                 (Port 8002)
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
│   ├── morfologico/               (Port 8000)
│   │   ├── app/                   
│   │   │   ├── main.py           
│   │   │   ├── models/           
│   │   │   │   ├── facial_analysis_pipeline.py
│   │   │   │   └── anthropometric_detection.py
│   │   │   └── utils/            
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/               
│   │   │   ├── facial_landmarks_detection_model.pth
│   │   │   ├── facial_points_detection_model.pth
│   │   │   └── best_facial_landmark_classifier.pth
│   │   ├── Dockerfile            
│   │   ├── docker-compose.yml     
│   │   ├── requirements.txt      
│   │   └── results/              
│   └── antropometrico/           (Port 8001)
│       ├── app/                  
│       │   ├── main.py          
│       │   ├── models/          
│       │   │   └── anthropometric_pipeline.py
│       │   └── utils/           
│       │       ├── visualization.py
│       │       └── image_processing.py
│       ├── models/              
│       │   ├── facial_points_detection_model.pth
│       │   └── shape_predictor_68_face_landmarks.dat
│       ├── Dockerfile           
│       ├── docker-compose.yml   
│       ├── requirements.txt     
│       └── results/             
├── profile_prod/
│   ├── validacion/                 (Port 8005)
│   │   ├── app/                   
│   │   │   ├── main.py           
│   │   │   ├── models/           
│   │   │   │   └── profile_validation_pipeline.py
│   │   │   └── utils/            
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/               
│   │   │   └── profile_best.pt
│   │   ├── Dockerfile            
│   │   ├── docker-compose.yml     
│   │   ├── requirements.txt      
│   │   └── results/              
│   ├── morfologico/               (Port 8003)
│   │   ├── app/                   
│   │   │   ├── main.py           
│   │   │   ├── models/           
│   │   │   │   ├── profile_analysis_pipeline.py
│   │   │   │   └── profile_detection.py
│   │   │   └── utils/            
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/               
│   │   │   ├── bbox_detection_model.pth
│   │   │   ├── profile_landmark_classifier_final.pth
│   │   │   └── profile_aware_point_detection_model.pth
│   │   ├── Dockerfile            
│   │   ├── docker-compose.yml     
│   │   ├── requirements.txt      
│   │   └── results/              
│   └── antropometrico/           (Port 8004)
│       ├── app/                  
│       │   ├── main.py          
│       │   ├── models/          
│       │   │   └── profile_anthropometric_pipeline.py
│       │   └── utils/           
│       │       ├── visualization.py
│       │       └── image_processing.py
│       ├── models/              
│       │   └── profile_aware_point_detection_model.pth
│       ├── Dockerfile           
│       ├── docker-compose.yml   
│       ├── requirements.txt     
│       └── results/             
├── .gitattributes              
├── .gitignore                  
└── README.md                   
```
## Features
### Frontal Analysis (Ports 8000-8002)
#### Facial Feature Validation (Port 8002) 
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
#### Morphological Facial Analysis (Port 8000)
- **3-Model Ensemble Architecture**:
  - Facial landmark detection (Faster R-CNN)
  - Characteristic classification (CNN)
  - Anthropometric point detection
- **GPU Acceleration**: Full CUDA support
- **Beautiful Visualizations**: Modern, clean annotations
- **Production Ready**: Docker containerization
#### Anthropometric Facial Analysis (Port 8001)
- **Hybrid Detection System**:
  - 68 standard dlib facial landmarks
  - Custom Faster R-CNN for 3 key anthropometric points
  - Enhanced facial proportion calculations
- **Advanced Measurements**:
  - Facial thirds analysis with model-enhanced precision
  - Eye relationship analysis
  - Mouth-pupil proportions
  - Eyebrow slope calculations
### Profile Analysis (Ports 8003-8005)
#### Profile Morphological Analysis (Port 8003) ✅ **COMPLETE**
- **3-Model Ensemble Architecture**:
  - Profile bounding box detection (Faster R-CNN) with 8 facial feature classes
  - Profile landmark classification (CNN) with 18 morphological tags
  - Profile anthropometric point detection with 80+ point classes
- **Advanced Filtering System**:
  - Duplicate bbox removal (keeps highest confidence per class)
  - Spurious point filtering by suffix majority (_i vs _d)
  - Smart profile side inference from detected points
- **Intelligent Analysis**:
  - Automatic left/right profile determination
  - Excluded problematic classes (hair coverage, objects)
  - Adaptive confidence thresholds
- **GPU Acceleration**: Full CUDA support with CPU fallback
- **Clean API Responses**: No neural network profile predictions, inference from actual detected points
#### Profile Anthropometric Analysis (Port 8004) ✅ **ACTIVE**
- **Profile-Specific Point Detection**: Custom trained model for profile anthropometric points
- **Advanced Profile Measurements**:
  - Nasal profile analysis (protrusion, angle, classification)
  - Facial thirds in profile view
  - Mandible classification (Sanguinea, Bilosa, Nerviosa, Linfática)
  - Angular measurements (nose tip, forehead, chin angles)
  - Ear morphology analysis (width, trago-antitrago proportions)
- **Side Detection**: Automatic left/right profile determination with vector analysis
- **Spurious Prediction Filtering**: Intelligent filtering of minority-side predictions
- **Profile-Specific Visualizations**: Detailed analysis plots with measurement overlays
- **GPU Acceleration**: Full CUDA support with CPU fallback
#### Profile Validation (Port 8005) 🔄 **PLANNED**
- Profile angle validation (true profile vs 3/4 view)
- Profile-specific quality assessment
- Hair coverage detection in profile
- Profile accessories validation
- Lighting and background analysis
## Quick Start
### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit (for GPU support)
### Deploy All Active Modules
```bash
# Clone the repository
git clone https://github.com/quantileMX/SG_prod.git
cd SG_prod
# Deploy Frontal Modules
cd frontal_prod
# Frontal Validacion (Port 8002)
cd validacion && docker compose up --build -d && cd ..
# Frontal Morfologico (Port 8000)
cd morfologico && docker compose up --build -d && cd ..
# Frontal Antropometrico (Port 8001)
cd antropometrico && docker compose up --build -d && cd ..
# Deploy Profile Modules
cd ../profile_prod
# Profile Morfologico (Port 8003) ✅ COMPLETE
cd morfologico && docker compose up --build -d && cd ..
# Profile Antropometrico (Port 8004) ✅ ACTIVE
cd antropometrico && docker compose up --build -d && cd ..
# Check all active services
curl http://localhost:8000/health  # Frontal Morfologico
curl http://localhost:8001/health  # Frontal Antropometrico
curl http://localhost:8002/health  # Frontal Validacion
curl http://localhost:8003/health  # Profile Morfologico ✅
curl http://localhost:8004/health  # Profile Antropometrico ✅
```
### Deploy Individual Modules
#### Profile Morfologico Module ✅
```bash
cd profile_prod/morfologico
docker compose up --build -d
curl http://localhost:8003/health
```
#### Profile Antropometrico Module ✅
```bash
cd profile_prod/antropometrico
docker compose up --build -d
curl http://localhost:8004/health
```
## API Documentation
### Active Services
- **Frontal Validacion (Port 8002)**: http://localhost:8002/docs
- **Frontal Morfologico (Port 8000)**: http://localhost:8000/docs
- **Frontal Antropometrico (Port 8001)**: http://localhost:8001/docs
- **Profile Morfologico (Port 8003)**: http://localhost:8003/docs ✅ **COMPLETE**
- **Profile Antropometrico (Port 8004)**: http://localhost:8004/docs ✅
### Planned Services
- **Profile Validacion (Port 8005)**: *Coming Soon*
## API Endpoints
### Profile Morfologico Module (Port 8003) ✅
#### Complete Profile Morphological Analysis
```bash
curl -X POST "http://localhost:8003/analyze-profile-morphological" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "bbox_threshold=0.5" \
  -F "include_visualization=true"
```
#### Profile Object Detection Only
```bash
curl -X POST "http://localhost:8003/detect-profile-objects" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5"
```
#### Profile Landmark Classification
```bash
curl -X POST "http://localhost:8003/classify-profile-landmarks" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "bbox_threshold=0.5"
```
#### Profile Point Detection
```bash
curl -X POST "http://localhost:8003/detect-profile-points" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.15"
```
#### Health Check
```bash
curl http://localhost:8003/health
```
### Profile Antropometrico Module (Port 8004) ✅
#### Complete Profile Anthropometric Analysis
```bash
curl -X POST "http://localhost:8004/analyze-profile-anthropometric" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.15" \
  -F "include_visualization=true"
```
#### Profile Point Detection Only
```bash
curl -X POST "http://localhost:8004/detect-profile-points" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.15"
```
#### Health Check
```bash
curl http://localhost:8004/health
```
## Response Formats
### Profile Morfologico Response ✅
```json
{
  "analysis_id": "uuid-string",
  "morphological_analysis": {
    "total_detected_objects": 5,
    "total_classified_landmarks": 4,
    "total_anthropometric_points": 12,
    "bbox_threshold_used": 0.5,
    "profile_side": "left"
  },
  "detected_objects": [
    {
      "class": "nariz",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.856
    }
  ],
  "landmark_classifications": [
    {
      "original_class": "nariz",
      "classified_tag": "nariz_aguileña",
      "bbox": [x1, y1, x2, y2],
      "tag_confidence": 0.923,
      "bbox_confidence": 0.856
    }
  ],
  "anthropometric_points": [
    {
      "class": "nasion_i",
      "coordinates": [x, y],
      "confidence": 0.789
    }
  ],
  "analysis_summary": {
    "timestamp": "2025-06-19T...",
    "processing_successful": true,
    "filtering_applied": {
      "duplicate_bbox_removal": true,
      "spurious_point_filtering": true,
      "profile_prediction_removed": true
    }
  },
  "visualization_path": "/app/results/profile_morphological_xxx.png",
  "visualization_url": "/visualization/profile_morphological_xxx.png"
}
```
### Profile Antropometrico Response ✅
```json
{
  "analysis_id": "uuid-string",
  "profile_analysis": {
    "profile_side": "left|right|unknown",
    "total_detected_points": 15,
    "filtered_points": 12,
    "anthropometric_points": [
      {
        "class": "24",
        "coordinates": [x, y],
        "confidence": 0.856
      }
    ]
  },
  "anthropometric_measurements": {
    "reference_distance": 245.67,
    "nose_classification": "nariz normal",
    "nose_normalized": 0.152,
    "tercio_superior_normalized": 0.334,
    "tercio_medio_normalized": 0.331,
    "tercio_inferior_normalized": 0.335,
    "mandibula_classification": "Mandibula Bilosa",
    "nose_tip_angle": 12.5,
    "nose_tip_classification": "punta de nariz promedio",
    "ear_width": 78.45,
    "trago_antitrago_proportion": 0.245
  },
  "analysis_summary": {
    "confidence_threshold": 0.15,
    "has_measurements": true,
    "profile_determination": "left"
  },
  "visualization_path": "/app/results/profile_anthropometric_xxx.png",
  "visualization_url": "/visualization/profile_anthropometric_xxx.png"
}
```
## Model Information
### Profile Models ✅
#### Profile Morphological Analysis (3-Model Ensemble)
- **Bbox Detection Model**: Faster R-CNN for 8 profile feature classes
- **Landmark Classifier**: CNN for 18 morphological classification tags
- **Point Detection Model**: Custom ResNet50-based with attention for 80+ anthropometric points
- **Features**: Duplicate filtering, spurious point removal, profile side inference
- **Input Size**: 224x224 pixels
- **Output**: Clean filtered detections with smart side determination
#### Profile Anthropometric Point Detection (Custom)
- **Architecture**: Custom ResNet50-based model with attention mechanisms
- **Classes**: 25+ profile anthropometric points with left/right variants
- **Features**: Profile-aware training, side detection, spurious prediction filtering
- **Input Size**: 224x224 pixels
- **Output**: Heatmaps + profile classification logits
### Profile Analysis Classifications
#### Profile Morphological Tags (18 Categories)
- **Nasal Types**: nariz_aguileña, nariz_recta, nariz_respingada, nariz_chata
- **Forehead Types**: frente_amplia, frente_estrecha, frente_prominente
- **Chin Types**: menton_prominente, menton_retrasado, menton_puntiagudo
- **Facial Structure**: cara_alargada, cara_redonda, cara_cuadrada
- **Profile Types**: perfil_convexo, perfil_concavo, perfil_recto
#### Nasal Profile Analysis
- **Protrusion**: nariz protruyente, nariz normal, nariz corta
- **Tip Angle**: punta muy hacia arriba, punta hacia arriba, punta promedio, punta hacia abajo
#### Mandible Classification
- **Mandíbula Sanguínea**: Strong, prominent mandible (≥0.75 proportion)
- **Mandíbula Bilosa**: Medium mandible (0.20-0.65 proportion)
- **Mandíbula Nerviosa**: Weak mandible (≤0.10 proportion)
- **Mandíbula Linfática**: Absent mandible definition
#### Facial Thirds (Profile)
- **Tercio Superior**: Hairline to eyebrow level
- **Tercio Medio**: Eyebrow to nasal base
- **Tercio Inferior**: Nasal base to chin
- Classifications: largo/corto/standard for each third
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
### Local Development - Profile Morfologico ✅
```bash
cd profile_prod/morfologico
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8003
```
### Local Development - Profile Antropometrico ✅
```bash
cd profile_prod/antropometrico
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8004
```
## Production Deployment
### System Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **RAM**: 25GB+ system memory (for all active modules)
- **Storage**: 10GB+ for models and containers
- **CPU**: Multi-core processor for preprocessing
### Port Allocation
- **Frontal Analysis**: 8000-8002
  - Morfológico: 8000
  - Antropométrico: 8001  
  - Validación: 8002
- **Profile Analysis**: 8003-8005
  - Morfológico: 8003 ✅ **Complete**
  - Antropométrico: 8004 ✅ **Active**
  - Validación: 8005 (Planned)
### Independent Scaling
Each module can be scaled independently:
- Deploy multiple instances of any module
- Use load balancer for request distribution
- Configure GPU memory optimization per module
- Implement request queuing for batch processing
### Monitoring
```bash
# All active modules health
curl http://localhost:8000/health  # Frontal Morfologico
curl http://localhost:8001/health  # Frontal Antropometrico  
curl http://localhost:8002/health  # Frontal Validacion
curl http://localhost:8003/health  # Profile Morfologico ✅
curl http://localhost:8004/health  # Profile Antropometrico ✅
# Container status
docker ps
# GPU utilization
nvidia-smi
```
## Architecture Roadmap
### Current Status
- ✅ **Frontal Analysis Complete**: validacion, morfologico, antropometrico (Ports 8000-8002)
- ✅ **Profile Morfologico**: Advanced profile morphological analysis (Port 8003) ✅ **COMPLETE**
- ✅ **Profile Antropometrico**: Advanced profile anthropometric analysis (Port 8004)
- 🔄 **Profile Validacion**: Profile validation and quality assessment (Port 8005) - *In Development*
### Planned Extensions
- 🔄 **Profile Analysis Completion**: Validacion module
- 🔄 **Whole Body Analysis**: Full body anthropometric measurements
- 🔄 **Master Orchestration**: Multi-service deployment and result aggregation
- 🔄 **3D Analysis Pipeline**: Depth-aware facial reconstruction
## Usage Workflow
### Recommended Analysis Pipeline
#### For Frontal Images
1. **Frontal Validacion** (Port 8002): Validate image quality and detect issues
2. **Frontal Morfologico** (Port 8000): Perform morphological analysis if suitable
3. **Frontal Antropometrico** (Port 8001): Conduct detailed measurements
#### For Profile Images ✅
1. **Profile Morfologico** (Port 8003): Complete profile morphological analysis ✅
2. **Profile Antropometrico** (Port 8004): Advanced anthropometric measurements ✅
3. **Profile Validacion** (Port 8005): Profile quality validation *(Coming Soon)*
### Quality-First Approach
The **Validacion** modules serve as quality gates, identifying:
- Hair covering facial features
- Problematic accessories (glasses, objects)
- Poor lighting or image quality
- Unsuitable facial expressions
- Recommendations for better image capture
## Support
### Issues & Questions
- Repository: https://github.com/quantileMX/SG_prod
- Documentation: See `/docs` endpoint when APIs are running
### Model Access
All trained model weights are included in this repository. Large files use Git LFS for efficient repository management.
## License
[Add appropriate license information]
---
**quantileMX** - Advanced AI Solutions



