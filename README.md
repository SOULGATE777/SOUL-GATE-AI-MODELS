# SG - Facial Recognition Production Pipeline
Complete production-ready facial recognition system with morphological, anthropometric, and validation analysis capabilities for both frontal and profile views.

## Project Structure
```
SG_prod/
├── frontal_prod/                   
│   ├── validacion/                 (Port 8002) ✅ COMPLETE
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
│   ├── morfologico/               (Port 8000) ✅ COMPLETE
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
│   └── antropometrico/           (Port 8001) ✅ COMPLETE
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
│   ├── validacion/                 (Port 8005) ✅ COMPLETE
│   │   ├── app/                   
│   │   │   ├── main.py           
│   │   │   ├── models/           
│   │   │   │   └── profile_validation_pipeline.py
│   │   │   └── utils/            
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/               
│   │   │   └── occlusion_detection_model.pth
│   │   ├── Dockerfile            
│   │   ├── docker-compose.yml     
│   │   ├── requirements.txt      
│   │   └── results/              
│   ├── morfologico/               (Port 8003) ✅ COMPLETE
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
│   └── antropometrico/           (Port 8004) ✅ COMPLETE
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

### Frontal Analysis (Ports 8000-8002) ✅ **ALL COMPLETE**

#### Facial Feature Validation (Port 8002) ✅ **COMPLETE**
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

#### Morphological Facial Analysis (Port 8000) ✅ **COMPLETE**
- **3-Model Ensemble Architecture**:
  - Facial landmark detection (Faster R-CNN)
  - Characteristic classification (CNN)
  - Anthropometric point detection
- **GPU Acceleration**: Full CUDA support
- **Beautiful Visualizations**: Modern, clean annotations
- **Production Ready**: Docker containerization

#### Anthropometric Facial Analysis (Port 8001) ✅ **COMPLETE**
- **Hybrid Detection System**:
  - 68 standard dlib facial landmarks
  - Custom Faster R-CNN for 3 key anthropometric points
  - Enhanced facial proportion calculations
- **Advanced Measurements**:
  - Facial thirds analysis with model-enhanced precision
  - Eye relationship analysis
  - Mouth-pupil proportions
  - Eyebrow slope calculations

### Profile Analysis (Ports 8003-8005) ✅ **ALL COMPLETE**

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

#### Profile Anthropometric Analysis (Port 8004) ✅ **COMPLETE**
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

#### Profile Validation (Port 8005) ✅ **COMPLETE**
- **Advanced Occlusion Detection**: Custom trained Faster R-CNN model for profile-specific occlusions
- **Occlusion Categories**:
  - Hair coverage (cabello_tapando_oreja, cabello_tapando_frente)
  - Objects and accessories (objeto)
- **Comprehensive Quality Assessment**:
  - Image resolution validation
  - Brightness and contrast analysis
  - Sharpness detection (blur assessment)
  - Profile orientation validation
- **Smart Recommendations System**: AI-powered actionable suggestions for image improvement
- **Advanced Visualizations**: Multi-panel dashboard with quality metrics, occlusion detection, and recommendations
- **NMS Filtering**: Per-class non-maximum suppression for clean detections
- **GPU Acceleration**: Full CUDA support with CPU fallback

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit (for GPU support)

### Deploy All Complete Modules ✅

```bash
# Clone the repository
git clone https://github.com/quantileMX/SG_prod.git
cd SG_prod

# Deploy Frontal Modules ✅ ALL COMPLETE
cd frontal_prod

# Frontal Validacion (Port 8002) ✅
cd validacion && docker compose up --build -d && cd ..

# Frontal Morfologico (Port 8000) ✅
cd morfologico && docker compose up --build -d && cd ..

# Frontal Antropometrico (Port 8001) ✅
cd antropometrico && docker compose up --build -d && cd ..

# Deploy Profile Modules ✅ ALL COMPLETE
cd ../profile_prod

# Profile Morfologico (Port 8003) ✅
cd morfologico && docker compose up --build -d && cd ..

# Profile Antropometrico (Port 8004) ✅
cd antropometrico && docker compose up --build -d && cd ..

# Profile Validacion (Port 8005) ✅ NEW!
cd validacion && docker compose up --build -d && cd ..

# Check all active services ✅
curl http://localhost:8000/health  # Frontal Morfologico ✅
curl http://localhost:8001/health  # Frontal Antropometrico ✅
curl http://localhost:8002/health  # Frontal Validacion ✅
curl http://localhost:8003/health  # Profile Morfologico ✅
curl http://localhost:8004/health  # Profile Antropometrico ✅
curl http://localhost:8005/health  # Profile Validacion ✅ NEW!
```

### Deploy Individual Modules

#### Profile Validacion Module ✅ **NEW COMPLETE**
```bash
cd profile_prod/validacion
docker compose up --build -d
curl http://localhost:8005/health
```

## API Documentation ✅ **ALL SERVICES ACTIVE**

### Complete Active Services
- **Frontal Validacion (Port 8002)**: http://localhost:8002/docs ✅ **COMPLETE**
- **Frontal Morfologico (Port 8000)**: http://localhost:8000/docs ✅ **COMPLETE**
- **Frontal Antropometrico (Port 8001)**: http://localhost:8001/docs ✅ **COMPLETE**
- **Profile Morfologico (Port 8003)**: http://localhost:8003/docs ✅ **COMPLETE**
- **Profile Antropometrico (Port 8004)**: http://localhost:8004/docs ✅ **COMPLETE**
- **Profile Validacion (Port 8005)**: http://localhost:8005/docs ✅ **COMPLETE**

## API Endpoints

### Profile Validacion Module (Port 8005) ✅ **NEW COMPLETE**

#### Complete Profile Validation Analysis
```bash
curl -X POST "http://localhost:8005/analyze-profile-validation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"
```

#### Profile Occlusion Detection Only
```bash
curl -X POST "http://localhost:8005/detect-profile-occlusions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5"
```

#### Profile Quality Assessment
```bash
curl -X POST "http://localhost:8005/assess-profile-quality" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg"
```

#### Model Information
```bash
curl http://localhost:8005/model-info
```

#### Health Check
```bash
curl http://localhost:8005/health
```

## Response Formats

### Profile Validacion Response ✅ **NEW**
```json
{
  "analysis_id": "uuid-string",
  "validation_status": {
    "is_suitable": true,
    "overall_score": 85.5,
    "has_occlusions": false,
    "quality_passed": true
  },
  "occlusion_analysis": {
    "total_detections": 0,
    "confidence_threshold_used": 0.5,
    "detections": []
  },
  "quality_assessment": {
    "quality_score": 88.0,
    "blur_score": 125.8,
    "brightness": 142.3,
    "contrast": 45.2,
    "resolution": [1024, 768],
    "quality_issues": [],
    "is_suitable": true
  },
  "recommendations": [],
  "analysis_summary": {
    "timestamp": "2025-06-20T...",
    "processing_successful": true,
    "model_classes": ["objeto", "cabello_tapando_oreja", "cabello_tapando_frente"],
    "device_used": "cuda:0"
  },
  "visualization_path": "/app/results/profile_validation_xxx.png",
  "visualization_url": "/visualization/profile_validation_xxx.png"
}
```

## Model Information

### Profile Validation Model ✅ **NEW**
- **Architecture**: Custom trained Faster R-CNN for profile occlusion detection
- **Classes**: 3 occlusion categories (objeto, cabello_tapando_oreja, cabello_tapando_frente)
- **Features**: Advanced quality assessment, smart recommendations, NMS filtering
- **Input Size**: 224x224 pixels
- **Output**: Occlusion detections + comprehensive quality metrics + actionable recommendations

### Profile Validation Classifications ✅ **NEW**
#### Occlusion Categories (3 Classes)
- **objeto**: Objects or accessories obstructing the profile
- **cabello_tapando_oreja**: Hair covering the ear area
- **cabello_tapando_frente**: Hair covering the forehead area

#### Quality Assessment Metrics
- **Resolution Validation**: Minimum 224x224 pixels required
- **Blur Detection**: Laplacian variance analysis
- **Brightness Analysis**: Optimal range 50-200
- **Contrast Assessment**: Standard deviation analysis
- **Profile Orientation**: Basic validation of true profile vs frontal view

#### Smart Recommendations System
- **Quality-Based**: Resolution, lighting, stability suggestions
- **Occlusion-Based**: Hair positioning, accessory removal guidance
- **Profile-Specific**: Angle optimization, background improvement tips

## Configuration

### GPU Production Setup (All Modules) ✅
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

### CPU Fallback ✅
```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=-1
# Remove deploy section
```

## Production Deployment

### System Requirements
- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **RAM**: 30GB+ system memory (for all 6 active modules)
- **Storage**: 15GB+ for models and containers
- **CPU**: Multi-core processor for preprocessing

### Port Allocation ✅ **ALL OCCUPIED**
- **Frontal Analysis**: 8000-8002 ✅ **ALL COMPLETE**
  - Morfológico: 8000 ✅
  - Antropométrico: 8001 ✅
  - Validación: 8002 ✅
- **Profile Analysis**: 8003-8005 ✅ **ALL COMPLETE**
  - Morfológico: 8003 ✅
  - Antropométrico: 8004 ✅
  - Validación: 8005 ✅ **NEW!**

### Independent Scaling ✅
Each module can be scaled independently:
- Deploy multiple instances of any module
- Use load balancer for request distribution
- Configure GPU memory optimization per module
- Implement request queuing for batch processing

### Monitoring ✅ **ALL SERVICES**
```bash
# All complete modules health check
curl http://localhost:8000/health  # Frontal Morfologico ✅
curl http://localhost:8001/health  # Frontal Antropometrico ✅
curl http://localhost:8002/health  # Frontal Validacion ✅
curl http://localhost:8003/health  # Profile Morfologico ✅
curl http://localhost:8004/health  # Profile Antropometrico ✅
curl http://localhost:8005/health  # Profile Validacion ✅ NEW!

# Container status
docker ps

# GPU utilization
nvidia-smi
```

## Architecture Status ✅ **PROJECT COMPLETE**

### Current Status ✅ **ALL MODULES OPERATIONAL**
- ✅ **Frontal Analysis Complete**: validacion, morfologico, antropometrico (Ports 8000-8002) ✅ **ALL COMPLETE**
- ✅ **Profile Analysis Complete**: morfologico, antropometrico, validacion (Ports 8003-8005) ✅ **ALL COMPLETE**

### Core Pipeline Complete ✅
The SG_prod facial recognition production pipeline is now **COMPLETE** with all 6 modules operational:

#### **Frontal Image Processing Pipeline** ✅
1. **Frontal Validacion** (Port 8002): Validate image quality and detect issues ✅
2. **Frontal Morfologico** (Port 8000): Perform morphological analysis ✅
3. **Frontal Antropometrico** (Port 8001): Conduct detailed measurements ✅

#### **Profile Image Processing Pipeline** ✅
1. **Profile Validacion** (Port 8005): Profile quality validation and occlusion detection ✅ **COMPLETE**
2. **Profile Morfologico** (Port 8003): Complete profile morphological analysis ✅
3. **Profile Antropometrico** (Port 8004): Advanced anthropometric measurements ✅

### Future Extensions 🔄
- 🔄 **Whole Body Analysis**: Full body anthropometric measurements
- 🔄 **Master Orchestration**: Multi-service deployment and result aggregation
- 🔄 **3D Analysis Pipeline**: Depth-aware facial reconstruction
- 🔄 **Real-time Processing**: WebRTC integration for live analysis

## Usage Workflow ✅ **COMPLETE PIPELINES**

### Recommended Analysis Pipeline

#### For Frontal Images ✅ **COMPLETE WORKFLOW**
1. **Frontal Validacion** (Port 8002): Validate image quality and detect issues ✅
2. **Frontal Morfologico** (Port 8000): Perform morphological analysis if suitable ✅
3. **Frontal Antropometrico** (Port 8001): Conduct detailed measurements ✅

#### For Profile Images ✅ **COMPLETE WORKFLOW**
1. **Profile Validacion** (Port 8005): Profile quality validation and occlusion detection ✅ **NEW!**
2. **Profile Morfologico** (Port 8003): Complete profile morphological analysis ✅
3. **Profile Antropometrico** (Port 8004): Advanced anthropometric measurements ✅

### Quality-First Approach ✅
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

### Model Access ✅
All trained model weights are included in this repository. Large files use Git LFS for efficient repository management.

## License
[Add appropriate license information]

---
**quantileMX** - Advanced AI Solutions  
**Status**: ✅ **PRODUCTION READY - ALL MODULES COMPLETE** ✅
