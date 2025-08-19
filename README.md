# SG - Complete AI Analysis Production Pipeline
Complete production-ready AI analysis system with facial recognition, body analysis, morphological, anthropometric, and validation capabilities for frontal, profile, and full body views.

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
│   ├── antropometrico/           (Port 8001) ✅ COMPLETE
│   │   ├── app/                  
│   │   │   ├── main.py          
│   │   │   ├── models/          
│   │   │   │   └── anthropometric_pipeline.py
│   │   │   └── utils/           
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/              
│   │   │   ├── facial_points_detection_model.pth
│   │   │   └── shape_predictor_68_face_landmarks.dat
│   │   ├── Dockerfile           
│   │   ├── docker-compose.yml   
│   │   ├── requirements.txt     
│   │   └── results/             
│   ├── espejo/                   (Port 8008) ✅ COMPLETE
│   │   ├── app/                  
│   │   │   ├── main.py          
│   │   │   ├── models/          
│   │   │   │   └── espejo_pipeline.py
│   │   │   └── utils/           
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/              
│   │   │   ├── binary_region_classifier_best.pth
│   │   │   ├── facial_points_detection_model.pth
│   │   │   ├── frente_best_model.pth
│   │   │   ├── rostro_menton_best_model.pth
│   │   │   └── shape_predictor_68_face_landmarks.dat
│   │   ├── Dockerfile           
│   │   ├── docker-compose.yml   
│   │   ├── requirements.txt     
│   │   └── results/             
│   └── rotacion/                 (Port 8012) ✅ **NEW COMPLETE**
│       ├── app/                  
│       │   ├── main.py          
│       │   ├── models/          
│       │   │   └── frontal_rotation_pipeline.py
│       │   └── utils/           
│       │       ├── visualization.py
│       │       └── image_processing.py
│       ├── models/              
│       │   └── improved_supervisely_head_rotation_model_MULTILABEL_CORRECTED.pth
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
│   ├── antropometrico/           (Port 8004) ✅ COMPLETE ✅ **UPDATED CALIBRATIONS**
│   │   ├── app/                  
│   │   │   ├── main.py          
│   │   │   ├── models/          
│   │   │   │   └── profile_anthropometric_pipeline.py
│   │   │   └── utils/           
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/              
│   │   │   └── profile_aware_point_detection_model.pth
│   │   ├── Dockerfile           
│   │   ├── docker-compose.yml   
│   │   ├── requirements.txt     
│   │   └── results/             
│   ├── preprocesamiento/        (Port 8010) ✅ **NEW COMPLETE**
│   │   ├── app/                  
│   │   │   ├── main.py          
│   │   │   ├── models/          
│   │   │   │   └── profile_preprocessing_pipeline.py
│   │   │   └── utils/           
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/              
│   │   │   └── profile_detection_model.pth
│   │   ├── Dockerfile           
│   │   ├── docker-compose.yml   
│   │   ├── requirements.txt     
│   │   └── results/             
│   └── rotacion/                 (Port 8011) ✅ **NEW COMPLETE**
│       ├── app/                  
│       │   ├── main.py          
│       │   ├── models/          
│       │   │   └── profile_rotation_pipeline.py
│       │   └── utils/           
│       │       ├── visualization.py
│       │       └── image_processing.py
│       ├── models/              
│       │   └── best_profile_classifier_multilabel.pth
│       ├── Dockerfile           
│       ├── docker-compose.yml   
│       ├── requirements.txt     
│       └── results/             
├── body_prod/                      ✅ **NEW COMPLETE**
│   ├── morfologico/               (Port 8006) ✅ **NEW COMPLETE**
│   │   ├── app/                   
│   │   │   ├── main.py           
│   │   │   ├── models/           
│   │   │   │   └── body_analysis_pipeline.py
│   │   │   └── utils/            
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/               
│   │   │   └── lightweight_body_classifier.pth
│   │   ├── Dockerfile            
│   │   ├── docker-compose.yml     
│   │   ├── requirements.txt      
│   │   └── results/              
│   ├── antropometrico/           (Port 8007) ✅ **NEW COMPLETE**
│   │   ├── app/                  
│   │   │   ├── main.py          
│   │   │   ├── models/          
│   │   │   │   └── anthropometric_pipeline.py
│   │   │   └── utils/           
│   │   │       ├── visualization.py
│   │   │       └── image_processing.py
│   │   ├── models/              
│   │   │   └── yolov8n-pose.pt
│   │   ├── Dockerfile           
│   │   ├── docker-compose.yml   
│   │   ├── requirements.txt     
│   │   └── results/             
│   └── manos/                     (Port 8009) ✅ **NEW COMPLETE**
│       ├── app/                  
│       │   ├── main.py          
│       │   ├── models/          
│       │   │   └── hand_analysis_pipeline.py
│       │   └── utils/           
│       │       ├── visualization.py
│       │       └── image_processing.py
│       ├── models/              
│       │   └── dorso_palma_classifier.pth
│       ├── Dockerfile           
│       ├── docker-compose.yml   
│       ├── requirements.txt     
│       └── results/             
├── .gitattributes              
├── .gitignore                  
└── README.md                   
```

## Features

### Frontal Analysis (Ports 8000-8002, 8008, 8012) ✅ **ALL COMPLETE**

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

#### Morphological Facial Analysis (Port 8000) ✅ **COMPLETE** ✅ **OPTIMIZED MODELS**
- **3-Model Ensemble Architecture**:
  - Facial landmark detection (Faster R-CNN)
  - Characteristic classification (CNN)
  - Anthropometric point detection
- **Enhanced Model Performance**: Optimized model versions for improved accuracy and speed
- **GPU Acceleration**: Full CUDA support
- **Beautiful Visualizations**: Modern, clean annotations
- **Production Ready**: Docker containerization

#### Anthropometric Facial Analysis (Port 8001) ✅ **COMPLETE** ✅ **OPTIMIZED MODELS**
- **Hybrid Detection System**:
  - 68 standard dlib facial landmarks
  - Custom Faster R-CNN for 3 key anthropometric points
  - Enhanced facial proportion calculations
- **Advanced Measurements**:
  - Facial thirds analysis with model-enhanced precision
  - Eye relationship analysis
  - Mouth-pupil proportions
  - Eyebrow slope calculations
- **Enhanced Model Performance**: Optimized model versions for improved accuracy and speed

#### Espejo Mirror Analysis (Port 8008) ✅ **COMPLETE**
- **Mirror Face Generation**: Creates left and right mirrored faces for comprehensive asymmetry analysis
- **Anthropometric Measurements**: 
  - Face, forehead, and temporal proportion calculations
  - 68-point facial landmark detection with dlib
  - Custom 13-point anthropometric point detection (Faster R-CNN)
- **Decision Tree Classification**: Excel-based decision rules for facial region analysis
- **Dual Region Analysis**:
  - FRENTE region classification (7 classes: jupiter, marte, mercurio, neptuno, solar/lunar, tierra, venus)
  - rostro_menton region classification (8 classes: jupiter/luna, marte/tierra, mercurio, pluton-venus, pluton, saturno, sol_neptuno, venus)
- **Hybrid Class Splitting**: Proportion-based diagnosis refinement with confidence thresholds
- **Comprehensive Reporting**: Detailed analysis reports with visualizations and dashboards

#### Frontal Rotation Assessment (Port 8012) ✅ **NEW COMPLETE**
- **Multi-label CNN Classification**: EfficientNet-B0 based model for frontal face rotation assessment
- **Rotation Categories**:
  - Aceptable: Suitable frontal orientation for analysis
  - Upward Tilt: Face tilted upward or camera positioned too low
  - Downward Tilt: Face tilted downward or camera positioned too high
  - Horizontal: Horizontal face orientation issues
  - Diagonal: Diagonal face tilt problems
- **Viability Assessment**: Determines suitability for anthropometric and morphological analysis
- **Pattern-Aware Predictions**: Respects annotation patterns (aceptable is standalone)
- **Comprehensive Visualizations**: 4-panel analysis with confidence scores, recommendations, and detailed reports
- **Quality Enhancement**: Optional image preprocessing and enhancement
- **Batch Processing**: Support for analyzing multiple frontal images simultaneously
- **GPU Acceleration**: Optimized for CUDA with CPU fallback

### Profile Analysis (Ports 8003-8005, 8010-8011) ✅ **ALL COMPLETE**

#### Profile Morphological Analysis (Port 8003) ✅ **COMPLETE** ✅ **OPTIMIZED MODELS**
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
- **Enhanced Model Performance**: Optimized model versions for improved accuracy and speed
- **GPU Acceleration**: Full CUDA support with CPU fallback
- **Clean API Responses**: No neural network profile predictions, inference from actual detected points

#### Profile Anthropometric Analysis (Port 8004) ✅ **COMPLETE** ✅ **OPTIMIZED MODELS** ✅ **UPDATED CALIBRATIONS**
- **Profile-Specific Point Detection**: Custom trained model for profile anthropometric points
- **Advanced Profile Measurements**:
  - Nasal profile analysis (protrusion, angle, classification)
  - Facial thirds in profile view with updated vector calculations
  - Mandible classification (Sanguinea, Bilosa, Nerviosa, Linfática)
  - Angular measurements (nose tip, forehead, chin angles)
  - Ear morphology analysis (width, trago-antitrago proportions)
- **Enhanced Vector Analysis**: Updated reference calculations using 24-18 vector baseline
- **Enhanced Model Performance**: Optimized model versions for improved accuracy and speed
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

#### Profile Rotation Assessment (Port 8011) ✅ **NEW COMPLETE**
- **Multi-label CNN Classification**: EfficientNet-B0 based model for profile rotation assessment
- **Profile Rotation Categories**:
  - Aceptable: Suitable profile orientation for analysis
  - Upward/Downward Tilt: Profile head tilted in various directions
  - Camera Position Issues: Camera too high/low or positioned incorrectly
  - Frontal/Back Rotation: Profile rotated toward or away from camera
- **Viability Assessment**: Determines suitability for anthropometric and morphological analysis
- **Pattern-Aware Predictions**: Respects annotation patterns (aceptable is standalone)
- **Comprehensive Visualizations**: 4-panel analysis with confidence scores, recommendations, and detailed reports
- **Quality Enhancement**: Optional image preprocessing and enhancement
- **Batch Processing**: Support for analyzing multiple profile images simultaneously
- **GPU Acceleration**: Optimized for CUDA with CPU fallback

### Body Analysis (Ports 8006-8007, 8009) ✅ **ALL COMPLETE** ✅ **NEW!**

#### Body Morphological Analysis (Port 8006) ✅ **NEW COMPLETE** ✅ **UPDATED MODEL**
- **LightweightHierarchicalModel**: ResNet18-based architecture optimized for body type classification
- **7 Body Type Classifications**:
  - Bilioso/NormalPocaGrasa (Normal Poca Grasa)
  - Nervioso/Delgado (Delgado)
  - SanguineoLinfatico/MusculosoGordo (Musculoso Gordo)
  - Sanguineo/Musculoso (Musculoso)
  - Flematico/Gordograsacuelga (Gordo Grasa Cuelga)
  - Linfatico/Gordo (Gordo)
  - BiliosoSanguineo/NormalMusculoso (Normal Musculoso)
- **Enhanced Model Architecture**: Updated morphological classification algorithms
- **Gender Classification**: Hombre/Mujer prediction with confidence scores
- **Morphological Insights**: Body composition, metabolic tendencies, physical characteristics
- **Advanced Analysis**: Confidence metrics, prediction certainty levels, consistency assessment
- **Hierarchical Classification**: Coarse and fine-grained body type predictions
- **GPU Acceleration**: Full CUDA support with CPU fallback

#### Body Anthropometric Analysis (Port 8007) ✅ **NEW COMPLETE**
- **YOLOv8n Pose Detection**: 17-keypoint full body pose estimation
- **Precise Skull Detection**: Anatomical proportions + contour refinement methodology
- **Advanced Skull Measurements**:
  - Skull-to-body ratio calculations (adult: 12.5-14.3%, child: 16-18%)
  - Head orientation analysis and tilt compensation
  - Anatomical assessment and age estimation
  - Multi-method skull detection (nose-centered, eye-centered, contour-refined)
- **Body Proportion Analysis**:
  - Full body keypoint detection and grouping
  - Body part measurements and relationships
  - Pose quality assessment for anthropometric reliability
- **Comprehensive Analysis**:
  - Detailed anatomical insights and recommendations
  - Confidence analysis for all detected keypoints
  - Quality metrics and measurement reliability assessment
- **Advanced Visualizations**: Multi-panel anthropometric dashboards with detailed reports
- **GPU Acceleration**: Full CUDA support with CPU fallback

#### Hand Analysis (Port 8009) ✅ **NEW COMPLETE** ✅ **UPDATED COLORIMETRY**
- **CNN Hand Classification**: ResNet50-based dorso/palma (back/palm) classification with 89%+ accuracy
- **Advanced Colorimetry Analysis**: Multi-color-space palm skin analysis (HSV + YCrCb filtering)
- **Traditional Color Classification**: 5 palm color types classification system:
  - rosa/sanguineo-linfatico oscuro (Pink/sanguine-lymphatic dark)
  - rojo/sanguineo (Red/sanguine) 
  - amarillo/nervioso (Yellow/nervous)
  - blanco/linfatico (White/lymphatic)
  - bilioso/cafe_o_oscuro (Bilious/brown or dark)
- **Enhanced Colorimetry Calibrations**: Updated color analysis algorithms for improved accuracy
- **K-means Color Clustering**: Dominant color extraction with percentage analysis
- **Comprehensive Analysis**: CNN prediction + colorimetry + color type classification
- **Intelligent Skin Detection**: Advanced skin masking with morphological operations
- **Production API**: FastAPI with async processing and batch analysis support
- **Rich Visualizations**: Multi-panel analysis dashboards with color palettes and detailed reports
- **GPU Acceleration**: Full CUDA support with CPU fallback

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit (for GPU support)

### Deploy All Complete Modules ✅ **ALL 12 MODULES**

```bash
# Clone the repository
git clone https://github.com/quantileMX/SG_prod.git
cd SG_prod

# Deploy Frontal Modules ✅ ALL COMPLETE
cd frontal_prod

# Frontal Validacion (Port 8002) ✅
cd validacion && docker compose up --build -d && cd ..

# Frontal Morfologico (Port 8000) ✅ OPTIMIZED MODELS
cd morfologico && docker compose up --build -d && cd ..

# Frontal Antropometrico (Port 8001) ✅ OPTIMIZED MODELS
cd antropometrico && docker compose up --build -d && cd ..

# Frontal Espejo (Port 8008) ✅
cd espejo && docker compose up --build -d && cd ..

# Frontal Rotation (Port 8012) ✅ NEW!
cd rotacion && docker compose up --build -d && cd ..

# Deploy Profile Modules ✅ ALL COMPLETE
cd ../profile_prod

# Profile Morfologico (Port 8003) ✅ OPTIMIZED MODELS
cd morfologico && docker compose up --build -d && cd ..

# Profile Antropometrico (Port 8004) ✅ OPTIMIZED MODELS + UPDATED CALIBRATIONS
cd antropometrico && docker compose up --build -d && cd ..

# Profile Validacion (Port 8005) ✅
cd validacion && docker compose up --build -d && cd ..

# Profile Preprocessing (Port 8010) ✅ NEW!
cd preprocesamiento && docker compose up --build -d && cd ..

# Profile Rotation (Port 8011) ✅ NEW!
cd rotacion && docker compose up --build -d && cd ..

# Deploy Body Modules ✅ ALL COMPLETE ✅ NEW!
cd ../body_prod

# Body Morfologico (Port 8006) ✅ NEW!
cd morfologico && docker compose up --build -d && cd ..

# Body Antropometrico (Port 8007) ✅ NEW!
cd antropometrico && docker compose up --build -d && cd ..

# Hand Analysis (Port 8009) ✅ NEW!
cd manos && docker compose up --build -d && cd ..

# Check all active services ✅ ALL 12 MODULES
curl http://localhost:8000/health  # Frontal Morfologico ✅
curl http://localhost:8001/health  # Frontal Antropometrico ✅
curl http://localhost:8002/health  # Frontal Validacion ✅
curl http://localhost:8003/health  # Profile Morfologico ✅
curl http://localhost:8004/health  # Profile Antropometrico ✅ UPDATED
curl http://localhost:8005/health  # Profile Validacion ✅
curl http://localhost:8006/health  # Body Morfologico ✅ UPDATED MODEL
curl http://localhost:8007/health  # Body Antropometrico ✅
curl http://localhost:8008/health  # Frontal Espejo ✅
curl http://localhost:8009/health  # Hand Analysis ✅ UPDATED COLORIMETRY
curl http://localhost:8010/health  # Profile Preprocessing ✅ NEW!
curl http://localhost:8011/health  # Profile Rotation ✅ NEW!
curl http://localhost:8012/health  # Frontal Rotation ✅ NEW!
```

### Deploy Individual Body Modules ✅ **NEW**

#### Body Morphological Analysis Module ✅ **NEW COMPLETE**
```bash
cd body_prod/morfologico
docker compose up --build -d
curl http://localhost:8006/health
```

#### Body Anthropometric Analysis Module ✅ **NEW COMPLETE**
```bash
cd body_prod/antropometrico
docker compose up --build -d
curl http://localhost:8007/health
```

#### Frontal Espejo Mirror Analysis Module ✅ **COMPLETE**
```bash
cd frontal_prod/espejo
docker compose up --build -d
curl http://localhost:8008/health
```

#### Hand Analysis Module ✅ **NEW COMPLETE**
```bash
cd body_prod/manos
docker compose up --build -d
curl http://localhost:8009/health
```

#### Frontal Rotation Assessment Module ✅ **NEW COMPLETE**
```bash
cd frontal_prod/rotacion
docker compose up --build -d
curl http://localhost:8012/health
```

#### Profile Rotation Assessment Module ✅ **NEW COMPLETE**
```bash
cd profile_prod/rotacion
docker compose up --build -d
curl http://localhost:8011/health
```

## API Documentation ✅ **ALL 12 SERVICES ACTIVE**

### Complete Active Services
- **Frontal Validacion (Port 8002)**: http://localhost:8002/docs ✅ **COMPLETE**
- **Frontal Morfologico (Port 8000)**: http://localhost:8000/docs ✅ **OPTIMIZED MODELS**
- **Frontal Antropometrico (Port 8001)**: http://localhost:8001/docs ✅ **OPTIMIZED MODELS**
- **Frontal Espejo (Port 8008)**: http://localhost:8008/docs ✅ **COMPLETE**
- **Profile Morfologico (Port 8003)**: http://localhost:8003/docs ✅ **OPTIMIZED MODELS**
- **Profile Antropometrico (Port 8004)**: http://localhost:8004/docs ✅ **OPTIMIZED MODELS** + **UPDATED CALIBRATIONS**
- **Profile Validacion (Port 8005)**: http://localhost:8005/docs ✅ **COMPLETE**
- **Body Morfologico (Port 8006)**: http://localhost:8006/docs ✅ **NEW COMPLETE**
- **Body Antropometrico (Port 8007)**: http://localhost:8007/docs ✅ **NEW COMPLETE**
- **Hand Analysis (Port 8009)**: http://localhost:8009/docs ✅ **NEW COMPLETE**
- **Profile Rotation (Port 8011)**: http://localhost:8011/docs ✅ **NEW COMPLETE**
- **Frontal Rotation (Port 8012)**: http://localhost:8012/docs ✅ **NEW COMPLETE**

## API Endpoints

### Frontal Espejo Mirror Analysis Module (Port 8008) ✅ **COMPLETE**

#### Complete Espejo Analysis
```bash
curl -X POST "http://localhost:8008/analyze-espejo" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "include_dashboard=true"
```

#### Mirror Generation Only
```bash
curl -X POST "http://localhost:8008/generate-mirrors" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5"
```

#### Final Diagnosis
```bash
curl -X POST "http://localhost:8008/get-diagnosis" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "format=json"
```

#### Region Classification
```bash
curl -X POST "http://localhost:8008/classify-regions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Body Morphological Analysis Module (Port 8006) ✅ **NEW COMPLETE**

#### Complete Body Type Analysis
```bash
curl -X POST "http://localhost:8006/analyze-body-morphology" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "detailed_analysis=true"
```

#### Body Type Classification Only
```bash
curl -X POST "http://localhost:8006/classify-body-type" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5"
```

#### Batch Body Classification
```bash
curl -X POST "http://localhost:8006/batch-classify" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "confidence_threshold=0.5"
```

### Body Anthropometric Analysis Module (Port 8007) ✅ **NEW COMPLETE**

#### Complete Anthropometric Analysis
```bash
curl -X POST "http://localhost:8007/analyze-body-anthropometry" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "detailed_analysis=true"
```

#### Skull Measurements Only
```bash
curl -X POST "http://localhost:8007/detect-skull-measurements" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_contour_refinement=true"
```

#### Pose Detection Only
```bash
curl -X POST "http://localhost:8007/detect-pose-keypoints" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Hand Analysis Module (Port 8009) ✅ **NEW COMPLETE**

#### Complete Hand Analysis
```bash
curl -X POST "http://localhost:8009/analyze-hand-comprehensive" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_colorimetry=true" \
  -F "include_visualization=true"
```

#### Hand Side Classification Only
```bash
curl -X POST "http://localhost:8009/classify-hand-side" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand_image.jpg" \
  -F "confidence_threshold=0.7"
```

#### Colorimetry Analysis Only
```bash
curl -X POST "http://localhost:8009/analyze-colorimetry" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand_image.jpg" \
  -F "include_visualization=true"
```

#### With Bounding Box
```bash
curl -X POST "http://localhost:8009/analyze-hand-comprehensive" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand_image.jpg" \
  -F "bbox=100,50,300,250"
```

#### Batch Hand Analysis
```bash
curl -X POST "http://localhost:8009/batch-analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@hand1.jpg" \
  -F "files=@hand2.jpg" \
  -F "confidence_threshold=0.5"
```

### Profile Rotation Assessment Module (Port 8011) ✅ **NEW COMPLETE**

#### Complete Profile Rotation Analysis
```bash
curl -X POST "http://localhost:8011/analyze-profile-rotation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "enhance_image=true"
```

#### Simple Rotation Classification
```bash
curl -X POST "http://localhost:8011/classify-rotation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5"
```

#### Viability Assessment Only
```bash
curl -X POST "http://localhost:8011/assess-viability" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Frontal Rotation Assessment Module (Port 8012) ✅ **NEW COMPLETE**

#### Complete Frontal Rotation Analysis
```bash
curl -X POST "http://localhost:8012/analyze-frontal-rotation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@frontal_face_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "enhance_image=true"
```

#### Simple Rotation Classification
```bash
curl -X POST "http://localhost:8012/classify-rotation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@frontal_face_image.jpg" \
  -F "confidence_threshold=0.5"
```

#### Viability Assessment Only
```bash
curl -X POST "http://localhost:8012/assess-viability" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@frontal_face_image.jpg" \
  -F "confidence_threshold=0.5"
```

#### Health Checks
```bash
curl http://localhost:8006/health  # Body Morfologico
curl http://localhost:8007/health  # Body Antropometrico
curl http://localhost:8008/health  # Frontal Espejo
curl http://localhost:8009/health  # Hand Analysis
curl http://localhost:8011/health  # Profile Rotation
curl http://localhost:8012/health  # Frontal Rotation
curl http://localhost:8006/model-info  # Body Type Model Info
curl http://localhost:8007/model-info  # Pose Detection Model Info
curl http://localhost:8008/model-info  # Espejo Model Info
curl http://localhost:8009/model-info  # Hand Analysis Model Info
curl http://localhost:8011/model-info  # Profile Rotation Model Info
curl http://localhost:8012/model-info  # Frontal Rotation Model Info
```

## Response Formats

### Frontal Espejo Mirror Analysis Response ✅ **COMPLETE**
```json
{
  "analysis_id": "uuid-string",
  "face_detected": true,
  "anthropometric_analysis": {
    "face_proportions": {
      "right": 0.456,
      "left": 0.443
    },
    "forehead_proportions": {
      "right": 0.387,
      "left": 0.392
    },
    "temporal_proportions": {
      "right": 0.523,
      "left": 0.518
    },
    "custom_model_points": {...},
    "landmarks_detected": 68
  },
  "mirror_analysis": {
    "right_mirrored": {...},
    "left_mirrored": {...}
  },
  "final_diagnosis": {
    "right_side": {
      "frente_diagnosis": "solar",
      "rostro_diagnosis": "venus_corazon",
      "confidence_scores": {...}
    },
    "left_side": {
      "frente_diagnosis": "luna",
      "rostro_diagnosis": "pluton-venus",
      "confidence_scores": {...}
    }
  },
  "decision_tree_analysis": {
    "right_side": {
      "frente_applied_rules": [...],
      "rostro_applied_rules": [...],
      "frente_split_rules": [...],
      "rostro_split_rules": [...]
    },
    "left_side": {...}
  },
  "analysis_summary": {...},
  "visualization_path": "/app/results/espejo_analysis_20241215_143022_abc123.png",
  "dashboard_path": "/app/results/espejo_dashboard_20241215_143022_def456.png"
}
```

### Body Morphological Analysis Response ✅ **NEW**
```json
{
  "analysis_id": "uuid-string",
  "body_type_analysis": {
    "predicted_class": "Sanguineo/Musculoso",
    "predicted_class_simple": "Musculoso",
    "confidence": 0.85,
    "meets_threshold": true,
    "all_probabilities": {...},
    "top_3_predictions": [...]
  },
  "gender_analysis": {
    "predicted_gender": "Hombre",
    "confidence": 0.92,
    "all_probabilities": {...}
  },
  "analysis_metrics": {
    "overall_confidence": 0.88,
    "prediction_certainty": "high",
    "gender_body_consistency": "high"
  },
  "morphological_insights": {
    "body_composition": "Mesomorphic build with well-developed musculature",
    "metabolic_tendency": "Efficient metabolism, responds well to exercise",
    "physical_characteristics": "Athletic build, defined muscle structure"
  },
  "classification_summary": {...},
  "analysis_summary": {...}
}
```

### Body Anthropometric Analysis Response ✅ **NEW**
```json
{
  "analysis_id": "uuid-string",
  "num_persons": 1,
  "anthropometric_analysis": [
    {
      "person_id": 1,
      "keypoint_summary": {
        "total_keypoints": 15,
        "detection_percentage": 88.2,
        "keypoint_completeness": "excellent"
      },
      "body_proportions": {
        "skull_height": 156,
        "skull_width": 142,
        "body_height": 1089,
        "skull_to_body_ratio": 0.143,
        "skull_percentage": 14.3,
        "anatomical_assessment": "Normal adult skull proportions",
        "head_orientation": "frontal (2.1°)",
        "detection_method": "nose_anatomical+contour_refined"
      },
      "detailed_analysis": {
        "skull_analysis": {...},
        "age_assessment": "adult_proportions",
        "anthropometric_insights": {...}
      }
    }
  ],
  "analysis_summary": {...}
}
```

### Hand Analysis Response ✅ **NEW**
```json
{
  "analysis_id": "uuid-string",
  "analysis_type": "comprehensive_hand_analysis",
  "image_path": "/app/temp/temp_uuid.jpg",
  "bbox": [x_min, y_min, x_max, y_max],
  "cnn_prediction": {
    "predicted_class": "Palma",
    "confidence": 0.89,
    "probabilities": {
      "Dorso": 0.11,
      "Palma": 0.89
    },
    "meets_threshold": true
  },
  "colorimetry": {
    "average_color_rgb": [185, 142, 125],
    "average_color_hsv": [12.5, 32.4, 72.5],
    "dominant_colors": [
      [[190, 145, 128], 35.2],
      [[180, 138, 120], 28.7],
      [[175, 135, 115], 22.1]
    ],
    "hue_mean": 12.8,
    "hue_std": 8.5,
    "total_pixels": 15432
  },
  "color_classification": {
    "average_color": {
      "rosa/sanguineo-linfatico oscuro": 65.0,
      "rojo/sanguineo": 35.0
    },
    "main_color": {
      "rosa/sanguineo-linfatico oscuro": 100.0
    }
  },
  "visualization_url": "/visualization/hand_analysis_uuid.png"
}
```

## Model Information

### Frontal Espejo Mirror Analysis Model ✅ **COMPLETE**
- **Architecture**: Multi-model ensemble with decision tree classification
- **Models**: 
  - dlib shape predictor (68 facial landmarks)
  - Faster R-CNN (13 anthropometric points)
  - Binary region classifier (FRENTE/rostro_menton)
  - FRENTE classifier (7 classes)
  - rostro_menton classifier (8 classes)
- **Classifications**: 
  - FRENTE: jupiter, marte, mercurio, neptuno, solar/lunar, tierra, venus
  - rostro_menton: jupiter/luna, marte/tierra, mercurio, pluton-venus, pluton, saturno, sol_neptuno, venus
- **Features**: Mirror generation, proportion calculation, decision tree rules, hybrid splitting
- **Input Size**: Variable (auto-resized for processing)
- **Output**: Dual-side analysis with comprehensive diagnosis and visualizations

### Body Morphological Analysis Model ✅ **NEW**
- **Architecture**: LightweightHierarchicalModel (ResNet18 backbone)
- **Classifications**: 7 body types + 2 genders + 4 coarse types
- **Features**: Attention mechanism, hierarchical classification, morphological insights
- **Input Size**: 224x224 pixels
- **Output**: Multi-class predictions with confidence scores and detailed analysis

### Body Anthropometric Analysis Model ✅ **NEW**
- **Architecture**: YOLOv8n-pose for 17-keypoint detection
- **Measurements**: Skull dimensions, body proportions, anatomical assessments
- **Features**: Head orientation analysis, contour refinement, age estimation
- **Input Format**: RGB images (any resolution, auto-resized)
- **Output**: Pose keypoints + skull measurements + anthropometric analysis

### Hand Analysis Model ✅ **NEW**
- **CNN Architecture**: ResNet50 with custom classifier head
- **Classifications**: Binary dorso/palma (back/palm) classification
- **Colorimetry Pipeline**: HSV + YCrCb color space filtering with K-means clustering
- **Color Types**: 5 traditional palm color classifications
- **Features**: Skin detection, dominant color extraction, morphological operations
- **Input Size**: 224x224 pixels (auto-resized from any input)
- **Output**: CNN predictions + colorimetry analysis + color type classification

### Profile Rotation Assessment Model ✅ **NEW**
- **Architecture**: EfficientNet-B0 with multi-label classification head
- **Classifications**: 7 profile rotation classes including 'aceptable'
- **Features**: Multi-label predictions, pattern-aware inference, viability assessment
- **Input Size**: 224x224 pixels (auto-resized from any input)
- **Output**: Multi-label rotation predictions with confidence scores and recommendations
- **Pattern Recognition**: Respects annotation patterns (aceptable is standalone)

### Frontal Rotation Assessment Model ✅ **NEW**
- **Architecture**: EfficientNet-B0 with multi-label classification head
- **Classifications**: 5 frontal face rotation classes including 'aceptable'
- **Features**: Multi-label predictions, pattern-aware inference, viability assessment
- **Input Size**: 224x224 pixels (auto-resized from any input)
- **Output**: Multi-label rotation predictions with confidence scores and recommendations
- **Pattern Recognition**: Respects annotation patterns (aceptable is standalone)

### Body Model Classifications ✅ **NEW**

#### Body Type Categories (7 Classes)
- **Bilioso/NormalPocaGrasa**: Normal build with low body fat
- **Nervioso/Delgado**: Ectomorphic, lean build
- **SanguineoLinfatico/MusculosoGordo**: Muscular with higher body fat
- **Sanguineo/Musculoso**: Mesomorphic, athletic build
- **Flematico/Gordograsacuelga**: Endomorphic with soft tissue
- **Linfatico/Gordo**: Endomorphic, higher body fat
- **BiliosoSanguineo/NormalMusculoso**: Balanced muscular build

#### Anthropometric Measurements
- **Skull Ratio Analysis**: Adult (12.5-14.3%), Child (16-18%)
- **17 Body Keypoints**: Full pose detection including head, torso, limbs
- **Head Orientation**: Frontal, tilted left/right with angle measurements
- **Detection Methods**: Anatomical estimation + contour refinement

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
- **GPU**: NVIDIA GPU with 6GB+ VRAM (recommended for all 11 modules)
- **RAM**: 55GB+ system memory (for all 11 active modules)
- **Storage**: 30GB+ for models and containers
- **CPU**: Multi-core processor for preprocessing

### Port Allocation ✅ **ALL 11 PORTS OCCUPIED**
- **Frontal Analysis**: 8000-8002, 8008 ✅ **ALL COMPLETE**
  - Morfológico: 8000 ✅
  - Antropométrico: 8001 ✅
  - Validación: 8002 ✅
  - Espejo: 8008 ✅
- **Profile Analysis**: 8003-8005, 8010 ✅ **ALL COMPLETE**
  - Morfológico: 8003 ✅
  - Antropométrico: 8004 ✅ **UPDATED CALIBRATIONS**
  - Validación: 8005 ✅
  - Preprocessing: 8010 ✅ **NEW!**
- **Body Analysis**: 8006-8007, 8009 ✅ **ALL COMPLETE** ✅ **UPDATED!**
  - Morfológico: 8006 ✅ **UPDATED MODEL**
  - Antropométrico: 8007 ✅
  - Hand Analysis: 8009 ✅ **UPDATED COLORIMETRY**

### Independent Scaling ✅
Each module can be scaled independently:
- Deploy multiple instances of any module
- Use load balancer for request distribution
- Configure GPU memory optimization per module
- Implement request queuing for batch processing

### Monitoring ✅ **ALL 10 SERVICES**
```bash
# All complete modules health check
curl http://localhost:8000/health  # Frontal Morfologico ✅
curl http://localhost:8001/health  # Frontal Antropometrico ✅
curl http://localhost:8002/health  # Frontal Validacion ✅
curl http://localhost:8008/health  # Frontal Espejo ✅
curl http://localhost:8003/health  # Profile Morfologico ✅
curl http://localhost:8004/health  # Profile Antropometrico ✅
curl http://localhost:8005/health  # Profile Validacion ✅
curl http://localhost:8006/health  # Body Morfologico ✅ NEW!
curl http://localhost:8007/health  # Body Antropometrico ✅ NEW!
curl http://localhost:8009/health  # Hand Analysis ✅ NEW!
curl http://localhost:8011/health  # Profile Rotation ✅ NEW!
curl http://localhost:8012/health  # Frontal Rotation ✅ NEW!

# Container status
docker ps

# GPU utilization
nvidia-smi
```

## Architecture Status ✅ **PROJECT EXPANDED**

### Current Status ✅ **ALL 12 MODULES OPERATIONAL**
- ✅ **Frontal Analysis Complete**: validacion, morfologico, antropometrico, espejo, rotacion (Ports 8000-8002, 8008, 8012) ✅ **ALL COMPLETE**
- ✅ **Profile Analysis Complete**: morfologico, antropometrico, validacion, preprocesamiento, rotacion (Ports 8003-8005, 8010-8011) ✅ **ALL COMPLETE**
- ✅ **Body Analysis Complete**: morfologico, antropometrico, manos (Ports 8006-8007, 8009) ✅ **ALL COMPLETE** ✅ **NEW!**

### Complete AI Analysis Pipeline ✅ **EXPANDED**
The SG_prod AI analysis production pipeline is now **EXPANDED** with all 12 modules operational:

#### **Frontal Image Processing Pipeline** ✅
1. **Frontal Rotation** (Port 8012): Assess face orientation suitability for analysis ✅ **NEW!**
2. **Frontal Validacion** (Port 8002): Validate image quality and detect issues ✅
3. **Frontal Morfologico** (Port 8000): Perform morphological analysis ✅
4. **Frontal Antropometrico** (Port 8001): Conduct detailed measurements ✅
5. **Frontal Espejo** (Port 8008): Mirror-based comprehensive analysis with decision tree classification ✅

#### **Profile Image Processing Pipeline** ✅
1. **Profile Rotation** (Port 8011): Assess profile orientation suitability for analysis ✅ **NEW!**
2. **Profile Validacion** (Port 8005): Profile quality validation and occlusion detection ✅
3. **Profile Morfologico** (Port 8003): Complete profile morphological analysis ✅
4. **Profile Antropometrico** (Port 8004): Advanced anthropometric measurements ✅

#### **Body Image Processing Pipeline** ✅ **NEW COMPLETE**
1. **Body Morfologico** (Port 8006): Body type classification and morphological analysis ✅ **NEW!**
2. **Body Antropometrico** (Port 8007): Skull detection and body anthropometric measurements ✅ **NEW!**
3. **Hand Analysis** (Port 8009): Hand classification and advanced palm colorimetry analysis ✅ **NEW!**

### Future Extensions 🔄
- 🔄 **Master Orchestration**: Multi-service deployment and result aggregation
- 🔄 **3D Analysis Pipeline**: Depth-aware facial and body reconstruction
- 🔄 **Real-time Processing**: WebRTC integration for live analysis
- 🔄 **Multi-modal Analysis**: Combined facial, profile, and body analysis workflows

## Usage Workflow ✅ **COMPLETE PIPELINES**

### Recommended Analysis Pipeline

#### For Frontal Images ✅ **COMPLETE WORKFLOW**
1. **Frontal Rotation** (Port 8012): Assess face orientation and suitability for analysis ✅ **NEW!**
2. **Frontal Validacion** (Port 8002): Validate image quality and detect issues ✅
3. **Frontal Morfologico** (Port 8000): Perform morphological analysis if suitable ✅
4. **Frontal Antropometrico** (Port 8001): Conduct detailed measurements ✅
5. **Frontal Espejo** (Port 8008): Mirror-based comprehensive analysis with decision tree classification ✅

#### For Profile Images ✅ **COMPLETE WORKFLOW**
1. **Profile Rotation** (Port 8011): Assess profile orientation and suitability for analysis ✅ **NEW!**
2. **Profile Validacion** (Port 8005): Profile quality validation and occlusion detection ✅
3. **Profile Morfologico** (Port 8003): Complete profile morphological analysis ✅
4. **Profile Antropometrico** (Port 8004): Advanced anthropometric measurements ✅

#### For Body Images ✅ **NEW COMPLETE WORKFLOW**
1. **Body Morfologico** (Port 8006): Body type classification and morphological insights ✅ **NEW!**
2. **Body Antropometrico** (Port 8007): Skull measurements and body anthropometric analysis ✅ **NEW!**

#### For Hand Images ✅ **NEW COMPLETE WORKFLOW**
1. **Hand Analysis** (Port 8009): Comprehensive hand side classification and palm colorimetry analysis ✅ **NEW!**

### Quality-First Approach ✅
The **Validacion** modules serve as quality gates, identifying:
- Hair covering facial features
- Problematic accessories (glasses, objects)
- Poor lighting or image quality
- Unsuitable facial expressions or poses
- Recommendations for better image capture

### Multi-Modal Analysis ✅ **NEW**
The **Body Analysis** modules provide:
- Body type classification with morphological insights
- Comprehensive anthropometric measurements
- Skull-to-body proportion analysis
- Age assessment based on anatomical proportions
- Full body pose detection and keypoint analysis

The **Hand Analysis** module provides:
- CNN-based hand side classification (dorso/palma)
- Advanced palm colorimetry analysis
- Traditional color type classification
- Dominant color extraction and analysis
- Comprehensive visualizations and reports

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
**Status**: ✅ **PRODUCTION READY - ALL 10 MODULES COMPLETE** ✅
