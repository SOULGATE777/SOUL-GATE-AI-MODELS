# Frontal Morphological Analysis Module (V2)

## Overview

The Frontal Morphological Analysis module is a comprehensive facial analysis system designed for frontal face images. It combines advanced deep learning models for facial landmark detection, morphological characteristic classification, eyebrow size classification, and anthropometric point detection to provide detailed facial analysis suitable for medical, anthropological, and biometric applications.

**Version 2.0** introduces a **3-model ensemble** with bbox confinement validation and specialized eyebrow analysis.

## Features

### Core Capabilities
- **3-Model Ensemble**: Detection + Shape Classification + Eyebrow Size Classification
- **Facial Landmark Detection**: 23 anatomical landmark classes using Faster R-CNN
- **Shape Classification**: 45 morphological tags with bbox confinement validation
- **Eyebrow Size Classification**: 3-class dedicated model for eyebrow characteristics (ap, g, ngna)
- **Bbox Confinement**: Validates predictions per landmark type to prevent cross-landmark errors
- **Anthropometric Point Detection**: Precise anatomical point localization (13 points)
- **Multi-Model Integration**: Seamless combination of all detection and classification results
- **Real-time Visualization**: Beautiful annotated outputs with comprehensive analysis
- **Batch Processing**: Efficient processing of multiple images

## API Endpoints

### Base URL
```
http://localhost:8005
```

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "shape_tags": 45,
  "eyebrow_size_tags": 3,
  "eyebrow_size_model_loaded": true,
  "bbox_confinement_enabled": true
}
```

### 2. Complete Facial Analysis
```http
POST /analyze-face
```
**Parameters:**
- `file` (required): Image file (JPG, PNG, etc.)
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)
- `include_visualization` (optional): Boolean (default: true)

**Response:**
```json
{
  "facial_landmarks": {
    "count": 12,
    "detections": [
      {
        "landmark_class": "cj_d",
        "tag": "tag_38",
        "tag_name": "rc",
        "score": 0.92,
        "tag_confidence": 0.87,
        "size_tag": "g",
        "size_confidence": 0.91,
        "top_tags": [
          {"tag": "rc", "confidence": 0.87, "rank": 1},
          {"tag": "el", "confidence": 0.09, "rank": 2},
          {"tag": "cv", "confidence": 0.04, "rank": 3}
        ],
        "box": [145.2, 78.4, 165.8, 98.6]
      },
      {
        "landmark_class": "nariz",
        "tag": "tag_13",
        "tag_name": "grueso",
        "score": 0.89,
        "tag_confidence": 0.76,
        "top_tags": [...],
        "box": [155.1, 95.3, 175.4, 125.7]
      }
    ]
  },
  "eyebrow_size_classification": {
    "count": 2,
    "detections": [
      {
        "landmark_class": "cj_d",
        "size_tag": "g",
        "size_confidence": 0.91,
        "box": [145.2, 78.4, 165.8, 98.6]
      },
      {
        "landmark_class": "cj_i",
        "size_tag": "ap",
        "size_confidence": 0.88,
        "box": [120.4, 78.1, 140.6, 98.3]
      }
    ],
    "classes": ["ap", "g", "ngna"]
  },
  "anthropometric_points": {
    "count": 8,
    "detections": [
      {
        "point_class": "Point_1",
        "label_idx": 1,
        "score": 0.94,
        "bbox": [160.5, 85.2, 165.3, 90.1],
        "center_point": [162.9, 87.65],
        "point": [162.9, 87.65]
      }
    ]
  },
  "summary": {
    "total_detections": 20,
    "eyebrow_size_detections": 2,
    "confidence_threshold": 0.5,
    "image_processed": true,
    "bbox_confinement_applied": true
  },
  "visualization_path": "/app/results/analysis_uuid.jpg"
}
```

### 3. Facial Landmark Detection Only
```http
POST /detect-landmarks
```
**Parameters:**
- `file` (required): Image file
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)
- `include_visualization` (optional): Boolean (default: true)

**Response:**
```json
{
  "landmarks": [
    {
      "landmark_class": "CjD",
      "tag": "tag_4",
      "tag_name": "a_n",
      "score": 0.84,
      "tag_confidence": 0.71,
      "box": [120.4, 65.8, 140.2, 85.6]
    },
    {
      "landmark_class": "F",
      "tag": "tag_18",
      "tag_name": "fleco",
      "score": 0.78,
      "tag_confidence": 0.69,
      "box": [145.1, 45.3, 185.7, 75.9]
    }
  ],
  "count": 2,
  "confidence_threshold": 0.5,
  "visualization_path": "/app/results/landmarks_uuid.jpg"
}
```

### 4. Anthropometric Point Detection Only
```http
POST /detect-points
```
**Parameters:**
- `file` (required): Image file
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)
- `include_visualization` (optional): Boolean (default: true)

**Response:**
```json
{
  "points": [
    {
      "point_id": "1",
      "coordinates": [160.5, 85.2],
      "confidence": 0.94
    },
    {
      "point_id": "3",
      "coordinates": [175.8, 95.4],
      "confidence": 0.89
    }
  ],
  "count": 2,
  "confidence_threshold": 0.5,
  "visualization_path": "/app/results/points_uuid.jpg"
}
```

### 5. Tag Mapping Information
```http
GET /tag-mapping
```
**Response:**
```json
{
  "shape_tags": [
    "0", "1", "2", "3", "a_n", "ab", "al", "ar",
    "crl", "cv", "delgada", "el", "fr", "grueso", "h", "hn", "i",
    "lineas_sonriza", "lineas_verticales", "ll", "lunar", "md", "md_a",
    "mercurial", "nd", "normal", "nrml", "nt", "on", "pc", "pg", "pl",
    "planos", "pliegue", "pm", "pn", "ptosis", "pursed", "rc", "rd",
    "salido", "sl", "solar", "sp_sl", "uniceja"
  ],
  "eyebrow_size_tags": ["ap", "g", "ngna"],
  "eyebrow_classes": ["cj_d", "cj_i"],
  "total_shape_tags": 45,
  "total_eyebrow_size_tags": 3,
  "bbox_confinement_mappings": {
    "cj_d": ["rc", "el", "cv"],
    "cj_i": ["rc", "el", "cv"],
    "nariz": ["delgada", "nrml", "grueso"],
    "bc": ["lunar", "mercurial", "pursed", "solar"],
    "n": ["i", "pn", "rd"],
    "oj_d": ["al", "crl", "fr", "md", "md_a"],
    "oj_i": ["al", "crl", "fr", "md", "md_a"],
    "entrecejo": ["lineas_verticales", "normal", "uniceja"],
    "parpado_dr": ["pliegue", "ptosis"],
    "parpado_i": ["pliegue", "ptosis"]
  },
  "sample_shape_tags": ["0", "1", "2", "3", "a_n", "ab", "al", "ar", "crl", "cv", "delgada", "el", "fr", "grueso", "h"]
}
```

## Facial Landmark Classes

The system detects 23 distinct facial landmark classes:

### Eye & Eyebrow Region (8 classes)
- **cj_d**: Right eyebrow (Ceja Derecha) - *includes size classification*
- **cj_i**: Left eyebrow (Ceja Izquierda) - *includes size classification*
- **oj_d**: Right eye (Ojo Derecho)
- **oj_i**: Left eye (Ojo Izquierdo)
- **parpado_dr**: Right eyelid
- **parpado_i**: Left eyelid
- **entrecejo**: Between eyebrows (glabella)
- **ac_d**, **ac_i**: Eye corners

### Cheek Region (4 classes)
- **cch_d**: Right cheek (Cachete Derecho)
- **cch_i**: Left cheek (Cachete Izquierdo)
- **pml_d**: Right cheekbone (Pomulo Derecho)
- **pml_i**: Left cheekbone (Pomulo Izquierdo)

### Central Facial Features (3 classes)
- **nariz**: Nose
- **n**: Nasal area
- **f**: Forehead (Frente)

### Mouth Region (1 class)
- **bc**: Mouth/lips (Boca)

### Ear Region (6 classes)
- **tr_ex_cj_dr**: Right external ear tragus
- **tr_ex_cj_i**: Left external ear tragus
- **tr_in_cj_d**: Right internal ear tragus
- **tr_in_cj_i**: Left internal ear tragus
- **o_d**: Right ear (Oreja Derecha)
- **o_i**: Left ear (Oreja Izquierda)

## Morphological Characteristics

### Shape Classification (45 Tags)

**Complete list of 45 shape tags:**
```
0, 1, 2, 3, a_n, ab, al, ar, crl, cv, delgada, el, fr, grueso, h, hn, i,
lineas_sonriza, lineas_verticales, ll, lunar, md, md_a, mercurial, nd,
normal, nrml, nt, on, pc, pg, pl, planos, pliegue, pm, pn, ptosis,
pursed, rc, rd, salido, sl, solar, sp_sl, uniceja
```

**Removed tags from V1** (improved data quality):
- `abierta`, `adelgazamiento`, `bigote`, `carnosos`, `cabellos_sueltos`, `fleco`, `sonriendo`

### Eyebrow Size Classification (3 Tags - Eyebrows Only)

Applied **only to `cj_d` and `cj_i`** landmarks:

- **ap**: Apretadas (narrow/thin eyebrows)
- **g**: Gruesas (thick/wide eyebrows)
- **ngna**: Normal/intermediate (neither narrow nor thick)

### Bbox Confinement Validation

The system validates shape predictions per landmark type. Examples:

- **Eyebrows** (`cj_d`, `cj_i`): Only accepts `rc`, `el`, `cv`
- **Nose** (`nariz`): Only accepts `delgada`, `nrml`, `grueso`
- **Mouth** (`bc`): Only accepts `lunar`, `mercurial`, `pursed`, `solar`
- **Eyes** (`oj_d`, `oj_i`): Only accepts `al`, `crl`, `fr`, `md`, `md_a`
- **Nostril** (`n`): Only accepts `i`, `pn`, `rd`
- **Eyelids** (`parpado_dr`, `parpado_i`): Only accepts `pliegue`, `ptosis`
- **Glabella** (`entrecejo`): Only accepts `lineas_verticales`, `normal`, `uniceja`

If the model predicts an invalid tag, the system automatically selects the highest-confidence valid tag.

## Technical Specifications

### Detection Model Architecture
- **Base Model**: Faster R-CNN with ResNet-50 FPN backbone
- **Classes**: 23 landmark classes + background
- **Input Processing**: 224×224 pixel normalization
- **Output**: Bounding boxes with confidence scores

### Shape Classification Model Architecture
- **Architecture**: Custom CNN with 4 convolutional layers
- **Feature Layers**: 32 → 64 → 128 → 256 channels
- **Classifier**: 1024 → 512 → 45 classes
- **Input Size**: 64×64 pixels
- **Normalization**: ImageNet standard (mean/std)
- **Validation**: Bbox confinement per landmark type

### Eyebrow Size Classification Model Architecture
- **Architecture**: Enhanced CNN with 4 convolutional layers + BatchNorm
- **Feature Layers**: 64 → 128 → 256 → 512 channels
- **Classifier**: 2048 → 1024 → 512 → 3 classes
- **Input Size**: 64×64 pixels
- **Normalization**: ImageNet standard (mean/std)
- **Applies to**: Only `cj_d` and `cj_i` landmarks

### Anthropometric Point Detector
- **Detection**: Precise anatomical point localization
- **Output**: Point coordinates with confidence scores
- **Integration**: Seamless combination with landmark detection

### Input Requirements
- **Image Formats**: JPG, PNG, BMP, TIFF
- **Orientation**: Frontal face view required
- **Quality**: Clear facial features, minimal occlusion
- **Lighting**: Even illumination preferred
- **Resolution**: Any resolution (auto-processed to 224×224)

### Output Formats
- **JSON**: Structured analysis data
- **Images**: Annotated visualization with overlays
- **Coordinates**: Pixel-level landmark and point locations

## Processing Pipeline

### 1. Image Preprocessing
```
Input Image → Resize (224×224) → Tensor Conversion → Normalization
```

### 2. Landmark Detection
```
Preprocessed Image → Faster R-CNN → Bounding Boxes → Confidence Filtering
```

### 3. Region Classification
```
For each detected landmark:
  Extract Region → Resize (64×64) → CNN Classification → Tag Assignment
```

### 4. Anthropometric Detection
```
Original Image → Point Detection Model → Coordinate Extraction → Confidence Scoring
```

### 5. Result Integration
```
Combine Landmarks + Tags + Points → Generate Visualization → Return Results
```

## Installation and Setup

### Docker Deployment (Recommended)
```bash
# Build the container
docker-compose build

# Run the service
docker-compose up -d

# Check service status
curl http://localhost:8005/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model files exist
ls models/facial_landmarks_detection_model.pth
ls models/best_facial_landmark_classifier.pth
ls models/facial_points_detection_model.pth

# Create required directories
mkdir -p /app/results

# Run the application
python app/main.py
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `PORT`: Service port (default: 8005)

## Usage Examples

### Python Client
```python
import requests

# Complete facial analysis
with open('frontal_face.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8005/analyze-face',
        files={'file': f},
        data={
            'confidence_threshold': 0.5,
            'include_visualization': True
        }
    )

results = response.json()

# Extract landmark information
landmarks = results['facial_landmarks']['detections']
for landmark in landmarks:
    print(f"Landmark: {landmark['landmark_class']}")
    print(f"Characteristic: {landmark['tag_name']}")
    print(f"Confidence: {landmark['score']:.2f}")
    print(f"Box: {landmark['box']}")
    print("---")

# Extract anthropometric points
points = results['anthropometric_points']['detections']
for point in points:
    print(f"Point {point['point_id']}: {point['coordinates']} (conf: {point['confidence']:.2f})")
```

### Landmark Detection Only
```python
import requests

with open('face_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8005/detect-landmarks',
        files={'file': f},
        data={'confidence_threshold': 0.6}
    )

landmarks = response.json()['landmarks']
print(f"Detected {len(landmarks)} landmarks")

# Group by landmark class
landmark_groups = {}
for landmark in landmarks:
    cls = landmark['landmark_class']
    if cls not in landmark_groups:
        landmark_groups[cls] = []
    landmark_groups[cls].append(landmark)

for cls, group in landmark_groups.items():
    print(f"{cls}: {len(group)} detections")
    for item in group:
        print(f"  - {item['tag_name']} (conf: {item['score']:.2f})")
```

### Anthropometric Points Only
```python
import requests

with open('face_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8005/detect-points',
        files={'file': f},
        data={'confidence_threshold': 0.5}
    )

points = response.json()['points']
print(f"Detected {len(points)} anthropometric points")

for point in points:
    x, y = point['coordinates']
    print(f"Point {point['point_id']}: ({x:.1f}, {y:.1f}) - conf: {point['confidence']:.2f}")
```

### cURL Examples
```bash
# Complete analysis
curl -X POST "http://localhost:8005/analyze-face" \
  -F "file=@frontal_face.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"

# Landmarks only
curl -X POST "http://localhost:8005/detect-landmarks" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.6"

# Points only
curl -X POST "http://localhost:8005/detect-points" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5"

# Get tag mapping
curl -X GET "http://localhost:8005/tag-mapping"
```

## Model Files Required

Place these files in the `models/` directory:

1. `facial_landmarks_detection_model.pth` - Faster R-CNN landmark detection (23 classes)
2. `best_shape_classifier.pth` - **NEW V2** Shape classification model (45 classes)
3. `best_eyebrow_size_classifier.pth` - **NEW V2** Eyebrow size model (3 classes)
4. `facial_points_detection_model.pth` - Anthropometric point detection (13 points)

**Note**: V1 used `best_facial_landmark_classifier.pth` (52 classes). V2 replaces this with 2 separate models.

## Results Directory

Generated files are stored in `/app/results/`:
- `analysis_{uuid}.jpg` - Complete analysis visualizations
- `landmarks_{uuid}.jpg` - Landmark detection visualizations
- `points_{uuid}.jpg` - Anthropometric point visualizations

## Error Handling

### Common Error Responses
- **400**: Invalid image format or parameters
- **503**: Models not loaded or initialization failed
- **500**: Processing error during analysis

### Troubleshooting
1. **Model Loading Issues**: Ensure all three model files exist and are valid
2. **CUDA Errors**: Check GPU availability and memory
3. **Memory Issues**: Reduce image resolution or use CPU mode
4. **Poor Detection**: Ensure frontal face orientation and clear visibility
5. **Tag Mapping Issues**: Verify tag consistency between models

## Performance Notes

- **Processing Time**: ~0.3-1.0 seconds per image (GPU)
- **Memory Usage**: ~2-4GB GPU memory
- **Accuracy**: 85-92% landmark detection, 78-88% characteristic classification
- **Throughput**: ~60-200 images/minute (depending on hardware)

## Applications

### Medical and Clinical
- **Craniofacial Analysis**: Morphological assessment for medical diagnosis
- **Genetic Studies**: Facial dysmorphology research
- **Orthodontics**: Treatment planning and progress monitoring
- **Plastic Surgery**: Pre/post-operative analysis

### Anthropological Research
- **Population Studies**: Facial morphology across ethnicities
- **Evolutionary Research**: Comparative morphological analysis
- **Forensic Applications**: Facial reconstruction and identification
- **Archaeological Studies**: Historical population analysis

### Biometric Applications
- **Identity Verification**: Enhanced facial recognition systems
- **Security Systems**: Multi-modal biometric authentication
- **Access Control**: High-precision identification
- **Surveillance**: Detailed facial analysis for security

### Research and Development
- **Computer Vision**: Advanced facial analysis algorithms
- **Machine Learning**: Training data generation and validation
- **Medical Imaging**: Automated diagnostic assistance
- **Aesthetic Analysis**: Beauty and symmetry assessment

## Version Information

- **API Version**: 1.0.0
- **Detection Model**: Faster R-CNN ResNet-50 FPN
- **Classification Model**: Custom CNN (54 classes)
- **Framework**: FastAPI + PyTorch + torchvision
- **Dependencies**: OpenCV, NumPy, PIL
- **Last Updated**: 2025

## Support

For technical support or questions about the frontal morphological analysis module, please refer to the main project documentation or contact the development team.
