# Frontal Morphological Analysis Module

## Overview

The Frontal Morphological Analysis module is a comprehensive facial analysis system designed for frontal face images. It combines advanced deep learning models for facial landmark detection, morphological characteristic classification, and anthropometric point detection to provide detailed facial analysis suitable for medical, anthropological, and biometric applications.

## Features

### Core Capabilities
- **Dual-Model Architecture**: Separate detection and classification models for optimal accuracy
- **Facial Landmark Detection**: 18 anatomical landmark classes using Faster R-CNN
- **Morphological Classification**: 54 facial characteristics with confidence scoring
- **Anthropometric Point Detection**: Precise anatomical point localization
- **Multi-Model Integration**: Seamless combination of detection and classification results
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
  "tag_mapping_entries": 54,
  "total_tags": 54
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
        "landmark_class": "OjD",
        "tag": "tag_33",
        "tag_name": "normal",
        "score": 0.92,
        "tag_confidence": 0.87,
        "box": [145.2, 78.4, 165.8, 98.6]
      },
      {
        "landmark_class": "Nariz",
        "tag": "tag_21",
        "tag_name": "grueso",
        "score": 0.89,
        "tag_confidence": 0.76,
        "box": [155.1, 95.3, 175.4, 125.7]
      }
    ]
  },
  "anthropometric_points": {
    "count": 8,
    "detections": [
      {
        "point_id": "1",
        "coordinates": [160.5, 85.2],
        "confidence": 0.94
      },
      {
        "point_id": "2", 
        "coordinates": [142.8, 102.6],
        "confidence": 0.88
      }
    ]
  },
  "summary": {
    "total_detections": 20,
    "confidence_threshold": 0.5,
    "image_processed": true
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
  "tag_mapping": {
    "tag_0": "0",
    "tag_1": "1",
    "tag_2": "2",
    "tag_3": "3",
    "tag_4": "a_n",
    "tag_5": "ab",
    "tag_6": "abierta",
    "tag_7": "adelgazamiento",
    "tag_8": "al",
    "tag_9": "ap",
    "tag_10": "ar",
    "tag_11": "bigote",
    "tag_12": "cabellos_sueltos",
    "tag_13": "carnosos",
    "tag_14": "crl",
    "tag_15": "cv",
    "tag_16": "delgada",
    "tag_17": "el",
    "tag_18": "fleco",
    "tag_19": "fr",
    "tag_20": "g",
    "tag_21": "grueso",
    "tag_22": "h",
    "tag_23": "hn",
    "tag_24": "i",
    "tag_25": "lineas_sonriza",
    "tag_26": "lineas_verticales",
    "tag_27": "ll",
    "tag_28": "lunar",
    "tag_29": "md",
    "tag_30": "md_a",
    "tag_31": "mercurial",
    "tag_32": "nd",
    "tag_33": "normal",
    "tag_34": "nrml",
    "tag_35": "nt",
    "tag_36": "on",
    "tag_37": "pc",
    "tag_38": "pg",
    "tag_39": "pl",
    "tag_40": "planos",
    "tag_41": "pliegue",
    "tag_42": "pm",
    "tag_43": "pn",
    "tag_44": "ptosis",
    "tag_45": "pursed",
    "tag_46": "rc",
    "tag_47": "rd",
    "tag_48": "salido",
    "tag_49": "sl",
    "tag_50": "solar",
    "tag_51": "sonriendo",
    "tag_52": "sp_sl",
    "tag_53": "uniceja"
  },
  "total_tags": 54,
  "sample_tags": ["tag_0", "tag_1", "tag_2", "tag_3", "tag_4"]
}
```

## Facial Landmark Classes

The system detects 18 distinct facial landmark classes:

### Eye Region (4 classes)
- **CjD**: Right eyebrow (Ceja Derecha)
- **CjIz**: Left eyebrow (Ceja Izquierda)
- **OjD**: Right eye (Ojo Derecho)
- **OjIz**: Left eye (Ojo Izquierdo)

### Cheek Region (2 classes)
- **CchD**: Right cheek (Cachete Derecho)
- **CchIzq**: Left cheek (Cachete Izquierdo)

### Central Facial Features (3 classes)
- **Nariz**: Nose
- **N**: Nasal area
- **F**: Forehead (Frente)

### Mouth Region (3 classes)
- **Bc**: Mouth/lips (Boca)
- **PmlD**: Right cheekbone (Pomulo Derecho)
- **PmlIz**: Left cheekbone (Pomulo Izquierdo)

### Ear Region (6 classes)
- **TrExCjDr**: Right external ear tragus
- **TrExCjIz**: Left external ear tragus
- **TrInCjDr**: Right internal ear tragus
- **TrInCjIz**: Left internal ear tragus
- **OD**: Right ear (Oreja Derecha)
- **OIz**: Left ear (Oreja Izquierda)

## Morphological Characteristics (54 Tags)

### Numeric Classifications (0-3)
- **tag_0-tag_3**: Numeric morphological codes

### Facial Structure
- **a_n**: Aquiline nose
- **ab**: Open/prominent
- **abierta**: Open feature
- **adelgazamiento**: Thinning
- **delgada**: Thin/narrow
- **grueso**: Thick/coarse
- **carnosos**: Fleshy/full

### Eye Characteristics
- **al**: Eye-related measurement
- **ap**: Eye appearance
- **ar**: Eye area
- **el**: Eye line
- **ptosis**: Drooping eyelid
- **planos**: Flat/plane

### Hair and Forehead
- **bigote**: Mustache
- **cabellos_sueltos**: Loose hair
- **fleco**: Bangs/fringe
- **fr**: Forehead-related
- **uniceja**: Unibrow

### Nose Features
- **h**: Height-related
- **hn**: Nasal height
- **normal**: Normal/standard
- **nrml**: Normal variant
- **nt**: Nasal tip
- **salido**: Protruding

### Mouth and Expression
- **lineas_sonriza**: Smile lines
- **lineas_verticales**: Vertical lines
- **pursed**: Pursed lips
- **sonriendo**: Smiling
- **pliegue**: Fold/crease

### Facial Features
- **lunar**: Mole/beauty mark
- **mercurial**: Mercury-like (reflective)
- **solar**: Sun-related (pigmentation)
- **cv**: Curve-related
- **crl**: Curl-related

### Measurements
- **md**: Mid-distance
- **md_a**: Mid-distance variant
- **nd**: Nasal distance
- **on**: Other measurement
- **pc**: Percentage/proportion
- **pg**: Page/position
- **pl**: Plane/flat
- **pm**: Mid-point
- **pn**: Nasal point
- **rc**: Radius/curve
- **rd**: Radius/distance
- **sl**: Side length
- **sp_sl**: Special side length

### General
- **g**: General measurement
- **i**: Index/indicator
- **ll**: Line length

## Technical Specifications

### Detection Model Architecture
- **Base Model**: Faster R-CNN with ResNet-50 FPN backbone
- **Classes**: 18 landmark classes + background
- **Input Processing**: 224×224 pixel normalization
- **Output**: Bounding boxes with confidence scores

### Classification Model Architecture
- **Architecture**: Custom CNN with 4 convolutional layers
- **Feature Layers**: 32 → 64 → 128 → 256 channels
- **Classifier**: 1024 → 512 → 54 classes
- **Input Size**: 64×64 pixels
- **Normalization**: ImageNet standard (mean/std)

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

- `models/facial_landmarks_detection_model.pth` - Faster R-CNN landmark detection model
- `models/best_facial_landmark_classifier.pth` - CNN characteristic classification model  
- `models/facial_points_detection_model.pth` - Anthropometric point detection model

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
