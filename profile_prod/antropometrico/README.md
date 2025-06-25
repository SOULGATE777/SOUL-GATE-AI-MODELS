# Profile Anthropometric Analysis Module

## Overview

The Profile Anthropometric Analysis module is an advanced facial analysis system designed specifically for profile (lateral) facial images. It performs comprehensive anthropometric measurements using deep learning-based point detection and geometric analysis to classify facial features according to established anthropometric standards.

## Features

### Core Capabilities
- **Profile-Aware Point Detection**: Detects 30+ anatomical landmarks specifically optimized for profile images
- **Automatic Profile Side Detection**: Determines whether the image shows left or right profile using vector analysis
- **Spurious Point Filtering**: Intelligent filtering to remove false detections from the minority side
- **Comprehensive Measurements**: 20+ different anthropometric measurements and classifications
- **Angular Analysis**: Advanced geometric calculations for facial angles and proportions
- **Real-time Visualization**: Generates annotated images with detected points and measurement summaries

## API Endpoints

### Base URL
```
http://localhost:8004
```

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "profile-anthropometric",
  "model_loaded": true,
  "device": "cuda"
}
```

### 2. Complete Profile Analysis
```http
POST /analyze-profile-anthropometric
```
**Parameters:**
- `file` (required): Profile image file (JPG, PNG)
- `confidence_threshold` (optional): Float between 0.05-0.9 (default: 0.15)
- `include_visualization` (optional): Boolean (default: true)

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "profile_analysis": {
    "profile_side": "left|right|unknown",
    "total_detected_points": 25,
    "filtered_points": 22,
    "anthropometric_points": [
      {
        "class": "1",
        "coordinates": [x, y],
        "confidence": 0.85
      }
    ]
  },
  "anthropometric_measurements": {
    "reference_distance": 156.2,
    "nose_classification": "nariz normal",
    "mandibula_classification": "Mandibula Sanguinea",
    "forehead_classification": "frente neutra",
    "chin_classification": "menton biloso/linfatico",
    "ear_length_classification": "oreja normal",
    "nasal_triangulation_classification": "sin triangulacion de fosa"
  },
  "analysis_summary": {
    "confidence_threshold": 0.15,
    "has_measurements": true,
    "profile_determination": "left"
  },
  "visualization_path": "/app/results/profile_anthropometric_uuid.png",
  "visualization_url": "/visualization/profile_anthropometric_uuid.png"
}
```

### 3. Point Detection Only
```http
POST /detect-profile-points
```
**Parameters:**
- `file` (required): Profile image file
- `confidence_threshold` (optional): Float between 0.05-0.9

**Response:**
```json
{
  "total_detected_points": 28,
  "filtered_points": 24,
  "profile_side": "left",
  "detected_points": [...],
  "filtered_anthropometric_points": [...],
  "detection_summary": {
    "confidence_threshold": 0.15,
    "spurious_points_removed": 4,
    "profile_determination": "left"
  }
}
```

### 4. Model Information
```http
GET /model-info
```
**Response:**
```json
{
  "model_type": "Profile Anthropometric Point Detection",
  "device": "cuda",
  "point_classes": ["1", "2", "3", ..., "30"],
  "num_classes": 30,
  "heatmap_size": 112,
  "input_size": 224
}
```

## Anthropometric Points

The system detects 30 anatomical landmarks:

### Facial Structure Points
- **1-11**: Primary facial contour points
- **16-19**: Nasal structure points
- **22-26**: Forehead and nasal bridge points
- **30**: Additional nasal reference point

### Ear Structure Points
- **2, 6**: Ear width boundaries
- **3-5**: Ear length and lobe measurements
- **7-8**: Tragus and antitragus points

## Measurements and Classifications

### 1. Basic Measurements
- **Reference Distance (24-10)**: Primary scaling measurement
- **Head Direction**: Left/right profile determination via vector analysis

### 2. Nose Analysis
- **Distance (18-17)**: Nasal protrusion measurement
- **Classifications**:
  - `nariz protruyente` (proportion > 0.165)
  - `nariz normal` (0.14 ≤ proportion ≤ 0.165)
  - `nariz corta` (proportion < 0.14)
- **Tip Angle**: Angular measurement of nose tip orientation
- **Tip Classifications**:
  - `punta muy hacia arriba` (≥ 26°)
  - `punta de nariz hacia arriba` (≥ 19°)
  - `punta de nariz promedio` (0° to 19°)
  - `punta hacia abajo` (< 0°)

### 3. Facial Thirds
- **Superior Third (24-22)**: Upper facial region
- **Middle Third (22-16)**: Mid-facial region
- **Inferior Third (18-10)**: Lower facial region

### 4. Mandible Analysis
- **Distance (5-9)**: Mandibular width measurement
- **Classifications**:
  - `Mandibula Sanguinea` (proportion ≥ 0.75)
  - `Mandibula intermedia sanguineo/bilosa` (0.65-0.75)
  - `Mandibula Bilosa` (0.20-0.65)
  - `Mandibula intermedia bilosa/nerviosa` (0.10-0.20)
  - `Mandibula Nerviosa` (< 0.10)
  - `Mandibula Linfatica` (point 9 missing)

### 5. Angular Analysis
- **Forehead Angle (24-22)**:
  - `frente inclinada hacia atras` (> 15°)
  - `frente neutra` (11-15°)
  - `frente vertical` (< 11°)
- **Chin Angle (18-11)**:
  - `menton nervioso` (≤ -5°)
  - `menton biloso/linfatico` (-5° to 5.5°)
  - `menton sanguineo` (> 5.5°)

### 6. Ear Analysis
- **Ear Width (2-6)**: Basic ear width measurement
- **Ear Length (4-5)**: Proportion to face length
  - `oreja larga` (proportion > 0.33)
  - `oreja normal` (0.20-0.33)
  - `oreja corta` (< 0.20)
- **Ear Lobe (3-5)**: Proportion to ear length
  - `lobulo grande` (proportion > 0.25)
  - `lobulo normal` (0.19-0.25)
  - `lobulo chico` (< 0.19)
- **Tragus-Antitragus (7-8)**: Proportion to ear width
  - `grande` (proportion ≥ 0.2)
  - `normal` (0.1-0.2)
  - `corta` (< 0.1)

### 7. Implantation Analysis
- **Superior Implantation (18-4)**: Upper ear attachment
  - `implantacion alta` (angle ≤ -9°)
  - `implantacion estandard` (angle > -9°)
- **Inferior Implantation (18-5)**: Lower ear attachment
  - `implantacion baja` (angle ≤ -10°)
  - `implantacion estandard` (angle > -10°)

### 8. Nasal Triangulation
- **Orifice Distance (26-17)**: Nasal opening measurement
- **Reference Distance (18-30)**: Nasal baseline
- **Classifications**:
  - `triangulacion de fosa` (proportion > 0.1)
  - `sin triangulacion de fosa` (proportion ≤ 0.1)

## Technical Specifications

### Model Architecture
- **Base Model**: ResNet-50 backbone with custom decoder
- **Point Detection**: Heatmap-based keypoint detection
- **Profile Classification**: Integrated left/right profile classifier
- **Input Size**: 224x224 pixels
- **Heatmap Resolution**: 112x112 pixels

### Input Requirements
- **Image Formats**: JPG, PNG
- **Orientation**: Profile (lateral) view required
- **Quality**: Clear facial features, minimal occlusion
- **Lighting**: Even illumination preferred

### Output Formats
- **JSON**: Structured measurement data
- **PNG**: Annotated visualization images
- **Base64**: Embedded image data in API responses

## Installation and Setup

### Docker Deployment (Recommended)
```bash
# Build the container
docker-compose build

# Run the service
docker-compose up -d

# Check service status
curl http://localhost:8004/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model file exists
ls models/profile_aware_point_detection_model.pth

# Run the application
python app/main.py
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: auto-detect)
- `PORT`: Service port (default: 8004)

## Usage Examples

### Python Client
```python
import requests
import json

# Analyze profile image
with open('profile_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8004/analyze-profile-anthropometric',
        files={'file': f},
        data={
            'confidence_threshold': 0.15,
            'include_visualization': True
        }
    )

results = response.json()
print(f"Profile side: {results['profile_analysis']['profile_side']}")
print(f"Nose type: {results['anthropometric_measurements']['nose_classification']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8004/analyze-profile-anthropometric" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.15" \
  -F "include_visualization=true"
```

## Model Files Required

- `models/profile_aware_point_detection_model.pth` - Main point detection model

## Results Directory

Generated files are stored in `/app/results/`:
- `profile_anthropometric_{uuid}.png` - Visualization images

## Error Handling

### Common Error Responses
- **400**: Invalid image format or parameters
- **503**: Model not loaded or initialization failed
- **500**: Analysis processing error

### Troubleshooting
1. **Model Loading Issues**: Ensure model file exists and is accessible
2. **CUDA Errors**: Check GPU availability and CUDA installation
3. **Memory Issues**: Reduce batch size or use CPU mode
4. **Poor Detection**: Ensure profile orientation and good image quality

## Performance Notes

- **Processing Time**: ~0.5-2 seconds per image (GPU)
- **Memory Usage**: ~2-4GB GPU memory
- **Accuracy**: Optimized for clear profile images
- **Throughput**: ~30-60 images/minute (depending on hardware)

## Version Information

- **API Version**: 1.0.0
- **Model Version**: Profile-aware point detection
- **Framework**: FastAPI + PyTorch
- **Last Updated**: 2025

## Support

For technical support or questions about the anthropometric analysis module, please refer to the main project documentation or contact the development team.
