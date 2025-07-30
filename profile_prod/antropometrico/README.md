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
  - `nariz protruyente` (proportion > 0.2)
  - `nariz normal` (0.17 ≤ proportion ≤ 0.2)
  - `nariz corta` (proportion < 0.17)
- **Tip Angle**: Angular measurement of nose tip orientation
- **Tip Classifications**:
  - `punta de nariz hacia arriba` (≥ 27°)
  - `punta de nariz promedio` (12° to 27°)
  - `punta hacia abajo` (< 12°)

### 3. Facial Thirds
- **Superior Third (34-22)**: Upper facial region
- **Middle Third (22-18)**: Mid-facial region
- **Inferior Third (18-10)**: Lower facial region

### 4. Mandible Analysis
- **Distance (3-9)**: Mandibular width measurement
- **Classifications**:
  - `Mandibula Sanguinea` (proportion ≥ 0.8)
  - `Mandibula intermedia sanguineo/bilosa` (0.75-0.8)
  - `Mandibula Bilosa` (0.40-0.75)
  - `Mandibula intermedia bilosa/nerviosa` (0.35-0.40)
  - `Mandibula Nerviosa` (< 0.35)
  - `Mandibula Linfatica` (point 9 missing)

### 5. Angular Analysis

#### 5.1 Mathematical Foundation
All angular measurements use the reference vector from point 22 to point 18 as the baseline. The system determines head direction using vector analysis: `head_direction = "right" if vector_22_18[0] > 0 else "left"`.

**Key Implementation Details:**
- **Vector Normalization**: All vectors are normalized using `vector / np.linalg.norm(vector)` before angle calculations
- **Dot Product Clamping**: `np.clip(dot_product, -1.0, 1.0)` prevents numerical errors in arccos
- **Angle Conversion**: All angles converted from radians using `np.degrees(np.arccos(cos_angle))`
- **Profile Awareness**: Left/right profile detection automatically adjusts angle signs and interpretations
- **Robust Edge Cases**: Handles infinite slopes and zero-division scenarios gracefully

#### 5.2 Forehead Angle Calculation (Points 24-22)
**Mathematical Process:**
1. **Reference Vector**: `vector_22_18 = [point_18[0] - point_22[0], point_18[1] - point_22[1]]`
2. **Measurement Vector**: `vector_22_24 = [point_24[0] - point_22[0], point_24[1] - point_22[1]]`
3. **Angle Calculation**: Uses dot product formula between normalized vectors
   - `cos_angle = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)`
   - `angle_degrees = np.degrees(np.arccos(cos_angle))`
4. **Direction Adjustment**:
   - **Left Profile**: `angle = abs(angle)` if turning right, `-angle` if turning left
   - **Right Profile**: `angle = -angle` if turning right, `abs(angle)` if turning left
5. **Normalization**: Angle normalized to 0-90° range (`abs(angle)` if negative, `180 - angle` if > 90°)

**Classification Thresholds:**
- `frente inclinada hacia atras`: angle > 15°
- `frente neutra`: 11° ≤ angle ≤ 15°
- `frente vertical`: angle < 11°

#### 5.3 Chin Angle Calculation (Points 18-11)
**Mathematical Process:**
1. **Reference Vector**: `vector_22_18` (same as above)
2. **Measurement Vector**: `vector_18_11 = [point_11[0] - point_18[0], point_11[1] - point_18[1]]`
3. **Angle Calculation**: Dot product between normalized vectors
4. **Face Side Detection**: Uses point 17 position relative to point 18
   - If `point_17[0] < point_18[0]`: Left side of face
   - Otherwise: Right side of face
5. **Direction Logic**:
   - **Left Side**: Positive angles = turning right, negative = turning left
   - **Right Side**: Positive angles = turning left, negative = turning right
6. **Normalization**: Angle clamped to -90° to +90° range

**Classification Thresholds:**
- `menton nervioso`: angle ≤ -5°
- `menton biloso/linfatico`: -5° < angle ≤ 5.5°
- `menton sanguineo`: angle > 5.5°

#### 5.4 Nose Tip Angle Calculation (Points 18-17)
**Mathematical Process:**
1. **Reference Vector**: `vector_22_18` (baseline)
2. **Perpendicular Calculation**:
   - `ref_slope = vector_22_18[1] / vector_22_18[0]`
   - `perp_slope = -1/ref_slope` (negative reciprocal)
   - `perp_vector = [1, perp_slope]`
3. **Measurement Vector**: `vector_18_17 = [point_17[0] - point_18[0], point_17[1] - point_18[1]]`
4. **Angle Calculation**: Dot product between `vector_18_17` and perpendicular vector
5. **Sign Correction**:
   - If `point_17[0] < point_18[0]`: Head turning right → negative angle
   - If `point_17[0] > point_18[0]`: Head turning left → positive angle
6. **Normalization**: Clamped to -90° to +90° range

**Classification Thresholds:**
- `punta de nariz hacia arriba`: angle ≥ 27°
- `punta de nariz promedio`: 12° ≤ angle < 27°
- `punta hacia abajo`: angle < 12°

#### 5.5 Implantation Angle Calculations

##### Superior Implantation (Point 22-4)
**Mathematical Process:**
1. **Reference Setup**: Uses same perpendicular vector as nose tip calculation
2. **Measurement Vector**: `vector_22_4 = [point_4[0] - point_22[0], point_4[1] - point_22[1]]`
3. **Angle Calculation**: Dot product between `vector_22_4` and perpendicular vector
4. **Normalization**: Absolute value applied, then clamped to 0-180° range
5. **Final Adjustment**: If angle > 90°, converted to `180 - angle`

**Classification:**
- `implantacion alta`: angle ≤ -9°
- `implantacion estandard`: angle > -9°

##### Inferior Implantation (Point 18-5)
**Mathematical Process:**
1. **Measurement Vector**: `vector_18_5 = [point_5[0] - point_18[0], point_5[1] - point_18[1]]`
2. **Same calculation process as superior implantation**

**Classification:**
- `implantacion baja`: angle ≤ -10°
- `implantacion estandard`: angle > -10°

##### Vector Intersection Angle (Vectors 22-18 and 1-3)
**Mathematical Process:**
1. **Vector 1**: `vector_22_18` (reference line)
2. **Vector 2**: `vector_1_3 = [point_1[0] - point_3[0], point_1[1] - point_3[1]]`
3. **Angle Calculation**: Direct dot product between normalized vectors
4. **Classification**:
   - `wide intersection`: angle > 90°
   - `right intersection`: angle = 90°
   - `acute intersection`: angle < 90°

### 6. Ear Analysis
- **Ear Width (2-6)**: Basic ear width measurement
- **Ear Length (4-5)**: Proportion to face length
  - `oreja larga` (proportion > 0.432)
  - `oreja normal` (0.38-0.432)
  - `oreja corta` (< 0.38)
- **Ear Lobe (3-5)**: Proportion to ear length
  - `lobulo grande` (proportion > 0.31)
  - `lobulo normal` (0.28-0.31)
  - `lobulo chico` (< 0.28)
- **Tragus-Antitragus (7-8)**: Proportion to ear width
  - `grande` (proportion ≥ 0.255)
  - `normal` (0.22-0.255)
  - `corta` (< 0.22)

### 7. Implantation Analysis
**Note**: Detailed mathematical calculations are provided in Section 5.5 above.
- **Superior Implantation (22-4)**: Upper ear attachment angle relative to perpendicular
  - `implantacion alta` (angle ≤ -9°)
  - `implantacion estandard` (angle > -9°)
- **Inferior Implantation (18-5)**: Lower ear attachment angle relative to perpendicular
  - `implantacion baja` (angle ≤ -10°)
  - `implantacion estandard` (angle > -10°)

### 8. Nasal Triangulation
- **Orifice Distance (26-17)**: Nasal opening measurement
- **Reference Distance (18-30)**: Nasal baseline
- **Classifications**:
  - `triangulacion de fosa` (proportion > 0.27)
  - `sin triangulacion de fosa` (proportion ≤ 0.27)

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
