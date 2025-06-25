# Body Anthropometric Analysis Module

## Overview

The Body Anthropometric Analysis module is a production-ready system for comprehensive body measurements and skull detection using advanced YOLO pose estimation. It performs anatomical assessments, calculates body proportions, and provides detailed anthropometric measurements suitable for clinical, research, and fitness applications.

## Features

### Core Capabilities
- **Full Body Pose Detection**: 17-point YOLO pose estimation with high accuracy
- **Intelligent Skull Detection**: Anatomical skull bounding box calculation using facial landmarks
- **Contour Refinement**: Advanced image processing for precise skull measurements
- **Body Proportion Analysis**: Skull-to-body ratio calculations with anatomical assessments
- **Head Orientation Detection**: Automatic detection of head tilt and orientation
- **Multi-Person Support**: Simultaneous analysis of multiple persons in single image
- **Batch Processing**: Efficient processing of multiple images
- **Real-time Visualization**: Annotated images with measurements and keypoints

## API Endpoints

### Base URL
```
http://localhost:8007
```

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "body-anthropometric-analysis",
  "model_loaded": true,
  "device": "cuda",
  "model_type": "YOLOv8n-pose",
  "keypoints_supported": 17
}
```

### 2. Complete Body Anthropometric Analysis
```http
POST /analyze-body-anthropometry
```
**Parameters:**
- `file` (required): Image file (JPG, PNG, etc.)
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)
- `include_visualization` (optional): Boolean (default: true)
- `detailed_analysis` (optional): Boolean (default: true)

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "image_path": "/path/to/image",
  "image_shape": [height, width, channels],
  "num_persons": 1,
  "anthropometric_analysis": [
    {
      "person_id": 1,
      "keypoint_summary": {
        "total_keypoints": 15,
        "total_possible": 17,
        "detection_percentage": 88.2,
        "body_parts_detected": 6,
        "head_keypoints": 5,
        "torso_keypoints": 4,
        "keypoint_completeness": "excellent"
      },
      "body_proportions": {
        "skull_height": 156.3,
        "skull_width": 112.8,
        "body_height": 1245.6,
        "skull_to_body_ratio": 0.125,
        "skull_percentage": 12.5,
        "measurements_available": true,
        "detection_method": "nose_anatomical+contour_refined",
        "skull_bbox": [x_min, y_min, x_max, y_max],
        "head_orientation": "frontal (2.3°)",
        "anatomical_assessment": "Normal adult skull proportions"
      },
      "confidence_analysis": {
        "overall_average": 0.82,
        "confidence_range": {"min": 0.51, "max": 0.94},
        "head_average": 0.89,
        "reliability_assessment": "excellent - very reliable measurements"
      },
      "detailed_analysis": {
        "skull_analysis": {
          "skull_to_body_percentage": 12.5,
          "anatomical_classification": "Normal adult skull proportions",
          "comparison_to_norms": {
            "adult_range": "12.5-14.3%",
            "child_range": "16-18%",
            "measured_value": "12.5%"
          }
        },
        "age_assessment": {
          "classification": "adult_proportions",
          "description": "Skull proportions consistent with adult anatomy",
          "confidence": "high"
        },
        "body_composition": {
          "torso_detection_quality": "good",
          "limb_detection_quality": "good",
          "pose_assessment": "excellent - suitable for detailed anthropometric analysis"
        },
        "anthropometric_insights": {
          "head_orientation": "frontal (2.3°)",
          "measurement_reliability": "high",
          "recommended_measurements": ["Good detection quality - measurements are reliable"]
        }
      }
    }
  ],
  "analysis_summary": {
    "total_persons_detected": 1,
    "successful_skull_measurements": 1,
    "measurement_success_rate": 100.0,
    "average_detection_confidence": 0.82,
    "overall_quality": "excellent",
    "recommendations": ["Analysis completed successfully with good quality measurements"]
  },
  "visualization_path": "/app/results/anthropometry_uuid.png",
  "visualization_url": "/visualization/anthropometry_uuid.png"
}
```

### 3. Skull Detection Only
```http
POST /detect-skull-measurements
```
**Parameters:**
- `file` (required): Image file
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)
- `include_contour_refinement` (optional): Boolean (default: true)

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "image_path": "/path/to/image",
  "skull_detections": [
    {
      "person_id": 1,
      "skull_height": 156.3,
      "skull_width": 112.8,
      "body_height": 1245.6,
      "skull_to_body_ratio": 0.125,
      "skull_percentage": 12.5,
      "measurements_available": true,
      "detection_method": "nose_anatomical+contour_refined",
      "skull_bbox": [x_min, y_min, x_max, y_max],
      "head_orientation": "frontal (2.3°)",
      "anatomical_assessment": "Normal adult skull proportions",
      "head_keypoints_detected": 5,
      "total_keypoints_detected": 15,
      "head_keypoint_confidences": {
        "nose": 0.94,
        "left_eye": 0.89,
        "right_eye": 0.91,
        "left_ear": 0.76,
        "right_ear": 0.82
      },
      "average_head_confidence": 0.86
    }
  ],
  "processing_successful": true,
  "num_persons": 1
}
```

### 4. Pose Detection Only
```http
POST /detect-pose-keypoints
```
**Parameters:**
- `file` (required): Image file
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "image_path": "/path/to/image",
  "image_shape": [height, width, channels],
  "detections": [
    {
      "keypoints": {
        "nose": [x, y],
        "left_eye": [x, y],
        "right_eye": [x, y],
        "left_shoulder": [x, y],
        "right_shoulder": [x, y]
      },
      "body_parts": {
        "head": {"nose": [x, y], "left_eye": [x, y], "right_eye": [x, y]},
        "torso": {"left_shoulder": [x, y], "right_shoulder": [x, y]},
        "left_arm": {"left_shoulder": [x, y], "left_elbow": [x, y]},
        "right_arm": {"right_shoulder": [x, y], "right_elbow": [x, y]}
      },
      "confidence_scores": {
        "nose": 0.94,
        "left_eye": 0.89,
        "right_eye": 0.91
      }
    }
  ],
  "num_persons": 1,
  "processing_successful": true
}
```

### 5. Batch Analysis
```http
POST /batch-analyze
```
**Parameters:**
- `files` (required): List of image files (max 5)
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)
- `detailed_analysis` (optional): Boolean (default: false)

**Response:**
```json
{
  "batch_id": "uuid-string",
  "total_images": 3,
  "successful_analyses": 2,
  "results": [
    {
      "analysis_id": "uuid-string",
      "batch_index": 0,
      "filename": "image1.jpg",
      "anthropometric_analysis": [...]
    },
    {
      "analysis_id": "uuid-string",
      "batch_index": 1,
      "filename": "image2.jpg",
      "error": "Analysis failed"
    }
  ]
}
```

### 6. Model Information
```http
GET /model-info
```
**Response:**
```json
{
  "model_architecture": "YOLOv8n-pose",
  "model_type": "pose_detection",
  "input_format": "RGB image",
  "keypoints_detected": 17,
  "keypoint_names": ["nose", "left_eye", "right_eye", ...],
  "body_parts": {
    "head": [0, 1, 2, 3, 4],
    "torso": [5, 6, 11, 12],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16]
  },
  "measurements_provided": [
    "skull_dimensions",
    "body_proportions", 
    "skull_to_body_ratio",
    "anatomical_assessment",
    "head_orientation"
  ],
  "device": "cuda"
}
```

## YOLO Keypoints Detected

The system detects 17 anatomical keypoints using YOLOv8n-pose:

### Facial/Head Points (0-4)
- **0**: `nose` - Central facial landmark
- **1**: `left_eye` - Left eye center
- **2**: `right_eye` - Right eye center  
- **3**: `left_ear` - Left ear position
- **4**: `right_ear` - Right ear position

### Upper Body Points (5-10)
- **5**: `left_shoulder` - Left shoulder joint
- **6**: `right_shoulder` - Right shoulder joint
- **7**: `left_elbow` - Left elbow joint
- **8**: `right_elbow` - Right elbow joint
- **9**: `left_wrist` - Left wrist joint
- **10**: `right_wrist` - Right wrist joint

### Lower Body Points (11-16)
- **11**: `left_hip` - Left hip joint
- **12**: `right_hip` - Right hip joint
- **13**: `left_knee` - Left knee joint
- **14**: `right_knee` - Right knee joint
- **15**: `left_ankle` - Left ankle joint
- **16**: `right_ankle` - Right ankle joint

## Body Part Groupings

Keypoints are automatically grouped into anatomical regions:

- **Head**: nose, eyes, ears (points 0-4)
- **Torso**: shoulders, hips (points 5, 6, 11, 12)
- **Left Arm**: left shoulder, elbow, wrist (points 5, 7, 9)
- **Right Arm**: right shoulder, elbow, wrist (points 6, 8, 10)
- **Left Leg**: left hip, knee, ankle (points 11, 13, 15)
- **Right Leg**: right hip, knee, ankle (points 12, 14, 16)

## Anthropometric Measurements

### 1. Skull Measurements
- **Skull Height**: Vertical dimension of skull bounding box
- **Skull Width**: Horizontal dimension of skull bounding box
- **Skull Bounding Box**: Precise anatomical skull boundaries
- **Detection Method**: Algorithm used (nose_anatomical, eye_anatomical, contour_refined)

### 2. Body Proportions
- **Body Height**: Total vertical span from detected keypoints
- **Skull-to-Body Ratio**: Skull height / body height
- **Skull Percentage**: Ratio expressed as percentage

### 3. Head Orientation Analysis
- **Orientation Detection**: frontal, tilted_left, tilted_right
- **Tilt Angle**: Precise angle measurement in degrees
- **Reliability Assessment**: Impact on measurement accuracy

### 4. Anatomical Classifications

#### Adult vs Child Assessment
- **Adult Proportions**: 12.5-14.3% skull-to-body ratio
- **Child Proportions**: 16-18% skull-to-body ratio
- **Intermediate**: Between typical ranges

#### Measurement Quality
- **Excellent**: ≥15 keypoints, ≥4 head keypoints
- **Good**: ≥12 keypoints, ≥3 head keypoints
- **Fair**: ≥8 keypoints, ≥2 head keypoints
- **Poor**: <8 keypoints or <2 head keypoints

#### Confidence Assessment
- **Excellent**: Head avg ≥0.8, Overall avg ≥0.7
- **Good**: Head avg ≥0.6, Overall avg ≥0.6
- **Fair**: Head avg ≥0.4, Overall avg ≥0.5
- **Poor**: Below fair thresholds

## Skull Detection Methods

### 1. Nose-Anatomical Method
- **Primary Method**: Uses nose position as skull center
- **Scaling**: Eye distance × 2.4 for skull width
- **Proportions**: 1.4:1 height-to-width ratio
- **Positioning**: Nose centered, 15% down from skull center

### 2. Eye-Anatomical Method
- **Backup Method**: When nose not detected
- **Scaling**: Eye distance × 2.3 for skull width
- **Proportions**: 1.6:1 height-to-width ratio
- **Positioning**: Eyes 45% down from skull top

### 3. Contour Refinement
- **Enhancement**: Image processing refinement
- **Techniques**: Gaussian blur, Canny edge detection, morphological operations
- **Validation**: Area comparison with anatomical estimates
- **Fallback**: Returns anatomical estimate if contour unreasonable

## Technical Specifications

### Model Architecture
- **Base Model**: YOLOv8n-pose (Ultralytics)
- **Input Size**: Variable (auto-resized)
- **Output**: 17 keypoints with confidence scores
- **Processing**: Real-time capable

### Input Requirements
- **Image Formats**: JPG, PNG, BMP, TIFF
- **Orientation**: Full body or upper body visible
- **Quality**: Clear keypoints, minimal occlusion
- **Lighting**: Even illumination preferred
- **Resolution**: Any resolution (auto-processed)

### Output Formats
- **JSON**: Structured measurement data
- **PNG**: Annotated visualization images
- **Batch Results**: Multiple image analysis

## Installation and Setup

### Docker Deployment (Recommended)
```bash
# Build the container
docker-compose build

# Run the service
docker-compose up -d

# Check service status
curl http://localhost:8007/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model file exists
ls models/yolov8n-pose.pt

# Create required directories
mkdir -p /app/temp /app/results

# Run the application
python app/main.py
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `PORT`: Service port (default: 8007)

## Usage Examples

### Python Client
```python
import requests

# Complete anthropometric analysis
with open('body_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8007/analyze-body-anthropometry',
        files={'file': f},
        data={
            'confidence_threshold': 0.5,
            'include_visualization': True,
            'detailed_analysis': True
        }
    )

results = response.json()
skull_ratio = results['anthropometric_analysis'][0]['body_proportions']['skull_to_body_ratio']
assessment = results['anthropometric_analysis'][0]['body_proportions']['anatomical_assessment']
print(f"Skull-to-body ratio: {skull_ratio:.3f}")
print(f"Assessment: {assessment}")
```

### Batch Processing
```python
import requests

files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]

response = requests.post(
    'http://localhost:8007/batch-analyze',
    files=files,
    data={'confidence_threshold': 0.5, 'detailed_analysis': False}
)

batch_results = response.json()
print(f"Processed {batch_results['total_images']} images")
print(f"Successful: {batch_results['successful_analyses']}")
```

### cURL Examples
```bash
# Complete analysis
curl -X POST "http://localhost:8007/analyze-body-anthropometry" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"

# Skull detection only
curl -X POST "http://localhost:8007/detect-skull-measurements" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5"

# Pose detection only
curl -X POST "http://localhost:8007/detect-pose-keypoints" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5"
```

## Model Files Required

- `models/yolov8n-pose.pt` - YOLOv8 pose detection model

## Results Directory

Generated files are stored in `/app/results/`:
- `anthropometry_{uuid}.png` - Visualization images with annotations

## Error Handling

### Common Error Responses
- **400**: Invalid image format or parameters
- **503**: Model not loaded or initialization failed
- **500**: Analysis processing error

### Troubleshooting
1. **Model Loading Issues**: Ensure YOLOv8 model file exists
2. **CUDA Errors**: Check GPU availability and drivers
3. **Memory Issues**: Reduce batch size or image resolution
4. **Poor Detection**: Ensure full body visibility and good lighting
5. **No Skull Detection**: Verify face/head is clearly visible

## Performance Notes

- **Processing Time**: ~0.2-1.0 seconds per image (GPU)
- **Memory Usage**: ~1-3GB GPU memory
- **Accuracy**: 88-95% keypoint detection rate on clear images
- **Throughput**: ~60-300 images/minute (depending on hardware)
- **Batch Limit**: 5 images per batch (configurable)

## Clinical Applications

### Use Cases
- **Growth Monitoring**: Child development tracking
- **Fitness Assessment**: Body proportion analysis
- **Medical Research**: Anthropometric studies
- **Ergonomic Design**: Human factors research
- **Sports Science**: Athletic performance analysis

### Measurement Reliability
- **High Reliability**: Frontal poses, good lighting, clear keypoints
- **Moderate Reliability**: Slight angles, partial occlusion
- **Low Reliability**: Profile poses, poor lighting, motion blur

## Version Information

- **API Version**: 1.0.0
- **Model Version**: YOLOv8n-pose
- **Framework**: FastAPI + Ultralytics YOLO
- **Dependencies**: OpenCV, NumPy, SciPy, PIL
- **Last Updated**: 2025

## Support

For technical support or questions about the body anthropometric analysis module, please refer to the main project documentation or contact the development team.
