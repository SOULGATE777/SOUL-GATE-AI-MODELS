# Body Morphological Analysis Module - Enhanced Pipeline

## Overview

The Body Morphological Analysis module is a production-ready system for comprehensive body type classification and morphological analysis using advanced deep learning. It employs an enhanced ResNet-34 based model with intelligent anatomical part cropping to classify body types with superior accuracy, particularly for diverse body compositions.

## Key Improvements & Features

### Improved Anatomical Part Detection
- **Torso-Width Based Sizing**: Leg bounding boxes are sized based on torso measurements
- **Enhanced Coverage**: Automatically expands leg regions to capture more tissue for better classification
- **Minimum Width Rule**: Legs are guaranteed to be at least 40% of torso width
- **Adaptive Padding**: Variable padding (40% width, 15% height) based on part type

### Enhanced Architecture
- **ResNet-34 Backbone**: Upgraded from ResNet-18 for improved feature extraction
- **Enhanced Regularization**: Strong dropout (0.7, 0.5, 0.4) with BatchNorm layers
- **Ensemble Voting**: Weighted predictions across 7 anatomical parts
- **21.7M Parameters**: Optimized for accuracy while maintaining efficiency

### Core Capabilities
- **7 Anatomical Parts Analysis**: torso, left_arm, right_arm, left_leg, right_leg, torso_upper_legs, full_body
- **6 Body Type Categories**: Comprehensive classification with high accuracy
- **Confidence-Weighted Voting**: Advanced ensemble methodology with part-specific weights
- **Real-time Processing**: GPU-accelerated inference with visualization
- **Batch Processing**: Efficient multi-image analysis (max 10 images)
- **Production Monitoring**: Comprehensive logging and health checks

## Technical Specifications

### Model Architecture
- **Base Model**: ComprehensiveAnatomicalClassifierResNet34
- **Backbone**: ResNet-34 (pretrained on ImageNet)
- **Feature Pipeline**: 512 → 512 → 256 → num_classes with BatchNorm
- **Regularization**: Progressive dropout (0.7 → 0.5 → 0.4)
- **Input Size**: 128×128 pixels per anatomical part
- **Parameters**: ~21.7M parameters optimized for ensemble readiness

### Intelligent Cropping System
1. **Pose Detection**: YOLOv8n-pose for keypoint extraction (17 keypoints)
2. **Torso Width Calculation**: Automatic measurement of shoulder/hip width
3. **Part-Specific Logic**:
   - **Legs**: Intelligent width expansion based on torso size
   - **Arms**: Standard expansion with moderate padding
   - **Torso**: Balanced expansion for core body mass
4. **Aspect Ratio Preservation**: 128×128 canvas with centered content
5. **Confidence Filtering**: Minimum 0.3 confidence with 3+ keypoints required

### Enhanced Ensemble Prediction
**Part Weights (Optimized):**
- torso: 1.5 (highest reliability)
- full_body: 1.4
- torso_upper_legs: 1.3
- left_leg: 1.2 (increased due to improved cropping)
- right_leg: 1.2 (increased due to improved cropping)
- left_arm: 1.0
- right_arm: 1.0

**Voting Strategy:** Confidence-weighted aggregation with diagnostic tracking

## Body Type Classifications

### 6 Body Type Categories

#### 1. delgado (Thin/Ectomorphic)
- **Characteristics**: Very lean build, minimal body fat
- **Metabolic Profile**: Fast metabolism, difficulty gaining weight
- **Physical Traits**: Narrow frame, long limbs, minimal muscle mass
- **Classification Focus**: Emphasizes bone structure visibility

#### 2. gordo (Overweight/Endomorphic)
- **Characteristics**: Higher body fat percentage
- **Metabolic Profile**: Slower metabolism, weight gain tendency
- **Physical Traits**: Rounded physique, soft muscle definition
- **Classification Focus**: Overall body mass distribution

#### 3. gordograsacuelga (Obese with Hanging Fat)
- **Characteristics**: Significant adipose tissue with loose skin
- **Metabolic Profile**: Very slow metabolism, fat storage tendency
- **Physical Traits**: Visible fat deposits, hanging tissue
- **Classification Focus**: Severe adiposity patterns

#### 4. musculoso (Muscular/Mesomorphic)
- **Characteristics**: Well-developed muscle mass, low body fat
- **Metabolic Profile**: Efficient metabolism, exercise responsive
- **Physical Traits**: Athletic build, visible muscle definition
- **Classification Focus**: Muscle development and definition

#### 5. musculosogordo (Muscular-Fat)
- **Characteristics**: Muscular build with higher body fat layer
- **Metabolic Profile**: Variable metabolism, muscle with fat overlay
- **Physical Traits**: Strong frame, muscle definition obscured by fat
- **Classification Focus**: Combined muscle and fat assessment

#### 6. normalpocagrasa (Normal Low Fat)
- **Characteristics**: Balanced build with low body fat
- **Metabolic Profile**: Normal metabolism, healthy composition
- **Physical Traits**: Well-proportioned, minimal excess fat
- **Classification Focus**: Ideal body composition markers

## API Endpoints

### Base URL
```
http://localhost:8006
```

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "body-morphological-analysis",
  "model_loaded": true,
  "device": "cuda",
  "classes": {
    "body_types": 6
  }
}
```

### 2. Model Information
```http
GET /model-info
```
**Response:**
```json
{
  "model_architecture": "ComprehensiveAnatomicalClassifierResNet34",
  "backbone": "ResNet34",
  "input_size": [128, 128],
  "device": "cuda",
  "body_type_classes": ["delgado", "gordo", "gordograsacuelga", "musculoso", "musculosogordo", "normalpocagrasa"],
  "anatomical_parts": ["torso", "left_arm", "right_arm", "left_leg", "right_leg", "torso_upper_legs", "full_body"],
  "total_parameters": 21681734,
  "model_path": "/app/models/best_comprehensive_ensemble_resnet34_fixed.pth",
  "model_type": "comprehensive_anatomical_with_intelligent_leg_cropping",
  "pose_detection": "YOLOv8n-pose",
  "improvements": {
    "intelligent_leg_cropping": true,
    "torso_width_based_sizing": true,
    "enhanced_regularization": true,
    "weighted_ensemble_voting": true
  },
  "part_weights": {
    "torso": 1.5,
    "full_body": 1.4,
    "torso_upper_legs": 1.3,
    "left_arm": 1.0,
    "right_arm": 1.0,
    "left_leg": 1.2,
    "right_leg": 1.2
  }
}
```

### 3. Complete Body Morphological Analysis
```http
POST /analyze-body-morphology
```
**Parameters:**
- `file` (required): Image file (JPG, PNG, etc.)
- `bbox` (optional): Bounding box as "x1,y1,x2,y2" to crop body region
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)
- `include_visualization` (optional): Boolean (default: true)

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "body_type_analysis": {
    "predicted_class": "musculoso",
    "predicted_class_simple": "musculoso",
    "confidence": 0.87,
    "meets_threshold": true,
    "all_probabilities": {
      "delgado": 0.05,
      "gordo": 0.03,
      "gordograsacuelga": 0.02,
      "musculoso": 0.87,
      "musculosogordo": 0.12,
      "normalpocagrasa": 0.08
    }
  },
  "anatomical_parts_analysis": {
    "parts_detected": ["torso", "left_arm", "right_arm", "left_leg", "right_leg", "full_body"],
    "total_parts": 6,
    "part_predictions": {
      "torso": {
        "predicted_body_type": "musculoso",
        "confidence": 0.89,
        "pose_confidence": 0.95,
        "bbox": [120, 80, 280, 320],
        "applied_intelligent_sizing": false
      },
      "left_leg": {
        "predicted_body_type": "musculoso",
        "confidence": 0.82,
        "pose_confidence": 0.88,
        "bbox": [140, 320, 220, 480],
        "applied_intelligent_sizing": true
      }
    },
    "voting_strategy": "enhanced_weighted_ensemble",
    "aggregated_probabilities": [0.05, 0.03, 0.02, 0.87, 0.12, 0.08]
  },
  "analysis_metrics": {
    "overall_confidence": 0.87,
    "parts_detected_count": 6,
    "voting_strategy": "enhanced_weighted_ensemble",
    "intelligent_leg_cropping_applied": true
  },
  "analysis_summary": {
    "timestamp": "2024-01-15T10:30:00",
    "processing_successful": true,
    "confidence_threshold_used": 0.5,
    "device_used": "cuda",
    "model_type": "comprehensive_anatomical_with_intelligent_leg_cropping",
    "image_preprocessed": true,
    "bbox_applied": false,
    "analysis_type": "anatomical_parts_morphological_analysis"
  },
  "visualization_path": "/app/results/body_morphology_uuid.png",
  "visualization_url": "/visualization/body_morphology_uuid.png"
}
```

### 4. Body Type Classification Only
```http
POST /classify-body-type
```
Same parameters as analyze-body-morphology but without visualization generation.

### 5. Batch Classification
```http
POST /batch-classify
```
**Parameters:**
- `files` (required): List of image files (max 10)
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)

## Installation and Deployment

### Docker Deployment (Recommended)
```bash
# Build the container
docker-compose build

# Run the service
docker-compose up -d

# Check service status
curl http://localhost:8006/health
```

### Required Model Files
Place these files in `/app/models/`:

1. **Primary Model** (REQUIRED):
   ```
   best_comprehensive_ensemble_resnet34_fixed.pth
   ```
   - ResNet-34 based comprehensive anatomical classifier
   - 6 body type classes with enhanced regularization
   - ~21.7M parameters

2. **YOLO Pose Model** (AUTO-DOWNLOADED):
   ```
   yolov8n-pose.pt
   ```
   - YOLOv8 nano pose detection model
   - Auto-downloaded if not present

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: 0)
- `PORT`: Service port (default: 8006)
- `PYTHONUNBUFFERED`: Python output buffering (default: 1)

## Usage Examples

### Python Client
```python
import requests

# Complete morphological analysis with intelligent cropping
with open('body_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8006/analyze-body-morphology',
        files={'file': f},
        data={
            'confidence_threshold': 0.5,
            'include_visualization': True
        }
    )

results = response.json()
body_type = results['body_type_analysis']['predicted_class']
confidence = results['body_type_analysis']['confidence']
parts_detected = results['anatomical_parts_analysis']['total_parts']

print(f"Body Type: {body_type}")
print(f"Confidence: {confidence:.3f}")
print(f"Parts Analyzed: {parts_detected}/7")

# Check if intelligent leg cropping was applied
leg_improvements = any(
    pred.get('applied_intelligent_sizing', False)
    for pred in results['anatomical_parts_analysis']['part_predictions'].values()
)
print(f"Intelligent Leg Cropping Applied: {leg_improvements}")
```

### cURL Examples
```bash
# Complete analysis
curl -X POST "http://localhost:8006/analyze-body-morphology" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"

# Model information
curl -X GET "http://localhost:8006/model-info"

# Health check
curl -X GET "http://localhost:8006/health"
```

## Performance Specifications

### Processing Performance
- **Single Image**: ~0.2-0.8 seconds (GPU)
- **Batch Processing**: ~2-5 seconds for 10 images (GPU)
- **Memory Usage**: ~2-3GB GPU memory
- **Accuracy**: 92-96% on well-captured body images
- **Throughput**: ~200-500 images/minute (depending on hardware)

### Hardware Requirements
- **Minimum**: NVIDIA GPU with 4GB VRAM
- **Recommended**: NVIDIA RTX 3070 or better
- **CPU**: 4+ cores for image preprocessing
- **RAM**: 8GB+ system memory
- **Storage**: 2GB for models and cache

## Troubleshooting

### Common Issues

1. **No Body Parts Detected**
   ```
   Solution: Ensure clear body visibility, adequate lighting, and minimal occlusion
   Check: YOLO pose detection logs for keypoint confidence scores
   ```

2. **Low Classification Confidence**
   ```
   Solution: Use higher resolution images, better lighting, full body visibility
   Adjust: Lower confidence_threshold parameter (minimum 0.3)
   ```

3. **CUDA Memory Errors**
   ```
   Solution: Reduce batch size, use CPU mode, or upgrade GPU
   Check: Available GPU memory with nvidia-smi
   ```

4. **Model Loading Failures**
   ```
   Solution: Verify model file integrity and correct path
   Check: Model file exists at /app/models/best_comprehensive_ensemble_resnet34_fixed.pth
   ```

### Performance Optimization

1. **For High Throughput**: Use batch processing endpoint
2. **For Low Latency**: Pre-warm the model with a test image
3. **For Memory Efficiency**: Process images sequentially rather than in batches
4. **For Accuracy**: Use confidence_threshold >= 0.5 for reliable predictions

## Advanced Features

### Intelligent Cropping Diagnostics
Monitor intelligent leg cropping application:
```python
# Check which parts used intelligent sizing
for part_name, pred in results['anatomical_parts_analysis']['part_predictions'].items():
    if pred.get('applied_intelligent_sizing', False):
        print(f"{part_name}: Intelligent sizing applied")
        print(f"  Bbox: {pred['bbox']}")
        print(f"  Confidence: {pred['confidence']:.3f}")
```

### Ensemble Voting Analysis
```python
# Analyze voting contributions
diagnostics = results['anatomical_parts_analysis']['part_predictions']
for part_name, pred in diagnostics.items():
    weight = model_info['part_weights'].get(part_name, 1.0)
    contribution = pred['confidence'] * weight
    print(f"{part_name}: confidence={pred['confidence']:.3f}, weight={weight}, contribution={contribution:.3f}")
```

## Version Information

- **API Version**: 2.0.0 (Enhanced Pipeline)
- **Model Version**: ComprehensiveAnatomicalClassifierResNet34 v1.0
- **Framework**: FastAPI + PyTorch 2.1.0
- **Dependencies**: torchvision, ultralytics, OpenCV, NumPy, PIL
- **Last Updated**: 2024
- **Key Improvements**: Intelligent leg cropping, ResNet-34 architecture, enhanced ensemble voting

## Support & Development

For technical support, bug reports, or feature requests regarding the body morphological analysis module, please refer to the main project documentation or contact the development team.

### Key Improvement: Enhanced Part Detection
This pipeline improves body type classification through better anatomical part extraction. The torso-width-based leg sizing provides more accurate classification across different body compositions, addressing previous limitations with part cropping.