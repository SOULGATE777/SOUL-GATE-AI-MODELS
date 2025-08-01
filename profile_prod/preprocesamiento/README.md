# Profile Preprocessing Module

## Overview

The Profile Preprocessing module is a specialized face detection and preprocessing service designed to prepare profile (lateral) facial images for downstream analysis. It uses a trained Faster R-CNN model to detect profile faces that may be too far away in images, crops them with appropriate padding, resizes them while maintaining proportions, and outputs them in base64 format for seamless integration with other services.

## Features

### Core Capabilities
- **Profile Face Detection**: Uses Faster R-CNN trained specifically for profile face detection
- **Intelligent Cropping**: Crops detected faces with configurable padding around bounding boxes
- **Aspect Ratio Preservation**: Maintains original proportions while resizing to target dimensions
- **Base64 Output**: Converts processed images to base64 format for easy API integration
- **Flexible Parameters**: Configurable confidence thresholds, target sizes, and output formats
- **Real-time Visualization**: Generates debug visualizations showing detection results

### Service Architecture
- **FastAPI Backend**: High-performance async API with automatic documentation
- **GPU Acceleration**: CUDA support for faster inference with CPU fallback
- **Docker Containerization**: Complete containerized deployment with health checks
- **Modular Design**: Separated pipeline, processing, and visualization components

## API Endpoints

### Base URL
```
http://localhost:8010
```

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "profile-preprocessing",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

### 2. Complete Profile Preprocessing
```http
POST /preprocess-profile
```
**Parameters:**
- `file` (required): Input image file (JPG, PNG)
- `confidence_threshold` (optional): Float 0.1-0.9 (default: 0.5)
- `target_width` (optional): Integer 100-2048 (default: 600)
- `target_height` (optional): Integer 100-2048 (default: 600)
- `padding_factor` (optional): Float 0.0-0.5 (default: 0.15)
- `output_format` (optional): String 'JPEG'|'PNG' (default: 'JPEG')
- `quality` (optional): Integer 1-100 (default: 95, JPEG only)
- `include_visualization` (optional): Boolean (default: false)

**Response:**
```json
{
  "processing_id": "uuid-string",
  "status": "success",
  "total_faces_detected": 2,
  "faces_processed": 2,
  "original_image_size": [1080, 1920],
  "processing_parameters": {
    "confidence_threshold": 0.5,
    "target_size": [600, 600],
    "padding_factor": 0.15,
    "output_format": "JPEG",
    "quality": 95
  },
  "processed_faces": [
    {
      "face_id": 1,
      "detection_confidence": 0.85,
      "class_name": "profile_face",
      "original_bbox": [120, 150, 420, 450],
      "cropped_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "crop_info": {
        "target_size": [600, 600],
        "padding_factor": 0.15
      }
    }
  ],
  "visualizations": {
    "detection_grid": "/visualization/preprocessing_viz_uuid_detection_grid.png",
    "original_with_detections": "/visualization/preprocessing_viz_uuid_original.png"
  }
}
```

### 3. Face Detection Only
```http
POST /detect-faces
```
**Parameters:**
- `file` (required): Input image file
- `confidence_threshold` (optional): Float 0.1-0.9

**Response:**
```json
{
  "status": "success",
  "total_detections": 2,
  "image_size": [1080, 1920],
  "confidence_threshold": 0.5,
  "detections": [
    {
      "bbox": [120, 150, 420, 450],
      "confidence": 0.85,
      "label": 1,
      "class_name": "profile_face",
      "detection_id": 0
    }
  ]
}
```

### 4. Face Cropping from Bounding Boxes
```http
POST /crop-faces
```
**Parameters:**
- `file` (required): Input image file
- `bboxes` (required): JSON string of bounding boxes `[[x1,y1,x2,y2], ...]`
- `target_width` (optional): Target width (default: 600)
- `target_height` (optional): Target height (default: 600)
- `padding_factor` (optional): Padding factor (default: 0.15)
- `output_format` (optional): Output format (default: 'JPEG')
- `quality` (optional): JPEG quality (default: 95)

**Response:**
```json
{
  "status": "success",
  "total_faces_cropped": 2,
  "original_image_size": [1080, 1920],
  "cropped_faces": [
    {
      "face_id": 1,
      "bbox": [120, 150, 420, 450],
      "cropped_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "target_size": [600, 600],
      "padding_factor": 0.15
    }
  ],
  "processing_parameters": {
    "target_size": [600, 600],
    "padding_factor": 0.15,
    "output_format": "JPEG",
    "quality": 95
  }
}
```

### 5. Model Information
```http
GET /model-info
```
**Response:**
```json
{
  "model_type": "Faster R-CNN Profile Face Detection",
  "device": "cuda",
  "model_path": "/app/models/profile_detection_model.pth",
  "num_classes": 2,
  "all_classes": ["profile_face"],
  "default_confidence_threshold": 0.5,
  "default_target_size": [600, 600],
  "default_padding_factor": 0.15
}
```

### 6. Processing Statistics
```http
GET /processing-stats
```
**Response:**
```json
{
  "service_info": {
    "name": "Profile Preprocessing Service",
    "version": "1.0.0",
    "description": "Face detection, cropping and preprocessing for profile images"
  },
  "capabilities": {
    "face_detection": true,
    "face_cropping": true,
    "base64_output": true,
    "batch_processing": false,
    "visualization": true
  },
  "supported_formats": {
    "input": ["JPEG", "PNG", "JPG"],
    "output": ["JPEG", "PNG"]
  },
  "parameter_ranges": {
    "confidence_threshold": {"min": 0.1, "max": 0.9, "default": 0.5},
    "target_size": {"min": 100, "max": 2048, "default": 600},
    "padding_factor": {"min": 0.0, "max": 0.5, "default": 0.15},
    "quality": {"min": 1, "max": 100, "default": 95}
  }
}
```

## Technical Specifications

### Model Architecture
- **Base Model**: Faster R-CNN with ResNet-50 backbone
- **Detection Head**: Custom FastRCNNPredictor for profile face classes
- **Training Framework**: PyTorch with torchvision detection utilities
- **Input Processing**: Automatic image preprocessing and tensor conversion

### Processing Pipeline
1. **Image Validation**: Format and dimension checks
2. **Face Detection**: Faster R-CNN inference with confidence filtering
3. **Bounding Box Processing**: Padding calculation and boundary validation
4. **Cropping**: Intelligent cropping with aspect ratio preservation
5. **Resizing**: Target size fitting with letterboxing on black background
6. **Format Conversion**: Base64 encoding with configurable quality

### Input Requirements
- **Image Formats**: JPG, JPEG, PNG
- **Dimensions**: 32x32 to 8192x8192 pixels
- **Orientation**: Any orientation (model handles profile detection)
- **Quality**: Clear images preferred for better detection accuracy

### Output Specifications
- **Base64 Format**: RFC 4648 compliant encoding
- **Target Sizes**: Configurable from 100x100 to 2048x2048 pixels
- **Aspect Ratio**: Preserved with black letterboxing
- **Quality**: Configurable JPEG quality (1-100) or lossless PNG

## Installation and Setup

### Docker Deployment (Recommended)
```bash
# Navigate to the preprocesamiento directory
cd preprocesamiento

# Build the container
docker-compose build

# Run the service
docker-compose up -d

# Check service status
curl http://localhost:8010/health

# View logs
docker-compose logs -f
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model file exists
ls models/profile_detection_model.pth

# Set environment variables
export PYTHONPATH=/app
export CUDA_VISIBLE_DEVICES=0  # or -1 for CPU

# Run the application
python app/main.py
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection (0 for first GPU, -1 for CPU)
- `PYTHONPATH`: Python module path (set to /app in container)

## Usage Examples

### Python Client Example
```python
import requests
import base64
from PIL import Image
import io

# Complete preprocessing
with open('profile_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8010/preprocess-profile',
        files={'file': f},
        data={
            'confidence_threshold': 0.6,
            'target_width': 600,
            'target_height': 600,
            'padding_factor': 0.2,
            'output_format': 'JPEG',
            'quality': 90,
            'include_visualization': True
        }
    )

result = response.json()
print(f"Detected {result['total_faces_detected']} faces")

# Decode first face from base64
if result['processed_faces']:
    face_base64 = result['processed_faces'][0]['cropped_image_base64']
    face_bytes = base64.b64decode(face_base64)
    face_image = Image.open(io.BytesIO(face_bytes))
    face_image.save('processed_face.jpg')

# Detection only
with open('profile_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8010/detect-faces',
        files={'file': f},
        data={'confidence_threshold': 0.5}
    )

detections = response.json()
print(f"Found {len(detections['detections'])} faces")
```

### cURL Examples
```bash
# Complete preprocessing
curl -X POST "http://localhost:8010/preprocess-profile" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.6" \
  -F "target_width=600" \
  -F "target_height=600" \
  -F "include_visualization=true"

# Detection only
curl -X POST "http://localhost:8010/detect-faces" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5"

# Health check
curl http://localhost:8010/health

# Model information
curl http://localhost:8010/model-info
```

### JavaScript/Node.js Example
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function preprocessProfile(imagePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    form.append('confidence_threshold', '0.6');
    form.append('target_width', '600');
    form.append('target_height', '600');
    form.append('output_format', 'JPEG');
    form.append('quality', '95');

    try {
        const response = await axios.post(
            'http://localhost:8010/preprocess-profile',
            form,
            { headers: form.getHeaders() }
        );
        
        console.log(`Processed ${response.data.faces_processed} faces`);
        return response.data;
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

// Usage
preprocessProfile('profile_image.jpg');
```

## Model Files Required

Place your trained model file in the `models/` directory:
- `models/profile_detection_model.pth` - Faster R-CNN model trained for profile face detection

The model file should contain:
- `model_state_dict`: PyTorch model weights
- `all_classes`: List of class names
- `num_classes`: Number of classes (including background)

## Results Directory

Generated files are stored in `/app/results/`:
- `preprocessing_viz_{uuid}_detection_grid.png` - Detection grid visualizations
- `preprocessing_viz_{uuid}_original_with_detections.png` - Original images with bounding boxes
- `preprocessing_viz_{uuid}_processing_summary.png` - Processing statistics summaries

## Architecture Integration

This preprocessing service is designed to work as the first step in a profile analysis pipeline:

```
Input Image → [Preprocessing Service:8010] → Base64 Cropped Faces → [Downstream Services]
                                                                    ↓
                                                        [Antropometrico:8004]
                                                        [Morfologico:8003]
                                                        [Validacion:8005]
```

The base64 output format allows seamless integration with other services in the profile analysis ecosystem.

## Error Handling

### Common Error Responses
- **400**: Invalid parameters, unsupported format, or image processing errors
- **503**: Model not loaded or initialization failed
- **500**: Internal processing errors

### Troubleshooting
1. **Model Loading Issues**: Ensure `profile_detection_model.pth` exists in `models/` directory
2. **CUDA Errors**: Check GPU availability or set `CUDA_VISIBLE_DEVICES=-1` for CPU mode
3. **Memory Issues**: Reduce image size or use CPU mode for large images
4. **Poor Detection**: Adjust confidence threshold or ensure clear profile images
5. **Base64 Errors**: Check output format and quality parameters

## Performance Notes

- **Processing Time**: ~0.5-3 seconds per image (GPU), ~2-10 seconds (CPU)
- **Memory Usage**: ~2-4GB GPU memory, ~1-2GB system RAM
- **Throughput**: ~20-60 images/minute depending on hardware and image size
- **Accuracy**: Optimized for clear profile images with faces visible from the side

## Development and Testing

### Running Tests
```bash
# Test health endpoint
curl http://localhost:8010/health

# Test with sample image
curl -X POST "http://localhost:8010/preprocess-profile" \
  -F "file=@test_image.jpg" \
  -F "include_visualization=true"
```

### Adding Custom Classes
To support additional face classes, retrain the model with your custom dataset and update the model file with the new class list.

## Version Information

- **API Version**: 1.0.0
- **Model Architecture**: Faster R-CNN ResNet-50
- **Framework**: FastAPI + PyTorch
- **Container Base**: Python 3.9 slim
- **Port**: 8010
- **Last Updated**: 2025

## Support and Integration

This service integrates seamlessly with the existing profile analysis ecosystem:
- **Antropometrico Service** (port 8004): Anthropometric measurements
- **Morfologico Service** (port 8003): Morphological analysis  
- **Validacion Service** (port 8005): Profile validation

For technical support or integration questions, refer to the main project documentation or contact the development team.