# Frontal Preprocessing Module

## Overview

The Frontal Preprocessing module is a specialized head detection and preprocessing service designed to prepare frontal facial images for downstream analysis. It uses YOLOv8 to detect heads/faces in frontal images, crops them with appropriate padding, resizes them while maintaining proportions, and outputs them in base64 format for seamless integration with other services.

## Features

### Core Capabilities
- **Frontal Head Detection**: Uses YOLOv8 for robust head/face detection in frontal images
- **Intelligent Cropping**: Crops detected heads with configurable padding around bounding boxes
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
http://localhost:8014
```

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "frontal-preprocessing",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

### 2. Complete Frontal Preprocessing
```http
POST /preprocess-frontal
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
  "total_heads_detected": 2,
  "heads_processed": 2,
  "original_image_size": [1080, 1920],
  "processing_parameters": {
    "confidence_threshold": 0.5,
    "target_size": [600, 600],
    "padding_factor": 0.15,
    "output_format": "JPEG",
    "quality": 95
  },
  "processed_heads": [
    {
      "head_id": 1,
      "detection_confidence": 0.85,
      "class_name": "head",
      "original_bbox": [120, 150, 420, 450],
      "cropped_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "crop_info": {
        "target_size": [600, 600],
        "padding_factor": 0.15
      },
      "detection_type": "head_from_person"
    }
  ],
  "visualizations": {
    "detection_grid": "/visualization/preprocessing_viz_uuid_detection_grid.png",
    "original_with_detections": "/visualization/preprocessing_viz_uuid_original.png"
  }
}
```

### 3. Head Detection Only
```http
POST /detect-heads
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
      "label": 0,
      "class_name": "head",
      "detection_id": 0,
      "detection_type": "head_from_person"
    }
  ]
}
```

### 4. Head Cropping from Bounding Boxes
```http
POST /crop-heads
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
  "total_heads_cropped": 2,
  "original_image_size": [1080, 1920],
  "cropped_heads": [
    {
      "head_id": 1,
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
  "model_type": "YOLOv8 Frontal Head Detection",
  "device": "cuda",
  "model_path": "yolov8n.pt (pre-trained)",
  "model_classes": ["person", "bicycle", "car", "..."],
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
    "name": "Frontal Preprocessing Service",
    "version": "1.0.0",
    "description": "Head detection, cropping and preprocessing for frontal images"
  },
  "capabilities": {
    "head_detection": true,
    "head_cropping": true,
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
- **Base Model**: YOLOv8 (nano version by default)
- **Detection Strategy**: Uses "person" class detection and extracts head region (top 30% of person bounding box)
- **Training Framework**: Ultralytics YOLOv8 with PyTorch backend
- **Input Processing**: Automatic image preprocessing and tensor conversion

### Processing Pipeline
1. **Image Validation**: Format and dimension checks
2. **Head Detection**: YOLOv8 inference with confidence filtering
3. **Head Region Extraction**: Intelligent head region estimation from person detections
4. **Bounding Box Processing**: Padding calculation and boundary validation
5. **Cropping**: Intelligent cropping with aspect ratio preservation
6. **Resizing**: Target size fitting with letterboxing on black background
7. **Format Conversion**: Base64 encoding with configurable quality

### Input Requirements
- **Image Formats**: JPG, JPEG, PNG
- **Dimensions**: 32x32 to 8192x8192 pixels
- **Orientation**: Any orientation (model handles frontal detection)
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
cd frontal_prod/preprocesamiento

# Build the container
docker-compose build

# Run the service (GPU version)
docker-compose up -d

# Run CPU version (if no GPU available)
# Edit docker-compose.yml to uncomment CPU service
docker-compose up -d frontal-preprocessing-cpu

# Check service status
curl http://localhost:8014/health

# View logs
docker-compose logs -f
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Create model directory (optional for custom models)
mkdir -p models

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
with open('frontal_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8014/preprocess-frontal',
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
print(f"Detected {result['total_heads_detected']} heads")

# Decode first head from base64
if result['processed_heads']:
    head_base64 = result['processed_heads'][0]['cropped_image_base64']
    head_bytes = base64.b64decode(head_base64)
    head_image = Image.open(io.BytesIO(head_bytes))
    head_image.save('processed_head.jpg')

# Detection only
with open('frontal_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8014/detect-heads',
        files={'file': f},
        data={'confidence_threshold': 0.5}
    )

detections = response.json()
print(f"Found {len(detections['detections'])} heads")
```

### cURL Examples
```bash
# Complete preprocessing
curl -X POST "http://localhost:8014/preprocess-frontal" \
  -F "file=@frontal_image.jpg" \
  -F "confidence_threshold=0.6" \
  -F "target_width=600" \
  -F "target_height=600" \
  -F "include_visualization=true"

# Detection only
curl -X POST "http://localhost:8014/detect-heads" \
  -F "file=@frontal_image.jpg" \
  -F "confidence_threshold=0.5"

# Health check
curl http://localhost:8014/health

# Model information
curl http://localhost:8014/model-info
```

### JavaScript/Node.js Example
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function preprocessFrontal(imagePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    form.append('confidence_threshold', '0.6');
    form.append('target_width', '600');
    form.append('target_height', '600');
    form.append('output_format', 'JPEG');
    form.append('quality', '95');

    try {
        const response = await axios.post(
            'http://localhost:8014/preprocess-frontal',
            form,
            { headers: form.getHeaders() }
        );

        console.log(`Processed ${response.data.heads_processed} heads`);
        return response.data;
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

// Usage
preprocessFrontal('frontal_image.jpg');
```

## Model Files

### Default Behavior
The service automatically downloads and uses the pre-trained YOLOv8n model on first run.

### Custom Models (Optional)
Place your trained YOLOv8 model file in the `models/` directory:
- `models/frontal_head_detection_model.pt` - Custom YOLOv8 model trained specifically for head detection

If no custom model is found, the service will use the pre-trained YOLOv8n model from Ultralytics.

## Results Directory

Generated files are stored in `/app/results/`:
- `preprocessing_viz_{uuid}_detection_grid.png` - Detection grid visualizations
- `preprocessing_viz_{uuid}_original_with_detections.png` - Original images with bounding boxes
- `preprocessing_viz_{uuid}_processing_summary.png` - Processing statistics summaries

## Architecture Integration

This preprocessing service is designed to work as the first step in a frontal analysis pipeline:

```
Input Image → [Frontal Preprocessing:8014] → Base64 Cropped Heads → [Downstream Services]
                                                                    ↓
                                                        [Antropometrico:8001]
                                                        [Morfologico:8000]
                                                        [Validacion:8002]
                                                        [Espejo:8008]
                                                        [Rotacion:8012]
```

The base64 output format allows seamless integration with other services in the frontal analysis ecosystem.

## Error Handling

### Common Error Responses
- **400**: Invalid parameters, unsupported format, or image processing errors
- **503**: Model not loaded or initialization failed
- **500**: Internal processing errors

### Troubleshooting
1. **Model Loading Issues**: Service uses pre-trained YOLOv8n automatically, no additional model files required
2. **CUDA Errors**: Check GPU availability or set `CUDA_VISIBLE_DEVICES=-1` for CPU mode
3. **Memory Issues**: Reduce image size or use CPU mode for large images
4. **Poor Detection**: Adjust confidence threshold or ensure clear frontal images
5. **Base64 Errors**: Check output format and quality parameters

## Performance Notes

- **Processing Time**: ~0.2-1.5 seconds per image (GPU), ~1-5 seconds (CPU)
- **Memory Usage**: ~1-3GB GPU memory, ~0.5-1GB system RAM
- **Throughput**: ~40-120 images/minute depending on hardware and image size
- **Accuracy**: Optimized for clear frontal images with visible heads/faces

## Development and Testing

### Running Tests
```bash
# Test health endpoint
curl http://localhost:8014/health

# Test with sample image
curl -X POST "http://localhost:8014/preprocess-frontal" \
  -F "file=@test_image.jpg" \
  -F "include_visualization=true"
```

### Adding Custom Models
To use a custom YOLOv8 model:
1. Train your model using Ultralytics YOLOv8
2. Save the model as `models/frontal_head_detection_model.pt`
3. Restart the service

## Version Information

- **API Version**: 1.0.0
- **Model Architecture**: YOLOv8 (nano)
- **Framework**: FastAPI + Ultralytics
- **Container Base**: Python 3.9 slim
- **Port**: 8014
- **Last Updated**: 2025

## Support and Integration

This service integrates seamlessly with the existing frontal analysis ecosystem:
- **Antropometrico Service** (port 8001): Anthropometric measurements
- **Morfologico Service** (port 8000): Morphological analysis
- **Validacion Service** (port 8002): Frontal validation
- **Espejo Service** (port 8008): Mirror analysis
- **Rotacion Service** (port 8012): Rotation analysis

For technical support or integration questions, refer to the main project documentation or contact the development team.