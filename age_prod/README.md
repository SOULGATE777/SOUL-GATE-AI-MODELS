# SG Age Estimation - AI Age Prediction Module

Advanced age estimation module using InsightFace for high-accuracy age prediction from facial images. Supports both frontal and profile face orientations with comprehensive analysis and visualization capabilities.

## Features

### Core Capabilities ✅
- **InsightFace Integration**: Production-ready age estimation using InsightFace's state-of-the-art models
- **Multi-Orientation Support**: Handles frontal, semi-profile, and profile face images
- **High Accuracy**: ±3-5 years accuracy on production datasets
- **Face Detection**: Automatic face detection with quality assessment
- **Batch Processing**: Support for multiple image processing
- **Comprehensive Analysis**: Age categorization, confidence scoring, and detailed metrics

### Age Categories
- **Child**: 1-12 years
- **Teenager**: 13-19 years  
- **Young Adult**: 20-29 years
- **Adult**: 30-49 years
- **Middle Aged**: 50-64 years
- **Senior**: 65+ years

### Advanced Features
- **Quality Assessment**: Automatic evaluation of face detection quality
- **Confidence Scoring**: Reliability metrics for age predictions
- **Face Orientation Detection**: Automatic classification of face angle
- **Rich Visualizations**: Detailed analysis charts and overlays
- **GPU Acceleration**: CUDA support with CPU fallback
- **Production API**: FastAPI with async processing

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit (for GPU support)

### Deploy the Module

```bash
# Navigate to age_prod directory
cd age_prod

# Build and start the container
docker compose up --build -d

# Check health status
curl http://localhost:8013/health
```

### API Documentation
- **Interactive Docs**: http://localhost:8013/docs
- **ReDoc**: http://localhost:8013/redoc

## API Endpoints

### 1. Single Age Estimation
Estimate age from a single face image with full analysis and visualization.

```bash
curl -X POST "http://localhost:8013/estimate-age" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "include_visualization=true"
```

### 2. Quick Age Estimation
Fast age estimation without visualization for high-throughput scenarios.

```bash
curl -X POST "http://localhost:8013/quick-age-estimate" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg"
```

### 3. Batch Age Estimation
Process multiple images simultaneously (up to 20 images per batch).

```bash
curl -X POST "http://localhost:8013/batch-estimate-ages" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "include_visualization=true"
```

### 4. Health Check
Check service status and model readiness.

```bash
curl http://localhost:8013/health
```

### 5. Model Information
Get detailed information about the loaded model.

```bash
curl http://localhost:8013/model-info
```

### 6. Visualization Access
Retrieve generated visualization files.

```bash
# Get specific visualization
curl http://localhost:8013/visualization/age_analysis_20241215_143022_abc123.png

# List all available visualizations
curl http://localhost:8013/list-visualizations
```

## Response Formats

### Single Age Estimation Response
```json
{
  "analysis_id": "uuid-string",
  "success": true,
  "processing_time": 1.234,
  "image_info": {
    "path": "/tmp/temp_image.jpg",
    "dimensions": {
      "width": 800,
      "height": 600
    }
  },
  "face_detection": {
    "faces_detected": 1,
    "primary_face_info": {
      "bbox": [150, 100, 350, 300],
      "face_dimensions": {
        "width": 200,
        "height": 200,
        "area": 40000
      },
      "image_coverage_ratio": 0.0833,
      "detection_confidence": 0.95,
      "age_confidence": 0.87,
      "orientation": "frontal",
      "quality_assessment": {
        "detection_quality": "excellent",
        "size_adequacy": "good"
      }
    },
    "detection_message": "Successfully detected 1 face(s)"
  },
  "age_estimation": {
    "estimated_age": 28.5,
    "age_category": "young_adult",
    "confidence": 0.87,
    "age_range": {
      "min": 25,
      "max": 32
    },
    "reliability": "high"
  },
  "analysis_summary": {
    "estimated_age": 28.5,
    "age_category": "young_adult",
    "confidence_level": "high",
    "face_orientation": "frontal",
    "detection_quality": "excellent"
  },
  "visualization": {
    "created": true,
    "path": "/app/results/age_analysis_20241215_143022_abc123.png"
  },
  "model_info": {
    "model_type": "InsightFace",
    "device": "cuda:0",
    "detection_size": "640x640"
  }
}
```

### Quick Age Estimation Response
```json
{
  "success": true,
  "estimated_age": 28.5,
  "age_category": "young_adult",
  "confidence": 0.87,
  "processing_time": 0.856,
  "analysis_id": "uuid-string"
}
```

### Batch Processing Response
```json
{
  "batch_id": "uuid-string",
  "total_images": 3,
  "successful_analyses": 3,
  "success_rate": 1.0,
  "total_processing_time": 3.45,
  "average_processing_time": 1.15,
  "individual_results": [
    {
      "analysis_id": "uuid-1",
      "success": true,
      "age_estimation": {
        "estimated_age": 25.2,
        "age_category": "young_adult",
        "confidence": 0.89
      }
    }
  ],
  "batch_summary": {
    "ages_estimated": [25.2, 34.8, 42.1],
    "average_age": 34.0,
    "age_categories": {
      "young_adult": 1,
      "adult": 2
    }
  }
}
```

## Model Information

### InsightFace Age Estimation Model
- **Architecture**: Deep learning-based face analysis with age regression
- **Framework**: MXNet/ONNX optimized for production
- **Detection Size**: 640x640 pixels (optimal for accuracy)
- **Age Range**: 1-100 years
- **Expected Accuracy**: ±3-5 years on diverse datasets
- **Supported Orientations**: Frontal, semi-profile, profile faces
- **Input Formats**: JPG, JPEG, PNG, BMP, TIFF

### Performance Characteristics
- **GPU Processing**: ~0.5-1.0 seconds per image
- **CPU Processing**: ~2-4 seconds per image
- **Batch Efficiency**: Linear scaling with multi-GPU support
- **Memory Usage**: ~2GB VRAM (GPU mode)

## Configuration

### GPU Production Setup
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

### CPU-Only Setup
```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=-1
# Remove deploy section for CPU-only
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU selection (0, 1, etc.) or -1 for CPU
- `PYTHONPATH`: Set to /app (automatically configured)

## Usage Examples

### Python Client Example
```python
import requests

# Single image age estimation
with open('face_image.jpg', 'rb') as f:
    files = {'file': f}
    data = {'include_visualization': True}
    response = requests.post('http://localhost:8013/estimate-age', 
                           files=files, data=data)
    result = response.json()
    print(f"Estimated age: {result['age_estimation']['estimated_age']} years")

# Batch processing
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]
data = {'include_visualization': True}
response = requests.post('http://localhost:8013/batch-estimate-ages', 
                       files=files, data=data)
batch_result = response.json()
print(f"Average age: {batch_result['batch_summary']['average_age']} years")
```

### JavaScript/Node.js Example
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function estimateAge(imagePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(imagePath));
  form.append('include_visualization', 'true');
  
  const response = await axios.post('http://localhost:8013/estimate-age', form, {
    headers: form.getHeaders()
  });
  
  return response.data;
}

estimateAge('face_image.jpg').then(result => {
  console.log(`Estimated age: ${result.age_estimation.estimated_age} years`);
  console.log(`Confidence: ${(result.age_estimation.confidence * 100).toFixed(1)}%`);
});
```

## Quality Guidelines

### Optimal Image Characteristics
- **Face Size**: At least 100x100 pixels
- **Image Quality**: Sharp, well-lit images
- **Face Coverage**: Face should occupy 5-20% of image area
- **Orientation**: Any angle supported (frontal preferred for highest accuracy)
- **Lighting**: Even, natural lighting preferred

### Factors Affecting Accuracy
- **Image Quality**: Blurry or low-resolution images reduce accuracy
- **Face Angle**: Extreme profile angles may have slightly lower accuracy
- **Age Range**: Accuracy typically higher for ages 10-60 years
- **Occlusions**: Glasses, masks, or hair covering face features may impact results

## Troubleshooting

### Common Issues

**1. "Pipeline not initialized" error**
- Check if InsightFace models downloaded correctly
- Verify CUDA/GPU setup if using GPU mode
- Check container logs: `docker logs sg_age_estimation`

**2. "No faces detected" error**
- Ensure image contains clearly visible faces
- Try different image formats or resolutions
- Check if face is too small or poorly lit

**3. GPU not being used**
- Verify NVIDIA Container Toolkit installation
- Check `nvidia-smi` output
- Ensure CUDA_VISIBLE_DEVICES is set correctly

**4. Out of memory errors**
- Reduce batch size for batch processing
- Use CPU mode for lower memory usage
- Ensure sufficient GPU VRAM (2GB+ recommended)

### Performance Optimization

**For High Throughput:**
- Use `/quick-age-estimate` endpoint (no visualization)
- Batch process multiple images when possible
- Use GPU acceleration
- Optimize image sizes (resize to 640x640 for best performance)

**For Production Deployment:**
- Set up load balancing for multiple instances
- Implement request queuing for batch processing
- Monitor GPU memory usage
- Set up log aggregation and monitoring

## Monitoring and Maintenance

### Health Monitoring
```bash
# Check service health
curl http://localhost:8013/health

# Monitor container logs
docker logs -f sg_age_estimation

# Check GPU utilization
nvidia-smi
```

### Cleanup Operations
```bash
# Clean up old visualizations (older than 7 days)
curl -X DELETE "http://localhost:8013/cleanup-old-visualizations" \
  -H "Content-Type: multipart/form-data" \
  -F "days=7"

# List current visualizations
curl http://localhost:8013/list-visualizations
```

## Integration with SG Pipeline

This age estimation module complements the existing SG analysis pipeline:

### Recommended Workflow
1. **Image Quality Check**: Use validation modules first
2. **Age Estimation**: Apply this module for demographic analysis
3. **Morphological Analysis**: Combine with existing facial analysis
4. **Comprehensive Report**: Integrate age data with other measurements

### Port Configuration
- **Age Estimation**: Port 8013
- **Compatible with**: All existing SG modules (ports 8000-8012)

## License
[Add appropriate license information]

---

**SG Age Estimation Module**  
**Status**: ✅ **PRODUCTION READY**  
**Model**: InsightFace Age Estimation  
**Port**: 8013