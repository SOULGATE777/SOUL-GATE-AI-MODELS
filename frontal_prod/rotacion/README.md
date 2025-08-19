# Frontal Face Rotation Assessment API

Advanced frontal face rotation classification to assess viability for anthropometric and morphological analysis.

## Overview

This module uses a multi-label CNN model (EfficientNet-B0) to classify frontal face orientations and determine whether they are suitable for accurate anthropometric and morphological analysis. The model can detect various rotation issues and provides actionable recommendations.

## Features

- **Multi-label Classification**: Detects multiple rotation issues simultaneously
- **Viability Assessment**: Determines if frontal faces are suitable for further analysis
- **Quality Enhancement**: Optional image preprocessing and quality assessment
- **Comprehensive Visualizations**: Detailed analysis reports with charts and recommendations
- **Batch Processing**: Support for analyzing multiple images at once
- **GPU Acceleration**: Optimized for CUDA with CPU fallback

## Model Classes

The model classifies frontal faces into the following categories:

- `aceptable`: Frontal face orientation is acceptable for analysis
- `hacia_arriba_o_tomadao_desde_abajo`: Face tilted upward or camera positioned too low
- `horizontal`: Horizontal face orientation
- `diagonal`: Diagonal face tilt
- `hacia_abajo_o_tomado_desde_arriba`: Face tilted downward or camera positioned too high

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit (for GPU support)
- Trained model file: `improved_supervisely_head_rotation_model_MULTILABEL_CORRECTED.pth`

### Installation

1. **Place the trained model**:
   ```bash
   # Copy your trained model to the models directory
   cp /path/to/improved_supervisely_head_rotation_model_MULTILABEL_CORRECTED.pth ./models/
   ```

2. **Build and run with Docker**:
   ```bash
   # Build and start the service
   docker compose up --build -d
   
   # Check health
   curl http://localhost:8012/health
   ```

3. **Alternative: Run locally**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the API
   uvicorn app.main:app --host 0.0.0.0 --port 8012 --reload
   ```

## API Endpoints

### Complete Frontal Rotation Analysis
```bash
curl -X POST "http://localhost:8012/analyze-frontal-rotation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@frontal_face_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "enhance_image=true"
```

### Simple Rotation Classification
```bash
curl -X POST "http://localhost:8012/classify-rotation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@frontal_face_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Viability Assessment Only
```bash
curl -X POST "http://localhost:8012/assess-viability" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@frontal_face_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Image Quality Assessment
```bash
curl -X POST "http://localhost:8012/assess-image-quality" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@frontal_face_image.jpg"
```

### Batch Processing
```bash
curl -X POST "http://localhost:8012/batch-analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "confidence_threshold=0.5"
```

### Health Check
```bash
curl http://localhost:8012/health
```

### Model Information
```bash
curl http://localhost:8012/model-info
```

## Response Format

### Complete Analysis Response
```json
{
  "analysis_id": "uuid-string",
  "image_info": {
    "width": 800,
    "height": 600,
    "channels": 3
  },
  "rotation_assessment": {
    "predicted_tags": ["aceptable"],
    "rotation_issues": [],
    "is_suitable": true,
    "suitability_reason": "Frontal face orientation is acceptable for analysis",
    "aceptable_confidence": 0.85,
    "max_confidence": 0.85,
    "prediction_certainty": "high",
    "all_probabilities": {
      "aceptable": 0.85,
      "hacia_arriba_o_tomadao_desde_abajo": 0.12,
      "horizontal": 0.08,
      "diagonal": 0.05,
      "hacia_abajo_o_tomado_desde_arriba": 0.03
    },
    "threshold_used": 0.5
  },
  "viability_for_analysis": {
    "suitable_for_anthropometric": true,
    "suitable_for_morphological": true,
    "recommendation": "Frontal face orientation is acceptable. Proceed with anthropometric and morphological analysis.",
    "confidence_level": "high"
  },
  "image_quality": {
    "sharpness": {
      "value": 150.5,
      "level": "good"
    },
    "brightness": {
      "value": 120.3,
      "level": "good"
    },
    "contrast": {
      "value": 45.8,
      "level": "moderate"
    },
    "overall_quality": {
      "score": 75,
      "level": "good"
    },
    "is_suitable": true
  },
  "analysis_summary": {
    "processing_successful": true,
    "main_finding": "Frontal face orientation is acceptable for analysis",
    "predicted_orientation": "aceptable",
    "max_confidence": 0.85,
    "threshold_used": 0.5,
    "timestamp": "2024-12-15T14:30:22.123456"
  },
  "visualization_path": "/app/results/frontal_rotation_analysis_uuid.png",
  "visualization_url": "/visualization/frontal_rotation_analysis_uuid.png"
}
```

## Configuration

### GPU Production Setup
```yaml
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

### CPU Fallback
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=-1
# Remove deploy section
```

## Integration with Other Modules

This frontal rotation assessment module should be used as a preprocessing step before:

- **Frontal Morfologico** (Port 8000): Morphological analysis
- **Frontal Antropometrico** (Port 8001): Anthropometric measurements
- **Frontal Validacion** (Port 8002): Additional validation checks
- **Frontal Espejo** (Port 8008): Mirror analysis

### Recommended Workflow

1. **Frontal Rotation Assessment** (Port 8012): Determine if frontal face is oriented correctly
2. **Frontal Validacion** (Port 8002): Check for occlusions and quality issues
3. **Frontal Morfologico** (Port 8000): Perform morphological analysis if suitable
4. **Frontal Antropometrico** (Port 8001): Conduct measurements if suitable
5. **Frontal Espejo** (Port 8008): Advanced mirror-based analysis if suitable

## Model Training

The model was trained using the corrected multi-label approach including the 'aceptable' tag for quality control. 

Key features:
- **Architecture**: EfficientNet-B0 with multi-label head
- **Training Strategy**: Progressive unfreezing with data augmentation
- **Pattern-Aware**: Respects annotation patterns (aceptable is standalone)
- **Multi-label Support**: Can detect multiple rotation issues simultaneously
- **Quality Control**: Includes aceptable tag for frontal face quality assessment

## Troubleshooting

### Model Not Found
```
Detail: Model file not found at /app/models/improved_supervisely_head_rotation_model_MULTILABEL_CORRECTED.pth
```
**Solution**: Place the trained model file `improved_supervisely_head_rotation_model_MULTILABEL_CORRECTED.pth` in the `./models/` directory.

### GPU Issues
If GPU is not available, the model will automatically fall back to CPU processing.

### Memory Issues
For large images or batch processing, consider:
- Reducing batch size
- Using image enhancement: `enhance_image=false`
- Lowering confidence threshold for faster processing

## Performance

- **Single Image**: ~0.1-0.3 seconds (GPU) / ~0.5-1.0 seconds (CPU)
- **Batch Processing**: Up to 10 images per request
- **Memory Usage**: ~2GB GPU VRAM / ~4GB system RAM
- **Image Support**: JPEG, PNG, common formats
- **Max Image Size**: 2048x2048 pixels (auto-resized)

## Comparison with Profile Module

This frontal rotation module complements the profile rotation module (Port 8011):

| Feature | Frontal (Port 8012) | Profile (Port 8011) |
|---------|--------------------|--------------------|
| **Target** | Frontal faces | Profile faces |
| **Classes** | 5 (upward/downward tilt, horizontal, diagonal, aceptable) | 7 (various profile rotations, aceptable) |
| **Use Case** | Frontal anthropometric/morphological analysis | Profile anthropometric/morphological analysis |
| **Model** | Supervisely-trained frontal model | Profile-specific trained model |

## License

Part of the SG_prod AI Analysis Pipeline.

---
**Service Port**: 8012  
**API Documentation**: http://localhost:8012/docs  
**Health Check**: http://localhost:8012/health