# Profile Rotation Assessment API

Advanced profile rotation classification to assess viability for anthropometric and morphological analysis.

## Overview

This module uses a multi-label CNN model (EfficientNet-B0) to classify profile image orientations and determine whether they are suitable for accurate anthropometric and morphological analysis. The model can detect various rotation issues and provides actionable recommendations.

## Features

- **Multi-label Classification**: Detects multiple rotation issues simultaneously
- **Viability Assessment**: Determines if profiles are suitable for further analysis
- **Quality Enhancement**: Optional image preprocessing and quality assessment
- **Comprehensive Visualizations**: Detailed analysis reports with charts and recommendations
- **Batch Processing**: Support for analyzing multiple images at once
- **GPU Acceleration**: Optimized for CUDA with CPU fallback

## Model Classes

The model classifies profiles into the following categories:

- `aceptable`: Profile orientation is acceptable for analysis
- `hacia_abajo`: Profile tilted downward
- `hacia_arriba`: Profile tilted upward  
- `desde_abajo_o_mandibula_hacia_primer_plano`: Camera positioned below subject
- `desde_arriba_o_superior_hacia_primer_plano`: Camera positioned above subject
- `rotado_lejos_o_desde_atras`: Profile rotated away from camera
- `rotado_hacia_desde_adelante`: Profile rotated toward camera

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with drivers (recommended)
- NVIDIA Container Toolkit (for GPU support)
- Trained model file: `best_profile_classifier_multilabel.pth`

### Installation

1. **Place the trained model**:
   ```bash
   # Copy your trained model to the models directory
   cp /path/to/best_profile_classifier_multilabel.pth ./models/
   ```

2. **Build and run with Docker**:
   ```bash
   # Build and start the service
   docker compose up --build -d
   
   # Check health
   curl http://localhost:8011/health
   ```

3. **Alternative: Run locally**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the API
   uvicorn app.main:app --host 0.0.0.0 --port 8011 --reload
   ```

## API Endpoints

### Complete Rotation Analysis
```bash
curl -X POST "http://localhost:8011/analyze-profile-rotation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "enhance_image=true"
```

### Simple Rotation Classification
```bash
curl -X POST "http://localhost:8011/classify-rotation" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Viability Assessment Only
```bash
curl -X POST "http://localhost:8011/assess-viability" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Image Quality Assessment
```bash
curl -X POST "http://localhost:8011/assess-image-quality" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@profile_image.jpg"
```

### Batch Processing
```bash
curl -X POST "http://localhost:8011/batch-analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "confidence_threshold=0.5"
```

### Health Check
```bash
curl http://localhost:8011/health
```

### Model Information
```bash
curl http://localhost:8011/model-info
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
    "suitability_reason": "Profile orientation is acceptable for analysis",
    "aceptable_confidence": 0.85,
    "max_confidence": 0.85,
    "prediction_certainty": "high",
    "all_probabilities": {
      "aceptable": 0.85,
      "hacia_abajo": 0.12,
      "hacia_arriba": 0.08,
      "desde_abajo_o_mandibula_hacia_primer_plano": 0.05,
      "desde_arriba_o_superior_hacia_primer_plano": 0.03,
      "rotado_lejos_o_desde_atras": 0.02,
      "rotado_hacia_desde_adelante": 0.04
    },
    "threshold_used": 0.5
  },
  "viability_for_analysis": {
    "suitable_for_anthropometric": true,
    "suitable_for_morphological": true,
    "recommendation": "Profile orientation is acceptable. Proceed with anthropometric and morphological analysis.",
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
    "main_finding": "Profile orientation is acceptable for analysis",
    "predicted_orientation": "aceptable",
    "max_confidence": 0.85,
    "threshold_used": 0.5,
    "timestamp": "2024-12-15T14:30:22.123456"
  },
  "visualization_path": "/app/results/rotation_analysis_uuid.png",
  "visualization_url": "/visualization/rotation_analysis_uuid.png"
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

This rotation assessment module should be used as a preprocessing step before:

- **Profile Morfologico** (Port 8003): Morphological analysis
- **Profile Antropometrico** (Port 8004): Anthropometric measurements
- **Profile Validacion** (Port 8005): Additional validation checks

### Recommended Workflow

1. **Rotation Assessment** (Port 8011): Determine if profile is oriented correctly
2. **Profile Validacion** (Port 8005): Check for occlusions and quality issues
3. **Profile Morfologico** (Port 8003): Perform morphological analysis if suitable
4. **Profile Antropometrico** (Port 8004): Conduct measurements if suitable

## Model Training

The model was trained using the notebook:
`/home/carlos/Documents/SG/Cnn_rotacion_perfil/multilabel_cnn_rotacion_perfil.ipynb`

Key features:
- **Architecture**: EfficientNet-B0 with multi-label head
- **Training Strategy**: Progressive unfreezing with data augmentation
- **Pattern-Aware**: Respects annotation patterns (aceptable is standalone)
- **Multi-label Support**: Can detect multiple rotation issues simultaneously

## Troubleshooting

### Model Not Found
```
Detail: Model file not found at /app/models/best_profile_classifier_multilabel.pth
```
**Solution**: Place the trained model file `best_profile_classifier_multilabel.pth` in the `./models/` directory.

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

## License

Part of the SG_prod AI Analysis Pipeline.

---
**Service Port**: 8011  
**API Documentation**: http://localhost:8011/docs  
**Health Check**: http://localhost:8011/health