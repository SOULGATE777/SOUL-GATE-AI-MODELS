# Hand Analysis Module

A production-ready Docker module for comprehensive hand analysis including CNN-based classification (dorso/palma) and advanced colorimetry analysis for palm regions.

## Features

- **CNN Hand Classification**: ResNet50-based model for distinguishing between dorso (back) and palma (palm) of hands
- **Colorimetry Analysis**: Advanced color analysis of palm regions using multiple color spaces
- **Color Classification**: Classification into predefined palm color types based on traditional analysis
- **Comprehensive Visualization**: Detailed visual reports of analysis results
- **FastAPI Integration**: Production-ready REST API with async processing
- **Docker Support**: Complete containerization with CUDA GPU support

## Architecture

```
manos/
├── Dockerfile                    # Container configuration
├── docker-compose.yml           # Service orchestration
├── requirements.txt             # Python dependencies
├── app/
│   ├── main.py                  # FastAPI application
│   ├── models/
│   │   └── hand_analysis_pipeline.py  # Core analysis pipeline
│   └── utils/
│       ├── image_processing.py  # Image processing utilities
│       └── visualization.py     # Visualization creation
├── models/                      # ML model files
│   └── dorso_palma_classifier.pth
└── results/                     # Analysis outputs
```

## API Endpoints

### Core Analysis
- `POST /analyze-hand-comprehensive` - Complete hand analysis (CNN + colorimetry)
- `POST /classify-hand-side` - CNN classification only (dorso/palma)
- `POST /analyze-colorimetry` - Colorimetry analysis only
- `POST /batch-analyze` - Batch processing for multiple images

### Information
- `GET /health` - Service health check
- `GET /model-info` - Model and pipeline information
- `GET /` - Service overview and available endpoints

### Visualizations
- `GET /visualization/{filename}` - Serve generated visualization files

## Color Classification Types

The module includes classification for traditional palm color analysis:

1. **rosa/sanguineo-linfatico oscuro** - Pink/sanguine-lymphatic dark
2. **rojo/sanguineo** - Red/sanguine
3. **amarillo/nervioso** - Yellow/nervous
4. **blanco/linfatico** - White/lymphatic
5. **bilioso/cafe_o_oscuro** - Bilious/brown or dark

## Usage

### Docker Deployment

1. **Build and run the service:**
```bash
cd manos
docker-compose up --build
```

2. **Service will be available at:**
- API: `http://localhost:8009`
- Documentation: `http://localhost:8009/docs`
- Health check: `http://localhost:8009/health`

### API Usage Examples

#### Complete Hand Analysis
```bash
curl -X POST "http://localhost:8009/analyze-hand-comprehensive" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_colorimetry=true" \
  -F "include_visualization=true"
```

#### Hand Side Classification Only
```bash
curl -X POST "http://localhost:8009/classify-hand-side" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand_image.jpg" \
  -F "confidence_threshold=0.7"
```

#### With Bounding Box
```bash
curl -X POST "http://localhost:8009/analyze-hand-comprehensive" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@hand_image.jpg" \
  -F "bbox=100,50,300,250"
```

## Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES` - GPU device selection (default: 0)
- `PYTHONUNBUFFERED` - Python output buffering (default: 1)

### Model Requirements
Place your trained CNN model at:
- `models/dorso_palma_classifier.pth`

The model should be a PyTorch checkpoint containing:
- `model_state_dict`: The trained ResNet50 model weights
- Binary classification for dorso (0) vs palma (1)

## Response Format

### Complete Analysis Response
```json
{
  "analysis_id": "unique-uuid",
  "analysis_type": "comprehensive_hand_analysis",
  "image_path": "/app/temp/temp_uuid.jpg",
  "bbox": [x_min, y_min, x_max, y_max],
  "cnn_prediction": {
    "predicted_class": "Palma",
    "confidence": 0.89,
    "probabilities": {
      "Dorso": 0.11,
      "Palma": 0.89
    },
    "meets_threshold": true
  },
  "colorimetry": {
    "average_color_rgb": [185, 142, 125],
    "average_color_hsv": [12.5, 32.4, 72.5],
    "dominant_colors": [
      [[190, 145, 128], 35.2],
      [[180, 138, 120], 28.7]
    ],
    "hue_mean": 12.8,
    "hue_std": 8.5,
    "total_pixels": 15432
  },
  "color_classification": {
    "dominant_color_1": {
      "rgb": [190, 145, 128],
      "percentage": 35.2,
      "classification": {
        "rosa/sanguineo-linfatico oscuro": 50.0,
        "rojo/sanguineo": 50.0
      }
    },
    "dominant_color_2": {
      "rgb": [180, 138, 120],
      "percentage": 28.7,
      "classification": {
        "rosa/sanguineo-linfatico oscuro": 100.0
      }
    },
    "dominant_color_3": {
      "rgb": [175, 135, 115],
      "percentage": 22.1,
      "classification": {
        "bilioso/cafe_o_oscuro": 100.0
      }
    }
  },
  "visualization_url": "/visualization/hand_analysis_uuid.png"
}
```

## Technical Details

### CNN Model Architecture
- **Base**: ResNet50 with ImageNet pre-training
- **Input Size**: 224x224x3 RGB images
- **Output**: 2 classes (Dorso, Palma)
- **Features**: Dropout layers, custom classifier head

### Colorimetry Pipeline
- **Skin Detection**: HSV + YCrCb color space filtering
- **Color Clustering**: K-means clustering for dominant color extraction
- **Color Analysis**: RGB, HSV analysis with statistical measures
- **Classification**: Rule-based classification into traditional palm color types

### Performance
- **GPU Support**: CUDA acceleration for CNN inference
- **Async Processing**: Non-blocking API operations
- **Batch Processing**: Support for multiple image analysis
- **Memory Management**: Automatic cleanup of temporary files

## Monitoring

### Health Checks
The service includes comprehensive health monitoring:
- Container health checks every 30 seconds
- API endpoint availability
- Model loading status
- GPU availability

### Logging
Structured logging for:
- Request processing
- Model inference
- Error tracking
- Performance metrics

## Integration

This module follows the established pattern for the body analysis ecosystem:
- **Port**: 8009 (sequential with other modules)
- **Network**: Shared `body-network` for inter-service communication
- **Volumes**: Persistent storage for results and models
- **GPU**: NVIDIA GPU support with proper resource allocation

## Development

For development and testing:

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run locally:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8009 --reload
```

3. **Test the API:**
```bash
pytest tests/
```