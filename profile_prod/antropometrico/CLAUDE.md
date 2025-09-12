# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Profile Anthropometric Analysis Module** - a FastAPI-based microservice for advanced facial analysis on profile (lateral) images. The system performs comprehensive anthropometric measurements using an enhanced ensemble of deep learning models with ResNet-50 CNN, optional Graph Neural Network (GNN) validation, and sophisticated false positive filtering.

## Architecture

### Core Components
- **FastAPI Web Service** (`app/main.py`) - REST API with 4 main endpoints
- **Enhanced Pipeline** (`app/models/enhanced_pipeline.py`) - Primary analysis engine with CNN + GNN ensemble
- **Legacy Pipeline** (`app/models/profile_anthropometric_pipeline.py`) - Fallback compatibility layer  
- **Image Processing** (`app/utils/image_processing.py`) - Computer vision utilities
- **Visualization** (`app/utils/visualization.py`) - Result rendering and annotation

### Model Architecture
The system uses a multi-tiered approach:
1. **Primary CNN Model**: ResNet-50 + profile classification + 4-stage progressive decoder
2. **Optional GNN Model**: Graph Neural Network for anatomical relationship validation
3. **Fallback Model**: Legacy point detection for compatibility

### Key Features
- **30 Anatomical Landmarks**: Comprehensive facial point detection
- **20+ Anthropometric Measurements**: Nose, mandible, forehead, chin, ear analysis
- **Profile Classification**: Automatic left/right profile determination
- **False Positive Filtering**: Advanced ensemble reduces spurious detections by 60-80%
- **Angular Analysis**: Complex geometric calculations for facial proportions
- **Real-time Visualization**: Annotated result images

## Development Commands

### Docker Development (Recommended)
```bash
# Build and start service
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f profile-anthropometric

# Restart after code changes
docker-compose restart profile-anthropometric

# Stop service
docker-compose down
```

### Manual Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python app/main.py
# OR
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload

# Health check
curl http://localhost:8004/health
```

### Testing
```bash
# Basic API test
curl -X POST "http://localhost:8004/analyze-profile-anthropometric" \
  -F "file=@test_profile.jpg" \
  -F "confidence_threshold=0.15"

# Point detection only
curl -X POST "http://localhost:8004/detect-profile-points" \
  -F "file=@test_profile.jpg"

# Model information
curl http://localhost:8004/model-info
```

## Model Configuration

### Required Model Files
Place in `./models/` directory:

**Enhanced Mode (Recommended):**
- `best_point_detection_model_v2.pth` - Primary CNN model (ResNet-50 + profile classification)
- `facial_landmark_gnn.pth` - Optional GNN validation model

**Fallback Mode:**
- `profile_aware_point_detection_model.pth` - Legacy model for compatibility

### Automatic Model Detection
The system automatically configures based on available models:
1. **Full Enhanced**: CNN v2 + GNN → Best accuracy with false positive filtering
2. **CNN-only**: CNN v2 only → Good accuracy, faster processing
3. **Fallback**: Legacy model → Basic functionality maintained

## API Endpoints

### Core Endpoints
- `GET /health` - Service health and model status
- `POST /analyze-profile-anthropometric` - Complete analysis with measurements
- `POST /detect-profile-points` - Point detection only
- `GET /model-info` - Model architecture information

### Parameters
- `confidence_threshold`: 0.05-0.9 (default: 0.15) - Detection confidence
- `include_visualization`: boolean (default: true) - Generate annotated images

## Key Implementation Details

### Point Detection System
- **80 Point Classes**: Comprehensive facial landmark coverage
- **Heatmap-based Detection**: 112x112 resolution heatmaps from 224x224 input
- **Profile-Aware Processing**: Automatic filtering of wrong-side landmarks
- **Sub-pixel Accuracy**: 4-stage progressive decoder refinement

### Anthropometric Calculations
The system calculates complex measurements including:
- **Reference Distance (24-10)**: Primary scaling measurement
- **Facial Thirds**: Superior/Middle/Inferior facial regions
- **Angular Analysis**: Forehead, chin, nose tip angles with mathematical rigor
- **Implantation Angles**: Superior/Inferior ear attachment analysis
- **Nasal Triangulation**: Advanced orifice measurements

### Angular Mathematics
All angular calculations use:
- Reference vector from point 24 to point 18 as baseline
- Dot product between normalized vectors: `np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)`
- Conversion: `np.degrees(np.arccos(cos_angle))`
- Profile-aware direction adjustment for left vs right profiles

## File Structure

```
antropometrico/
├── app/
│   ├── main.py                          # FastAPI application
│   ├── models/
│   │   ├── enhanced_pipeline.py         # Enhanced CNN + GNN pipeline
│   │   └── profile_anthropometric_pipeline.py  # Legacy pipeline
│   └── utils/
│       ├── image_processing.py          # OpenCV utilities
│       └── visualization.py             # Result rendering
├── models/                              # Model files (.pth)
├── results/                             # Generated visualizations
├── requirements.txt                     # Python dependencies
├── docker-compose.yml                   # Docker configuration
└── Dockerfile                          # Container build
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: 0, set to -1 for CPU)
- `PORT`: Service port (default: 8004)
- `PYTHONPATH`: Set to /app in container

## Performance Characteristics

### Processing Times (GPU)
- **Full Enhanced Mode**: 0.8-2.0 seconds per image
- **CNN-only Mode**: 0.5-1.2 seconds per image  
- **Legacy Fallback**: 0.5-2.0 seconds per image

### Memory Requirements
- **GPU Memory**: 2-4GB for model inference
- **System Memory**: 1-2GB for image processing

### Quality Improvements
- **False Positive Reduction**: 60-80% fewer spurious detections vs legacy
- **Profile Classification**: More robust left/right determination
- **Edge Case Handling**: Better performance on challenging images

## Development Notes

### Import Strategy
The codebase uses a robust multi-tier import system in `app/main.py`:
1. Relative imports for Docker environment
2. Fallback to direct module imports
3. Comprehensive error logging for debugging

### Error Handling
- Graceful model loading failures with fallback modes
- Comprehensive HTTP status codes (400, 503, 500)
- Detailed health check responses with model status

### Visualization System  
- Results stored in `/app/results/` directory
- PNG format with UUID-based naming
- Static file serving via FastAPI StaticFiles
- URL format: `/visualization/profile_anthropometric_{uuid}.png`

## Docker Configuration

### GPU Support
The docker-compose.yml includes NVIDIA GPU support:
- Uses `driver: nvidia` with `count: 1`
- Requires nvidia-docker runtime
- Falls back to CPU mode if GPU unavailable

### Volume Mounts
- `./results:/app/results` - Generated visualizations
- `./models:/app/models:ro` - Read-only model files

### Health Checks
Automatic health monitoring with:
- 30s intervals, 10s timeout
- 3 retries with 40s startup grace period
- Endpoint: `http://localhost:8004/health`