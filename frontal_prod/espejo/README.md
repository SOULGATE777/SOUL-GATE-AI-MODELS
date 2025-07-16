# Espejo Analysis Module

Advanced mirror face analysis with anthropometric measurements, decision tree classification, and hybrid splitting for comprehensive facial diagnosis.

## Features

- **Mirror Face Generation**: Create left and right mirrored faces for asymmetry analysis
- **Anthropometric Analysis**: Calculate face, forehead, and temporal proportions
- **Decision Tree Classification**: Apply Excel-based decision rules for FRENTE and rostro_menton regions
- **Hybrid Class Splitting**: Proportion-based class splitting for enhanced accuracy
- **Comprehensive Reporting**: Detailed analysis reports and visualizations
- **GPU Acceleration**: CUDA support with CPU fallback

## Architecture

The espejo module implements a complete pipeline:

1. **Face Detection & Landmarks**: 68-point facial landmark detection using dlib
2. **Custom Point Detection**: Faster R-CNN model for 13 custom anthropometric points
3. **Proportion Calculations**: Face, forehead, and temporal proportion measurements
4. **Mirror Generation**: Left and right mirrored face creation with alignment
5. **Region Classification**: Binary, FRENTE, and rostro_menton classification
6. **Decision Tree Processing**: Excel-based decision rules application
7. **Hybrid Splitting**: Proportion-based final diagnosis refinement

## API Endpoints

### Primary Analysis
- `POST /analyze-espejo` - Complete espejo analysis with all features
- `POST /generate-mirrors` - Generate mirror images only
- `POST /classify-regions` - Classify facial regions with decision tree
- `POST /analyze-proportions` - Analyze facial proportions only
- `POST /get-diagnosis` - Get final diagnosis with rules applied

### Utility Endpoints
- `GET /health` - Health check
- `GET /model-info` - Model information
- `GET /api-info` - API information
- `GET /results/{filename}` - Retrieve generated images

## Models Required

Place the following models in the `/models` directory:

1. **facial_points_detection_model.pth** - Custom Faster R-CNN for anthropometric points
2. **binary_region_classifier_best.pth** - Binary FRENTE/rostro_menton classifier
3. **frente_best_model.pth** - FRENTE region shape classifier
4. **rostro_menton_best_model.pth** - rostro_menton region shape classifier
5. **shape_predictor_68_face_landmarks.dat** - dlib facial landmark detector

## Usage Examples

### Complete Analysis
```bash
curl -X POST "http://localhost:8008/analyze-espejo" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "include_dashboard=true"
```

### Mirror Generation Only
```bash
curl -X POST "http://localhost:8008/generate-mirrors" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Final Diagnosis
```bash
curl -X POST "http://localhost:8008/get-diagnosis" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "format=json"
```

## Response Format

### Complete Analysis Response
```json
{
  "face_detected": true,
  "anthropometric_analysis": {
    "face_proportions": {
      "right": 0.456,
      "left": 0.443
    },
    "forehead_proportions": {
      "right": 0.387,
      "left": 0.392
    },
    "temporal_proportions": {
      "right": 0.523,
      "left": 0.518
    },
    "custom_model_points": {...},
    "landmarks_detected": 68
  },
  "mirror_analysis": {
    "right_mirrored": {...},
    "left_mirrored": {...}
  },
  "final_diagnosis": {
    "right_side": {
      "frente_diagnosis": "solar",
      "rostro_diagnosis": "venus_corazon",
      "confidence_scores": {...}
    },
    "left_side": {
      "frente_diagnosis": "luna",
      "rostro_diagnosis": "pluton-venus",
      "confidence_scores": {...}
    }
  },
  "decision_tree_analysis": {
    "right_side": {
      "frente_applied_rules": [...],
      "rostro_applied_rules": [...],
      "frente_split_rules": [...],
      "rostro_split_rules": [...]
    },
    "left_side": {...}
  },
  "analysis_summary": {...},
  "visualization_path": "/app/results/espejo_analysis_20241215_143022_abc123.png",
  "dashboard_path": "/app/results/espejo_dashboard_20241215_143022_def456.png"
}
```

## Decision Tree Rules

### FRENTE Region Rules
- **Exclusion Rules**: confidence-based filtering
- **Proportion Rules**: neptuno exclusion based on face proportion
- **Hybrid Splitting**: solar/lunar split using forehead proportion (threshold: 0.35)

### rostro_menton Region Rules
- **Exclusion Rules**: confidence thresholds per class
- **Complex Logic**: multi-class conflict resolution
- **Anthropometric Rules**: proportion-based diagnosis (face proportion thresholds: 0.99, 1.17)

## Classification Classes

### FRENTE Classes
- jupiter_aplio_base_ancha
- marte_rectangular
- mercurio_triangulo
- neptuno_ovalo/capsula
- solar/lunar_redonda
- tierra_cuadrada
- venus_corazon_o_trapezoide_angosto

### rostro_menton Classes
- jupiter/luna_redondo_ancho
- marte/tierra_rectangulo
- mercurio_triangular
- pluton-venus
- pluton_hexagonal
- saturno_trapezoide_base_angosta
- sol_neptuno_ovalo
- venus_corazon

## Deployment

### Docker Deployment
```bash
# Build and run with GPU support
docker compose up --build -d

# Check health
curl http://localhost:8008/health
```

### Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run with GPU
CUDA_VISIBLE_DEVICES=0 uvicorn app.main:app --host 0.0.0.0 --port 8008

# Run with CPU only
CUDA_VISIBLE_DEVICES=-1 uvicorn app.main:app --host 0.0.0.0 --port 8008
```

## Configuration

### GPU Configuration
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

### CPU Configuration
```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=-1
# Remove deploy section
```

## System Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM (recommended)
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ for models and containers
- **CPU**: Multi-core processor for preprocessing

## Port Information

- **Service Port**: 8008
- **Health Check**: http://localhost:8008/health
- **API Documentation**: http://localhost:8008/docs
- **Next Available Port**: 8009

## Integration

### With Other Modules
The espejo module can be used in conjunction with other frontal analysis modules:

1. **Frontal Validacion (Port 8002)**: Validate image quality first
2. **Frontal Morfologico (Port 8000)**: Compare morphological results
3. **Frontal Antropometrico (Port 8001)**: Cross-reference measurements
4. **Espejo Analysis (Port 8008)**: Mirror-based comprehensive analysis

### Workflow Integration
```bash
# 1. Validate image quality
curl -X POST "http://localhost:8002/validate-image" -F "file=@image.jpg"

# 2. Perform espejo analysis
curl -X POST "http://localhost:8008/analyze-espejo" -F "file=@image.jpg"

# 3. Compare with standard analysis
curl -X POST "http://localhost:8000/analyze-morphological" -F "file=@image.jpg"
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure all model files are in `/models` directory
   - Check model file permissions
   - Verify GPU/CUDA availability

2. **Memory Issues**
   - Reduce batch size for large images
   - Use CPU fallback if GPU memory insufficient
   - Check system memory availability

3. **Face Detection Failures**
   - Ensure face is clearly visible
   - Check image quality and lighting
   - Verify image format compatibility

4. **Classification Errors**
   - Check confidence thresholds
   - Verify model compatibility
   - Review input image preprocessing

### Performance Optimization

- Use GPU acceleration for faster inference
- Implement image preprocessing optimization
- Consider model quantization for deployment
- Use appropriate confidence thresholds

## Development

### Adding New Classification Classes
1. Update class mappings in `espejo_pipeline.py`
2. Retrain classification models
3. Update decision tree rules
4. Add new hybrid splitting logic

### Extending Decision Rules
1. Modify decision tree functions
2. Add new proportion calculations
3. Update hybrid splitting criteria
4. Test with validation dataset

## Support

For issues and questions:
- Check health endpoint: `/health`
- Review logs in container
- Verify model files and permissions
- Test with sample images

---

**Status**: ✅ **PRODUCTION READY** - Complete mirror analysis pipeline with decision tree classification and hybrid splitting
**Port**: 8008
**Next Module Port**: 8009