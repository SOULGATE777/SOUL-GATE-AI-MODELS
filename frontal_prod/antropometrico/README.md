# Antropometrico Analysis API v2.0

Advanced anthropometric facial analysis system with comprehensive feature detection, custom model integration, and detailed reporting capabilities.

## 🚀 New Features in v2.0

### Enhanced Analysis Features
- **Eyebrow Length Analysis**: Classifies eyebrow length relative to eye length (corta/normal/larga)
- **Eye Angle Analysis**: Measures and classifies eye angles (normal/hacia arriba/hacia abajo)
- **Face Area Proportions**: Analyzes inner/outer face area ratios and eye-to-face proportions
- **Enhanced Model Integration**: Uses all 13 custom model points with confidence scoring
- **Comprehensive Reporting**: Detailed text and JSON reports with all measurements

### API Endpoints
- **Complete Analysis**: `/analyze-anthropometric` - Full facial analysis with all features
- **Focused Analysis**: Specialized endpoints for eyebrows, eyes, and face areas
- **Detailed Reporting**: `/get-detailed-report` - Comprehensive analysis reports
- **Enhanced Visualization**: Multiple visualization options including detailed report images

## 📁 Directory Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                          # Enhanced FastAPI application
│   ├── models/
│   │   ├── anthropometric_pipeline.py   # Core analysis engine with new features
│   │   └── __init__.py
│   └── utils/
│       ├── image_processing.py          # Image processing utilities
│       ├── visualization.py             # Enhanced visualization functions
│       └── __init__.py
├── models/
│   ├── facial_points_detection_model.pth    # Custom trained model
│   └── shape_predictor_68_face_landmarks.dat # Dlib predictor
├── results/                             # Generated visualizations and reports
├── dataset_frontal/                     # Test images
├── docker-compose.yml                   # Enhanced Docker composition
├── Dockerfile                          # Updated container definition
├── requirements.txt                     # Updated dependencies
└── README.md                           # This file
```

## 🔧 Installation & Setup

### Prerequisites
- Docker and Docker Compose
- At least 4GB RAM available for the container
- The required model files:
  - `facial_points_detection_model.pth` (custom trained model)
  - `shape_predictor_68_face_landmarks.dat` (dlib model)

### Quick Start
1. **Clone and setup:**
   ```bash
   git clone <repository>
   cd antropometrico
   ```

2. **Ensure model files are in place:**
   ```bash
   # Place your model files in the models/ directory
   ls models/
   # Should show:
   # facial_points_detection_model.pth
   # shape_predictor_68_face_landmarks.dat
   ```

3. **Build and run:**
   ```bash
   docker-compose up --build
   ```

4. **Verify installation:**
   ```bash
   curl http://localhost:8001/health
   ```

## 📊 API Documentation

### Complete Analysis Endpoint
```bash
POST /analyze-anthropometric
```

**Parameters:**
- `file`: Image file (multipart/form-data)
- `confidence_threshold`: Model confidence threshold (0.0-1.0, default: 0.5)
- `include_visualization`: Generate visualization image (default: true)
- `include_detailed_report`: Generate detailed report image (default: false)

**Response includes:**
- Facial landmarks (68 points + extended points)
- All model predictions (up to 13 custom points)
- Facial thirds analysis
- Eyebrow length classifications
- Eye angle measurements
- Face area proportions
- Comprehensive summary with classifications

### Specialized Analysis Endpoints

#### Eyebrow Analysis
```bash
POST /analyze-eyebrows
```
Returns focused eyebrow analysis with length classifications and proportions.

#### Eye Analysis
```bash
POST /analyze-eyes
```
Returns eye angle analysis, classifications, and face proportion measurements.

#### Face Area Analysis
```bash
POST /analyze-face-areas
```
Returns inner/outer face area analysis and proportional measurements.

#### Eye Colorimetry Analysis
```bash
POST /analyze-eye-colorimetry
```
Returns comprehensive iris color analysis with RGB and HSV classification systems.

```bash
POST /analyze-iris-color
```
Returns focused iris color classification using dominant or average color methods.

```bash
POST /compare-eye-colors
```
Compares eye color classifications across different analysis methods.

### Detailed Reporting
```bash
POST /get-detailed-report?format=text
POST /get-detailed-report?format=json
```
Generates comprehensive analysis reports in text or JSON format.

## 🎯 Analysis Features

### Facial Thirds Analysis
- **Primer Tercio**: Top of head to between eyebrows
- **Segundo Tercio**: Between eyebrows to nose base  
- **Tercer Tercio**: Nose base to chin
- Classifications: largo/corto/standard based on proportional thresholds

### Eyebrow Analysis
- **Length Classification**: Compares eyebrow length to corresponding eye length
  - `ceja corta`: < 1.0 ratio
  - `ceja normal`: 1.0-1.4 ratio
  - `ceja larga`: > 1.4 ratio
- **Detailed Measurements**: Absolute lengths and proportional ratios

### Eye Analysis
- **Angle Classification**: Measures eye slope relative to horizontal
  - `angulo normal`: -5° to +5°
  - `angulo hacia arriba`: > +5° (upward tilt)
  - `angulo hacia abajo`: < -5° (downward tilt)
- **Proportional Analysis**: Eye spacing and face proportions

### Face Area Analysis
- **Inner/Outer Ratio**: Proportion of inner facial features to total face area
- **Eye-to-Face Proportions**: Individual eye area relative to total face
- **Detailed Measurements**: Absolute areas and percentage calculations

### Eye Colorimetry Analysis
- **Iris Color Classification**: Advanced RGB and HSV-based eye color detection
- **Multiple Classification Systems**: RGB range-based and HSV hue-based classification
- **Comprehensive Color Analysis**: K-means clustering for dominant colors and average color calculation
- **Bilateral Analysis**: Separate analysis for left and right eyes
- **Color Categories**: Supports 8 distinct eye color classifications including café oscuro, café claro/hazel, verde, azul claro/gris, azul oscuro, azul intenso/morado, amarillo, and azul verde

### Model Integration
- **13 Custom Points**: Utilizes all available model predictions
- **Enhanced Accuracy**: Model points replace inferred calculations where available
- **Confidence Scoring**: Adjustable thresholds for point detection
- **Fallback System**: Graceful degradation when model points unavailable

## 🖼️ Visualization Options

### Standard Visualization
- All 68 facial landmarks
- Extended points (69-72)
- Model predictions with color coding
- Facial thirds reference lines
- Key measurements overlay

### Detailed Report Image
- Side-by-side image and comprehensive measurements
- All analysis results in organized format
- Model integration status
- Classification results

### Specialized Visualizations
- Eyebrow analysis with length comparisons
- Eye angle visualization with degree measurements
- Face area boundary visualization
- Model point detection visualization

## 🔍 Usage Examples

### Complete Analysis with Visualization
```bash
curl -X POST "http://localhost:8001/analyze-anthropometric" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true" \
  -F "include_detailed_report=true"
```

### Get Text Report
```bash
curl -X POST "http://localhost:8001/get-detailed-report?format=text" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5"
```

### Focused Eyebrow Analysis
```bash
curl -X POST "http://localhost:8001/analyze-eyebrows" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg"
```

### Eye Colorimetry Analysis
```bash
curl -X POST "http://localhost:8001/analyze-eye-colorimetry" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"
```

### Iris Color Classification
```bash
curl -X POST "http://localhost:8001/analyze-iris-color" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face_image.jpg" \
  -F "use_dominant_color=true"
```

## 📈 Performance & Requirements

### System Requirements
- **Memory**: Minimum 2GB, Recommended 4GB
- **CPU**: Multi-core recommended for model inference
- **Storage**: 1GB for models and temporary files
- **Network**: Port 8001 available

### Processing Performance
- **Analysis Time**: 2-5 seconds per image (depending on resolution)
- **Model Inference**: ~1 second for point detection
- **Visualization Generation**: ~1 second additional
- **Supported Formats**: JPG, PNG, BMP, TIFF

### Image Requirements
- **Minimum Resolution**: 100x100 pixels
- **Maximum Resolution**: 4000x4000 pixels (auto-resized)
- **Face Requirements**: Clear frontal face view
- **Optimal Conditions**: Good lighting, minimal occlusion

## 🚨 Error Handling

### Common Issues
- **No Face Detected**: Returns error with empty analysis
- **Model Not Loaded**: Check model file paths and permissions
- **Low Confidence**: Adjust confidence_threshold parameter
- **Memory Issues**: Reduce image size or increase container memory

### Troubleshooting
1. **Check Health Endpoint**: `curl http://localhost:8001/health`
2. **Verify Model Files**: Ensure both model files are present and readable
3. **Check Logs**: `docker-compose logs antropometrico-api`
4. **Image Quality**: Ensure clear, well-lit frontal face images

## 🔄 API Changes from v1.0

### New Endpoints
- `/analyze-eyebrows` - Focused eyebrow analysis
- `/analyze-eyes` - Eye angle and proportion analysis
- `/analyze-face-areas` - Face area proportion analysis
- `/get-detailed-report` - Comprehensive reporting

### Enhanced Responses
- Additional analysis fields in `/analyze-anthropometric`
- Detailed classifications and measurements
- Model integration status and confidence scores
- Enhanced error reporting and validation

### Backward Compatibility
- All v1.0 endpoints remain functional
- Response format extended (not breaking)
- Previous visualization format supported

## 📝 Development & Testing

### Running Tests
```bash
# Test with sample image
curl -X POST "http://localhost:8001/analyze-anthropometric" \
  -F "file=@dataset_frontal/sample.jpg"

# Health check
curl http://localhost:8001/health

# API information
curl http://localhost:8001/api-info
```

### Development Setup
```bash
# For local development without Docker
pip install -r requirements.txt
python -m app.main
```

## 📄 License & Credits

This enhanced version includes advanced anthropometric analysis capabilities developed for comprehensive facial feature assessment. The system integrates classical facial landmark detection with custom-trained deep learning models for enhanced accuracy and detailed analysis.
