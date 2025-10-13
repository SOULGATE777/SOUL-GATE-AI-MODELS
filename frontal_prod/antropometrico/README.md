# Antropometrico Analysis API v2.1

Advanced anthropometric facial analysis system with comprehensive feature detection, custom model integration, and detailed reporting capabilities.

## 🚀 New Features in v2.1

### Latest Updates (v2.1)
- **Improved Angle Measurements**: Eyebrow and eye angles now calculated relative to vertical midline reference (point 9 to M3) instead of horizontal baseline
- **Enhanced Anatomical Accuracy**: Standardized angle calculations ensure symmetrical features produce consistent measurements
- **Reference Vector Integration**: Uses Model Point 3 (M3) when available for accurate vertical reference alignment
- **Consistent Bilateral Measurements**: Both left and right features now measured with anatomically equivalent directions
- **Bug Fixes**: Face alignment preprocessing improvements, updated object detection model integration
- **Classification Improvements**: Enhanced eyebrow type classifier integration with morfológico pipeline

### Enhanced Analysis Features
- **Eyebrow Length Analysis**: Classifies eyebrow length relative to eye length (corta/normal/larga)
- **Eye Angle Analysis**: Measures and classifies eye angles relative to facial vertical axis (normal/hacia arriba/hacia abajo)
- **Eyebrow Slope Analysis**: Three-segment eyebrow angle analysis relative to perpendicular of vertical midline
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

## 🎯 Detailed Analysis Features

### Facial Landmark Detection
The system uses 68 standard dlib facial landmarks (points 0-67) plus 4 extended points (68-71) and up to 13 custom model-detected points:

#### Standard Landmarks (0-67)
- **Face contour**: Points 0-16 (jaw line from ear to ear)
- **Right eyebrow**: Points 17-21 (outer to inner edge)
- **Left eyebrow**: Points 22-26 (inner to outer edge)
- **Nose bridge**: Points 27-30 (top to tip)
- **Nose base**: Points 31-35 (nostrils and base)
- **Right eye**: Points 36-41 (outer corner to inner corner)
- **Left eye**: Points 42-47 (inner corner to outer corner)
- **Mouth outer**: Points 48-59 (outer lip contour)
- **Mouth inner**: Points 60-67 (inner lip contour)

#### Extended Landmarks (68-71)
- **Point 68**: Between eyebrows (uses model point 2 when available)
- **Point 69**: Top of head (uses model point 3 when available)
- **Point 70**: Left pupil center (calculated from eye landmarks)
- **Point 71**: Right pupil center (calculated from eye landmarks)

#### Custom Model Points (1-13)
Advanced deep learning model detects up to 13 specialized anthropometric points with confidence scoring:
- **Point 1**: Superior facial implantation reference
- **Point 2**: Between eyebrows (replaces calculated point 68)
- **Point 3**: Superior head point (replaces calculated point 69)
- **Points 4-13**: Additional facial landmarks for enhanced accuracy

### Facial Thirds Analysis (Golden Ratio Assessment)
Divides the face into three vertical sections using key anthropometric points:

#### Measurement Points
- **Top reference**: Point 69 (top of head) or model point 3
- **Upper division**: Point 68 (between eyebrows) or model point 2
- **Middle division**: Point 34 (nose base/subnasale)
- **Lower reference**: Point 9 (menton/chin)

#### Proportional Calculations
- **Primer Tercio**: Distance 69-68 / Total head height
  - `tercio superior largo`: > 0.38 proportion
  - `tercio superior standard`: 0.27-0.38 proportion
  - `tercio superior corto`: < 0.27 proportion

- **Segundo Tercio**: Distance 68-34 / Total head height
  - `tercio medio largo`: > 0.38 proportion
  - `tercio medio standard`: 0.27-0.38 proportion
  - `tercio medio corto`: < 0.27 proportion

- **Tercer Tercio**: Distance 34-9 / Total head height
  - `tercio inferior largo`: > 0.38 proportion
  - `tercio inferior standard`: 0.27-0.38 proportion
  - `tercio inferior corto`: < 0.27 proportion

### Eyebrow Morphometry Analysis
Analyzes eyebrow length relative to corresponding eye length for aesthetic proportion assessment:

#### Measurement Points
- **Right eyebrow**: Points 17 (outer) to 21 (inner)
- **Left eyebrow**: Points 22 (outer) to 26 (inner)
- **Right eye**: Points 36 (outer corner) to 39 (inner corner)
- **Left eye**: Points 42 (inner corner) to 45 (outer corner)

#### Length Classifications
- **Ceja corta**: Eyebrow/eye ratio < 1.0
- **Ceja normal**: Eyebrow/eye ratio 1.0-1.4
- **Ceja larga**: Eyebrow/eye ratio > 1.4

#### Slope Analysis (v2.1 Update)
Calculates eyebrow angles in three segments relative to the perpendicular of the vertical midline reference:
- **Reference System**: Angles measured relative to perpendicular of vertical axis (point 9 to M3)
- **Portion 1**: Inner segment angle (points 17-18 for right, 22-23 for left)
- **Portion 2**: Middle arch segment (points 18-20 for right, 23-25 for left)
- **Portion 3**: Outer descending segment (points 20-21 for right, 25-26 for left)
- **Anatomical Consistency**: Both eyebrows measured in equivalent anatomical directions for direct comparison

### Eye Angle Analysis (Palpebral Fissure Inclination) - v2.1 Update
Measures the angle of eye openings relative to the perpendicular of the vertical midline reference:

#### Measurement Points
- **Right eye**: Outer corner (point 36) to inner corner (point 39)
- **Left eye**: Inner corner (point 42) to outer corner (point 45)
- **Reference Vector**: Vertical midline from chin (point 9) to top of head (M3/point 69)

#### Angle Classifications
- **Ángulo normal**: -5° to +5° (alignment with perpendicular to vertical axis)
- **Ángulo hacia arriba**: > +5° (upward slanting relative to reference, positive canthal tilt)
- **Ángulo hacia abajo**: < -5° (downward slanting relative to reference, negative canthal tilt)

#### Calculation Method (v2.1)
1. Establishes vertical midline reference vector (point 9 to M3)
2. Calculates perpendicular to vertical reference as "horizontal" baseline
3. Measures eye angle relative to this perpendicular reference
4. Ensures both eyes measured in anatomically equivalent directions for bilateral symmetry assessment
5. Prefers Model Point 3 (M3) over calculated point 69 for enhanced accuracy

### Intercanthal Distance Analysis
Evaluates eye spacing proportions:

#### Measurements
- **Inner eye distance**: Distance between points 39 and 42 (inner corners)
- **Outer eye distance**: Distance between points 36 and 45 (outer corners)
- **Eye distance proportion**: Inner distance / Outer distance

#### Classifications
- **Cercanos**: < 0.3 proportion (hypotelorism)
- **Standard**: 0.3-0.37 proportion (normal spacing)
- **Lejanos**: > 0.37 proportion (hypertelorism)

### Face Area Analysis
Calculates proportional relationships between facial regions:

#### Area Measurements
- **Outer face area**: Convex hull of jaw contour + all model points
- **Inner face area**: Region from eyebrows (points 17-26) to mouth (points 48-67)
- **Eye areas**: Individual eye regions (points 36-41 and 42-47)

#### Proportional Analysis
- **Inner/Outer percentage**: (Inner area / Outer area) × 100
- **Eye-to-face proportions**: Individual eye area relative to total face area
- Provides detailed area measurements in pixels²

### Mouth-to-Eye Proportional Analysis
Assesses facial harmony through mouth and eye relationships:

#### Measurements
- **Mouth length**: Distance between points 49 and 54 (mouth corners)
- **Pupil distance**: Distance between pupil centers (points 70 and 71)
- **Proportion ratio**: Mouth length / Pupil distance

#### Classifications
- **Boca grande**: > 1.0 ratio (mouth wider than interpupillary distance)
- **Relación estándar**: 0.7-1.0 ratio (harmonious proportion)
- **Boca pequeña**: < 0.7 ratio (mouth narrower than typical)

### Eye Colorimetry Analysis
Advanced iris color classification using multiple methodologies:

#### Color Detection Methods
1. **HSV-based classification**: Uses hue, saturation, and brightness values
2. **RGB average classification**: Analyzes average color within iris region
3. **RGB dominant classification**: Uses K-means clustering for dominant colors

#### Color Categories
- **color_de_ojo_negro/cafe_oscuro**: Very dark brown/black eyes
- **cafe_claro/hazel**: Light brown/hazel eyes
- **verde**: Green eyes
- **azul_claro/gris**: Light blue/gray eyes
- **azul_oscuro**: Dark blue eyes
- **azul_intenso/morado**: Intense blue/violet eyes
- **amarillo**: Amber/yellow eyes
- **azul_verde**: Blue-green/turquoise eyes

#### Analysis Process
1. **Landmark detection**: Identifies eye regions using facial landmarks
2. **Iris extraction**: Isolates iris region excluding pupil and sclera
3. **Color analysis**: K-means clustering and average color calculation
4. **Multi-system classification**: RGB and HSV-based color categorization
5. **Bilateral comparison**: Separate analysis for left and right eyes

### Model Integration and Confidence Scoring
The system integrates a custom-trained deep learning model for enhanced accuracy:

#### Model Architecture
- **Base model**: Faster R-CNN with ResNet-50 backbone
- **Detection classes**: 13 specialized anthropometric points + background
- **Input processing**: 224×224 pixel images with normalization
- **Output**: Bounding boxes, confidence scores, and point classifications

#### Confidence Thresholds
- **Adjustable threshold**: 0.0-1.0 (default 0.5)
- **Point validation**: Only points above threshold are used
- **Fallback system**: Traditional calculations when model points unavailable
- **Quality indicators**: Model integration status reported in results

#### Enhanced Accuracy Features
- **Coordinate scaling**: Automatic scaling from model input to original image
- **Point replacement**: Model points replace calculated estimates when available
- **Hybrid approach**: Combines traditional landmarks with AI-detected points
- **Confidence reporting**: Detailed confidence scores for each detected point

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

## 🔄 Version History

### v2.1 (Current)
**Breaking Changes:**
- Eye angle and eyebrow slope calculations now return different values due to new reference system
- Response fields `face_line_slope` removed from eyebrow analysis
- Response fields `left_eye_slope` and `right_eye_slope` removed from eye analysis

**New Fields:**
- `vertical_reference_used`: Boolean indicating if M3 was used (vs fallback point 69)
- `vertical_reference_angle_deg`: Angle of vertical midline reference
- `perpendicular_reference_angle_deg`: Angle of perpendicular reference used for measurements

**Improvements:**
- More accurate angle measurements that account for head tilt and pose
- Consistent bilateral measurements for improved symmetry assessment
- Better anatomical accuracy using facial vertical axis as reference

### v2.0
**New Endpoints:**
- `/analyze-eyebrows` - Focused eyebrow analysis
- `/analyze-eyes` - Eye angle and proportion analysis
- `/analyze-face-areas` - Face area proportion analysis
- `/get-detailed-report` - Comprehensive reporting
- `/analyze-eye-colorimetry` - Iris color analysis
- `/analyze-iris-color` - Focused iris classification
- `/compare-eye-colors` - Color classification comparison

**Enhanced Responses:**
- Additional analysis fields in `/analyze-anthropometric`
- Detailed classifications and measurements
- Model integration status and confidence scores
- Enhanced error reporting and validation

**Backward Compatibility:**
- All v1.0 endpoints remain functional
- Response format extended (not breaking from v1.0)

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
