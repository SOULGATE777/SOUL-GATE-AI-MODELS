# Antropometric Analysis API v2.1

Advanced anthropometric facial analysis system with comprehensive feature detection, custom model integration, and detailed reporting capabilities.

## Latest Updates (v2.1)

### Critical Changes in Angle Measurement System
- **Vertical Midline Reference**: All angle measurements now calculated relative to vertical midline (point 9 to M3) instead of horizontal baseline
- **Enhanced Anatomical Accuracy**: Standardized calculations ensure symmetrical features produce consistent measurements across left and right sides
- **Model Point 3 (M3) Integration**: Uses M3 when available for accurate vertical reference alignment
- **Calculated Point C1**: Combines X coordinate from M2 (between eyebrows) with Y coordinate from M9 to avoid widow's peak interference
- **Consistent Bilateral Measurements**: Both left and right features measured in anatomically equivalent directions

### Bug Fixes
- Face alignment preprocessing improvements
- Updated object detection model integration
- Enhanced eyebrow type classifier integration with morfológico pipeline

## Directory Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application
│   ├── models/
│   │   ├── anthropometric_pipeline.py   # Core analysis engine
│   │   ├── eye_colorimetry_analyzer.py  # Iris color analysis
│   │   └── __init__.py
│   └── utils/
│       ├── image_processing.py          # Image processing utilities
│       ├── visualization.py             # Visualization functions
│       └── __init__.py
├── models/
│   ├── facial_points_detection_model.pth    # Custom trained model
│   └── shape_predictor_68_face_landmarks.dat # Dlib predictor
├── results/                             # Generated visualizations and reports
├── dataset_frontal/                     # Test images
├── docker-compose.yml                   # Docker composition
├── Dockerfile                          # Container definition
├── requirements.txt                     # Dependencies
└── README.md                           # This file
```

## Installation & Setup

### Prerequisites
- Docker and Docker Compose
- At least 4GB RAM available for the container
- Required model files:
  - `facial_points_detection_model.pth` (custom trained model)
  - `shape_predictor_68_face_landmarks.dat` (dlib model)

### Quick Start

1. Clone and setup:
   ```bash
   git clone <repository>
   cd antropometrico
   ```

2. Ensure model files are in place:
   ```bash
   ls models/
   # Should show:
   # facial_points_detection_model.pth
   # shape_predictor_68_face_landmarks.dat
   ```

3. Build and run:
   ```bash
   docker-compose up --build
   ```

4. Verify installation:
   ```bash
   curl http://localhost:8001/health
   ```

## API Endpoints

### Complete Analysis
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

## Facial Landmark System

### Standard Landmarks (Points 0-67)
Uses dlib's 68-point facial landmark detector:
- **Face contour (0-16)**: Jaw line from left ear to right ear
- **Right eyebrow (17-21)**: Five points from outer to inner edge
- **Left eyebrow (22-26)**: Five points from inner to outer edge
- **Nose bridge (27-30)**: Four points from top to tip
- **Nose base (31-35)**: Five points defining nostrils and base
- **Right eye (36-41)**: Six points defining eye contour, outer to inner
- **Left eye (42-47)**: Six points defining eye contour, inner to outer
- **Mouth outer (48-59)**: Twelve points defining outer lip contour
- **Mouth inner (60-67)**: Eight points defining inner lip contour

### Extended Landmarks (Points 68-71)
Calculated or model-derived points:
- **Point 68**: Between eyebrows (uses model point 2 when available, otherwise calculated as midpoint of highest eyebrow points)
- **Point 69**: Top of head (uses calculated C1 point: X from M2, Y from M9 to avoid widow's peak interference; fallback to M3 or calculated estimate)
- **Point 70**: Left pupil center (calculated as midpoint of left eye landmarks 37 and 40)
- **Point 71**: Right pupil center (calculated as midpoint of right eye landmarks 43 and 46)

### Custom Model Points (M1-M13)
Deep learning model detects up to 13 specialized anthropometric points:
- **M1**: Superior facial implantation reference
- **M2**: Between eyebrows (replaces calculated point 68 when available)
- **M3**: Superior head point (used as vertical reference for all angle measurements)
- **M4-M13**: Additional facial landmarks for enhanced accuracy
- **C1 (Calculated)**: Hybrid point combining X from M2 and Y from M9 to avoid hair interference

## Comprehensive Measurement System

### 1. Facial Thirds Analysis

Divides face into three vertical sections to assess proportional harmony according to classical anthropometric ratios.

**Reference Points:**
- Top: Point 69 (top of head, preferably calculated C1)
- Upper division: Point 68 (between eyebrows, preferably M2)
- Middle division: Point 34 (nose base/subnasale)
- Lower reference: Point 9 (menton/chin)

**Measurements:**
- **Primer Tercio (Upper Third)**: Distance from point 69 to 68, divided by total head height (69 to 9)
- **Segundo Tercio (Middle Third)**: Distance from point 68 to 34, divided by total head height
- **Tercer Tercio (Lower Third)**: Distance from point 34 to 9, divided by total head height

**Classification Thresholds:**
- **Tercio Largo (Long)**: Proportion > 0.38
- **Tercio Standard (Normal)**: Proportion 0.27-0.38
- **Tercio Corto (Short)**: Proportion < 0.27

**Clinical Significance**: Ideal facial thirds should each be approximately 33% of total face height. Deviations indicate disproportionate facial development.

### 2. Eyebrow Morphometry Analysis

Analyzes eyebrow length relative to corresponding eye length for aesthetic proportion assessment.

**Measurement Points:**
- **Right eyebrow length**: Distance from point 17 (outer) to point 21 (inner)
- **Left eyebrow length**: Distance from point 26 (outer) to point 22 (inner)
- **Right eye length**: Distance from point 36 (outer corner) to point 39 (inner corner)
- **Left eye length**: Distance from point 42 (inner corner) to point 45 (outer corner)

**Proportional Calculation**: Eyebrow length / Corresponding eye length

**Classification Thresholds:**
- **Ceja Corta (Short Eyebrow)**: Ratio < 1.0 (eyebrow shorter than eye)
- **Ceja Normal (Normal Eyebrow)**: Ratio 1.0-1.4
- **Ceja Larga (Long Eyebrow)**: Ratio > 1.4 (eyebrow significantly longer than eye)

**Clinical Significance**: Normal eyebrows typically extend slightly beyond the eye length. Ratios outside normal range may indicate sparse eyebrows or structural variations.

### 3. Eyebrow Slope Analysis (Three-Segment Method)

Analyzes eyebrow angle in three anatomical segments relative to the perpendicular of the vertical midline reference.

**Reference System:**
- **Vertical Midline**: Vector from point 9 (chin) to M3 (top of head)
- **Perpendicular Reference**: 90-degree rotation of vertical midline, serves as "horizontal" baseline
- **Angle Measurement**: Degrees relative to perpendicular reference

**Right Eyebrow Segments** (measured from inner to outer for anatomical consistency):
- **Portion 1**: Inner segment angle (points 21 to 20, reversed direction)
- **Portion 2**: Middle arch segment (points 20 to 18, reversed direction)
- **Portion 3**: Outer descending segment (points 18 to 17, reversed direction)

**Left Eyebrow Segments** (measured from inner to outer, natural direction):
- **Portion 1**: Inner segment angle (points 22 to 23)
- **Portion 2**: Middle arch segment (points 23 to 25)
- **Portion 3**: Outer descending segment (points 25 to 26)

**Classification Thresholds (Portions 1 and 2):**
- **Ascendente (Ascending)**: Angle 5° to 75°
- **Recto (Straight)**: Angle -1° to 5°
- **Descendente (Descending)**: Angle ≤ 0°

**Classification Thresholds (Portion 3):**
- **Descendente (Descending)**: Angle > 75°
- **Normal**: Angle 10° to 75°
- **Ascendente (Ascending)**: Angle < 10°

**Clinical Significance**: Eyebrow shape affects facial expression perception. Ascending outer portions create a lifted appearance, while descending portions can appear sad or aged.

### 4. Eye Angle Analysis (Palpebral Fissure Inclination)

Measures the inclination of both eyes relative to the perpendicular of the vertical midline reference.

**Reference System:**
- **Vertical Midline**: Vector from point 9 (chin) to M3 (top of head)
- **Perpendicular Reference**: 90-degree rotation of vertical midline
- **Measurement Direction**: Both eyes measured in anatomically equivalent directions for symmetry assessment

**Measurement Points:**
- **Right Eye**: From outer corner (point 36) to inner corner (point 39), reversed for rightward direction
- **Left Eye**: From inner corner (point 42) to outer corner (point 45), natural rightward direction

**Angle Calculation:**
1. Calculate angle of eye line relative to perpendicular reference
2. Normalize to range [-180°, +180°]
3. Positive values indicate upward slant at outer corner
4. Negative values indicate downward slant at outer corner

**Classification Thresholds:**
- **Ángulo Normal (Normal Angle)**: -5° to +5° (nearly horizontal alignment with perpendicular reference)
- **Ángulo Hacia Arriba (Upward Angle)**: > +5° (positive canthal tilt, outer corner higher than inner)
- **Ángulo Hacia Abajo (Downward Angle)**: < -5° (negative canthal tilt, outer corner lower than inner)

**Clinical Significance**: Positive canthal tilt is associated with youthful appearance and attractiveness. Negative canthal tilt can appear aged or sad.

### 5. Intercanthal Distance Analysis

Evaluates horizontal eye spacing proportions to classify eye separation.

**Measurement Points:**
- **Inner Eye Distance**: Distance between inner corners (point 39 to point 42)
- **Outer Eye Distance**: Distance between outer corners (point 36 to point 45)

**Proportional Calculation**: Inner eye distance / Outer eye distance

**Classification Thresholds:**
- **Cercanos (Hypotelorism)**: Proportion < 0.3 (eyes close together)
- **Standard (Normal)**: Proportion 0.3-0.37
- **Lejanos (Hypertelorism)**: Proportion > 0.37 (eyes widely spaced)

**Clinical Significance**: Abnormal intercanthal distances may indicate genetic syndromes or craniofacial conditions.

### 6. Eyebrow-Eyelid Distance Analysis

Measures vertical distance between eyebrow and upper eyelid, proportional to head height.

**Measurement Points:**
- **Left Side**: Distance from point 19 (left eyebrow) to point 37 (left upper eyelid)
- **Right Side**: Distance from point 24 (right eyebrow) to point 44 (right upper eyelid)

**Proportional Calculation**: Eyebrow-eyelid distance / Total head height (point 69 to point 9)

**Clinical Significance**: This measurement assesses eyebrow position and ptosis. Low proportions may indicate brow ptosis or excess upper eyelid skin.

### 7. Mouth Morphometry Analysis

Analyzes mouth structure including cupid's bow arches and lip proportions.

**Cupid's Bow Measurements:**
- **Left Cupid's Arch**: Distance from point 50 to point 61
- **Right Cupid's Arch**: Distance from point 52 to point 63
- **Proportional Calculation**: Arch distance / Head height

**Lip Measurements:**
- **Upper Lip Thickness**: Vertical distance from point 51 to point 62
- **Lower Lip Thickness**: Vertical distance from point 66 to point 57
- **Lips Ratio**: Upper lip thickness / Lower lip thickness

**Clinical Significance**: Cupid's bow definition contributes to lip aesthetics. Lips ratio normally favors slightly fuller lower lip (ratio < 1.0).

### 8. Mouth-to-Eye Proportional Analysis

Assesses facial harmony through relationship between mouth width and interpupillary distance.

**Measurement Points:**
- **Mouth Length**: Distance between mouth corners (point 49 to point 54)
- **Pupil Distance**: Distance between pupil centers (point 70 to point 71)

**Proportional Calculation**: Mouth length / Pupil distance

**Classification Thresholds:**
- **Boca Grande (Large Mouth)**: Ratio > 1.0 (mouth wider than interpupillary distance)
- **Relación Estándar (Standard Relation)**: Ratio 0.7-1.0
- **Boca Pequeña (Small Mouth)**: Ratio < 0.7 (mouth narrower than typical)

**Clinical Significance**: This proportion contributes to overall facial balance. Extreme deviations affect facial aesthetics.

### 9. Face Area Analysis

Calculates proportional relationships between different facial regions using polygon areas.

**Outer Face Area:**
- Defined by convex hull of jaw contour (points 0-16) plus all detected model points
- Represents total visible face boundary

**Inner Face Area:**
- Defined by region from eyebrows (points 17-26) to mouth (points 48-67)
- Represents central feature-rich zone

**Eye Areas:**
- **Right Eye Area**: Polygon defined by points 36-41
- **Left Eye Area**: Polygon defined by points 42-47

**Proportional Calculations:**
- **Inner/Outer Percentage**: (Inner area / Outer area) × 100
- **Right Eye to Face Proportion**: (Right eye area / Outer face area) × 100
- **Left Eye to Face Proportion**: (Left eye area / Outer face area) × 100

**Clinical Significance**: These proportions assess facial compactness and feature distribution. Eye-to-face ratios correlate with neoteny and attractiveness.

### 10. Additional Proportional Measurements

**Head Width Proportion:**
- Calculation: Head width (point 2 to point 16) / Head height (point 69 to point 9)
- Clinical Significance: Assesses facial shape (narrow, average, wide)

**Chin to Face Width Proportion:**
- Calculation: Chin width (point 7 to point 9) / Head width (point 2 to point 16)
- Clinical Significance: Evaluates jaw taper and chin prominence

**Outer Eye Distance Proportion:**
- Calculation: Outer eye distance (point 36 to point 45) / Head height (point 69 to point 9)
- Clinical Significance: Assesses eye position relative to face height

**Mouth Length Proportion:**
- Calculation: Mouth length (point 49 to point 54) / Head width (point 2 to point 16)
- Clinical Significance: Evaluates mouth width relative to facial width

## Eye Colorimetry Analysis

Advanced iris color classification using multiple methodologies.

### Color Detection Methods
1. **HSV-based Classification**: Uses hue, saturation, and value (brightness) analysis
2. **RGB Average Classification**: Analyzes average RGB values within iris region
3. **RGB Dominant Classification**: Uses K-means clustering to identify dominant colors

### Color Categories
- **color_de_ojo_negro/cafe_oscuro**: Very dark brown/black eyes
- **cafe_claro/hazel**: Light brown/hazel eyes
- **verde**: Green eyes
- **azul_claro/gris**: Light blue/gray eyes
- **azul_oscuro**: Dark blue eyes
- **azul_intenso/morado**: Intense blue/violet eyes
- **amarillo**: Amber/yellow eyes
- **azul_verde**: Blue-green/turquoise eyes

### Analysis Process
1. **Landmark Detection**: Identifies eye regions using facial landmarks
2. **Iris Extraction**: Isolates iris region excluding pupil and sclera
3. **Color Analysis**: K-means clustering and average color calculation
4. **Multi-system Classification**: RGB and HSV-based color categorization
5. **Bilateral Comparison**: Separate analysis for left and right eyes

## Model Integration and Confidence Scoring

### Model Architecture
- **Base Model**: Faster R-CNN with ResNet-50 backbone
- **Detection Classes**: 13 specialized anthropometric points + background class
- **Input Processing**: 224×224 pixel images with normalization
- **Output**: Bounding boxes, confidence scores, and point classifications

### Confidence Thresholds
- **Adjustable Threshold**: 0.0-1.0 (default 0.5)
- **Point Validation**: Only points above threshold are used in analysis
- **Fallback System**: Traditional calculations when model points unavailable
- **Quality Indicators**: Model integration status reported in results

### Enhanced Accuracy Features
- **Coordinate Scaling**: Automatic scaling from model input (224×224) to original image dimensions
- **Point Replacement**: Model points replace calculated estimates when available and confident
- **Hybrid Approach**: Combines traditional landmarks with AI-detected points
- **Confidence Reporting**: Detailed confidence scores for each detected point in results

## Visualization Options

### Standard Visualization
- All 68 facial landmarks with point numbers
- Extended points (68-71) highlighted
- Model predictions with color coding by confidence
- Facial thirds reference lines
- Key measurement overlays

### Detailed Report Image
- Side-by-side image and comprehensive measurements
- All analysis results in organized format
- Model integration status indicators
- Classification results with thresholds

### Specialized Visualizations
- Eyebrow analysis with length comparison markers
- Eye angle visualization with degree measurements and reference lines
- Face area boundary visualization with color-coded regions
- Model point detection visualization with confidence scores

## Usage Examples

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

## Performance & Requirements

### System Requirements
- **Memory**: Minimum 2GB, Recommended 4GB
- **CPU**: Multi-core recommended for model inference
- **Storage**: 1GB for models and temporary files
- **Network**: Port 8001 available

### Processing Performance
- **Analysis Time**: 2-5 seconds per image (depending on resolution and hardware)
- **Model Inference**: Approximately 1 second for point detection
- **Visualization Generation**: Approximately 1 second additional
- **Supported Formats**: JPG, PNG, BMP, TIFF

### Image Requirements
- **Minimum Resolution**: 100x100 pixels
- **Maximum Resolution**: 4000x4000 pixels (automatically resized if larger)
- **Face Requirements**: Clear frontal face view, minimal rotation
- **Optimal Conditions**: Good lighting, minimal occlusion, neutral expression recommended

## Error Handling

### Common Issues
- **No Face Detected**: Ensure clear frontal face view with adequate lighting
- **Model Not Loaded**: Check model file paths and permissions in logs
- **Low Confidence**: Adjust confidence_threshold parameter or improve image quality
- **Memory Issues**: Reduce image size or increase container memory allocation

### Troubleshooting
1. **Check Health Endpoint**: `curl http://localhost:8001/health`
2. **Verify Model Files**: Ensure both model files present and readable in models/ directory
3. **Check Logs**: `docker-compose logs antropometrico-api`
4. **Image Quality**: Ensure clear, well-lit frontal face images without heavy occlusion

## Version History

### v2.1 (Current)

**Breaking Changes:**
- Eye angle and eyebrow slope calculations now return different values due to new vertical midline reference system
- Removed response fields: `face_line_slope` from eyebrow analysis
- Removed response fields: `left_eye_slope` and `right_eye_slope` from eye analysis (replaced with angle measurements)

**New Fields:**
- `vertical_reference_used`: Boolean indicating if M3 was used as vertical reference (vs fallback point 69)
- `vertical_reference_angle_deg`: Angle of vertical midline reference vector (point 9 to M3)
- `perpendicular_reference_angle_deg`: Angle of perpendicular reference used for measurements
- `calculated_c1_used`: Boolean indicating if hybrid C1 point was calculated and used
- `c1_calculation`: Description of C1 calculation method when used

**Improvements:**
- More accurate angle measurements accounting for head tilt and pose variations
- Consistent bilateral measurements for improved symmetry assessment
- Better anatomical accuracy using facial vertical axis as reference
- Widow's peak interference avoided through C1 calculation method

### v2.0

**New Endpoints:**
- `/analyze-eyebrows` - Focused eyebrow analysis with classifications
- `/analyze-eyes` - Eye angle and proportion analysis
- `/analyze-face-areas` - Face area proportion analysis
- `/get-detailed-report` - Comprehensive reporting in text or JSON format
- `/analyze-eye-colorimetry` - Complete iris color analysis
- `/analyze-iris-color` - Focused iris classification
- `/compare-eye-colors` - Color classification comparison across methods

**Enhanced Responses:**
- Additional analysis fields in `/analyze-anthropometric` response
- Detailed classifications and measurements for all features
- Model integration status and confidence scores
- Enhanced error reporting and validation

**Backward Compatibility:**
- All v1.0 endpoints remain functional
- Response format extended (not breaking changes from v1.0)

## Development & Testing

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

## License & Credits

This enhanced version includes advanced anthropometric analysis capabilities developed for comprehensive facial feature assessment. The system integrates classical facial landmark detection with custom-trained deep learning models for enhanced accuracy and detailed analysis.
