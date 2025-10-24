# Antropometric Analysis API v2.2

Advanced anthropometric facial analysis system with comprehensive feature detection, custom model integration, and detailed reporting capabilities.

> **⚠️ IMPORTANT NOTE ON POINT NUMBERING:**
> This documentation uses **dlib point numbers (1-68)** for all landmark references, NOT Python array indices (0-67).
>
> The system combines two detection models:
> - **Dlib 68-point model**: Points 1-68 (standard facial landmarks)
> - **Custom detection model**: Points M1-M13 (specialized anthropometric points)
>
> **Point 68 overlap:** The last dlib landmark (point 68 - inner lower lip) and the first extended point (point 68 - between eyebrows) share the same number but represent different locations. In practice:
> - Dlib point 68 (mouth inner) is accessed via Python index `[67]`
> - Extended point 68 (between eyebrows) is accessed via Python index `[68]`
>
> Extended points 68-71 are calculated or derived from model predictions and are added after the 68 dlib points in the array.

## Latest Updates (v2.2.1)

### Eyebrow Slope Analysis Corrections (2025-10-24)
- **Fixed Right Eyebrow Direction**: Right eyebrow points now reversed to ensure medial→lateral consistency (points [22, 21, 20, 19, 18])
- **Implemented Mirrored Angle Calculation**: Right eyebrow now uses mirrored calculation (p2→p1 vector) to ensure symmetrical eyebrows produce identical angles
- **Corrected Classification Thresholds**: Fixed broken logic for portions 1 & 2
  - Before (broken): `Recto` condition caught angles from -1° to 5°, `Descendente` condition never properly triggered
  - After (fixed): `Ascendente` > 5°, `Recto` -1° to 5°, `Descendente` < -1°
- **Added Slope Classifications to Summary**: Eyebrow slope angle classifications now included in API response summary (previously only in detailed report)
- **Enhanced Documentation**: Added detailed explanation of mirrored calculation technique and anatomical consistency

### Calibration Updates
- **Mouth Measurement Thresholds**: Updated mouth-to-pupil proportion thresholds
  - Boca grande: > 0.685 (previously > 1.0)
  - Relación estándar: 0.65 - 0.685 (previously 0.7 - 1.0)
  - Boca pequeña: < 0.65 (previously < 0.7)
- **Eyebrow Proportions**: Now measured relative to middle third of face (point 68 to 34) instead of whole head height
- **Cupid's Bow Measurements**: Changed from absolute distances to proportional ratios
  - Right cupid arch: distance(51→62) / distance(52→63) - lip thickness / bow depth
  - Left cupid arch: distance(53→64) / distance(52→63) - lip thickness / bow depth
- **Lip Thickness Analysis**: Added new measurement from point 52 to 58, proportional to bottom third of face
- **Mouth Analysis Reference**: Cupid's bow and lip thickness now proportional to bottom third (point 34 to 9) instead of whole face
- **Eye Size Classification**: Added classification for eye-to-face proportion
  - Ojo pequeño: < 0.74%
  - Ojo mediano: 0.74% - 0.85%
  - Ojo grande: > 0.85%
- **Inner Face Size Classification**: Added classification for inner-to-outer face proportion
  - Cara interna pequeña: < 38%
  - Cara interna promedio: 38% - 44.5%
  - Cara interna grande: > 44.5%
- **Mouth Length Classification**: Added classification for mouth width relative to face width
  - Boca ancha (Wide Mouth): > 33.5%
  - Boca promedio (Average Mouth): 32% - 33.5%
  - Boca angosta (Narrow Mouth): < 32%
- **Integral Mouth Diagnosis**: Combined analysis of mouth_to_eye and mouth_length classifications
  - Boca grande: When both mouth_to_eye = "boca grande" AND mouth_length = "boca ancha"
  - Boca pequeña: When both mouth_to_eye = "boca pequeña" AND mouth_length = "boca angosta"
  - Boca estandar: All other combinations
- **Eyebrow-Eyelid Position Classification**: Classification based on eyebrow-eyelid distance proportion
  - high_eyebrows: > 0.31
  - normal_eyebrows: 0.225 - 0.31
  - low_eyebrows: < 0.225
- **Lip Thickness Classification**: Classification based on total lip thickness proportion
  - thick_lips: > 30%
  - normal_lips: 18% - 30%
  - thin_lips: < 18%
- **Upper Lip Thickness Classification**: Classification based on lips ratio (upper/lower lip)
  - thick_upper_lip: > 67%
  - normal_upper_lip: 49% - 67%
  - thin_upper_lip: < 49%
- **Cupid's Arch Classification**: Classification based on cupid's arch proportion
  - cupid_arch: > 1 (cupid's bow present)
  - no_cupid_arch: ≤ 1 (no cupid's bow)

## Previous Updates (v2.1)

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

### Standard Landmarks (Points 1-68)
Uses dlib's 68-point facial landmark detector:
- **Face contour (1-17)**: Jaw line from left ear to right ear
- **Right eyebrow (18-22)**: Five points from outer to inner edge
- **Left eyebrow (23-27)**: Five points from inner to outer edge
- **Nose bridge (28-31)**: Four points from top to tip
- **Nose base (32-36)**: Five points defining nostrils and base
- **Right eye (37-42)**: Six points defining eye contour, outer to inner
- **Left eye (43-48)**: Six points defining eye contour, inner to outer
- **Mouth outer (49-60)**: Twelve points defining outer lip contour
- **Mouth inner (61-68)**: Eight points defining inner lip contour

### Extended Landmarks (Points 68-71)
Calculated or model-derived points:
- **Point 68**: Between eyebrows (uses model point 2 when available, otherwise calculated as midpoint of highest eyebrow points)
- **Point 69**: Top of head (uses calculated C1 point: X from M2, Y from M9 to avoid widow's peak interference; fallback to M3 or calculated estimate)
- **Point 70**: Left pupil center (calculated as midpoint of left eye landmarks 44 and 47)
- **Point 71**: Right pupil center (calculated as midpoint of right eye landmarks 38 and 41)

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

### 2. Eyebrow Morphometry Analysis

Analyzes eyebrow length relative to the middle third of the face for aesthetic proportion assessment.

**Measurement Points:**
- **Right eyebrow length**: Distance from point 18 (outer) to point 22 (inner)
- **Left eyebrow length**: Distance from point 27 (outer) to point 23 (inner)
- **Middle third reference**: Distance from point 68 (between eyebrows) to point 34 (nose base)
- **Right eye length**: Distance from point 37 (outer corner) to point 40 (inner corner)
- **Left eye length**: Distance from point 43 (inner corner) to point 46 (outer corner)

**Proportional Calculation**: Eyebrow length / Middle third length

**Classification Thresholds:**
- **Ceja Corta (Short Eyebrow)**: Ratio < 1.0
- **Ceja Normal (Normal Eyebrow)**: Ratio 1.0-1.4
- **Ceja Larga (Long Eyebrow)**: Ratio > 1.4

### 3. Eyebrow Slope Analysis (Three-Segment Method)

Analyzes eyebrow angle in three anatomical segments relative to the perpendicular of the vertical midline reference. Both eyebrows are analyzed in a **consistent medial→lateral anatomical direction** with **mirrored angle calculations** to ensure symmetrical eyebrows produce identical angle values.

**Reference System:**
- **Vertical Midline**: Vector from point 9 (chin) to M3 (top of head)
- **Perpendicular Reference**: 90-degree rotation of vertical midline, serves as "horizontal" baseline
- **Angle Measurement**: Degrees relative to perpendicular reference

**Anatomical Direction Consistency:**
Both eyebrows are analyzed **medial→lateral** (from nose to temple):
- **Right Eyebrow**: Dlib natural order (18→22) is lateral→medial, so points are **reversed** to [22, 21, 20, 19, 18] for medial→lateral consistency
- **Left Eyebrow**: Dlib natural order (23→27) is already medial→lateral, no reversal needed

**Mirrored Angle Calculation Technique:**
To ensure symmetrical eyebrows produce identical angles, the right eyebrow uses a **mirrored calculation**:
- **Left Eyebrow**: Standard angle calculation (p1→p2 vector direction)
- **Right Eyebrow**: Mirrored calculation (p2→p1 vector direction, reversed)

This compensates for the fact that even with medial→lateral anatomical ordering, the eyebrows point in opposite directions on screen (left goes →, right goes ←). The mirroring ensures that horizontal eyebrows on both sides report ~0° instead of 0° and 180°.

**Right Eyebrow Segments** (medial→lateral with mirrored angles):
- **Portion 1**: Medial segment angle (points 22 to 21, mirrored calculation)
- **Portion 2**: Middle arch segment (points 21 to 19, mirrored calculation)
- **Portion 3**: Lateral descending segment (points 19 to 18, mirrored calculation)

**Left Eyebrow Segments** (medial→lateral with standard angles):
- **Portion 1**: Medial segment angle (points 23 to 24, standard calculation)
- **Portion 2**: Middle arch segment (points 24 to 26, standard calculation)
- **Portion 3**: Lateral descending segment (points 26 to 27, standard calculation)

**Classification Thresholds (Portions 1 and 2):**
- **Ascendente (Ascending)**: Angle > 5°
- **Recto (Straight)**: Angle -1° to 5°
- **Descendente (Descending)**: Angle < -1°

**Classification Thresholds (Portion 3):**
- **Descendente (Descending)**: Angle > 75°
- **Normal**: Angle 10° to 75°
- **Ascendente (Ascending)**: Angle < 10°

### 4. Eye Angle Analysis (Palpebral Fissure Inclination)

Measures the inclination of both eyes relative to the perpendicular of the vertical midline reference.

**Reference System:**
- **Vertical Midline**: Vector from point 9 (chin) to M3 (top of head)
- **Perpendicular Reference**: 90-degree rotation of vertical midline
- **Measurement Direction**: Both eyes measured in anatomically equivalent directions for symmetry assessment

**Measurement Points:**
- **Right Eye**: From outer corner (point 37) to inner corner (point 40), reversed for rightward direction
- **Left Eye**: From inner corner (point 43) to outer corner (point 46), natural rightward direction

**Angle Calculation:**
1. Calculate angle of eye line relative to perpendicular reference
2. Normalize to range [-180°, +180°]
3. Left eye angle is negated to ensure symmetrical eyes produce consistent signs
4. Positive values indicate upward slant at outer corner
5. Negative values indicate downward slant at outer corner

**Classification Thresholds:**
- **Ángulo Normal (Normal Angle)**: -5° to +5°
- **Ángulo Hacia Arriba (Upward Angle)**: > +5°
- **Ángulo Hacia Abajo (Downward Angle)**: < -5°

### 5. Intercanthal Distance Analysis

Evaluates horizontal eye spacing proportions to classify eye separation.

**Measurement Points:**
- **Inner Eye Distance**: Distance between inner corners (point 40 to point 43)
- **Outer Eye Distance**: Distance between outer corners (point 37 to point 46)

**Proportional Calculation**: Inner eye distance / Outer eye distance

**Classification Thresholds:**
- **cercanos**: Proportion < 0.40
- **standard**: Proportion 0.40-0.435
- **lejanos**: Proportion > 0.435

### 6. Eyebrow-Eyelid Distance Analysis

Measures vertical distance between eyebrow and upper eyelid, proportional to middle third of face.

**Measurement Points:**
- **Right Side**: Distance from point 20 (right eyebrow) to point 38 (right upper eyelid)
- **Left Side**: Distance from point 25 (left eyebrow) to point 45 (left upper eyelid)

**Proportional Calculation**: Eyebrow-eyelid distance / Middle third length (point 68 to point 34)

**Classification Thresholds:**
- **high_eyebrows**: Proportion > 0.31
- **normal_eyebrows**: Proportion 0.225-0.31
- **low_eyebrows**: Proportion < 0.225

### 7. Mouth Morphometry Analysis

Analyzes mouth structure including cupid's bow arches, lip thickness, and lip proportions.

**Cupid's Bow Measurements (Proportional Ratios):**
- **Right Cupid's Arch**: distance(51→62) / distance(52→63)
  - Proportion of lip thickness to cupid's bow depth at right side
- **Left Cupid's Arch**: distance(53→64) / distance(52→63)
  - Proportion of lip thickness to cupid's bow depth at left side

**Cupid's Arch Classification Thresholds:**
- **cupid_arch**: Proportion > 1 (cupid's bow present)
- **no_cupid_arch**: Proportion ≤ 1 (no cupid's bow)

**Lip Thickness Measurements:**
- **Total Lip Thickness**: Vertical distance from point 52 (top center of upper lip) to point 58 (bottom center of lower lip)
- **Proportional Calculation**: Lip thickness distance / Bottom third length (point 34 to 9)
- **Upper Lip Thickness**: Vertical distance from point 52 to point 63
- **Lower Lip Thickness**: Vertical distance from point 67 to point 58
- **Lips Ratio**: Upper lip thickness / Lower lip thickness

**Lip Thickness Classification Thresholds:**
- **thick_lips**: Proportion > 30%
- **normal_lips**: Proportion 18%-30%
- **thin_lips**: Proportion < 18%

**Upper Lip Thickness Classification Thresholds (based on lips ratio):**
- **thick_upper_lip**: Ratio > 67%
- **normal_upper_lip**: Ratio 49%-67%
- **thin_upper_lip**: Ratio < 49%

### 8. Mouth-to-Eye Proportional Analysis

Assesses facial harmony through relationship between mouth width and interpupillary distance.

**Measurement Points:**
- **Mouth Length**: Distance between mouth corners (point 49 to point 55)
- **Pupil Distance**: Distance between pupil centers (point 70 to point 71)

**Proportional Calculation**: Mouth length / Pupil distance

**Classification Thresholds:**
- **Boca Grande (Large Mouth)**: Ratio > 0.685
- **Relación Estándar (Standard Relation)**: Ratio 0.65-0.685
- **Boca Pequeña (Small Mouth)**: Ratio < 0.65

### 9. Face Area Analysis

Calculates proportional relationships between different facial regions using polygon areas.

**Outer Face Area:**
- Defined by convex hull of jaw contour (points 1-17) plus all detected model points
- Represents total visible face boundary

**Inner Face Area:**
- Defined by region from eyebrows (points 18-27) to mouth (points 49-68)
- Represents central feature-rich zone

**Eye Areas:**
- **Right Eye Area**: Polygon defined by points 37-42
- **Left Eye Area**: Polygon defined by points 43-48

**Proportional Calculations:**
- **Inner/Outer Percentage**: (Inner area / Outer area) × 100
- **Right Eye to Face Proportion**: (Right eye area / Outer face area) × 100
- **Left Eye to Face Proportion**: (Left eye area / Outer face area) × 100

**Classification Thresholds (Inner Face Size):**
- **Cara Interna Pequeña (Small Inner Face)**: < 38%
- **Cara Interna Promedio (Average Inner Face)**: 38% - 44.5%
- **Cara Interna Grande (Large Inner Face)**: > 44.5%

**Classification Thresholds (Eye Size):**
- **Ojo Pequeño (Small Eye)**: < 0.74%
- **Ojo Mediano (Medium Eye)**: 0.74% - 0.85%
- **Ojo Grande (Large Eye)**: > 0.85%

### 10. Additional Proportional Measurements

**Head Width Proportion:**
- Calculation: Head width (point 2 to point 16) / Head height (point 69 to point 9)

**Chin to Face Width Proportion:**
- Calculation: Chin width (point 8 to point 10) / Head width (point 2 to point 16)

**Outer Eye Distance Proportion:**
- Calculation: Outer eye distance (point 37 to point 46) / Head height (point 69 to point 9)

**Mouth Length Proportion:**
- Calculation: Mouth length (point 49 to point 55) / Head width (point 2 to point 16)
- **Classification Thresholds:**
  - **Boca Ancha (Wide Mouth)**: Percentage > 33.5%
  - **Boca Promedio (Average Mouth)**: Percentage 32% - 33.5%
  - **Boca Angosta (Narrow Mouth)**: Percentage < 32%

**Integral Mouth Diagnosis:**
- Combines mouth_to_eye_proportion and mouth_length_proportion classifications for comprehensive mouth size assessment
- **Classification Logic:**
  - **Boca Grande**: Both mouth_to_eye = "boca grande en relación a las pupilas" AND mouth_length = "boca ancha"
  - **Boca Pequeña**: Both mouth_to_eye = "boca pequeña en relación a las pupilas" AND mouth_length = "boca angosta"
  - **Boca Estandar**: All other combinations (mixed or standard classifications)

## Eye Colorimetry Analysis

Advanced iris color classification using multiple methodologies with **INCLUSIVE CRITERIA**: if an eye color matches multiple categories, all matches are returned with equal percentages.

### Color Detection Methods
1. **HSV-based Classification**: Uses hue, saturation, and value (brightness) analysis
2. **RGB Average Classification**: Analyzes average RGB values within iris region
3. **RGB Dominant Classification**: Uses K-means clustering to identify dominant colors

### Inclusive Classification System
When RGB values match multiple eye color categories, the system returns ALL matching classifications with equal percentages:
- **Single Match**: `{"classifications": ["verde"], "percentages": [100.0], "primary_classification": "verde"}`
- **Two Matches**: `{"classifications": ["azul_claro/gris", "gris"], "percentages": [50.0, 50.0], "primary_classification": "azul_claro/gris / gris"}`

### Color Categories and RGB Ranges

- **color_de_ojo_negro/cafe_oscuro**: Very dark brown/black eyes
  - R: 0-120, G: 0-100, B: 0-60
  - No additional conditions
- **cafe_claro/hazel**: Light brown/hazel eyes
  - R: 120-180, G: 80-140, B: 30-100
  - Condition: R > G
- **amarillo**: Amber/yellow eyes
  - R: 120-255, G: 120-255, B: 0-160
  - Conditions: R and G within 30% of each other, B less than 60% of R or G
- **verde**: Green eyes
  - R: 0-150, G: 70-255, B: 0-110
  - Condition: G >= R
- **azul_claro/gris**: Light blue/gray eyes
  - R: 0-220, G: 80-255, B: 70-255
  - Conditions: G >= R and B >= R, G not more than 40% greater than B
- **gris**: Gray eyes
  - R: 0-220, G: 60-255, B: 70-255
  - Condition: G not more than 40% greater than B
- **azul_oscuro**: Dark blue eyes
  - R: 0-90, G: 0-120, B: 70-135
  - Conditions: G >= R and B >= G
- **azul_intenso/morado**: Intense blue/violet eyes
  - R: 0-140, G: 0-130, B: 135-255
  - Condition: B > G
- **azul_verde**: Blue-green/turquoise eyes
  - R: 60-85, G: 70-170, B: 69-169
  - Conditions: G > B, R < 50% of G, B > R

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

### v2.2 (Current)

**Calibration Changes:**
- Updated mouth-to-pupil proportion classification thresholds for improved accuracy
- Changed eyebrow proportion reference from whole head height to middle third of face
- Modified cupid's bow measurements from absolute distances to proportional ratios
- Added lip thickness measurement (point 52 to 58) proportional to bottom third
- Updated mouth analysis measurements to use bottom third as reference

**New Fields:**
- `lip_thickness_distance`: Absolute distance for total lip thickness (52 to 58)
- `lip_thickness_proportion`: Lip thickness relative to bottom third of face
- `middle_third_length`: Reference length used for eyebrow proportions
- `bottom_third_length`: Reference length used for mouth measurements
- `left_eye_size_classification`: Classification label for left eye size (pequeño/mediano/grande)
- `right_eye_size_classification`: Classification label for right eye size (pequeño/mediano/grande)
- `inner_face_size_classification`: Classification label for inner face size (pequeña/promedio/grande)
- `mouth_length_proportion`: Mouth width proportion relative to head width
- `mouth_length_percentage`: Formatted percentage of mouth width (e.g., "33.50%")
- `mouth_length_classification`: Classification label for mouth width (ancha/promedio/angosta)
- `integral_diagnosis`: Combined mouth size diagnosis from both mouth measurements (grande/pequeña/estandar)
- `left_eyebrow_eyelid_classification`: Classification for left eyebrow position (high_eyebrows/normal_eyebrows/low_eyebrows)
- `right_eyebrow_eyelid_classification`: Classification for right eyebrow position (high_eyebrows/normal_eyebrows/low_eyebrows)
- `lip_thickness_classification`: Classification for lip thickness (thick_lips/normal_lips/thin_lips)
- `upper_lip_thickness_classification`: Classification for upper lip thickness (thick_upper_lip/normal_upper_lip/thin_upper_lip)
- `left_cupid_arch_classification`: Classification for left cupid's arch presence (cupid_arch/no_cupid_arch)
- `right_cupid_arch_classification`: Classification for right cupid's arch presence (cupid_arch/no_cupid_arch)

**Modified Calculations:**
- Eyebrow proportions now relative to middle third (point 68 to 34) instead of head height
- Cupid's bow arches now calculated as ratios of lip thickness to bow depth
  - Right: distance(51→62) / distance(52→63)
  - Left: distance(53→64) / distance(52→63)

### v2.1

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
