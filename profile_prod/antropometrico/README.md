# Profile Anthropometric Analysis Module

## Overview

The Profile Anthropometric Analysis module is an advanced facial analysis system designed specifically for profile (lateral) facial images. It performs comprehensive anthropometric measurements using an enhanced ensemble of deep learning models: a profile-aware point detection CNN with ResNet-50 backbone and an optional Graph Neural Network (GNN) for validation and false positive filtering.

This system provides precise measurements of facial features, proportional relationships, angular characteristics, and constitutional classifications based on established anthropometric principles. The module is designed for professional use in facial analysis applications requiring detailed morphological assessments.

## Features

### Core Capabilities
- **Enhanced Point Detection**: ResNet-50 based CNN with 4-stage progressive decoder for precise landmark detection
- **Profile Classification**: Integrated profile classifier (left/right) with class balancing for improved accuracy
- **GNN Validation**: Optional Graph Neural Network using anatomical relationships to validate and refine detections
- **False Positive Filtering**: Advanced ensemble approach significantly reduces out-of-bounds and spurious detections
- **Profile-Aware Processing**: Automatic filtering of wrong-side landmarks based on profile classification
- **Comprehensive Measurements**: 20+ different anthropometric measurements and classifications
- **Angular Analysis**: Advanced geometric calculations for facial angles and proportions
- **Real-time Visualization**: Generates annotated images with detected points and measurement summaries

## API Endpoints

### Base URL
```
http://localhost:8004
```

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "profile-anthropometric",
  "model_loaded": true,
  "device": "cuda",
  "gnn_enabled": true,
  "point_classes": 80
}
```

### 2. Complete Profile Analysis
```http
POST /analyze-profile-anthropometric
```
**Parameters:**
- `file` (required): Profile image file (JPG, PNG)
- `confidence_threshold` (optional): Float between 0.05-0.9 (default: 0.15)
- `include_visualization` (optional): Boolean (default: true)

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "profile_analysis": {
    "profile_side": "left|right|unknown",
    "total_detected_points": 25,
    "filtered_points": 22,
    "anthropometric_points": [
      {
        "class": "1",
        "coordinates": [x, y],
        "confidence": 0.85
      }
    ]
  },
  "anthropometric_measurements": {
    "reference_distance": 156.2,
    "nose_classification": "nariz normal",
    "mandibula_classification": "Mandibula Sanguinea",
    "forehead_classification": "frente neutra",
    "chin_classification": "menton biloso/linfatico",
    "ear_length_classification": "oreja normal",
    "nasal_triangulation_classification": "sin triangulacion de fosa"
  },
  "analysis_summary": {
    "confidence_threshold": 0.15,
    "has_measurements": true,
    "profile_determination": "left"
  },
  "visualization_path": "/app/results/profile_anthropometric_uuid.png",
  "visualization_url": "/visualization/profile_anthropometric_uuid.png"
}
```

### 3. Point Detection Only
```http
POST /detect-profile-points
```
**Parameters:**
- `file` (required): Profile image file
- `confidence_threshold` (optional): Float between 0.05-0.9

**Response:**
```json
{
  "total_detected_points": 28,
  "filtered_points": 24,
  "profile_side": "left",
  "detected_points": [...],
  "filtered_anthropometric_points": [...],
  "detection_summary": {
    "confidence_threshold": 0.15,
    "spurious_points_removed": 4,
    "profile_determination": "left"
  }
}
```

### 4. Model Information
```http
GET /model-info
```
**Response:**
```json
{
  "model_type": "Profile Anthropometric Point Detection",
  "device": "cuda",
  "point_classes": ["1", "2", "3", ..., "30"],
  "num_classes": 30,
  "heatmap_size": 112,
  "input_size": 224
}
```

## Anthropometric Points

The system detects 30 anatomical landmarks on the profile facial structure. Each point serves specific measurement purposes:

### Facial Structure Points
- **Point 1**: Superior ear attachment (used for ear implantation vector)
- **Point 2**: Posterior ear boundary (ear width measurement)
- **Point 3**: Inferior ear attachment and mandibular reference (used for ear implantation, ear lobe, and mandible measurements)
- **Point 4**: Superior ear tip (used for superior implantation angle and ear length)
- **Point 5**: Inferior ear lobe (used for inferior implantation angle, ear lobe, and ear length)
- **Point 6**: Anterior ear boundary (ear width measurement)
- **Point 7**: Tragus point (used for tragus-antitragus distance)
- **Point 8**: Antitragus point (used for tragus-antitragus distance)
- **Point 9**: Mandibular angle (used for mandibular width and angular analysis)
- **Point 10**: Menton (chin point, used as reference endpoint and lower third boundary)
- **Point 11**: Mental protuberance (used for chin angle calculation)
- **Point 16**: Nasal bridge superior
- **Point 17**: Nasal tip (pronasale, used for nose protrusion and tip angle)
- **Point 18**: Subnasale (nasal base, critical reference point for multiple measurements)
- **Point 19**: Nasal dorsum point (used for nasal slope calculation)
- **Point 22**: Glabella (nasal root, used for middle third boundary and forehead angle)
- **Point 24**: Trichion (hairline, used as primary reference start point)
- **Point 26**: Nostril base (used for nasal orifice triangulation)
- **Point 30**: Nasal septum reference (used for nasal orifice triangulation)
- **Point 33**: Alternative forehead reference (used for forehead angle calculation)
- **Point 34**: Vertex (crown, used for superior third measurement)
- **Point 37**: Orbital base (inner eye corner, used for eye protrusion orbital plane)
- **Point 38**: Cornea reference (eye protrusion measurement)
- **Point 39**: Orbital superior boundary (used for eye protrusion orbital plane)

### Point Classification System
Points may include suffixes to indicate profile side:
- **_i suffix**: Points detected on the left profile
- **_d suffix**: Points detected on the right profile
- Points without suffix are profile-agnostic landmarks

The system automatically filters spurious predictions by counting left and right suffixes to determine the dominant profile side, removing false positives from the minority side.

## Measurements and Classifications

### 1. Basic Measurements

#### 1.1 Reference Distance (Points 24 to 10)
The primary scaling measurement calculated as the Euclidean distance from the trichion (point 24, hairline) to the menton (point 10, chin point). This distance serves as the normalization factor for all proportional measurements in the analysis. All normalized proportions are calculated as: `measurement_distance / reference_distance`.

**Mathematical Formula:**
```
reference_distance = sqrt((point_10[x] - point_24[x])² + (point_10[y] - point_24[y])²)
```

#### 1.2 Head Direction Determination
Profile direction is determined via vector analysis using the reference vector from point 24 to point 18. The system calculates:
```
vector_24_18 = [point_18[x] - point_24[x], point_18[y] - point_24[y]]
head_direction = "right" if vector_24_18[x] > 0 else "left"
```
This infallible method determines whether the face is oriented to the left or right side of the image. The head direction is critical for proper angular measurement interpretation.

### 2. Nose Analysis

#### 2.1 Nasal Protrusion (Points 18 to 17)
Measures the horizontal projection of the nasal tip from the subnasale reference point.

**Measurement Process:**
1. Calculate Euclidean distance from point 18 (subnasale) to point 17 (pronasale/nasal tip)
2. Normalize by dividing by the reference distance (24-10)
3. Classify based on normalized proportion

**Formula:**
```
nasal_protrusion_distance = sqrt((point_17[x] - point_18[x])² + (point_17[y] - point_18[y])²)
nasal_proportion = nasal_protrusion_distance / reference_distance
```

**Classifications:**
- **nariz protruyente** (protruding nose): proportion > 0.2
- **nariz normal** (normal nose): 0.17 ≤ proportion ≤ 0.2
- **nariz corta** (short nose): proportion < 0.17

#### 2.2 Nasal Tip Angle (Points 18 to 17)
Measures the angular orientation of the nasal tip relative to a perpendicular reference line. This indicates whether the nose points upward, forward, or downward.

**Mathematical Process:**
1. Establish reference vector from point 24 to point 18: `vector_24_18`
2. Calculate perpendicular slope as negative reciprocal: `perp_slope = -1/ref_slope`
3. Create perpendicular vector: `perp_vector = [1, perp_slope]`
4. Calculate measurement vector from point 18 to point 17: `vector_18_17`
5. Normalize both vectors for dot product calculation
6. Calculate angle: `angle = arccos(clip(dot(v1_normalized, v2_normalized), -1, 1))`
7. Apply directional sign correction:
   - If `point_17[x] < point_18[x]`: Head turning right, angle becomes negative
   - If `point_17[x] > point_18[x]`: Head turning left, angle remains positive
8. Clamp angle to range of -90 to +90 degrees

**Classifications:**
- **punta de nariz hacia arriba** (upturned nose): angle ≥ 27 degrees
- **punta de nariz promedio** (average nose): 12 ≤ angle < 27 degrees
- **punta hacia abajo** (downturned nose): angle < 12 degrees

#### 2.3 Nasal Dorsum Slope (Points 19 to 17)
Calculates the slope of the nasal dorsum line for angular assessment.

**Formula:**
```
slope_19_17 = (point_19[y] - point_17[y]) / (point_19[x] - point_17[x])
angle_19_17 = atan2(slope_19_17, 1) * (180/π)
```

### 3. Facial Thirds Analysis

The face is divided into three vertical regions to assess proportional balance. Each third is measured and normalized to the reference distance.

#### 3.1 Superior Third (Points 34 to 22)
Measures the upper facial region from the vertex (crown) to the glabella (nasal root).

**Formula:**
```
superior_third_distance = sqrt((point_22[x] - point_34[x])² + (point_22[y] - point_34[y])²)
superior_third_proportion = superior_third_distance / reference_distance
```

**Label:** tercio superior (upper third)

#### 3.2 Middle Third (Points 22 to 18)
Measures the mid-facial region from the glabella to the subnasale.

**Formula:**
```
middle_third_distance = sqrt((point_18[x] - point_22[x])² + (point_18[y] - point_22[y])²)
middle_third_proportion = middle_third_distance / reference_distance
```

**Label:** tercio medio (middle third)

#### 3.3 Inferior Third (Points 18 to 10)
Measures the lower facial region from the subnasale to the menton. This measurement is identical to the component of the reference distance and serves as the baseline for mandibular analysis.

**Formula:**
```
inferior_third_distance = sqrt((point_10[x] - point_18[x])² + (point_10[y] - point_18[y])²)
inferior_third_proportion = inferior_third_distance / reference_distance
```

**Label:** tercio inferior (lower third)

### 4. Mandible Analysis

#### 4.1 Mandibular Width (Points 3 to 9)
Measures the width of the mandibular angle region, providing constitutional classification based on jaw structure.

**Measurement Process:**
1. Calculate Euclidean distance from point 3 (inferior ear attachment) to point 9 (mandibular angle)
2. Normalize by dividing by the inferior third distance (18-10)
3. Classify based on normalized proportion relative to lower facial height

**Formula:**
```
mandibular_distance = sqrt((point_9[x] - point_3[x])² + (point_9[y] - point_3[y])²)
mandibular_proportion = mandibular_distance / inferior_third_distance
```

**Classifications:**
- **Mandibula Sanguinea** (sanguine mandible): proportion ≥ 0.8 (wide, robust jaw)
- **Mandibula intermedia sanguineo/bilosa** (intermediate sanguine/bilious): 0.75 ≤ proportion < 0.8
- **Mandibula Bilosa** (bilious mandible): 0.40 ≤ proportion < 0.75 (moderate jaw)
- **Mandibula intermedia bilosa/nerviosa** (intermediate bilious/nervous): 0.35 ≤ proportion < 0.40
- **Mandibula Nerviosa** (nervous mandible): proportion < 0.35 (narrow, delicate jaw)
- **Mandibula Linfatica** (lymphatic mandible): classification when point 9 is not detected (indicates very recessed mandibular angle)

#### 4.2 Mandibular Intersection Angle (Vectors 24-18 and 3-9)
Calculates the angle between the facial reference line and the mandibular line to assess jaw alignment.

**Mathematical Process:**
1. Create reference vector: `vector_24_18 = [point_18 - point_24]`
2. Create mandible vector: `vector_3_9 = [point_9 - point_3]`
3. Calculate slopes for both vectors
4. Normalize both vectors to unit length
5. Calculate angle via dot product: `angle = arccos(abs(clip(dot(v1, v2), -1, 1)))`
6. Normalize angle to 0-180 degree range (use absolute value for acute angle measurement)

**Formula:**
```
reference_slope = vector_24_18[y] / vector_24_18[x]
mandible_slope = vector_3_9[y] / vector_3_9[x]
cos_angle = clip(dot(v1_normalized, v2_normalized), -1, 1)
angle = arccos(abs(cos_angle)) * (180/π)
if angle > 90: angle = 180 - angle
```

**Angle Classifications:**
- **acute mandible angle**: angle < 70 degrees (more angled jaw line)
- **normal mandible angle**: 70 ≤ angle ≤ 110 degrees (balanced jaw alignment)
- **obtuse mandible angle**: angle > 110 degrees (more parallel jaw line)

### 5. Angular Analysis

#### 5.1 Mathematical Foundation
All angular measurements use the reference vector from point 24 to point 18 as the baseline. The system determines head direction using vector analysis as described in section 1.2.

**Key Implementation Details:**
- **Vector Normalization**: All vectors are normalized to unit length using `vector / norm(vector)` before angle calculations to ensure consistent results
- **Dot Product Clamping**: Values are clamped to range [-1.0, 1.0] to prevent numerical errors in arccos function
- **Angle Conversion**: All angles are converted from radians to degrees using the formula `degrees = radians * (180/π)`
- **Profile Awareness**: Left and right profile orientations automatically adjust angle signs and interpretations for anatomically correct measurements
- **Robust Edge Cases**: Handles infinite slopes (vertical lines) and zero-division scenarios with conditional checks

#### 5.2 Forehead Angle Calculation (Points 24 to 33)
Measures the inclination of the forehead relative to the facial reference line to classify forehead posture.

**Mathematical Process:**
1. Establish reference vector: `vector_24_18 = [point_18[x] - point_24[x], point_18[y] - point_24[y]]`
2. Create forehead measurement vector: `vector_24_33 = [point_33[x] - point_24[x], point_33[y] - point_24[y]]`
3. Normalize both vectors to unit length
4. Calculate angle via dot product: `cos_angle = clip(dot(v1_normalized, v2_normalized), -1, 1)`
5. Convert to degrees: `angle = arccos(cos_angle) * (180/π)`
6. Apply directional adjustment based on head direction:
   - **Left Profile**: If vector turns right (vector_24_33[x] < vector_24_18[x]), angle is positive; if turns left, angle is negative
   - **Right Profile**: Opposite sign convention
7. Normalize to 0-90 degree range: take absolute value if negative; if > 90, convert to `180 - angle`

**Formula:**
```
v1 = vector_24_18 / |vector_24_18|
v2 = vector_24_33 / |vector_24_33|
angle = arccos(clip(dot(v1, v2), -1, 1)) * (180/π)
```

**Classification Thresholds:**
- **frente inclinada hacia atras** (reclined forehead): angle > 15 degrees
- **frente neutra** (neutral forehead): 11 ≤ angle ≤ 15 degrees
- **frente vertical** (vertical forehead): angle < 11 degrees

#### 5.3 Chin Angle Calculation (Points 18 to 11)
Measures the projection angle of the chin relative to the facial reference line for constitutional classification.

**Mathematical Process:**
1. Use reference vector: `vector_24_18` (established above)
2. Create chin measurement vector: `vector_18_11 = [point_11[x] - point_18[x], point_11[y] - point_18[y]]`
3. Normalize both vectors to unit length
4. Calculate angle via dot product: `angle = arccos(clip(dot(v1, v2), -1, 1)) * (180/π)`
5. Determine face side using point 17 position:
   - If `point_17[x] < point_18[x]`: Left side of face (nose points left)
   - Otherwise: Right side of face (nose points right)
6. Apply directional logic:
   - **Left Side**: Positive angles indicate chin turning right (protrusion); negative angles indicate chin turning left (recession)
   - **Right Side**: Opposite convention
7. Clamp final angle to -90 to +90 degree range

**Formula:**
```
v1 = vector_24_18 / |vector_24_18|
v2 = vector_18_11 / |vector_18_11|
angle = arccos(clip(dot(v1, v2), -1, 1)) * (180/π)
# Apply sign based on face side and vector direction
if angle > 90: angle = -(180 - angle)
if angle < -90: angle = 180 + angle
```

**Classification Thresholds:**
- **menton nervioso** (receding chin): angle ≤ -5 degrees
- **menton biloso/linfatico** (neutral chin): -5 < angle ≤ 5.5 degrees
- **menton sanguineo** (protruding chin): angle > 5.5 degrees

#### 5.4 Ear Implantation Angular Analysis

##### 5.4.1 Superior Implantation Angle (Point 22 to 4)
Measures the angle of the superior ear attachment point relative to a perpendicular reference line at the glabella.

**Mathematical Process:**
1. Calculate perpendicular vector at point 22:
   - `ref_slope = vector_24_18[y] / vector_24_18[x]`
   - `perp_slope = -1 / ref_slope` (negative reciprocal for perpendicular)
   - `perp_vector = [1, perp_slope]`
2. Create measurement vector: `vector_22_4 = [point_4[x] - point_22[x], point_4[y] - point_22[y]]`
3. Normalize both vectors
4. Calculate angle using atan2 for proper quadrant handling:
   - `angle = atan2(cross(v1, v2), dot(v1, v2)) * (180/π)`
5. Convert negative angles to 0-360 degree system: if angle < 0, add 360
6. Classification uses threshold in the 351-360 degree range

**Implementation Note:** The system uses a 0-360 degree measurement system converted from the original negative threshold system. An angle ≥ 351 degrees (equivalent to ≤ -9 degrees in the -180 to +180 system) indicates high implantation.

**Classifications:**
- **implantacion alta** (high implantation): angle ≥ 351 degrees (ear attaches above reference line)
- **implantacion estandard** (standard implantation): angle < 351 degrees

##### 5.4.2 Inferior Implantation Angle (Point 18 to 5)
Measures the angle of the inferior ear lobe attachment relative to a perpendicular reference line at the subnasale.

**Mathematical Process:**
1. Use same perpendicular vector calculation method as superior implantation
2. Create measurement vector: `vector_18_5 = [point_5[x] - point_18[x], point_5[y] - point_18[y]]`
3. Apply same atan2 calculation for angle determination
4. Convert to 0-360 degree system
5. Classification uses threshold at 350 degrees

**Classifications:**
- **implantacion baja** (low implantation): angle ≥ 350 degrees (ear lobe hangs below reference line)
- **implantacion estandard** (standard implantation): angle < 350 degrees

##### 5.4.3 Ear Implantation Vector Intersection (Vectors 24-18 and 1-3)
Measures the overall angle between the facial reference line and the ear implantation axis.

**Mathematical Process:**
1. Reference vector: `vector_24_18`
2. Ear implantation vector: `vector_1_3 = [point_3[x] - point_1[x], point_3[y] - point_1[y]]` (superior to inferior ear attachment)
3. Normalize both vectors
4. Calculate angle: `angle = arccos(abs(clip(dot(v1, v2), -1, 1))) * (180/π)`
5. Use absolute value to always measure acute angle
6. Normalize to 0-90 degree range: if angle > 90, convert to `180 - angle`

**Formula:**
```
v1 = vector_24_18 / |vector_24_18|
v2 = vector_1_3 / |vector_1_3|
angle = arccos(abs(clip(dot(v1, v2), -1, 1))) * (180/π)
if angle > 90: angle = 180 - angle
```

**Classifications:**
- **acute ear implantation**: angle < 60 degrees (more angled ear axis)
- **normal ear implantation**: 60 ≤ angle ≤ 120 degrees (balanced ear orientation)
- **obtuse ear implantation**: angle > 120 degrees (more parallel ear axis)

#### 5.5 Eye Protrusion Analysis (Points 37, 38, 39)
Assesses the degree of eyeball protrusion relative to the orbital plane using a sophisticated geometric method.

##### 5.5.1 Orbital Plane Definition
The orbital plane is defined by the vector from point 37 (orbital base, inner eye corner) to point 39 (orbital superior boundary). This plane represents the natural boundary of the eye socket.

##### 5.5.2 Perpendicular Distance Calculation
**Mathematical Process:**
1. Create orbital plane vector: `vector_37_39 = [point_39 - point_37]`
2. Create cornea vector: `vector_37_38 = [point_38 - point_37]`
3. Calculate 2D cross product (signed area):
   - `cross = (vector_37_39[x] * vector_37_38[y]) - (vector_37_39[y] * vector_37_38[x])`
4. Normalize by orbital plane length:
   - `perpendicular_distance = cross / |vector_37_39|`
5. Apply profile-aware sign correction:
   - **Left Profile**: `signed_distance = -perpendicular_distance` (negative indicates protrusion leftward)
   - **Right Profile**: `signed_distance = perpendicular_distance` (positive indicates protrusion rightward)

**Formula:**
```
cross_product = (v_orbital[x] * v_cornea[y]) - (v_orbital[y] * v_cornea[x])
perpendicular_distance = cross_product / |v_orbital|
signed_distance = apply_profile_correction(perpendicular_distance, head_direction)
```

**Physical Interpretation:**
- Positive signed distance: Cornea crosses beyond the orbital plane in the profile direction (protrusion)
- Near-zero signed distance: Cornea aligns with the orbital plane (normal)
- Negative signed distance: Cornea recedes behind the orbital plane (deep-set eyes)

**Classifications (with 3-pixel tolerance):**
- **pronounced eye protrusion**: signed_distance > 3 pixels (eyeball noticeably protrudes beyond orbital plane)
- **normal eye protrusion**: -3 ≤ signed_distance ≤ 3 pixels (eyeball aligned with orbital plane)
- **minimal eye protrusion**: signed_distance < -3 pixels (deep-set eyes, eyeball recessed behind orbital plane)

##### 5.5.3 Angular Analysis (Vectors 39-37 and 38-37)
Supplementary measurement calculating the opening angle of the eye structure.

**Mathematical Process:**
1. Create vectors pointing toward point 37:
   - `vector_39_37 = [point_37 - point_39]`
   - `vector_38_37 = [point_37 - point_38]`
2. Calculate angle between these vectors using dot product
3. Normalize to acute angle (0-90 degrees)

This provides additional angular context for eye shape assessment.

### 6. Ear Measurements

#### 6.1 Ear Width (Points 2 to 6)
Measures the anteroposterior diameter of the ear.

**Formula:**
```
ear_width = sqrt((point_6[x] - point_2[x])² + (point_6[y] - point_2[y])²)
```

This measurement serves as a baseline for tragus-antitragus proportional analysis.

#### 6.2 Ear Length (Points 4 to 5)
Measures the superoinferior height of the ear and assesses it proportionally to facial height.

**Measurement Process:**
1. Calculate ear length: `ear_length = distance(point_4, point_5)`
2. Use reference distance as face length: `face_length = distance(point_24, point_10)`
3. Calculate proportion: `ear_proportion = ear_length / face_length`

**Formula:**
```
ear_length = sqrt((point_5[x] - point_4[x])² + (point_5[y] - point_4[y])²)
ear_proportion = ear_length / reference_distance
```

**Classifications:**
- **oreja larga** (long ear): proportion > 0.432 (ear exceeds 43.2% of facial height)
- **oreja normal** (normal ear): 0.38 ≤ proportion ≤ 0.432 (ear is 38-43.2% of facial height)
- **oreja corta** (short ear): proportion < 0.38 (ear less than 38% of facial height)

#### 6.3 Ear Lobe Length (Points 3 to 5)
Measures the ear lobe size relative to total ear length.

**Measurement Process:**
1. Calculate ear lobe length: `lobe_length = distance(point_3, point_5)`
2. Calculate total ear length: `ear_length = distance(point_4, point_5)`
3. Calculate proportion: `lobe_proportion = lobe_length / ear_length`

**Formula:**
```
lobe_length = sqrt((point_5[x] - point_3[x])² + (point_5[y] - point_3[y])²)
lobe_proportion = lobe_length / ear_length
```

**Classifications:**
- **lobulo grande** (large lobe): proportion > 0.31 (lobe exceeds 31% of ear length)
- **lobulo normal** (normal lobe): 0.28 ≤ proportion ≤ 0.31 (lobe is 28-31% of ear length)
- **lobulo chico** (small lobe): proportion < 0.28 (lobe less than 28% of ear length)

#### 6.4 Tragus-Antitragus Distance (Points 7 to 8)
Measures the distance between tragus and antitragus cartilage structures relative to ear width.

**Measurement Process:**
1. Calculate tragus-antitragus distance: `distance_7_8 = distance(point_7, point_8)`
2. Calculate ear width: `ear_width = distance(point_2, point_6)`
3. Calculate proportion: `ta_proportion = distance_7_8 / ear_width`

**Formula:**
```
tragus_antitragus_distance = sqrt((point_8[x] - point_7[x])² + (point_8[y] - point_7[y])²)
ta_proportion = tragus_antitragus_distance / ear_width
```

**Classifications:**
- **grande** (large): proportion ≥ 0.255 (tragus-antitragus distance is at least 25.5% of ear width)
- **normal** (normal): 0.22 ≤ proportion < 0.255 (distance is 22-25.5% of ear width)
- **corta** (short): proportion < 0.22 (distance is less than 22% of ear width)

### 7. Nasal Triangulation Analysis (Points 17, 18, 26, 30)
Assesses the relationship between the nostril opening and nasal base to evaluate nasal cavity structure.

**Measurement Process:**
1. Calculate nasal orifice distance from nostril base to nasal tip:
   - `orifice_distance = distance(point_26, point_17)`
2. Calculate nasal reference baseline from subnasale to nasal septum:
   - `reference_distance = distance(point_18, point_30)`
3. Calculate triangulation proportion:
   - `triangulation_proportion = orifice_distance / reference_distance`

**Formula:**
```
nasal_orifice_distance = sqrt((point_17[x] - point_26[x])² + (point_17[y] - point_26[y])²)
nose_reference_distance = sqrt((point_30[x] - point_18[x])² + (point_30[y] - point_18[y])²)
nasal_proportion = nasal_orifice_distance / nose_reference_distance
```

**Physical Interpretation:**
A higher proportion indicates greater separation between the nostril base and the nasal tip, suggesting a more triangulated or opened nasal cavity structure. A lower proportion indicates a more compact, less triangulated nasal opening.

**Classifications:**
- **triangulacion de fosa** (nostril triangulation present): proportion > 0.27 (nostril opening is more than 27% of nasal baseline, indicating pronounced triangulation)
- **sin triangulacion de fosa** (no nostril triangulation): proportion ≤ 0.27 (nostril opening is 27% or less of nasal baseline, indicating minimal triangulation)

## Comprehensive Measurement Summary

This section provides a quick reference table of all measurements, their point dependencies, proportional thresholds, and classification ranges.

### Distance-Based Measurements

| Measurement | Points Used | Normalization | Thresholds | Classifications |
|------------|-------------|---------------|------------|-----------------|
| **Reference Distance** | 24-10 | None (baseline) | N/A | Primary scaling factor for all proportional measurements |
| **Nasal Protrusion** | 18-17 | Divide by reference distance | > 0.2<br>0.17-0.2<br>< 0.17 | nariz protruyente<br>nariz normal<br>nariz corta |
| **Superior Third** | 34-22 | Divide by reference distance | N/A | tercio superior (proportional measurement) |
| **Middle Third** | 22-18 | Divide by reference distance | N/A | tercio medio (proportional measurement) |
| **Inferior Third** | 18-10 | Divide by reference distance | N/A | tercio inferior (proportional measurement) |
| **Mandibular Width** | 3-9 | Divide by inferior third | ≥ 0.8<br>0.75-0.8<br>0.40-0.75<br>0.35-0.40<br>< 0.35<br>Point 9 missing | Mandibula Sanguinea<br>Mandibula intermedia sanguineo/bilosa<br>Mandibula Bilosa<br>Mandibula intermedia bilosa/nerviosa<br>Mandibula Nerviosa<br>Mandibula Linfatica |
| **Ear Width** | 2-6 | None | N/A | Used as baseline for tragus-antitragus proportion |
| **Ear Length** | 4-5 | Divide by reference distance | > 0.432<br>0.38-0.432<br>< 0.38 | oreja larga<br>oreja normal<br>oreja corta |
| **Ear Lobe Length** | 3-5 | Divide by ear length (4-5) | > 0.31<br>0.28-0.31<br>< 0.28 | lobulo grande<br>lobulo normal<br>lobulo chico |
| **Tragus-Antitragus** | 7-8 | Divide by ear width (2-6) | ≥ 0.255<br>0.22-0.255<br>< 0.22 | grande<br>normal<br>corta |
| **Nasal Triangulation** | 26-17 | Divide by nose reference (18-30) | > 0.27<br>≤ 0.27 | triangulacion de fosa<br>sin triangulacion de fosa |

### Angular Measurements

| Measurement | Points Used | Reference Vector | Angle Range | Thresholds | Classifications |
|------------|-------------|------------------|-------------|------------|-----------------|
| **Nasal Tip Angle** | 18-17 | 24-18 (perpendicular) | -90° to +90° | ≥ 27°<br>12° to 27°<br>< 12° | punta de nariz hacia arriba<br>punta de nariz promedio<br>punta hacia abajo |
| **Nasal Dorsum Slope** | 19-17 | N/A (slope calculation) | N/A | N/A | Provides slope angle for nasal bridge assessment |
| **Forehead Angle** | 24-33 | 24-18 | 0° to 90° | > 15°<br>11° to 15°<br>< 11° | frente inclinada hacia atras<br>frente neutra<br>frente vertical |
| **Chin Angle** | 18-11 | 24-18 | -90° to +90° | ≤ -5°<br>-5° to 5.5°<br>> 5.5° | menton nervioso<br>menton biloso/linfatico<br>menton sanguineo |
| **Superior Implantation** | 22-4 | 24-18 (perpendicular) | 0° to 360° | ≥ 351°<br>< 351° | implantacion alta<br>implantacion estandard |
| **Inferior Implantation** | 18-5 | 24-18 (perpendicular) | 0° to 360° | ≥ 350°<br>< 350° | implantacion baja<br>implantacion estandard |
| **Mandible Intersection** | 3-9 vs 24-18 | 24-18 | 0° to 180° | < 70°<br>70° to 110°<br>> 110° | acute mandible angle<br>normal mandible angle<br>obtuse mandible angle |
| **Ear Implantation Intersection** | 1-3 vs 24-18 | 24-18 | 0° to 90° | < 60°<br>60° to 120°<br>> 120° | acute ear implantation<br>normal ear implantation<br>obtuse ear implantation |
| **Eye Opening Angle** | 39-37 vs 38-37 | N/A | 0° to 90° | N/A | Supplementary angle for eye shape assessment |

### Eye Protrusion Measurements

| Measurement | Points Used | Method | Tolerance | Classifications |
|------------|-------------|--------|-----------|-----------------|
| **Eye Protrusion Distance** | 37-39-38 | Perpendicular distance from cornea (38) to orbital plane (37-39) | ±3 pixels | > 3 px: pronounced eye protrusion<br>-3 to 3 px: normal eye protrusion<br>< -3 px: minimal eye protrusion |

### Profile Direction Determination

| Method | Points Used | Logic | Output |
|--------|-------------|-------|--------|
| **Suffix Counting** | All detected points | Count points with `_i` suffix (left) vs `_d` suffix (right); majority determines profile | left / right / unknown |
| **Vector Analysis** | 24-18 | If `vector_24_18[x] > 0`: right profile, else: left profile | left / right |

### Critical Point Dependencies

The following measurements require specific point combinations to function. If any required point is missing, that measurement cannot be performed:

- **All normalized proportional measurements**: Require points 24 and 10 (reference distance)
- **Mandibular measurements**: Require points 3 and 9; if point 9 missing, classified as Mandibula Linfatica
- **Facial thirds**: Require points 34, 22, 18, and 10 for complete three-part analysis
- **Ear measurements**: Require points 2, 4, 5, 6 for comprehensive ear analysis; points 3, 7, 8 for lobe and tragus measurements
- **Nasal measurements**: Require points 17, 18 for basic nose analysis; points 19, 26, 30 for advanced nasal features
- **Angular measurements**: Most require point 24 and 18 for the baseline reference vector
- **Eye protrusion**: Requires all three points 37, 38, and 39 for orbital plane calculation
- **Implantation angles**: Require points 1, 3, 4, 5, 18, 22 for complete superior and inferior implantation analysis

### Adaptive Thresholding

The system uses an adaptive confidence threshold for point detection:
- **Primary threshold**: 0.15 (default)
- **Fallback threshold**: 0.05 (used when no points exceed 0.15)
- **Configurable range**: 0.05 to 0.9 via API parameter

Points with confidence below the active threshold are filtered out before measurement calculations.

## Technical Specifications

### Model Architecture
- **Base Model**: ResNet-50 backbone with custom decoder
- **Point Detection**: Heatmap-based keypoint detection
- **Profile Classification**: Integrated left/right profile classifier
- **Input Size**: 224x224 pixels
- **Heatmap Resolution**: 112x112 pixels

### Input Requirements
- **Image Formats**: JPG, PNG
- **Orientation**: Profile (lateral) view required
- **Quality**: Clear facial features, minimal occlusion
- **Lighting**: Even illumination preferred

### Output Formats
- **JSON**: Structured measurement data
- **PNG**: Annotated visualization images
- **Base64**: Embedded image data in API responses

## Installation and Setup

### Docker Deployment (Recommended)
```bash
# Build the container
docker-compose build

# Run the service
docker-compose up -d

# Check service status
curl http://localhost:8004/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model file exists
ls models/profile_aware_point_detection_model.pth

# Run the application
python app/main.py
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: auto-detect)
- `PORT`: Service port (default: 8004)

## Usage Examples

### Python Client
```python
import requests
import json

# Analyze profile image
with open('profile_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8004/analyze-profile-anthropometric',
        files={'file': f},
        data={
            'confidence_threshold': 0.15,
            'include_visualization': True
        }
    )

results = response.json()
print(f"Profile side: {results['profile_analysis']['profile_side']}")
print(f"Nose type: {results['anthropometric_measurements']['nose_classification']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8004/analyze-profile-anthropometric" \
  -F "file=@profile_image.jpg" \
  -F "confidence_threshold=0.15" \
  -F "include_visualization=true"
```

## Model Files Required

### Enhanced Pipeline (Recommended)
- `models/best_point_detection_model_v2.pth` - **Enhanced CNN model** (ResNet-50 + Profile Classification)
- `models/facial_landmark_gnn.pth` - **GNN validation model** (optional, improves accuracy)

### Fallback Compatibility
- `models/profile_aware_point_detection_model.pth` - Legacy model (fallback if v2 not available)

### Model Configuration
The system automatically detects available models and configures accordingly:
1. **Full Enhanced Mode**: Both CNN v2 + GNN models → Best accuracy with false positive filtering
2. **CNN-only Mode**: CNN v2 model only → Good accuracy, faster processing  
3. **Fallback Mode**: Legacy model → Basic functionality maintained

**Note**: The GNN model significantly improves detection quality by validating landmark positions using anatomical relationships. For production use, both models are recommended.

## Results Directory

Generated files are stored in `/app/results/`:
- `profile_anthropometric_{uuid}.png` - Visualization images

## Error Handling

### Common Error Responses
- **400**: Invalid image format or parameters
- **503**: Model not loaded or initialization failed
- **500**: Analysis processing error

### Troubleshooting
1. **Model Loading Issues**: Ensure model file exists and is accessible
2. **CUDA Errors**: Check GPU availability and CUDA installation
3. **Memory Issues**: Reduce batch size or use CPU mode
4. **Poor Detection**: Ensure profile orientation and good image quality

## Performance Notes

### Enhanced Pipeline Performance
- **Processing Time**: 
  - CNN-only mode: ~0.5-1.2 seconds per image (GPU)
  - Full enhanced mode (CNN + GNN): ~0.8-2.0 seconds per image (GPU)
- **Memory Usage**: ~2-4GB GPU memory (similar to legacy)
- **Accuracy**: Significantly improved false positive filtering and profile detection
- **Throughput**: ~25-50 images/minute (depending on hardware and GNN usage)

### Improvements over Legacy
- **False Positive Reduction**: ~60-80% fewer spurious detections
- **Profile Classification**: More robust left/right profile determination
- **Edge Case Handling**: Better performance on challenging images

## Version Information

- **API Version**: 1.0.0
- **Pipeline Version**: Enhanced v2 with GNN validation
- **Model Architecture**: ResNet-50 + Profile Classifier + Graph Neural Network
- **Framework**: FastAPI + PyTorch
- **Last Updated**: 2025

## Support

For technical support or questions about the anthropometric analysis module, please refer to the main project documentation or contact the development team.
