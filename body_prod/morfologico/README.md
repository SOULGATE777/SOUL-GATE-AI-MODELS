# Body Morphological Analysis Module

## Overview

The Body Morphological Analysis module is a production-ready system for comprehensive body type classification and morphological analysis using advanced deep learning. It employs a lightweight hierarchical model to classify body types according to classical somatotype theory while providing detailed morphological insights, gender classification, and health-related recommendations.

## Features

### Core Capabilities
- **Hierarchical Body Type Classification**: 7 distinct body type categories with simplified naming
- **Gender Classification**: Automatic gender detection with confidence scoring
- **Coarse-to-Fine Analysis**: Multi-level classification for improved accuracy
- **Morphological Insights**: Detailed body composition and metabolic tendency analysis
- **Confidence Assessment**: Multi-level confidence scoring and reliability assessment
- **Batch Processing**: Efficient processing of multiple images
- **Real-time Visualization**: Annotated results with classification details

## API Endpoints

### Base URL
```
http://localhost:8006
```

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "body-morphological-analysis",
  "model_loaded": true,
  "device": "cuda",
  "classes": {
    "body_types": 7,
    "genders": 2
  }
}
```

### 2. Complete Body Morphological Analysis
```http
POST /analyze-body-morphology
```
**Parameters:**
- `file` (required): Image file (JPG, PNG, etc.)
- `bbox` (optional): Bounding box as "x1,y1,x2,y2" to crop body region
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)
- `include_visualization` (optional): Boolean (default: true)

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "body_type_analysis": {
    "predicted_class": "Sanguineo/Musculoso",
    "predicted_class_simple": "Musculoso",
    "confidence": 0.87,
    "meets_threshold": true,
    "all_probabilities": {
      "Bilioso/NormalPocaGrasa": 0.05,
      "Nervioso/Delgado": 0.03,
      "SanguineoLinfatico/MusculosoGordo": 0.12,
      "Sanguineo/Musculoso": 0.87,
      "Flematico/Gordograsacuelga": 0.02,
      "Linfatico/Gordo": 0.01,
      "BiliosoSanguineo/NormalMusculoso": 0.08
    },
    "top_3_predictions": [
      {
        "class": "Sanguineo/Musculoso",
        "confidence": 0.87,
        "rank": 1
      },
      {
        "class": "SanguineoLinfatico/MusculosoGordo",
        "confidence": 0.12,
        "rank": 2
      },
      {
        "class": "BiliosoSanguineo/NormalMusculoso",
        "confidence": 0.08,
        "rank": 3
      }
    ]
  },
  "gender_analysis": {
    "predicted_gender": "Hombre",
    "confidence": 0.92,
    "meets_threshold": true,
    "all_probabilities": {
      "Hombre": 0.92,
      "Mujer": 0.08
    }
  },
  "coarse_analysis": {
    "predicted_coarse": "Coarse_Type_2",
    "confidence": 0.78,
    "all_probabilities": {
      "Coarse_Type_0": 0.05,
      "Coarse_Type_1": 0.12,
      "Coarse_Type_2": 0.78,
      "Coarse_Type_3": 0.05
    }
  },
  "analysis_metrics": {
    "overall_confidence": 0.895,
    "prediction_certainty": "high",
    "gender_body_consistency": "high"
  },
  "morphological_insights": {
    "body_composition": "Mesomorphic build with well-developed musculature",
    "metabolic_tendency": "Efficient metabolism, responds well to exercise",
    "physical_characteristics": "Athletic build, defined muscle structure",
    "health_considerations": "Maintain balanced training and nutrition",
    "analysis_note": "High confidence prediction - reliable classification"
  },
  "classification_summary": {
    "primary_classification": "Musculoso",
    "gender": "Hombre",
    "confidence_level": "high",
    "recommended_action": "Classification is reliable - proceed with analysis"
  },
  "analysis_summary": {
    "timestamp": "2025-06-25T10:30:00",
    "processing_successful": true,
    "confidence_threshold_used": 0.5,
    "device_used": "cuda",
    "image_preprocessed": true,
    "bbox_applied": false,
    "analysis_type": "complete_morphological_analysis"
  },
  "visualization_path": "/app/results/body_morphology_uuid.png",
  "visualization_url": "/visualization/body_morphology_uuid.png"
}
```

### 3. Body Type Classification Only
```http
POST /classify-body-type
```
**Parameters:**
- `file` (required): Image file
- `bbox` (optional): Bounding box as "x1,y1,x2,y2"
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "body_type_analysis": {
    "predicted_class": "Nervioso/Delgado",
    "predicted_class_simple": "Delgado",
    "confidence": 0.73,
    "meets_threshold": true,
    "all_probabilities": {...},
    "top_3_predictions": [...]
  },
  "gender_analysis": {
    "predicted_gender": "Mujer",
    "confidence": 0.85,
    "meets_threshold": true,
    "all_probabilities": {...}
  },
  "coarse_analysis": {...},
  "analysis_summary": {
    "timestamp": "2025-06-25T10:30:00",
    "processing_successful": true,
    "confidence_threshold_used": 0.5,
    "device_used": "cuda",
    "image_preprocessed": true,
    "bbox_applied": false
  },
  "analysis_type": "classification_only"
}
```

### 4. Batch Classification
```http
POST /batch-classify
```
**Parameters:**
- `files` (required): List of image files (max 10)
- `confidence_threshold` (optional): Float 0.0-1.0 (default: 0.5)

**Response:**
```json
{
  "batch_id": "uuid-string",
  "total_images": 3,
  "successful_classifications": 2,
  "results": [
    {
      "analysis_id": "uuid-string",
      "batch_index": 0,
      "filename": "image1.jpg",
      "body_type_analysis": {...},
      "gender_analysis": {...}
    },
    {
      "analysis_id": "uuid-string",
      "batch_index": 1,
      "filename": "image2.jpg",
      "error": "Classification failed"
    }
  ]
}
```

### 5. Model Information
```http
GET /model-info
```
**Response:**
```json
{
  "model_architecture": "LightweightHierarchicalModel",
  "backbone": "ResNet18",
  "input_size": [224, 224],
  "device": "cuda",
  "body_type_classes": [
    "Bilioso/NormalPocaGrasa",
    "Nervioso/Delgado",
    "SanguineoLinfatico/MusculosoGordo",
    "Sanguineo/Musculoso",
    "Flematico/Gordograsacuelga",
    "Linfatico/Gordo",
    "BiliosoSanguineo/NormalMusculoso"
  ],
  "gender_classes": ["Hombre", "Mujer"],
  "total_parameters": 11789318,
  "model_path": "/app/models/lightweight_body_classifier.pth"
}
```

## Body Type Classifications

### Primary Classifications (7 Types)

#### 1. Bilioso/NormalPocaGrasa → "Normal Poca Grasa"
- **Characteristics**: Normal build with low body fat
- **Metabolic Profile**: Balanced metabolism
- **Physical Traits**: Well-proportioned, minimal excess fat
- **Health Focus**: Maintain current composition

#### 2. Nervioso/Delgado → "Delgado" 
- **Characteristics**: Ectomorphic build, very lean
- **Metabolic Profile**: Fast metabolism, difficulty gaining weight
- **Physical Traits**: Narrow frame, long limbs, minimal muscle mass
- **Health Focus**: Strength training, increased caloric intake

#### 3. SanguineoLinfatico/MusculosoGordo → "Musculoso Gordo"
- **Characteristics**: Muscular build with higher body fat
- **Metabolic Profile**: Variable metabolism, muscle with fat layer
- **Physical Traits**: Strong frame, muscle definition obscured by fat
- **Health Focus**: Fat reduction while preserving muscle

#### 4. Sanguineo/Musculoso → "Musculoso"
- **Characteristics**: Mesomorphic build, well-developed muscle
- **Metabolic Profile**: Efficient metabolism, exercise responsive
- **Physical Traits**: Athletic build, visible muscle definition
- **Health Focus**: Balanced training and nutrition

#### 5. Flematico/Gordograsacuelga → "Gordo Grasa Cuelga"
- **Characteristics**: Endomorphic build with significant adipose tissue
- **Metabolic Profile**: Slow metabolism, fat storage tendency
- **Physical Traits**: Soft tissue, visible fat deposits
- **Health Focus**: Cardiovascular exercise, dietary management

#### 6. Linfatico/Gordo → "Gordo"
- **Characteristics**: Endomorphic build, high body fat percentage
- **Metabolic Profile**: Slower metabolism, weight gain tendency
- **Physical Traits**: Rounded physique, soft muscle definition
- **Health Focus**: Weight management, metabolic improvement

#### 7. BiliosoSanguineo/NormalMusculoso → "Normal Musculoso"
- **Characteristics**: Balanced build with good muscle development
- **Metabolic Profile**: Moderate metabolism, training responsive
- **Physical Traits**: Proportioned frame, moderate muscle definition
- **Health Focus**: Maintain active lifestyle

### Gender Classifications
- **Hombre** (Male): Masculine body characteristics
- **Mujer** (Female): Feminine body characteristics

## Technical Specifications

### Model Architecture
- **Base Model**: LightweightHierarchicalModel
- **Backbone**: ResNet18 (pretrained)
- **Feature Extraction**: 512 → 256 → 128 dimensions
- **Attention Mechanism**: Sigmoid-gated attention
- **Multi-Head Classification**: Body type, gender, coarse classification
- **Input Size**: 224×224 pixels
- **Parameters**: ~11.8M parameters (optimized for limited GPU memory)

### Hierarchical Classification
1. **Feature Extraction**: ResNet18 backbone with custom feature layers
2. **Attention Module**: Learns important feature regions
3. **Coarse Classification**: 4 broad categories for guidance
4. **Fine Classification**: 7 specific body types using coarse + features
5. **Gender Classification**: Independent gender prediction head

### Input Requirements
- **Image Formats**: JPG, PNG, BMP, TIFF
- **Content**: Full body or torso visible
- **Quality**: Clear body outline, minimal occlusion
- **Lighting**: Even illumination preferred
- **Resolution**: Any resolution (auto-resized to 224×224)

### Output Formats
- **JSON**: Structured classification data
- **PNG**: Annotated visualization images
- **Batch Results**: Multiple image analysis

## Morphological Insights System

### Body Composition Analysis
Based on classified body type, the system provides:
- **Somatotype Assessment**: Ectomorphic, Mesomorphic, Endomorphic traits
- **Muscle Development**: Muscle mass and definition levels
- **Fat Distribution**: Adipose tissue patterns and locations
- **Frame Size**: Bone structure and overall build assessment

### Metabolic Tendency Prediction
- **Fast Metabolism**: High caloric needs, difficulty gaining weight
- **Efficient Metabolism**: Balanced energy use, exercise responsive
- **Slower Metabolism**: Lower energy needs, weight gain tendency
- **Variable Metabolism**: Context-dependent metabolic patterns

### Physical Characteristics
- **Body Frame**: Narrow, athletic, broad, or rounded
- **Muscle Definition**: From minimal to well-defined
- **Limb Proportions**: Relative length and muscle development
- **Overall Appearance**: Lean, athletic, balanced, or soft

### Health Considerations
Automated recommendations based on body type:
- **Exercise Focus**: Strength training, cardio, or balanced approach
- **Nutritional Guidance**: Caloric needs and macronutrient focus
- **Metabolic Support**: Strategies for metabolic optimization
- **Body Composition Goals**: Realistic targets for improvement

## Confidence and Reliability Assessment

### Confidence Levels
- **High (>0.8)**: Very reliable classification
- **Medium (0.6-0.8)**: Moderately reliable classification
- **Low (<0.6)**: Consider additional assessment

### Gender-Body Consistency Check
- **High**: Gender and body type predictions align well
- **Medium**: Reasonable alignment between predictions
- **Low**: Inconsistencies detected, manual review recommended

### Recommendation System
- **High Certainty**: "Classification is reliable - proceed with analysis"
- **Medium Certainty**: "Classification is moderately reliable - consider additional assessment"
- **Low Certainty**: "Low confidence classification - recommend manual review or better image quality"

## Installation and Setup

### Docker Deployment (Recommended)
```bash
# Build the container
docker-compose build

# Run the service
docker-compose up -d

# Check service status
curl http://localhost:8006/health
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model file exists
ls models/lightweight_body_classifier.pth

# Create required directories
mkdir -p /app/results

# Run the application
python app/main.py
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `PORT`: Service port (default: 8006)

## Usage Examples

### Python Client
```python
import requests

# Complete morphological analysis
with open('body_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8006/analyze-body-morphology',
        files={'file': f},
        data={
            'confidence_threshold': 0.5,
            'include_visualization': True
        }
    )

results = response.json()
body_type = results['body_type_analysis']['predicted_class_simple']
gender = results['gender_analysis']['predicted_gender']
confidence = results['analysis_metrics']['overall_confidence']

print(f"Body Type: {body_type}")
print(f"Gender: {gender}")
print(f"Confidence: {confidence:.2f}")
print(f"Insights: {results['morphological_insights']['body_composition']}")
```

### With Bounding Box
```python
import requests

# Analyze specific body region
with open('full_body_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8006/analyze-body-morphology',
        files={'file': f},
        data={
            'bbox': '100,50,400,600',  # x1,y1,x2,y2
            'confidence_threshold': 0.6,
            'include_visualization': True
        }
    )

results = response.json()
```

### Batch Processing
```python
import requests

files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]

response = requests.post(
    'http://localhost:8006/batch-classify',
    files=files,
    data={'confidence_threshold': 0.5}
)

batch_results = response.json()
for result in batch_results['results']:
    if 'error' not in result:
        body_type = result['body_type_analysis']['predicted_class_simple']
        print(f"{result['filename']}: {body_type}")
```

### cURL Examples
```bash
# Complete analysis
curl -X POST "http://localhost:8006/analyze-body-morphology" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "include_visualization=true"

# Classification only
curl -X POST "http://localhost:8006/classify-body-type" \
  -F "file=@body_image.jpg" \
  -F "confidence_threshold=0.6"

# With bounding box
curl -X POST "http://localhost:8006/analyze-body-morphology" \
  -F "file=@body_image.jpg" \
  -F "bbox=100,50,400,600" \
  -F "confidence_threshold=0.5"
```

## Model Files Required

- `models/lightweight_body_classifier.pth` - Trained hierarchical classification model

## Results Directory

Generated files are stored in `/app/results/`:
- `body_morphology_{uuid}.png` - Visualization images with classification results

## Error Handling

### Common Error Responses
- **400**: Invalid image format, bounding box format, or parameters
- **503**: Model not loaded or initialization failed
- **500**: Classification processing error

### Troubleshooting
1. **Model Loading Issues**: Ensure model checkpoint file exists and is valid
2. **CUDA Errors**: Check GPU availability and memory
3. **Memory Issues**: Reduce batch size or use CPU mode
4. **Poor Classification**: Ensure clear body visibility and good image quality
5. **Bounding Box Errors**: Verify bbox format as "x1,y1,x2,y2" with valid coordinates

## Performance Notes

- **Processing Time**: ~0.1-0.5 seconds per image (GPU)
- **Memory Usage**: ~1-2GB GPU memory
- **Accuracy**: 85-92% on well-captured body images
- **Throughput**: ~100-500 images/minute (depending on hardware)
- **Batch Limit**: 10 images per batch (configurable)

## Applications

### Fitness and Health
- **Body Composition Tracking**: Monitor changes over time
- **Personalized Programs**: Tailor exercise and nutrition plans
- **Progress Assessment**: Objective body type classification
- **Health Screening**: Initial morphological assessment

### Research and Clinical
- **Anthropometric Studies**: Population body type analysis
- **Clinical Assessment**: Objective body composition evaluation
- **Nutrition Research**: Metabolic type categorization
- **Exercise Science**: Training response prediction

### Commercial Applications
- **Fashion and Retail**: Size recommendation systems
- **Fitness Apps**: Personalized coaching
- **Health Platforms**: Comprehensive wellness assessment
- **Insurance**: Risk assessment applications

## Version Information

- **API Version**: 1.0.0
- **Model Version**: LightweightHierarchicalModel v1.0
- **Framework**: FastAPI + PyTorch
- **Dependencies**: torchvision, OpenCV, NumPy, PIL
- **Last Updated**: 2025

## Support

For technical support or questions about the body morphological analysis module, please refer to the main project documentation or contact the development team.
