# Migration Guide: V1 to V2 (3-Model System)

## Overview of Changes

The Frontal Morphological Analysis system has been upgraded from a 2-model to a **3-model ensemble** with improved data quality and bbox confinement validation.

### Key Improvements

1. **Tag Reduction**: 52 → 45 shape classification tags (removed problematic tags)
2. **Separate Eyebrow Size Model**: New dedicated model for eyebrow size (3 classes: ap, g, ngna)
3. **Bbox Confinement**: Shape predictions are validated per landmark type
4. **Improved Data Quality**: Models retrained with cleaned dataset

## Model Changes

### Old System (V1)
- **Detection Model**: Faster R-CNN (23 landmark classes)
- **Classification Model**: Single CNN (52 tags for all landmarks)
- **Total Models**: 2

### New System (V2)
- **Detection Model**: Faster R-CNN (23 landmark classes) - *unchanged*
- **Shape Classification Model**: CNN (45 tags with bbox confinement)
- **Eyebrow Size Model**: Enhanced CNN (3 tags, eyebrows only: ap, g, ngna)
- **Total Models**: 3

## Required Model Files

Place these files in `/app/models/`:

1. `facial_landmarks_detection_model.pth` - Detection model (unchanged)
2. `best_shape_classifier.pth` - **NEW** Shape classification model (45 classes)
3. `best_eyebrow_size_classifier.pth` - **NEW** Eyebrow size model (3 classes)
4. `facial_points_detection_model.pth` - Anthropometric points (unchanged)

## API Response Changes

### `/analyze-face` Response Structure

**New fields added:**

```json
{
  "facial_landmarks": {
    "count": 12,
    "detections": [
      {
        "landmark_class": "cj_d",
        "tag": "tag_38",
        "tag_name": "rc",
        "score": 0.92,
        "tag_confidence": 0.87,
        "size_tag": "g",              // NEW: Only for eyebrows
        "size_confidence": 0.91,      // NEW: Only for eyebrows
        "top_tags": [...],
        "box": [145.2, 78.4, 165.8, 98.6]
      }
    ]
  },
  "eyebrow_size_classification": {   // NEW: Separate table
    "count": 2,
    "detections": [
      {
        "landmark_class": "cj_d",
        "size_tag": "g",
        "size_confidence": 0.91,
        "box": [145.2, 78.4, 165.8, 98.6]
      },
      {
        "landmark_class": "cj_i",
        "size_tag": "ap",
        "size_confidence": 0.88,
        "box": [145.2, 78.4, 165.8, 98.6]
      }
    ],
    "classes": ["ap", "g", "ngna"]
  },
  "summary": {
    "bbox_confinement_applied": true,  // NEW
    "eyebrow_size_detections": 2       // NEW
  }
}
```

## Shape Tags Changes

### Removed Tags (7 total)
- `abierta` - Removed
- `adelgazamiento` - Removed
- `bigote` - Removed
- `carnosos` - Removed
- `cabellos_sueltos` - Removed
- `fleco` - Removed
- `sonriendo` - Removed

### New Shape Tags (45 total)
```python
[
    '0', '1', '2', '3', 'a_n', 'ab', 'al', 'ar',
    'crl', 'cv', 'delgada', 'el', 'fr', 'grueso', 'h', 'hn', 'i',
    'lineas_sonriza', 'lineas_verticales', 'll', 'lunar', 'md', 'md_a',
    'mercurial', 'nd', 'normal', 'nrml', 'nt', 'on', 'pc', 'pg', 'pl',
    'planos', 'pliegue', 'pm', 'pn', 'ptosis', 'pursed', 'rc', 'rd',
    'salido', 'sl', 'solar', 'sp_sl', 'uniceja'
]
```

## Bbox Confinement Mappings

The system now validates that shape predictions match the landmark type:

```python
valid_shape_tags = {
    'cj_d': ['rc', 'el', 'cv'],           # Right eyebrow shapes only
    'cj_i': ['rc', 'el', 'cv'],           # Left eyebrow shapes only
    'nariz': ['delgada', 'nrml', 'grueso'],  # Nose shapes only
    'bc': ['lunar', 'mercurial', 'pursed', 'solar'],  # Mouth shapes only
    'n': ['i', 'pn', 'rd'],               # Nostril shapes only
    'oj_d': ['al', 'crl', 'fr', 'md', 'md_a'],  # Right eye shapes only
    'oj_i': ['al', 'crl', 'fr', 'md', 'md_a'],  # Left eye shapes only
    'entrecejo': ['lineas_verticales', 'normal', 'uniceja'],
    'parpado_dr': ['pliegue', 'ptosis'],
    'parpado_i': ['pliegue', 'ptosis'],
    # ... etc
}
```

**How it works**: If the model predicts an invalid tag for a landmark (e.g., 'grueso' for an eyebrow), the system automatically selects the highest-confidence valid tag from the allowed list.

## Eyebrow Size Classification

**Only applies to**: `cj_d` (right eyebrow) and `cj_i` (left eyebrow)

**3 Size Classes**:
- `ap` - Apretadas (narrow/thin)
- `g` - Gruesas (thick/wide)
- `ngna` - Normal/intermediate

This classification is **separate** from shape classification, providing two independent characteristics per eyebrow.

## Migration Steps

1. **Place new model files** in `/app/models/`
   - `best_shape_classifier.pth`
   - `best_eyebrow_size_classifier.pth`

2. **Rebuild Docker container**:
   ```bash
   docker-compose down
   docker-compose build
   docker-compose up -d
   ```

3. **Verify models loaded**:
   ```bash
   curl http://localhost:8005/health
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "models_loaded": true,
     "shape_tags": 45,
     "eyebrow_size_tags": 3,
     "eyebrow_size_model_loaded": true,
     "bbox_confinement_enabled": true
   }
   ```

4. **Test the updated API**:
   ```bash
   curl -X POST "http://localhost:8005/analyze-face" \
     -F "file=@test_face.jpg" \
     -F "confidence_threshold=0.5"
   ```

5. **Check tag mappings**:
   ```bash
   curl http://localhost:8005/tag-mapping
   ```

## Backward Compatibility

- Existing clients will still receive `tag` and `tag_name` fields
- New `size_tag` and `size_confidence` fields are **optional** (only present for eyebrows)
- The `eyebrow_size_classification` section is a **new addition** to the response
- Old visualization endpoints remain unchanged

## Benefits of V2

1. **Higher Accuracy**: Bbox confinement prevents cross-landmark confusion
2. **Cleaner Data**: Removed problematic tags that caused training issues
3. **Specialized Eyebrow Analysis**: Dedicated model for eyebrow size characteristics
4. **Better Validation**: Predictions are validated against landmark-specific constraints
5. **More Interpretable**: Separate shape and size classifications for eyebrows

## Troubleshooting

### Model Loading Errors
- Ensure all 3 model files are present in `/app/models/`
- Check file permissions (readable by Docker container)
- Verify model files are not corrupted

### Missing size_tag in Response
- This is normal for non-eyebrow landmarks
- Only `cj_d` and `cj_i` will have `size_tag` and `size_confidence`

### Unexpected tag_name Values
- Check `/tag-mapping` endpoint for current mappings
- Verify you're using the V2 shape classifier (45 classes)
- Bbox confinement may have corrected an invalid prediction

## Support

For issues or questions about the migration, refer to the main README.md or contact the development team.
