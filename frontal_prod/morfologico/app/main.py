from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import uuid
from typing import List, Optional
import json
import aiofiles
from pathlib import Path

from app.models.facial_analysis_pipeline import FacialAnalysisPipeline
from app.models.anthropometric_detection import AnthropometricPointDetector
from app.utils.visualization import create_beautiful_visualization
from app.utils.image_processing import process_uploaded_image

# Initialize FastAPI app
app = FastAPI(
    title="Facial Recognition API",
    description="Complete facial analysis with landmark detection, classification, and anthropometric points",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
facial_pipeline = None
point_detector = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global facial_pipeline, point_detector

    try:
        print("Loading facial analysis pipeline...")
        # Initialize the pipeline with 3 models: detection + shape + eyebrow size
        facial_pipeline = FacialAnalysisPipeline(
            detection_model_path='/app/models/facial_landmarks_detection_model.pth',
            classification_model_path='/app/models/best_shape_classifier.pth',
            size_model_path='/app/models/best_eyebrow_size_classifier.pth',  # NEW: Eyebrow size model
            tag_mapping_path=None
        )

        print("Loading anthropometric point detector...")
        point_detector = AnthropometricPointDetector(
            model_path='/app/models/facial_points_detection_model.pth'
        )

        print("All models loaded successfully!")

        # Validate tag consistency
        print("=== MODEL CONFIGURATION ===")
        print(f"Detection classes: {len(facial_pipeline.landmark_classes)}")
        print(f"Shape tags: {len(facial_pipeline.shape_tags)} classes")
        print(f"Eyebrow size tags: {len(facial_pipeline.eyebrow_size_tags)} classes (for {', '.join(facial_pipeline.eyebrow_classes)})")
        print(f"Bbox confinement enabled: {len(facial_pipeline.valid_shape_tags)} landmark types")
        print("\nSample shape tags:")
        for i, tag in enumerate(facial_pipeline.shape_tags[:10]):
            print(f"  {i}: {tag}")
        print(f"\nEyebrow size tags: {', '.join(facial_pipeline.eyebrow_size_tags)}")
        print("=== CONFIGURATION COMPLETE ===")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": facial_pipeline is not None and point_detector is not None,
        "shape_tags": len(facial_pipeline.shape_tags) if facial_pipeline else 0,
        "eyebrow_size_tags": len(facial_pipeline.eyebrow_size_tags) if facial_pipeline else 0,
        "eyebrow_size_model_loaded": facial_pipeline.size_model is not None if facial_pipeline else False,
        "bbox_confinement_enabled": True
    }

@app.post("/analyze-face")
async def analyze_face(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    include_visualization: bool = True
):
    """
    Complete facial analysis with all three models:
    1. Facial landmark detection
    2. Characteristic classification
    3. Anthropometric point detection
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process uploaded image
        image_array, temp_path = await process_uploaded_image(file)
        
        # Run facial landmark detection and classification
        facial_results, facial_viz = facial_pipeline.process_image(
            image_array,
            confidence_threshold=confidence_threshold,
            display=False
        )
        
        # REMOVED: No need to add tag names here - they're already included in process_image
        # The pipeline now handles this internally with the same logic as Jupyter
        
        # Debug: Print first few results for validation
        print("=== FACIAL ANALYSIS RESULTS ===")
        for i, result in enumerate(facial_results[:3]):  # Show first 3
            print(f"Result {i+1}:")
            print(f"  Landmark: {result['landmark_class']}")
            print(f"  Tag: {result['tag']}")
            print(f"  Tag Name: {result['tag_name']}")
            print(f"  Score: {result['score']:.3f}")
        
        # Run anthropometric point detection
        point_results = point_detector.detect_points(
            image_array,
            confidence_threshold=confidence_threshold
        )

        # Separate eyebrow size results from shape results
        eyebrow_size_results = []
        for result in facial_results:
            if 'size_tag' in result:
                eyebrow_size_results.append({
                    'landmark_class': result['landmark_class'],
                    'size_tag': result['size_tag'],
                    'size_confidence': result['size_confidence'],
                    'box': result['box']
                })

        # Create beautiful combined visualization
        if include_visualization:
            viz_path = await create_beautiful_visualization(
                image_array, facial_results, point_results
            )
        else:
            viz_path = None

        # Prepare response
        response = {
            "facial_landmarks": {
                "count": len(facial_results),
                "detections": facial_results
            },
            "eyebrow_size_classification": {
                "count": len(eyebrow_size_results),
                "detections": eyebrow_size_results,
                "classes": facial_pipeline.eyebrow_size_tags if facial_pipeline else []
            },
            "anthropometric_points": {
                "count": len(point_results),
                "detections": point_results
            },
            "summary": {
                "total_detections": len(facial_results) + len(point_results),
                "eyebrow_size_detections": len(eyebrow_size_results),
                "confidence_threshold": confidence_threshold,
                "image_processed": True,
                "bbox_confinement_applied": True
            }
        }

        if viz_path:
            response["visualization_path"] = viz_path
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect-landmarks")
async def detect_landmarks(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    include_visualization: bool = True
):
    """Detect and classify facial landmarks only"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_array, temp_path = await process_uploaded_image(file)
        
        results, viz_image = facial_pipeline.process_image(
            image_array,
            confidence_threshold=confidence_threshold,
            display=False
        )
        
        # REMOVED: No need to add tag names - they're already included
        # Tag names are now handled internally in the pipeline
        
        # Debug output
        print(f"Detected {len(results)} landmarks")
        for i, result in enumerate(results[:2]):  # Show first 2
            print(f"  {i+1}. {result['landmark_class']}: {result['tag']} ({result['tag_name']}) (score: {result['score']:.2f})")
        
        # Save visualization if requested
        viz_path = None
        if include_visualization:
            viz_filename = f"landmarks_{uuid.uuid4().hex}.jpg"
            viz_path = f"/app/results/{viz_filename}"
            os.makedirs("/app/results", exist_ok=True)
            cv2.imwrite(viz_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        response = {
            "landmarks": results,
            "count": len(results),
            "confidence_threshold": confidence_threshold
        }
        
        if viz_path:
            response["visualization_path"] = viz_path
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect-points")
async def detect_anthropometric_points(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    include_visualization: bool = True
):
    """Detect anthropometric points only"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_array, temp_path = await process_uploaded_image(file)
        
        results = point_detector.detect_points(
            image_array,
            confidence_threshold=confidence_threshold
        )
        
        # Create visualization if requested
        viz_path = None
        if include_visualization:
            viz_path = await point_detector.create_visualization(
                image_array, results
            )
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        response = {
            "points": results,
            "count": len(results),
            "confidence_threshold": confidence_threshold
        }
        
        if viz_path:
            response["visualization_path"] = viz_path
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization images"""
    file_path = f"/app/results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Visualization not found")

@app.get("/tag-mapping")
async def get_tag_mapping():
    """Get the current tag mapping for debugging"""
    if facial_pipeline:
        return {
            "shape_tags": facial_pipeline.shape_tags,
            "eyebrow_size_tags": facial_pipeline.eyebrow_size_tags,
            "eyebrow_classes": facial_pipeline.eyebrow_classes,
            "total_shape_tags": len(facial_pipeline.shape_tags),
            "total_eyebrow_size_tags": len(facial_pipeline.eyebrow_size_tags),
            "bbox_confinement_mappings": facial_pipeline.valid_shape_tags,
            "sample_shape_tags": facial_pipeline.shape_tags[:15]
        }
    else:
        raise HTTPException(status_code=503, detail="Models not loaded")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Facial Recognition API - Enhanced 3-Model System",
        "version": "2.0.0",
        "endpoints": {
            "analyze_face": "/analyze-face",
            "detect_landmarks": "/detect-landmarks",
            "detect_points": "/detect-points",
            "tag_mapping": "/tag-mapping",
            "health": "/health"
        },
        "models_loaded": facial_pipeline is not None and point_detector is not None,
        "features": {
            "shape_classification": "45 classes with bbox confinement",
            "eyebrow_size_classification": "3 classes (ap, g, ngna) for eyebrows only",
            "bbox_confinement": "Validates predictions per landmark type",
            "anthropometric_points": "13 anatomical points"
        },
        "improvements": [
            "Improved data quality",
            "Tag reduction from 52 to 45 classes",
            "Separate eyebrow size model",
            "Bbox confinement validation"
        ]
    }
