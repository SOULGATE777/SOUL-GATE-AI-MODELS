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
from app.utils.lazy_model_loader import MultiModelLoader

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

# Initialize lazy model loader
model_loader = MultiModelLoader()

# Register models for lazy loading
def _load_facial_pipeline():
    """Lazy load function for facial analysis pipeline"""
    print("🔄 Loading facial analysis pipeline...")
    pipeline = FacialAnalysisPipeline(
        detection_model_path='/app/models/facial_landmarks_detection_model.pth',
        classification_model_path='/app/models/best_shape_classifier.pth',
        size_model_path='/app/models/best_eyebrow_size_classifier.pth',
        tag_mapping_path=None
    )
    print("✅ Facial analysis pipeline loaded successfully!")
    return pipeline

def _load_point_detector():
    """Lazy load function for anthropometric point detector"""
    print("🔄 Loading anthropometric point detector...")
    detector = AnthropometricPointDetector(
        model_path='/app/models/facial_points_detection_model.pth'
    )
    print("✅ Anthropometric point detector loaded successfully!")
    return detector

@app.on_event("startup")
async def startup_event():
    """Register models for lazy loading (no actual loading here)"""
    print("🚀 Initializing Facial Recognition API with lazy loading...")

    # Register models without loading them
    model_loader.register_model("facial_pipeline", _load_facial_pipeline)
    model_loader.register_model("point_detector", _load_point_detector)

    print("✅ Models registered for lazy loading. They will load on first request.")
    print("💾 RAM saved: Models will only load when needed!")

# Helper functions to get models
def get_facial_pipeline():
    """Get facial pipeline, loading it if necessary"""
    return model_loader.get_model("facial_pipeline")

def get_point_detector():
    """Get point detector, loading it if necessary"""
    return model_loader.get_model("point_detector")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = model_loader.get_loaded_models()
    return {
        "status": "healthy",
        "lazy_loading_enabled": True,
        "models_registered": ["facial_pipeline", "point_detector"],
        "models_currently_loaded": loaded_models,
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
        # Get models (lazy load if needed)
        facial_pipeline = get_facial_pipeline()
        point_detector = get_point_detector()

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
        # Get model (lazy load if needed)
        facial_pipeline = get_facial_pipeline()

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
        # Get model (lazy load if needed)
        point_detector = get_point_detector()

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
    # Get model (lazy load if needed)
    facial_pipeline = get_facial_pipeline()

    return {
        "shape_tags": facial_pipeline.shape_tags,
        "eyebrow_size_tags": facial_pipeline.eyebrow_size_tags,
        "eyebrow_classes": facial_pipeline.eyebrow_classes,
        "total_shape_tags": len(facial_pipeline.shape_tags),
        "total_eyebrow_size_tags": len(facial_pipeline.eyebrow_size_tags),
        "bbox_confinement_mappings": facial_pipeline.valid_shape_tags,
        "sample_shape_tags": facial_pipeline.shape_tags[:15]
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    loaded_models = model_loader.get_loaded_models()
    return {
        "message": "Facial Recognition API - Enhanced 3-Model System with Lazy Loading",
        "version": "2.0.0",
        "endpoints": {
            "analyze_face": "/analyze-face",
            "detect_landmarks": "/detect-landmarks",
            "detect_points": "/detect-points",
            "tag_mapping": "/tag-mapping",
            "health": "/health",
            "memory_stats": "/memory-stats"
        },
        "lazy_loading_enabled": True,
        "models_currently_loaded": loaded_models,
        "features": {
            "shape_classification": "45 classes with bbox confinement",
            "eyebrow_size_classification": "3 classes (ap, g, ngna) for eyebrows only",
            "bbox_confinement": "Validates predictions per landmark type",
            "anthropometric_points": "13 anatomical points",
            "lazy_loading": "Models load on first request to save RAM"
        },
        "improvements": [
            "Improved data quality",
            "Tag reduction from 52 to 45 classes",
            "Separate eyebrow size model",
            "Bbox confinement validation",
            "Lazy loading for reduced RAM usage"
        ]
    }

@app.get("/memory-stats")
async def memory_stats():
    """Get memory and model loading statistics"""
    return {
        "model_stats": model_loader.get_all_stats(),
        "loaded_models": model_loader.get_loaded_models()
    }
