from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import cv2
import numpy as np
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
import os

# Import our custom modules with correct paths
from app.models.profile_analysis_pipeline import ProfileAnalysisPipeline
from app.utils.image_processing import ImageProcessor
from app.utils.visualization import VisualizationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
pipeline = None
image_processor = None
viz_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global pipeline, image_processor, viz_manager
    
    # Startup
    try:
        logger.info("Loading Profile Morfologico models...")
        
        # Model paths
        bbox_model_path = "models/bbox_detection_model.pth"
        classifier_model_path = "models/profile_landmark_classifier_final.pth"
        point_model_path = "models/profile_aware_point_detection_model.pth"
        
        # Check if model files exist
        for model_path in [bbox_model_path, classifier_model_path, point_model_path]:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize pipeline
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline = ProfileAnalysisPipeline(
            bbox_model_path=bbox_model_path,
            classifier_model_path=classifier_model_path,
            point_model_path=point_model_path,
            device=device
        )
        
        # Initialize utilities
        image_processor = ImageProcessor()
        viz_manager = VisualizationManager("results")
        
        logger.info(f"Profile Morfologico models loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Profile Morfologico service...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Profile Morfologico Analysis API",
    description="Advanced profile morphological analysis with 3-model ensemble",
    version="1.0.0",
    lifespan=lifespan
)

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Mount static files for serving visualizations
app.mount("/visualization", StaticFiles(directory=str(RESULTS_DIR)), name="visualization")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Profile Morfologico Analysis",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-profile-morphological",
            "detect_objects": "/detect-profile-objects", 
            "classify_landmarks": "/classify-profile-landmarks",
            "detect_points": "/detect-profile-points",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    device = next(pipeline.bbox_model.parameters()).device
    
    return {
        "status": "healthy",
        "service": "Profile Morfologico Analysis",
        "models_loaded": True,
        "gpu_available": gpu_available,
        "device": str(device),
        "torch_version": torch.__version__,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze-profile-morphological")
async def analyze_profile_morphological(
    file: UploadFile = File(...),
    bbox_threshold: float = Form(0.5),
    include_visualization: bool = Form(True),
    save_results: bool = Form(True)
):
    """
    Complete profile morphological analysis pipeline
    
    Args:
        file: Input profile image
        bbox_threshold: Confidence threshold for bounding box detection
        include_visualization: Whether to generate visualization
        save_results: Whether to save results to disk
    
    Returns:
        Complete morphological analysis results (NO PROFILE PREDICTION)
    """
    global pipeline, viz_manager
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    analysis_id = str(uuid.uuid4())
    temp_path = None
    
    try:
        # Save uploaded file temporarily
        temp_path = RESULTS_DIR / f"temp_{analysis_id}.jpg"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Run analysis using the pipeline
        results = pipeline.analyze_image(
            image_path=str(temp_path),
            bbox_threshold=bbox_threshold,
            save_result=False  # We'll handle visualization ourselves
        )
        
        # Prepare response data - UPDATED TO MATCH NEW PIPELINE OUTPUT (NO PROFILE PREDICTION)
        response_data = {
            "analysis_id": analysis_id,
            "morphological_analysis": {
                "total_detected_objects": len(results["detected_objects"]),
                "total_classified_landmarks": len(results["landmark_classifications"]),
                "total_anthropometric_points": len(results["anthropometric_points"]),
                "bbox_threshold_used": bbox_threshold,
                "profile_side": results.get("profile_side", "")
            },
            "detected_objects": [
                {
                    "class": obj["class"],
                    "bbox": obj["bbox"].tolist() if hasattr(obj["bbox"], "tolist") else obj["bbox"],
                    "confidence": float(obj["confidence"])
                } for obj in results["detected_objects"]
            ],
            "landmark_classifications": [
                {
                    "original_class": cls["original_class"],
                    "classified_tag": cls["tag"],
                    "bbox": cls["bbox"].tolist() if hasattr(cls["bbox"], "tolist") else cls["bbox"],
                    "tag_confidence": float(cls["tag_confidence"]),
                    "bbox_confidence": float(cls["bbox_confidence"])
                } for cls in results["landmark_classifications"]
            ],
            "anthropometric_points": [
                {
                    "class": point["class"],
                    "coordinates": point["coordinates"].tolist() if hasattr(point["coordinates"], "tolist") else point["coordinates"],
                    "confidence": float(point["confidence"])
                } for point in results["anthropometric_points"]
            ],
            "analysis_summary": {
                "timestamp": datetime.now().isoformat(),
                "processing_successful": True,
                "filtering_applied": {
                    "duplicate_bbox_removal": True,
                    "spurious_point_filtering": True,
                    "profile_prediction_removed": True
                },
                "model_info": {
                    "bbox_classes": len(pipeline.bbox_classes),
                    "classifier_tags": len(pipeline.classifier_tags),
                    "point_classes": len(pipeline.point_classes)
                }
            }
        }
        
        # Generate visualization if requested
        if include_visualization and viz_manager:
            try:
                # Load original image for visualization
                original_image = cv2.imread(str(temp_path))
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
                # Create visualization
                viz_filename = f"profile_morphological_{analysis_id}.png"
                viz_path = RESULTS_DIR / viz_filename
                
                viz_manager.create_complete_visualization(
                    original_image=original_image,
                    results=results,
                    save_path=str(viz_path)
                )
                
                response_data["visualization_path"] = str(viz_path)
                response_data["visualization_url"] = f"/visualization/{viz_filename}"
                
            except Exception as e:
                logger.error(f"Visualization failed: {str(e)}")
                response_data["visualization_error"] = str(e)
        
        # Save detailed results if requested
        if save_results:
            results_filename = f"profile_morphological_{analysis_id}.json"
            results_path = RESULTS_DIR / results_filename
            
            with open(results_path, 'w') as f:
                json.dump(response_data, f, indent=2)
            
            response_data["results_saved_path"] = str(results_path)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()

@app.post("/detect-profile-objects")
async def detect_profile_objects(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """Detect profile objects using bounding box model"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    analysis_id = str(uuid.uuid4())
    temp_path = None
    
    try:
        # Save uploaded file temporarily
        temp_path = RESULTS_DIR / f"temp_bbox_{analysis_id}.jpg"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Preprocess image
        original_image, image_tensor = pipeline.preprocess_image(str(temp_path))
        
        # Detect objects
        detected_objects = pipeline.detect_bboxes(image_tensor, confidence_threshold)
        
        # Prepare response
        response_data = {
            "analysis_id": analysis_id,
            "detection_summary": {
                "total_detections": len(detected_objects),
                "confidence_threshold": confidence_threshold,
                "image_shape": list(original_image.shape),
                "duplicate_removal_applied": True
            },
            "detected_objects": [
                {
                    "class": obj["class"],
                    "bbox": obj["bbox"].tolist() if hasattr(obj["bbox"], "tolist") else obj["bbox"],
                    "confidence": float(obj["confidence"])
                } for obj in detected_objects
            ]
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Object detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()

@app.post("/classify-profile-landmarks")
async def classify_profile_landmarks(
    file: UploadFile = File(...),
    bbox_threshold: float = Form(0.5)
):
    """Classify profile landmarks from detected objects"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    analysis_id = str(uuid.uuid4())
    temp_path = None
    
    try:
        # Save uploaded file temporarily
        temp_path = RESULTS_DIR / f"temp_classify_{analysis_id}.jpg"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Preprocess image
        original_image, image_tensor = pipeline.preprocess_image(str(temp_path))
        
        # Detect objects first
        detected_objects = pipeline.detect_bboxes(image_tensor, bbox_threshold)
        
        # Classify landmarks
        classifications = pipeline.classify_landmarks(original_image, detected_objects)
        
        # Prepare response
        response_data = {
            "analysis_id": analysis_id,
            "classification_summary": {
                "total_objects_detected": len(detected_objects),
                "total_classifications": len(classifications),
                "bbox_threshold": bbox_threshold,
                "available_tags": pipeline.classifier_tags,
                "duplicate_removal_applied": True
            },
            "landmark_classifications": [
                {
                    "original_class": cls["original_class"],
                    "classified_tag": cls["tag"],
                    "bbox": cls["bbox"].tolist() if hasattr(cls["bbox"], "tolist") else cls["bbox"],
                    "tag_confidence": float(cls["tag_confidence"]),
                    "bbox_confidence": float(cls["bbox_confidence"])
                } for cls in classifications
            ]
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Landmark classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Landmark classification failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()

@app.post("/detect-profile-points")
async def detect_profile_points(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.15)
):
    """Detect anthropometric points in profile"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    analysis_id = str(uuid.uuid4())
    temp_path = None
    
    try:
        # Save uploaded file temporarily
        temp_path = RESULTS_DIR / f"temp_points_{analysis_id}.jpg"
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Preprocess image
        original_image, image_tensor = pipeline.preprocess_image(str(temp_path))
        
        # Detect points (NO PROFILE PREDICTION RETURNED)
        detected_points = pipeline.detect_points(image_tensor)
        
        # Prepare response
        response_data = {
            "analysis_id": analysis_id,
            "point_detection_summary": {
                "total_points_detected": len(detected_points),
                "confidence_threshold": confidence_threshold,
                "available_point_classes": pipeline.point_classes,
                "spurious_filtering_applied": True
            },
            "anthropometric_points": [
                {
                    "class": point["class"],
                    "coordinates": point["coordinates"].tolist() if hasattr(point["coordinates"], "tolist") else point["coordinates"],
                    "confidence": float(point["confidence"])
                } for point in detected_points
            ]
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Point detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Point detection failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization files"""
    file_path = RESULTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename
    )

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "bbox_model": {
            "classes": pipeline.bbox_classes,
            "num_classes": len(pipeline.bbox_classes)
        },
        "classifier_model": {
            "tags": pipeline.classifier_tags,
            "num_tags": len(pipeline.classifier_tags)
        },
        "point_model": {
            "classes": pipeline.point_classes,
            "num_classes": len(pipeline.point_classes),
            "heatmap_size": getattr(pipeline, 'heatmap_size', 112)
        },
        "excluded_classes": ["cabello_tapando_frente", "cabello_tapando_oreja", "objeto"],
        "device": str(next(pipeline.bbox_model.parameters()).device),
        "filtering_features": {
            "duplicate_bbox_removal": True,
            "spurious_point_filtering": True,
            "profile_prediction_removed": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

