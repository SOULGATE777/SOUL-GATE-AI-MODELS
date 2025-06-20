from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uuid
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

from app.models.profile_validation_pipeline import ProfileValidationPipeline
from app.utils.image_processing import ImageProcessor
from app.utils.visualization import ProfileValidationVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Profile Validation API",
    description="Advanced profile image validation with occlusion detection and quality assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
RESULTS_DIR = Path("/app/results")
MODELS_DIR = Path("/app/models")
RESULTS_DIR.mkdir(exist_ok=True)

# Mount static files for serving visualizations
app.mount("/visualization", StaticFiles(directory=str(RESULTS_DIR)), name="visualization")

# Global variables for model and utilities
pipeline = None
image_processor = None
visualizer = None

def get_pipeline():
    """Dependency to get the validation pipeline"""
    global pipeline
    if pipeline is None:
        model_path = MODELS_DIR / "occlusion_detection_model.pth"
        if not model_path.exists():
            raise HTTPException(
                status_code=500, 
                detail=f"Model file not found at {model_path}"
            )
        try:
            pipeline = ProfileValidationPipeline(str(model_path))
            logger.info("Profile validation pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {e}")
    return pipeline

def get_image_processor():
    """Dependency to get the image processor"""
    global image_processor
    if image_processor is None:
        image_processor = ImageProcessor()
    return image_processor

def get_visualizer():
    """Dependency to get the visualizer"""
    global visualizer
    if visualizer is None:
        visualizer = ProfileValidationVisualizer()
    return visualizer

@app.on_event("startup")
async def startup_event():
    """Initialize the model and utilities on startup"""
    try:
        get_pipeline()
        get_image_processor()
        get_visualizer()
        logger.info("Profile validation API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "profile-validation",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": pipeline is not None
    }

@app.post("/analyze-profile-validation")
async def analyze_profile_validation(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    include_visualization: bool = Form(True),
    pipeline: ProfileValidationPipeline = Depends(get_pipeline),
    processor: ImageProcessor = Depends(get_image_processor),
    visualizer: ProfileValidationVisualizer = Depends(get_visualizer)
):
    """
    Complete profile validation analysis with occlusion detection and quality assessment
    
    Args:
        file: Profile image file
        confidence_threshold: Minimum confidence threshold (0.0-1.0)
        include_visualization: Whether to generate visualization
        
    Returns:
        Complete validation analysis results
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise HTTPException(
                status_code=400, 
                detail="Confidence threshold must be between 0.0 and 1.0"
            )
        
        # Read and process image
        image_data = await file.read()
        image = processor.read_image_from_bytes(image_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Validate image dimensions
        if not processor.validate_image_dimensions(image):
            raise HTTPException(
                status_code=400, 
                detail="Image dimensions too small. Minimum size: 224x224 pixels"
            )
        
        # Run validation analysis
        results = pipeline.analyze_profile_validation(image, confidence_threshold)
        
        # Add timestamp
        results['analysis_summary']['timestamp'] = datetime.utcnow().isoformat()
        
        # Generate visualization if requested
        visualization_path = None
        visualization_url = None
        
        if include_visualization and results['analysis_summary']['processing_successful']:
            try:
                viz_filename = f"profile_validation_{results['analysis_id']}.png"
                visualization_path = RESULTS_DIR / viz_filename
                
                visualizer.create_validation_visualization(
                    image=image,
                    results=results,
                    save_path=str(visualization_path)
                )
                
                visualization_url = f"/visualization/{viz_filename}"
                logger.info(f"Visualization saved to {visualization_path}")
                
            except Exception as viz_error:
                logger.warning(f"Failed to create visualization: {viz_error}")
                # Don't fail the entire request for visualization errors
        
        # Add visualization info to results
        if visualization_path:
            results['visualization_path'] = str(visualization_path)
            results['visualization_url'] = visualization_url
        
        # Log analysis summary
        logger.info(f"Profile validation completed - ID: {results['analysis_id']}, "
                   f"Suitable: {results['validation_status']['is_suitable']}, "
                   f"Score: {results['validation_status']['overall_score']}")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in profile validation analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/detect-profile-occlusions")
async def detect_profile_occlusions(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    pipeline: ProfileValidationPipeline = Depends(get_pipeline),
    processor: ImageProcessor = Depends(get_image_processor)
):
    """
    Detect occlusions in profile images only (without full quality assessment)
    
    Args:
        file: Profile image file
        confidence_threshold: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        Occlusion detection results
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise HTTPException(
                status_code=400, 
                detail="Confidence threshold must be between 0.0 and 1.0"
            )
        
        # Read and process image
        image_data = await file.read()
        image = processor.read_image_from_bytes(image_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run occlusion detection
        results = pipeline.detect_occlusions_only(image, confidence_threshold)
        
        # Add timestamp
        results['timestamp'] = datetime.utcnow().isoformat()
        
        # Log results
        logger.info(f"Occlusion detection completed - ID: {results['analysis_id']}, "
                   f"Occlusions found: {results['has_occlusions']}, "
                   f"Total detections: {results['total_detections']}")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in occlusion detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/assess-profile-quality")
async def assess_profile_quality(
    file: UploadFile = File(...),
    processor: ImageProcessor = Depends(get_image_processor)
):
    """
    Assess profile image quality without occlusion detection
    
    Args:
        file: Profile image file
        
    Returns:
        Image quality assessment results
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read and process image
        image_data = await file.read()
        image = processor.read_image_from_bytes(image_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get a temporary pipeline instance for quality assessment
        temp_pipeline = get_pipeline()
        quality_assessment = temp_pipeline._assess_image_quality(image)
        
        # Add analysis metadata
        analysis_id = str(uuid.uuid4())
        results = {
            'analysis_id': analysis_id,
            'quality_assessment': quality_assessment,
            'timestamp': datetime.utcnow().isoformat(),
            'processing_successful': True
        }
        
        # Log results
        logger.info(f"Quality assessment completed - ID: {analysis_id}, "
                   f"Score: {quality_assessment['quality_score']}, "
                   f"Suitable: {quality_assessment['is_suitable']}")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quality assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@app.get("/model-info")
async def get_model_info(pipeline: ProfileValidationPipeline = Depends(get_pipeline)):
    """Get information about the loaded model"""
    return {
        "model_classes": pipeline.all_classes,
        "included_classes": pipeline.included_classes,
        "num_classes": pipeline.num_classes,
        "device": str(pipeline.device),
        "class_thresholds": pipeline.class_thresholds,
        "quality_criteria": pipeline.quality_criteria
    }

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization files"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(file_path)

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
