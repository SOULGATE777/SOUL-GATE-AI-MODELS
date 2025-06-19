from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from typing import Optional
import logging

from .models.facial_validation_pipeline import FacialValidationPipeline
from .utils.image_processing import save_uploaded_file, cleanup_temp_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SG Validacion - Facial Feature Detection",
    description="Advanced facial feature detection using YOLO ensemble",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline
try:
    pipeline = FacialValidationPipeline()
    logger.info("✅ Facial validation pipeline initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize pipeline: {e}")
    pipeline = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SG Validacion - Facial Feature Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": [
            "/health",
            "/analyze-validation",
            "/detect-features",
            "/docs",
            "/redoc"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "model_loaded": pipeline.model is not None if pipeline else False,
        "gpu_available": pipeline.device.type == 'cuda' if pipeline else False,
        "service": "validacion"
    }

@app.post("/analyze-validation")
async def analyze_validation(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.20),
    include_visualization: bool = Form(True)
):
    """
    Complete facial validation analysis
    
    Args:
        file: Input image file
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        include_visualization: Whether to generate visualization
    
    Returns:
        JSON response with detection results and analysis
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = await save_uploaded_file(file)
        
        # Run analysis
        results = await pipeline.analyze_complete(
            image_path=temp_path,
            confidence_threshold=confidence_threshold,
            include_visualization=include_visualization
        )
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in validation analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_path:
            cleanup_temp_files(temp_path)

@app.post("/detect-features")
async def detect_features(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.20)
):
    """
    Detect facial features only (no visualization)
    
    Args:
        file: Input image file
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        JSON response with feature detections
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = await save_uploaded_file(file)
        
        # Run detection only
        results = await pipeline.detect_features(
            image_path=temp_path,
            confidence_threshold=confidence_threshold
        )
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in feature detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_path:
            cleanup_temp_files(temp_path)

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Get visualization result file"""
    results_dir = "/app/results"
    file_path = os.path.join(results_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )
