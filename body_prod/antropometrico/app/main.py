from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
import numpy as np
from PIL import Image
import io
import logging
from typing import Optional, List
import asyncio
from contextlib import asynccontextmanager

from app.models.anthropometric_pipeline import AnthropometricAnalysisPipeline
from app.utils.image_processing import validate_image, preprocess_image
from app.utils.visualization import create_anthropometric_visualization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup"""
    global pipeline
    try:
        logger.info("🚀 Initializing Body Anthropometric Analysis Pipeline...")
        pipeline = AnthropometricAnalysisPipeline()
        await asyncio.to_thread(pipeline.load_model)
        logger.info("✅ Pipeline loaded successfully!")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down pipeline...")

# Create FastAPI app
app = FastAPI(
    title="Body Anthropometric Analysis API",
    description="Production-ready body anthropometric measurements and skull detection service",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving visualizations
os.makedirs("/app/results", exist_ok=True)
app.mount("/visualization", StaticFiles(directory="/app/results"), name="visualization")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Body Anthropometric Analysis API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "analyze": "/analyze-body-anthropometry",
            "skull_detection": "/detect-skull-measurements",
            "pose_detection": "/detect-pose-keypoints",
            "batch_analyze": "/batch-analyze",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "status": "healthy",
        "service": "body-anthropometric-analysis",
        "model_loaded": pipeline.model is not None,
        "device": str(pipeline.device),
        "model_type": "YOLOv8n-pose",
        "keypoints_supported": 17
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "model_architecture": "YOLOv8n-pose",
        "model_type": "pose_detection",
        "input_format": "RGB image",
        "keypoints_detected": 17,
        "keypoint_names": pipeline.keypoint_names,
        "body_parts": pipeline.body_parts,
        "measurements_provided": [
            "skull_dimensions",
            "body_proportions",
            "skull_to_body_ratio",
            "anatomical_assessment",
            "head_orientation"
        ],
        "device": str(pipeline.device),
        "model_path": "/app/models/yolov8n-pose.pt"
    }

@app.post("/analyze-body-anthropometry")
async def analyze_body_anthropometry(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    include_visualization: bool = Form(True),
    detailed_analysis: bool = Form(True)
):
    """
    Complete body anthropometric analysis including skull detection and body proportions
    
    Args:
        file: Image file (JPG, PNG, etc.)
        confidence_threshold: Minimum confidence for keypoint detection (0.0-1.0)
        include_visualization: Whether to generate visualization
        detailed_analysis: Whether to include detailed anatomical analysis
    
    Returns:
        Complete anthropometric analysis results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate image
        image_array = await validate_image(file)
        
        # Save temporary image for processing
        temp_id = str(uuid.uuid4())
        temp_path = f"/app/temp/temp_{temp_id}.jpg"
        os.makedirs("/app/temp", exist_ok=True)
        
        # Save image temporarily
        image_pil = Image.fromarray(image_array)
        image_pil.save(temp_path, "JPEG")
        
        # Run complete anthropometric analysis
        results = await asyncio.to_thread(
            pipeline.analyze_body_anthropometry,
            temp_path,
            confidence_threshold,
            detailed_analysis
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        results["analysis_id"] = analysis_id
        
        # Create visualization if requested
        if include_visualization:
            viz_path = await asyncio.to_thread(
                create_anthropometric_visualization,
                temp_path,
                results,
                f"/app/results/anthropometry_{analysis_id}.png"
            )
            
            if viz_path:
                results["visualization_path"] = viz_path
                results["visualization_url"] = f"/visualization/anthropometry_{analysis_id}.png"
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in anthropometric analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-skull-measurements")
async def detect_skull_measurements(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    include_contour_refinement: bool = Form(True)
):
    """
    Skull detection and measurement analysis only
    
    Args:
        file: Image file
        confidence_threshold: Minimum confidence for keypoint detection
        include_contour_refinement: Whether to use contour refinement for skull detection
    
    Returns:
        Skull detection and measurement results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate image
        image_array = await validate_image(file)
        
        # Save temporary image for processing
        temp_id = str(uuid.uuid4())
        temp_path = f"/app/temp/temp_{temp_id}.jpg"
        os.makedirs("/app/temp", exist_ok=True)
        
        # Save image temporarily
        image_pil = Image.fromarray(image_array)
        image_pil.save(temp_path, "JPEG")
        
        # Run skull detection analysis
        results = await asyncio.to_thread(
            pipeline.detect_skull_measurements_only,
            temp_path,
            confidence_threshold,
            include_contour_refinement
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Skull detection failed")
        
        # Add analysis metadata
        results["analysis_id"] = str(uuid.uuid4())
        results["analysis_type"] = "skull_detection_only"
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in skull detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-pose-keypoints")
async def detect_pose_keypoints(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Basic pose detection and keypoint extraction
    
    Args:
        file: Image file
        confidence_threshold: Minimum confidence for keypoint detection
    
    Returns:
        Pose keypoints and body part groupings
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate image
        image_array = await validate_image(file)
        
        # Save temporary image for processing
        temp_id = str(uuid.uuid4())
        temp_path = f"/app/temp/temp_{temp_id}.jpg"
        os.makedirs("/app/temp", exist_ok=True)
        
        # Save image temporarily
        image_pil = Image.fromarray(image_array)
        image_pil.save(temp_path, "JPEG")
        
        # Run pose detection only
        results = await asyncio.to_thread(
            pipeline.detect_pose_only,
            temp_path,
            confidence_threshold
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Pose detection failed")
        
        # Add analysis metadata
        results["analysis_id"] = str(uuid.uuid4())
        results["analysis_type"] = "pose_detection_only"
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in pose detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-analyze")
async def batch_analyze(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5),
    detailed_analysis: bool = Form(False)
):
    """
    Batch anthropometric analysis for multiple images
    
    Args:
        files: List of image files
        confidence_threshold: Minimum confidence for keypoint detection
        detailed_analysis: Whether to include detailed analysis
    
    Returns:
        List of analysis results for each image
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if len(files) > 5:  # Limit batch size for anthropometric analysis
        raise HTTPException(status_code=400, detail="Maximum 5 images per batch")
    
    try:
        results = []
        for i, file in enumerate(files):
            try:
                # Validate image
                image_array = await validate_image(file)
                
                # Save temporary image
                temp_id = str(uuid.uuid4())
                temp_path = f"/app/temp/temp_{temp_id}.jpg"
                os.makedirs("/app/temp", exist_ok=True)
                
                image_pil = Image.fromarray(image_array)
                image_pil.save(temp_path, "JPEG")
                
                # Run analysis
                result = await asyncio.to_thread(
                    pipeline.analyze_body_anthropometry,
                    temp_path,
                    confidence_threshold,
                    detailed_analysis
                )
                
                if result:
                    result["analysis_id"] = str(uuid.uuid4())
                    result["batch_index"] = i
                    result["filename"] = file.filename
                    results.append(result)
                else:
                    results.append({
                        "analysis_id": str(uuid.uuid4()),
                        "batch_index": i,
                        "filename": file.filename,
                        "error": "Analysis failed"
                    })
                
                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            except Exception as e:
                results.append({
                    "analysis_id": str(uuid.uuid4()),
                    "batch_index": i,
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return JSONResponse(content={
            "batch_id": str(uuid.uuid4()),
            "total_images": len(files),
            "successful_analyses": len([r for r in results if "error" not in r]),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization files"""
    file_path = f"/app/results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8007,
        reload=False,
        workers=1
    )
