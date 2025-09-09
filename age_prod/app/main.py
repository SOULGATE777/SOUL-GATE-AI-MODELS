from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from typing import List, Optional
import asyncio

from app.models.age_estimation_pipeline import AgeEstimationPipeline
from app.utils.image_processing import (
    save_uploaded_file, cleanup_temp_files, 
    validate_image_format, ensure_results_directory,
    save_multiple_uploaded_files
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SG Age Estimation - AI Age Prediction",
    description="Advanced age estimation using InsightFace for frontal and profile faces",
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

# Ensure results directory exists
ensure_results_directory()

# Initialize the pipeline
try:
    pipeline = AgeEstimationPipeline()
    logger.info("✅ Age estimation pipeline initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize pipeline: {e}")
    pipeline = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SG Age Estimation - AI Age Prediction API",
        "version": "1.0.0",
        "status": "active",
        "model": "InsightFace",
        "endpoints": [
            "/health",
            "/estimate-age",
            "/batch-estimate-ages", 
            "/model-info",
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
        "model_initialized": pipeline.is_initialized if pipeline else False,
        "gpu_available": str(pipeline.device).startswith('cuda') if pipeline else False,
        "device": str(pipeline.device) if pipeline else "unknown",
        "service": "age_estimation",
        "model_type": "InsightFace"
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return pipeline.get_model_info()

@app.post("/estimate-age")
async def estimate_age(
    file: UploadFile = File(...),
    include_visualization: bool = Form(True)
):
    """
    Estimate age from a single face image
    
    Args:
        file: Input image file (frontal or profile face)
        include_visualization: Whether to generate visualization
    
    Returns:
        JSON response with age estimation results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not validate_image_format(file.filename):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported image format. Supported: jpg, jpeg, png, bmp, tiff"
        )
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = await save_uploaded_file(file)
        
        # Run age estimation
        results = await pipeline.estimate_age(
            image_path=temp_path,
            include_visualization=include_visualization
        )
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in age estimation: {e}")
        raise HTTPException(status_code=500, detail=f"Age estimation failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if temp_path:
            cleanup_temp_files(temp_path)

@app.post("/batch-estimate-ages")
async def batch_estimate_ages(
    files: List[UploadFile] = File(...),
    include_visualization: bool = Form(True)
):
    """
    Estimate ages from multiple face images
    
    Args:
        files: List of input image files
        include_visualization: Whether to generate visualizations
    
    Returns:
        JSON response with batch age estimation results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate number of files
    if len(files) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed per batch")
    
    # Validate all files
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"All files must be images: {file.filename}")
        
        if not validate_image_format(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format for {file.filename}. Supported: jpg, jpeg, png, bmp, tiff"
            )
    
    temp_paths = []
    try:
        # Save all uploaded files
        temp_paths = await save_multiple_uploaded_files(files)
        
        # Run batch age estimation
        results = await pipeline.batch_estimate_ages(
            image_paths=temp_paths,
            include_visualization=include_visualization
        )
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in batch age estimation: {e}")
        raise HTTPException(status_code=500, detail=f"Batch age estimation failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if temp_paths:
            cleanup_temp_files(*temp_paths)

@app.post("/quick-age-estimate")
async def quick_age_estimate(
    file: UploadFile = File(...)
):
    """
    Quick age estimation without visualization (faster processing)
    
    Args:
        file: Input image file
    
    Returns:
        JSON response with basic age estimation
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = await save_uploaded_file(file)
        
        # Run age estimation without visualization
        results = await pipeline.estimate_age(
            image_path=temp_path,
            include_visualization=False
        )
        
        # Return simplified response
        if results["success"]:
            quick_results = {
                "success": True,
                "estimated_age": results["age_estimation"]["estimated_age"],
                "age_category": results["age_estimation"]["age_category"],
                "confidence": results["age_estimation"]["confidence"],
                "processing_time": results["processing_time"],
                "analysis_id": results["analysis_id"]
            }
        else:
            quick_results = {
                "success": False,
                "error": results.get("error", "Unknown error"),
                "analysis_id": results["analysis_id"]
            }
        
        return JSONResponse(content=quick_results)
        
    except Exception as e:
        logger.error(f"Error in quick age estimation: {e}")
        raise HTTPException(status_code=500, detail=f"Quick age estimation failed: {str(e)}")
    
    finally:
        if temp_path:
            cleanup_temp_files(temp_path)

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Get visualization result file"""
    results_dir = "/app/results"
    file_path = os.path.join(results_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    # Security check - ensure filename doesn't contain path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    return FileResponse(
        file_path,
        media_type="image/png",
        filename=filename
    )

@app.get("/list-visualizations")
async def list_visualizations():
    """List available visualization files"""
    results_dir = "/app/results"
    
    try:
        if not os.path.exists(results_dir):
            return {"visualizations": []}
        
        files = []
        for filename in os.listdir(results_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(results_dir, filename)
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                
                files.append({
                    "filename": filename,
                    "size": file_size,
                    "modified": file_mtime,
                    "url": f"/visualization/{filename}"
                })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"visualizations": files}
        
    except Exception as e:
        logger.error(f"Error listing visualizations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list visualizations")

@app.delete("/cleanup-old-visualizations")
async def cleanup_old_visualizations(days: int = Form(7)):
    """
    Clean up visualization files older than specified days
    
    Args:
        days: Number of days (default: 7)
    """
    results_dir = "/app/results"
    
    if days < 1:
        raise HTTPException(status_code=400, detail="Days must be at least 1")
    
    try:
        import time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        cleaned_files = []
        total_size_cleaned = 0
        
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(results_dir, filename)
                    if os.path.getmtime(file_path) < cutoff_time:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleaned_files.append(filename)
                        total_size_cleaned += file_size
        
        return {
            "cleaned_files": len(cleaned_files),
            "total_size_cleaned": total_size_cleaned,
            "files": cleaned_files
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up visualizations: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup visualizations")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8013,
        reload=False
    )