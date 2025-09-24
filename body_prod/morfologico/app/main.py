from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

from app.models.body_analysis_pipeline import BodyAnalysisPipeline
from app.utils.image_processing import validate_image, preprocess_image
from app.utils.visualization import create_body_analysis_visualization

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
        logger.info("🚀 Initializing Body Morphological Analysis Pipeline...")
        pipeline = BodyAnalysisPipeline()
        await asyncio.to_thread(pipeline.load_model)
        logger.info("Pipeline loaded successfully!")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise
    finally:
        logger.info("Shutting down pipeline...")

# Create FastAPI app
app = FastAPI(
    title="Body Morphological Analysis API",
    description="Production-ready body type classification and morphological analysis service",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files for serving visualizations
os.makedirs("/app/results", exist_ok=True)
app.mount("/visualization", StaticFiles(directory="/app/results"), name="visualization")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Body Morphological Analysis API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "analyze": "/analyze-body-morphology",
            "classify": "/classify-body-type",
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
        "service": "body-morphological-analysis",
        "model_loaded": pipeline.model is not None,
        "device": str(pipeline.device),
        "classes": {
            "body_types": len(pipeline.body_type_classes)
        }
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "model_architecture": "ComprehensiveAnatomicalClassifierResNet34",
        "backbone": "ResNet34",
        "input_size": [128, 128],
        "device": str(pipeline.device),
        "body_type_classes": pipeline.body_type_classes,
        "anatomical_parts": list(pipeline.ANATOMICAL_PARTS.keys()) if hasattr(pipeline, 'ANATOMICAL_PARTS') else [],
        "total_parameters": pipeline.get_model_parameters(),
        "model_path": "/app/models/best_comprehensive_ensemble_resnet34_fixed.pth",
        "model_type": "comprehensive_anatomical_with_intelligent_leg_cropping",
        "pose_detection": "YOLOv8n-pose",
        "improvements": {
            "intelligent_leg_cropping": True,
            "torso_width_based_sizing": True,
            "enhanced_regularization": True,
            "weighted_ensemble_voting": True
        },
        "part_weights": getattr(pipeline, 'part_weights', {})
    }

@app.post("/analyze-body-morphology")
async def analyze_body_morphology(
    file: UploadFile = File(...),
    bbox: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.5),
    include_visualization: bool = Form(True)
):
    """
    Complete body morphological analysis including classification and visualization
    
    Args:
        file: Image file (JPG, PNG, etc.)
        bbox: Optional bounding box as "x1,y1,x2,y2" to crop the body region
        confidence_threshold: Minimum confidence for predictions (0.0-1.0)
        include_visualization: Whether to generate visualization
    
    Returns:
        Complete analysis results with predictions and optional visualization
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate image
        image_array = await validate_image(file)
        
        # Parse bounding box if provided
        bbox_coords = None
        if bbox:
            try:
                bbox_coords = [int(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Bounding box must have 4 coordinates")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid bounding box format: {e}")
        
        # Run analysis
        results = await asyncio.to_thread(
            pipeline.analyze_body_type,
            image_array,
            bbox_coords,
            confidence_threshold
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        results["analysis_id"] = analysis_id
        
        # Create visualization if requested
        if include_visualization:
            viz_path = await asyncio.to_thread(
                create_body_analysis_visualization,
                image_array,
                results,
                bbox_coords,
                f"/app/results/body_morphology_{analysis_id}.png"
            )
            
            if viz_path:
                results["visualization_path"] = viz_path
                results["visualization_url"] = f"/visualization/body_morphology_{analysis_id}.png"
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in body morphological analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-body-type")
async def classify_body_type(
    file: UploadFile = File(...),
    bbox: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.5)
):
    """
    Body type classification only (no visualization)
    
    Args:
        file: Image file
        bbox: Optional bounding box as "x1,y1,x2,y2"
        confidence_threshold: Minimum confidence for predictions
    
    Returns:
        Body type classification results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Validate image
        image_array = await validate_image(file)
        
        # Parse bounding box if provided
        bbox_coords = None
        if bbox:
            try:
                bbox_coords = [int(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Bounding box must have 4 coordinates")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid bounding box format: {e}")
        
        # Run classification only
        results = await asyncio.to_thread(
            pipeline.classify_body_type_only,
            image_array,
            bbox_coords,
            confidence_threshold
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Classification failed")
        
        # Add analysis metadata
        results["analysis_id"] = str(uuid.uuid4())
        results["analysis_type"] = "classification_only"
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in body type classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization files"""
    file_path = f"/app/results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(file_path)

@app.post("/batch-classify")
async def batch_classify(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Batch body type classification for multiple images
    
    Args:
        files: List of image files
        confidence_threshold: Minimum confidence for predictions
    
    Returns:
        List of classification results for each image
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    try:
        results = []
        for i, file in enumerate(files):
            try:
                # Validate image
                image_array = await validate_image(file)
                
                # Run classification
                result = await asyncio.to_thread(
                    pipeline.classify_body_type_only,
                    image_array,
                    None,  # No bbox for batch processing
                    confidence_threshold
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
                        "error": "Classification failed"
                    })
                    
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
            "successful_classifications": len([r for r in results if "error" not in r]),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8006,
        reload=False,
        workers=1
    )
