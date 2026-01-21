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

from app.models.hand_analysis_pipeline import HandAnalysisPipeline
from app.utils.image_processing import validate_image, preprocess_image
from app.utils.visualization import create_hand_analysis_visualization

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
        logger.info("🚀 Initializing Hand Analysis Pipeline...")
        pipeline = HandAnalysisPipeline()
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
    title="Hand Analysis API",
    description="Production-ready hand classification and colorimetry analysis service",
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
        "service": "Hand Analysis API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "analyze": "/analyze-hand-comprehensive",
            "classify": "/classify-hand-side",
            "colorimetry": "/analyze-colorimetry",
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
    
    model_info = pipeline.get_model_info()
    
    return {
        "status": "healthy",
        "service": "hand-analysis",
        "cnn_model_loaded": model_info['cnn_model']['model_loaded'],
        "device": model_info['cnn_model']['device'],
        "supported_classes": model_info['cnn_model']['classes'],
        "color_types": len(model_info['colorimetry']['color_types'])
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return pipeline.get_model_info()

@app.post("/analyze-hand-comprehensive")
async def analyze_hand_comprehensive(
    file: UploadFile = File(...),
    bbox: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.5),
    include_colorimetry: bool = Form(True),
    include_visualization: bool = Form(True)
):
    """
    Complete hand analysis including CNN classification and colorimetry
    
    Args:
        file: Image file (JPG, PNG, etc.)
        bbox: Optional bounding box as "x1,y1,x2,y2" to crop the hand region
        confidence_threshold: Minimum confidence for CNN predictions (0.0-1.0)
        include_colorimetry: Whether to include colorimetry analysis
        include_visualization: Whether to generate visualization
    
    Returns:
        Complete hand analysis results
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
                bbox_coords = [float(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Bounding box must have 4 coordinates")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid bounding box format: {e}")
        
        # Save temporary image for processing
        temp_id = str(uuid.uuid4())
        temp_path = f"/app/temp/temp_{temp_id}.jpg"
        os.makedirs("/app/temp", exist_ok=True)
        
        # Save image temporarily
        image_pil = Image.fromarray(image_array)
        image_pil.save(temp_path, "JPEG")
        
        # Run comprehensive analysis
        results = await asyncio.to_thread(
            pipeline.analyze_hand_comprehensive,
            temp_path,
            bbox_coords,
            confidence_threshold,
            include_colorimetry
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Analysis failed")
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        results["analysis_id"] = analysis_id
        
        # Create visualization if requested
        if include_visualization:
            viz_path = await asyncio.to_thread(
                create_hand_analysis_visualization,
                temp_path,
                results,
                f"/app/results/hand_analysis_{analysis_id}.png"
            )
            
            if viz_path:
                results["visualization_path"] = viz_path
                results["visualization_url"] = f"/visualization/hand_analysis_{analysis_id}.png"
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in comprehensive hand analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-hand-side")
async def classify_hand_side(
    file: UploadFile = File(...),
    bbox: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.5)
):
    """
    Hand side classification only (dorso/palma)
    
    Args:
        file: Image file
        bbox: Optional bounding box as "x1,y1,x2,y2"
        confidence_threshold: Minimum confidence for predictions
    
    Returns:
        Hand side classification results
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
                bbox_coords = [float(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Bounding box must have 4 coordinates")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid bounding box format: {e}")
        
        # Save temporary image for processing
        temp_id = str(uuid.uuid4())
        temp_path = f"/app/temp/temp_{temp_id}.jpg"
        os.makedirs("/app/temp", exist_ok=True)
        
        # Save image temporarily
        image_pil = Image.fromarray(image_array)
        image_pil.save(temp_path, "JPEG")
        
        # Run classification only
        results = await asyncio.to_thread(
            pipeline.classify_hand_side_only,
            temp_path,
            bbox_coords,
            confidence_threshold
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Classification failed")
        
        # Add analysis metadata
        results["analysis_id"] = str(uuid.uuid4())
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in hand side classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-colorimetry")
async def analyze_colorimetry(
    file: UploadFile = File(...),
    bbox: Optional[str] = Form(None),
    include_visualization: bool = Form(True)
):
    """
    Colorimetry analysis only (no CNN classification)
    
    Args:
        file: Image file
        bbox: Optional bounding box as "x1,y1,x2,y2"
        include_visualization: Whether to generate visualization
    
    Returns:
        Colorimetry analysis results
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
                bbox_coords = [float(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Bounding box must have 4 coordinates")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid bounding box format: {e}")
        
        # Save temporary image for processing
        temp_id = str(uuid.uuid4())
        temp_path = f"/app/temp/temp_{temp_id}.jpg"
        os.makedirs("/app/temp", exist_ok=True)
        
        # Save image temporarily
        image_pil = Image.fromarray(image_array)
        image_pil.save(temp_path, "JPEG")
        
        # Run colorimetry analysis only
        results = await asyncio.to_thread(
            pipeline.analyze_colorimetry_only,
            temp_path,
            bbox_coords
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Colorimetry analysis failed")
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        results["analysis_id"] = analysis_id
        
        # Create visualization if requested
        if include_visualization and results.get('colorimetry'):
            viz_path = await asyncio.to_thread(
                create_hand_analysis_visualization,
                temp_path,
                results,
                f"/app/results/colorimetry_{analysis_id}.png"
            )
            
            if viz_path:
                results["visualization_path"] = viz_path
                results["visualization_url"] = f"/visualization/colorimetry_{analysis_id}.png"
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in colorimetry analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-analyze")
async def batch_analyze(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5),
    include_colorimetry: bool = Form(True)
):
    """
    Batch hand analysis for multiple images
    
    Args:
        files: List of image files
        confidence_threshold: Minimum confidence for CNN predictions
        include_colorimetry: Whether to include colorimetry analysis
    
    Returns:
        List of analysis results for each image
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if len(files) > 5:  # Limit batch size for hand analysis
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
                    pipeline.analyze_hand_comprehensive,
                    temp_path,
                    None,  # No bbox for batch processing
                    confidence_threshold,
                    include_colorimetry
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
        port=8009,
        reload=False,
        workers=1
    )