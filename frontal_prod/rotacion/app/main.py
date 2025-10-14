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

from app.models.frontal_rotation_pipeline import FrontalRotationPipeline
from app.utils.image_processing import ImageProcessor
from app.utils.visualization import FrontalRotationVisualizer
from app.utils.lazy_model_loader import MultiModelLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Frontal Face Rotation Assessment API",
    description="Advanced frontal face rotation classification to assess viability for anthropometric and morphological analysis",
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

# Initialize lazy model loader
model_loader = MultiModelLoader()

def _load_pipeline():
    """Lazy load frontal rotation pipeline"""
    model_path = MODELS_DIR / "improved_supervisely_head_rotation_model_MULTILABEL_CORRECTED.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please place the trained model in the /models directory."
        )
    try:
        pipeline = FrontalRotationPipeline(str(model_path))
        logger.info("✅ Frontal rotation pipeline loaded successfully!")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise e

def _load_image_processor():
    """Lazy load image processor"""
    logger.info("✅ Image processor loaded successfully!")
    return ImageProcessor()

def _load_visualizer():
    """Lazy load visualizer"""
    logger.info("✅ Visualizer loaded successfully!")
    return FrontalRotationVisualizer()

def get_pipeline():
    """Get frontal rotation pipeline, loading it if necessary"""
    return model_loader.get_model("pipeline")

def get_image_processor():
    """Get image processor, loading it if necessary"""
    return model_loader.get_model("image_processor")

def get_visualizer():
    """Get visualizer, loading it if necessary"""
    return model_loader.get_model("visualizer")

@app.on_event("startup")
async def startup_event():
    """Register models for lazy loading"""
    logger.info("🚀 Initializing Frontal Rotation API with lazy loading...")

    # Register models for lazy loading
    model_loader.register_model("pipeline", _load_pipeline)
    model_loader.register_model("image_processor", _load_image_processor)
    model_loader.register_model("visualizer", _load_visualizer)

    logger.info("✅ Models registered for lazy loading. They will load on first request.")
    logger.info("💾 RAM saved: Models will only load when needed!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "frontal-rotation",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "lazy_loading_enabled": True,
        "models_loaded": model_loader.get_loaded_models(),
        "port": 8012
    }

@app.post("/analyze-frontal-rotation")
async def analyze_frontal_rotation(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    include_visualization: bool = Form(True),
    enhance_image: bool = Form(True),
    pipeline: FrontalRotationPipeline = Depends(get_pipeline),
    processor: ImageProcessor = Depends(get_image_processor),
    visualizer: FrontalRotationVisualizer = Depends(get_visualizer)
):
    """
    Complete frontal face rotation analysis to assess viability for anthropometric and morphological analysis
    
    Args:
        file: Frontal face image file
        confidence_threshold: Minimum confidence threshold (0.0-1.0)
        include_visualization: Whether to generate visualization
        enhance_image: Whether to apply image enhancement
        
    Returns:
        Complete rotation analysis results with viability assessment
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
        
        # Prepare image for analysis
        processed_image = processor.prepare_image_for_analysis(image, enhance=enhance_image)
        
        # Assess basic image quality
        quality_assessment = processor.assess_image_quality(processed_image)
        
        # Run frontal rotation analysis
        results = pipeline.analyze_frontal_rotation(processed_image, confidence_threshold)
        
        # Add image quality to results
        results['image_quality'] = quality_assessment
        
        # Add timestamp
        results['analysis_summary']['timestamp'] = datetime.utcnow().isoformat()
        
        # Generate visualization if requested
        visualization_path = None
        visualization_url = None
        
        if include_visualization and results['analysis_summary']['processing_successful']:
            try:
                viz_filename = f"frontal_rotation_analysis_{results['analysis_id']}.png"
                visualization_path = RESULTS_DIR / viz_filename
                
                visualizer.create_rotation_analysis_visualization(
                    image=processed_image,
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
        rotation_assessment = results.get('rotation_assessment', {})
        logger.info(f"Frontal rotation analysis completed - ID: {results['analysis_id']}, "
                   f"Suitable: {rotation_assessment.get('is_suitable', False)}, "
                   f"Orientation: {rotation_assessment.get('predicted_tags', [])}")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in frontal rotation analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/classify-rotation")
async def classify_rotation(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    pipeline: FrontalRotationPipeline = Depends(get_pipeline),
    processor: ImageProcessor = Depends(get_image_processor)
):
    """
    Simple frontal rotation classification without full analysis
    
    Args:
        file: Frontal face image file
        confidence_threshold: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        Simple rotation classification results
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
        
        # Prepare image for analysis
        processed_image = processor.prepare_image_for_analysis(image)
        
        # Run rotation classification
        results = pipeline.classify_rotation_only(processed_image, confidence_threshold)
        
        # Add timestamp
        results['timestamp'] = datetime.utcnow().isoformat()
        
        # Log results
        logger.info(f"Frontal rotation classification completed - ID: {results['analysis_id']}, "
                   f"Orientation: {results.get('predicted_orientation', 'unclear')}, "
                   f"Acceptable: {results.get('is_acceptable', False)}")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in frontal rotation classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/assess-viability")
async def assess_viability(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    pipeline: FrontalRotationPipeline = Depends(get_pipeline),
    processor: ImageProcessor = Depends(get_image_processor)
):
    """
    Assess frontal face viability for anthropometric and morphological analysis
    
    Args:
        file: Frontal face image file
        confidence_threshold: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        Viability assessment results
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
        
        # Prepare image for analysis
        processed_image = processor.prepare_image_for_analysis(image)
        
        # Run viability assessment
        results = pipeline.assess_viability_only(processed_image, confidence_threshold)
        
        # Add timestamp
        results['timestamp'] = datetime.utcnow().isoformat()
        
        # Log results
        logger.info(f"Frontal viability assessment completed - ID: {results['analysis_id']}, "
                   f"Viable: {results.get('viable_for_analysis', False)}")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in frontal viability assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@app.post("/assess-image-quality")
async def assess_image_quality(
    file: UploadFile = File(...),
    processor: ImageProcessor = Depends(get_image_processor)
):
    """
    Assess basic image quality metrics without rotation analysis
    
    Args:
        file: Frontal face image file
        
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
        
        # Assess image quality
        quality_assessment = processor.assess_image_quality(image)
        
        # Add analysis metadata
        analysis_id = str(uuid.uuid4())
        results = {
            'analysis_id': analysis_id,
            'image_info': {
                'width': int(image.shape[1]),
                'height': int(image.shape[0]),
                'channels': int(image.shape[2]) if len(image.shape) == 3 else 1
            },
            'quality_assessment': quality_assessment,
            'timestamp': datetime.utcnow().isoformat(),
            'processing_successful': True
        }
        
        # Log results
        logger.info(f"Frontal image quality assessment completed - ID: {analysis_id}, "
                   f"Quality: {quality_assessment.get('overall_quality', {}).get('level', 'unknown')}")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in frontal image quality assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@app.get("/model-info")
async def get_model_info(pipeline: FrontalRotationPipeline = Depends(get_pipeline)):
    """Get information about the loaded frontal rotation model"""
    return {
        "model_classes": pipeline.class_names,
        "num_classes": pipeline.num_classes,
        "device": str(pipeline.device),
        "aceptable_index": pipeline.aceptable_idx,
        "rotation_indices": pipeline.rotation_indices,
        "default_threshold": pipeline.default_threshold,
        "model_architecture": "EfficientNet-B0-MultiLabel-Frontal-Rotation"
    }

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization files"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(file_path)

# Batch processing endpoint
@app.post("/batch-analyze")
async def batch_analyze_rotation(
    files: list[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5),
    pipeline: FrontalRotationPipeline = Depends(get_pipeline),
    processor: ImageProcessor = Depends(get_image_processor)
):
    """
    Batch process multiple frontal face images for rotation analysis
    
    Args:
        files: List of frontal face image files
        confidence_threshold: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        Batch analysis results
    """
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
        
        batch_id = str(uuid.uuid4())
        results = {
            'batch_id': batch_id,
            'total_files': len(files),
            'processed_files': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'results': [],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for i, file in enumerate(files):
            try:
                # Read and process image
                image_data = await file.read()
                image = processor.read_image_from_bytes(image_data)
                
                if image is None:
                    results['results'].append({
                        'file_index': i,
                        'filename': file.filename,
                        'success': False,
                        'error': 'Invalid image file'
                    })
                    results['failed_analyses'] += 1
                    continue
                
                # Prepare image and run analysis
                processed_image = processor.prepare_image_for_analysis(image)
                analysis_result = pipeline.classify_rotation_only(processed_image, confidence_threshold)
                
                results['results'].append({
                    'file_index': i,
                    'filename': file.filename,
                    'success': True,
                    'analysis': analysis_result
                })
                results['successful_analyses'] += 1
                
            except Exception as e:
                results['results'].append({
                    'file_index': i,
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
                results['failed_analyses'] += 1
            
            results['processed_files'] += 1
        
        logger.info(f"Batch frontal rotation analysis completed - ID: {batch_id}, "
                   f"Success: {results['successful_analyses']}/{len(files)}")
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch frontal rotation analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

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
    uvicorn.run(app, host="0.0.0.0", port=8012)