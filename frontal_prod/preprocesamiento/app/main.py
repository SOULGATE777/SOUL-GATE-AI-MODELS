from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uuid
import json
from pathlib import Path
import logging
from typing import Optional
import os
from datetime import datetime

from app.models.frontal_preprocessing_pipeline import FrontalPreprocessingPipeline
from app.utils.image_processing import ImageProcessor
from app.utils.visualization import FrontalVisualizationManager
from app.utils.lazy_model_loader import MultiModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Frontal Preprocessing API",
    description="Frontal head detection, cropping and preprocessing service for downstream analysis",
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

# Mount static files for visualization serving
app.mount("/visualization", StaticFiles(directory="results"), name="visualization")

# Initialize lazy model loader
model_loader = MultiModelLoader()

def _load_pipeline():
    """Lazy load preprocessing pipeline"""
    try:
        model_path = "/app/models/frontal_head_detection_model.pt"

        if not os.path.exists(model_path):
            logger.info(f"Custom model not found at {model_path}, using pre-trained YOLOv8n")
            model_path = None

        device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '-1' else 'cpu'
        logger.info(f"Using device: {device}")

        pipeline = FrontalPreprocessingPipeline(
            model_path=model_path,
            device=device
        )
        logger.info("✅ Frontal preprocessing pipeline loaded successfully!")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
        raise e

def _load_image_processor():
    """Lazy load image processor"""
    logger.info("✅ Image processor loaded successfully!")
    return ImageProcessor()

def _load_viz_manager():
    """Lazy load visualization manager"""
    logger.info("✅ Visualization manager loaded successfully!")
    return FrontalVisualizationManager()

def get_pipeline():
    """Get preprocessing pipeline, loading it if necessary"""
    return model_loader.get_model("pipeline")

def get_image_processor():
    """Get image processor, loading it if necessary"""
    return model_loader.get_model("image_processor")

def get_viz_manager():
    """Get visualization manager, loading it if necessary"""
    return model_loader.get_model("viz_manager")

@app.on_event("startup")
async def startup_event():
    """Register models for lazy loading"""
    logger.info("🚀 Initializing Frontal Preprocessing API with lazy loading...")

    # Register models for lazy loading
    model_loader.register_model("pipeline", _load_pipeline)
    model_loader.register_model("image_processor", _load_image_processor)
    model_loader.register_model("viz_manager", _load_viz_manager)

    logger.info("✅ Models registered for lazy loading. They will load on first request.")
    logger.info("💾 RAM saved: Models will only load when needed!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Frontal Preprocessing API",
        "version": "1.0.0",
        "description": "Head detection, cropping and preprocessing for downstream analysis",
        "endpoints": {
            "health": "/health",
            "preprocess": "/preprocess-frontal",
            "detect_only": "/detect-heads",
            "crop_only": "/crop-heads",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    pipeline = get_pipeline()
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Pipeline not initialized"}
        )

    return {
        "status": "healthy",
        "service": "frontal-preprocessing",
        "lazy_loading_enabled": True,
        "models_loaded": model_loader.get_loaded_models(),
        "device": str(pipeline.device) if pipeline else "unknown",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/preprocess-frontal")
async def preprocess_frontal(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    target_width: int = Form(600),
    target_height: int = Form(600),
    padding_factor: float = Form(0.15),
    output_format: str = Form("JPEG"),
    quality: int = Form(95),
    include_visualization: bool = Form(False),
    align_face: bool = Form(True)
):
    """
    Complete frontal preprocessing: detect heads, crop, resize and convert to base64

    - **file**: Input image file (JPG, PNG)
    - **confidence_threshold**: Minimum confidence for head detection (0.1-0.9, default: 0.5)
    - **target_width**: Target width for cropped heads (default: 600)
    - **target_height**: Target height for cropped heads (default: 600)
    - **padding_factor**: Padding around detected heads (0.0-0.5, default: 0.15)
    - **output_format**: Output format for base64 images ('JPEG', 'PNG', default: 'JPEG')
    - **quality**: JPEG quality 1-100 (default: 95)
    - **include_visualization**: Generate debug visualizations (default: false)
    - **align_face**: Align tilted faces to anatomical position (default: true)
    """

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Validate parameters
    if not 0.1 <= confidence_threshold <= 0.9:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.1 and 0.9")

    if not 100 <= target_width <= 2048 or not 100 <= target_height <= 2048:
        raise HTTPException(status_code=400, detail="Target dimensions must be between 100 and 2048 pixels")

    if not 0.0 <= padding_factor <= 0.5:
        raise HTTPException(status_code=400, detail="Padding factor must be between 0.0 and 0.5")

    if output_format.upper() not in ['JPEG', 'PNG']:
        raise HTTPException(status_code=400, detail="Output format must be 'JPEG' or 'PNG'")

    if not 1 <= quality <= 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")

    try:
        # Read and decode image
        contents = await file.read()
        image = get_image_processor().decode_image_from_bytes(contents)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        if not get_image_processor().validate_image(image):
            raise HTTPException(status_code=400, detail="Invalid image format or dimensions")

        # Process image through pipeline
        results = pipeline.process_image(
            image=image,
            confidence_threshold=confidence_threshold,
            target_size=(target_width, target_height),
            padding_factor=padding_factor,
            output_format=output_format,
            quality=quality,
            align_face=align_face
        )

        # Generate unique processing ID
        processing_id = str(uuid.uuid4())

        # Create response
        response = {
            "processing_id": processing_id,
            "status": "success",
            "total_heads_detected": results['total_detections'],
            "heads_processed": len(results['processed_heads']),
            "original_image_size": results['original_image_size'],
            "processing_parameters": results['processing_parameters'],
            "processed_heads": []
        }

        # Add alignment metadata if available
        if 'alignment' in results:
            response['alignment'] = results['alignment']

        # Add processed heads to response
        for i, head_data in enumerate(results['processed_heads']):
            response["processed_heads"].append({
                "head_id": i + 1,
                "detection_confidence": head_data['confidence'],
                "class_name": head_data['class_name'],
                "original_bbox": head_data['bbox'],
                "cropped_image_base64": head_data['cropped_image_base64'],
                "crop_info": {
                    "target_size": head_data['target_size'],
                    "padding_factor": head_data['padding_factor']
                },
                "detection_type": head_data.get('detection_type', 'head')
            })

        # Generate visualizations if requested
        if include_visualization and results['processed_heads']:
            try:
                viz_results = get_viz_manager().create_complete_visualization(
                    original_image=image,
                    processing_result=results,
                    include_summary=True
                )

                if viz_results['status'] == 'success':
                    # Save visualizations to results directory
                    viz_paths = {}
                    for viz_name, viz_base64 in viz_results['visualizations'].items():
                        viz_filename = f"preprocessing_viz_{processing_id}_{viz_name}.png"
                        viz_path = f"/app/results/{viz_filename}"

                        # Decode and save visualization
                        viz_data = base64.b64decode(viz_base64)
                        with open(viz_path, 'wb') as f:
                            f.write(viz_data)

                        viz_paths[viz_name] = f"/visualization/{viz_filename}"

                    response["visualizations"] = viz_paths

            except Exception as viz_error:
                logger.warning(f"Visualization generation failed: {str(viz_error)}")
                response["visualization_error"] = str(viz_error)

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.post("/detect-heads")
async def detect_heads_only(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Detect frontal heads in image without cropping

    - **file**: Input image file (JPG, PNG)
    - **confidence_threshold**: Minimum confidence for detection (0.1-0.9)
    """

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Validate confidence threshold
    if not 0.1 <= confidence_threshold <= 0.9:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.1 and 0.9")

    try:
        # Read and decode image
        contents = await file.read()
        image = get_image_processor().decode_image_from_bytes(contents)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        if not get_image_processor().validate_image(image):
            raise HTTPException(status_code=400, detail="Invalid image format or dimensions")

        # Detect heads
        detections = pipeline.detect_heads(image, confidence_threshold)

        response = {
            "status": "success",
            "total_detections": len(detections),
            "image_size": image.shape[:2],
            "confidence_threshold": confidence_threshold,
            "detections": detections
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Head detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Head detection failed: {str(e)}")

@app.post("/crop-heads")
async def crop_heads_from_bboxes(
    file: UploadFile = File(...),
    bboxes: str = Form(...),  # JSON string of bounding boxes
    target_width: int = Form(600),
    target_height: int = Form(600),
    padding_factor: float = Form(0.15),
    output_format: str = Form("JPEG"),
    quality: int = Form(95)
):
    """
    Crop heads from image using provided bounding boxes

    - **file**: Input image file (JPG, PNG)
    - **bboxes**: JSON string of bounding boxes [[x1,y1,x2,y2], ...]
    - **target_width**: Target width for cropped heads
    - **target_height**: Target height for cropped heads
    - **padding_factor**: Padding around bounding boxes (0.0-0.5)
    - **output_format**: Output format ('JPEG', 'PNG')
    - **quality**: JPEG quality 1-100
    """

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Parse bounding boxes
        bbox_list = json.loads(bboxes)
        if not isinstance(bbox_list, list):
            raise HTTPException(status_code=400, detail="Bboxes must be a list of bounding boxes")

        # Validate parameters
        if not 100 <= target_width <= 2048 or not 100 <= target_height <= 2048:
            raise HTTPException(status_code=400, detail="Target dimensions must be between 100 and 2048 pixels")

        if not 0.0 <= padding_factor <= 0.5:
            raise HTTPException(status_code=400, detail="Padding factor must be between 0.0 and 0.5")

        # Read and decode image
        contents = await file.read()
        image = get_image_processor().decode_image_from_bytes(contents)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        if not get_image_processor().validate_image(image):
            raise HTTPException(status_code=400, detail="Invalid image format or dimensions")

        # Crop heads
        cropped_heads = []
        for i, bbox in enumerate(bbox_list):
            if len(bbox) != 4:
                continue

            cropped_head = pipeline.crop_head_with_padding(
                image, bbox, (target_width, target_height), padding_factor
            )

            head_base64 = pipeline.image_to_base64(cropped_head, output_format, quality)

            cropped_heads.append({
                "head_id": i + 1,
                "bbox": bbox,
                "cropped_image_base64": head_base64,
                "target_size": [target_width, target_height],
                "padding_factor": padding_factor
            })

        response = {
            "status": "success",
            "total_heads_cropped": len(cropped_heads),
            "original_image_size": image.shape[:2],
            "cropped_heads": cropped_heads,
            "processing_parameters": {
                "target_size": [target_width, target_height],
                "padding_factor": padding_factor,
                "output_format": output_format,
                "quality": quality
            }
        }

        return JSONResponse(content=response)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for bboxes")
    except Exception as e:
        logger.error(f"Head cropping error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Head cropping failed: {str(e)}")

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization files"""
    file_path = Path(f"/app/results/{filename}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename
    )

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return pipeline.get_model_info()

@app.get("/processing-stats")
async def get_processing_stats():
    """Get processing statistics and capabilities"""
    pipeline = get_pipeline() if "pipeline" in model_loader.get_loaded_models() else None

    return {
        "service_info": {
            "name": "Frontal Preprocessing Service",
            "version": "1.0.0",
            "description": "Head detection, cropping and preprocessing for frontal images"
        },
        "capabilities": {
            "head_detection": True,
            "head_cropping": True,
            "base64_output": True,
            "batch_processing": False,
            "visualization": True
        },
        "supported_formats": {
            "input": ["JPEG", "PNG", "JPG"],
            "output": ["JPEG", "PNG"]
        },
        "parameter_ranges": {
            "confidence_threshold": {"min": 0.1, "max": 0.9, "default": 0.5},
            "target_size": {"min": 100, "max": 2048, "default": 600},
            "padding_factor": {"min": 0.0, "max": 0.5, "default": 0.15},
            "quality": {"min": 1, "max": 100, "default": 95}
        },
        "lazy_loading": {
            "enabled": True,
            "models_loaded": model_loader.get_loaded_models()
        },
        "device": str(pipeline.device) if pipeline else "unknown",
        "model_loaded": pipeline is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8014)