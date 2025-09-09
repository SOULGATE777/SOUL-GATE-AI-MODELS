from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import uuid
import json
from pathlib import Path
import logging
from typing import Optional
import os

# Try multiple import strategies for enhanced pipeline
try:
    from app.models.enhanced_pipeline import EnhancedCompatibilityPipeline
except ImportError:
    try:
        # Fallback for production environment
        import sys
        import os
        sys.path.append('/app')
        from app.models.enhanced_pipeline import EnhancedCompatibilityPipeline
    except ImportError:
        try:
            # Direct import fallback
            sys.path.append('/app/app/models')
            from app.models.enhanced_pipeline import EnhancedCompatibilityPipeline
        except ImportError as e:
            raise ImportError(f"Could not import EnhancedCompatibilityPipeline: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Profile Anthropometric Analysis API",
    description="Advanced profile facial anthropometric analysis with point detection and measurements",
    version="1.0.0"
)

# Mount static files for visualization serving
app.mount("/visualization", StaticFiles(directory="results"), name="visualization")

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced analysis pipeline on startup"""
    global pipeline
    
    try:
        # Primary model path (required)
        point_model_path = "/app/models/best_point_detection_model_v2.pth"
        
        # Fallback to older model name if new one doesn't exist
        if not os.path.exists(point_model_path):
            point_model_path = "/app/models/profile_aware_point_detection_model.pth"
            
        if not os.path.exists(point_model_path):
            logger.error(f"Point detection model not found at {point_model_path}")
            raise FileNotFoundError(f"Point detection model not found")
        
        # GNN model path (optional)
        gnn_model_path = "/app/models/facial_landmark_gnn.pth"
        if not os.path.exists(gnn_model_path):
            logger.warning(f"GNN model not found at {gnn_model_path}, will use point detection only")
            gnn_model_path = None
        
        # Determine device
        device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '-1' else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Initialize enhanced pipeline
        pipeline = EnhancedCompatibilityPipeline(
            point_model_path=point_model_path,
            gnn_model_path=gnn_model_path,
            device=device
        )
        
        logger.info("Enhanced profile anthropometric pipeline initialized successfully!")
        
        # Log pipeline configuration
        gnn_status = "with GNN validation" if gnn_model_path else "point detection only"
        logger.info(f"Pipeline mode: {gnn_status}")
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced pipeline: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Profile Anthropometric Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-profile-anthropometric",
            "detect_points": "/detect-profile-points",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Pipeline not initialized"}
        )
    
    return {
        "status": "healthy",
        "service": "profile-anthropometric",
        "model_loaded": pipeline is not None,
        "device": str(pipeline.device) if pipeline else "unknown",
        "gnn_enabled": pipeline.gnn_model is not None if pipeline else False,
        "point_classes": len(pipeline.point_classes) if pipeline else 0
    }

@app.post("/analyze-profile-anthropometric")
async def analyze_profile_anthropometric(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.15),
    include_visualization: bool = Form(True)
):
    """
    Complete profile anthropometric analysis including point detection and measurements
    
    - **file**: Profile image file (JPG, PNG)
    - **confidence_threshold**: Minimum confidence for point detection (0.05-0.9)
    - **include_visualization**: Whether to generate visualization
    """
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate confidence threshold
    if not 0.05 <= confidence_threshold <= 0.9:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.05 and 0.9")
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform analysis
        results = pipeline.analyze_image(
            image=image_rgb,
            include_visualization=include_visualization
        )
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Save visualization if generated
        visualization_path = None
        visualization_url = None
        
        if results.get('visualization'):
            import base64
            visualization_filename = f"profile_anthropometric_{analysis_id}.png"
            visualization_path = f"/app/results/{visualization_filename}"
            
            # Decode and save visualization
            visualization_data = base64.b64decode(results['visualization'])
            with open(visualization_path, 'wb') as f:
                f.write(visualization_data)
            
            visualization_url = f"/visualization/{visualization_filename}"
            
            # Remove base64 data from response (too large)
            del results['visualization']
        
        # Format response
        response = {
            "analysis_id": analysis_id,
            "profile_analysis": {
                "profile_side": results['profile_side'],
                "total_detected_points": results['total_detected_points'],
                "filtered_points": results['filtered_points'],
                "anthropometric_points": results['anthropometric_points']
            },
            "anthropometric_measurements": results['measurements'],
            "analysis_summary": {
                "confidence_threshold": confidence_threshold,
                "has_measurements": bool(results['measurements']),
                "profile_determination": results['profile_side']
            },
            "visualization_path": visualization_path,
            "visualization_url": visualization_url
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/detect-profile-points")
async def detect_profile_points(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.15)
):
    """
    Detect anthropometric points in profile images only
    
    - **file**: Profile image file (JPG, PNG)
    - **confidence_threshold**: Minimum confidence for point detection (0.05-0.9)
    """
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate confidence threshold
    if not 0.05 <= confidence_threshold <= 0.9:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.05 and 0.9")
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Convert BGR to RGB 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use enhanced pipeline's point detection (includes preprocessing)
        detected_points = pipeline.detect_points(
            pipeline.preprocess_image(image_rgb)[1]  # Get tensor from preprocessing
        )
        
        # Filter spurious predictions using enhanced logic
        filtered_points, actual_profile = pipeline.filter_spurious_predictions(detected_points)
        
        response = {
            "total_detected_points": len(detected_points),
            "filtered_points": len(filtered_points),
            "profile_side": actual_profile,
            "detected_points": detected_points,
            "filtered_anthropometric_points": filtered_points,
            "detection_summary": {
                "confidence_threshold": confidence_threshold,
                "spurious_points_removed": len(detected_points) - len(filtered_points),
                "profile_determination": actual_profile
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Point detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Point detection failed: {str(e)}")

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
    """Get information about the loaded enhanced model"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "model_type": "Enhanced Profile Anthropometric Point Detection",
        "architecture": "ResNet50 + Profile Classification + Optional GNN",
        "device": str(pipeline.device),
        "point_classes": pipeline.point_classes,
        "num_classes": len(pipeline.point_classes),
        "heatmap_size": pipeline.heatmap_size,
        "input_size": 224,
        "gnn_enabled": pipeline.gnn_model is not None,
        "features": [
            "Profile-aware point detection",
            "False positive filtering",
            "Profile classification (left/right)",
            "GNN validation" if pipeline.gnn_model is not None else "CNN-only mode"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
