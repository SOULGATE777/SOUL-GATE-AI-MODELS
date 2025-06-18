from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import cv2
import numpy as np
from typing import Optional
import uuid
from datetime import datetime

from .models.anthropometric_pipeline import AnthropometricAnalyzer
from .utils.visualization import create_visualization
from .utils.image_processing import process_uploaded_image

app = FastAPI(
    title="Antropometrico Analysis API",
    description="Advanced anthropometric facial analysis with custom point detection",
    version="1.0.0"
)

# Initialize the analyzer (will be loaded on startup)
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize the anthropometric analyzer on startup"""
    global analyzer
    analyzer = AnthropometricAnalyzer()
    print("✅ Antropometrico Analysis API initialized successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "antropometrico",
        "timestamp": datetime.now().isoformat(),
        "analyzer_loaded": analyzer is not None
    }

@app.post("/analyze-anthropometric")
async def analyze_anthropometric(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    include_visualization: bool = Form(True)
):
    """
    Complete anthropometric facial analysis
    
    Args:
        file: Input image file
        confidence_threshold: Confidence threshold for model predictions (0.0-1.0)
        include_visualization: Whether to generate visualization
    
    Returns:
        JSON response with analysis results
    """
    try:
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        # Process uploaded image
        image_array = await process_uploaded_image(file)
        
        # Perform analysis
        results = analyzer.analyze_face(
            image_array, 
            confidence_threshold=confidence_threshold
        )
        
        visualization_path = None
        if include_visualization and results:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"anthropometric_{timestamp}_{unique_id}.jpg"
            visualization_path = f"/app/results/{filename}"
            
            # Create visualization
            vis_image = create_visualization(
                image_array, 
                results['landmarks'], 
                results['model_predictions'],
                results['proportions'],
                results['slopes']
            )
            
            # Save visualization
            os.makedirs("/app/results", exist_ok=True)
            cv2.imwrite(visualization_path, vis_image)
        
        # Prepare response
        response = {
            "facial_landmarks": {
                "count": 68,
                "extended_points": len(results['extended_points']) if results else 0
            },
            "model_predictions": results['model_predictions'] if results else {},
            "proportions": results['proportions'] if results else {},
            "slopes": results['slopes'] if results else {},
            "analysis_summary": results['summary'] if results else {},
            "visualization_path": visualization_path,
            "confidence_threshold": confidence_threshold
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/detect-landmarks")
async def detect_landmarks(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Detect facial landmarks only
    """
    try:
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        landmarks_result = analyzer.detect_landmarks_only(image_array)
        
        return JSONResponse(content={
            "landmarks_count": landmarks_result["count"],
            "landmarks": landmarks_result["landmarks"],
            "confidence_threshold": confidence_threshold
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Landmark detection failed: {str(e)}")

@app.post("/detect-points")
async def detect_anthropometric_points(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Detect anthropometric points using custom model
    """
    try:
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        model_predictions = analyzer.detect_model_points_only(image_array, confidence_threshold)
        
        return JSONResponse(content={
            "model_predictions": model_predictions,
            "points_detected": len(model_predictions),
            "confidence_threshold": confidence_threshold
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Point detection failed: {str(e)}")

@app.get("/results/{filename}")
async def get_result_image(filename: str):
    """
    Retrieve generated visualization image
    """
    file_path = f"/app/results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Visualization not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
