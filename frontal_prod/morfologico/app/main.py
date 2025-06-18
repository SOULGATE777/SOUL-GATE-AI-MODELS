from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import uuid
from typing import List, Optional
import json
import aiofiles
from pathlib import Path

from app.models.facial_analysis_pipeline import FacialAnalysisPipeline
from app.models.anthropometric_detection import AnthropometricPointDetector
from app.utils.visualization import create_beautiful_visualization
from app.utils.image_processing import process_uploaded_image

# Initialize FastAPI app
app = FastAPI(
    title="Facial Recognition API",
    description="Complete facial analysis with landmark detection, classification, and anthropometric points",
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

# Global variables for models
facial_pipeline = None
point_detector = None

# Tag mapping for better readability
tag_mapping = {
    "tag_0": "0", "tag_1": "1", "tag_2": "2", "tag_3": "3", "tag_4": "AN", 
    "tag_5": "Ab", "tag_6": "Adelgazamiento", "tag_7": "Al", "tag_8": "Ap", 
    "tag_9": "Ar", "tag_10": "Bigote", "tag_11": "CabellosSueltos", 
    "tag_12": "Carnosos", "tag_13": "Crl", "tag_14": "Cv", "tag_15": "Delgada", 
    "tag_16": "El", "tag_17": "Fleco", "tag_18": "Fr", "tag_19": "G", 
    "tag_20": "Grueso", "tag_21": "H", "tag_22": "Hn", "tag_23": "I", 
    "tag_24": "LineasVerticales", "tag_25": "Ll", "tag_26": "Lunar", 
    "tag_27": "Md", "tag_28": "MdA", "tag_29": "Mercurial", "tag_30": "Nd", 
    "tag_31": "Normal", "tag_32": "Nt", "tag_33": "On", "tag_34": "Pc", 
    "tag_35": "Pg", "tag_36": "Pl", "tag_37": "Planos", "tag_38": "Pn", 
    "tag_39": "Pt", "tag_40": "Pursed", "tag_41": "Rc", "tag_42": "Rd", 
    "tag_43": "Salido", "tag_44": "Sl", "tag_45": "Solar", "tag_46": "Sonriendo", 
    "tag_47": "SpSl", "tag_48": "abierta", "tag_49": "uniceja"
}

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global facial_pipeline, point_detector
    
    try:
        print("Loading facial analysis pipeline...")
        facial_pipeline = FacialAnalysisPipeline(
            detection_model_path='/app/models/facial_landmarks_detection_model.pth',
            classification_model_path='/app/models/best_facial_landmark_classifier.pth',
            tag_mapping_path=None
        )
        
        print("Loading anthropometric point detector...")
        point_detector = AnthropometricPointDetector(
            model_path='/app/models/facial_points_detection_model.pth'
        )
        
        print("All models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": facial_pipeline is not None and point_detector is not None
    }

@app.post("/analyze-face")
async def analyze_face(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    include_visualization: bool = True
):
    """
    Complete facial analysis with all three models:
    1. Facial landmark detection
    2. Characteristic classification
    3. Anthropometric point detection
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process uploaded image
        image_array, temp_path = await process_uploaded_image(file)
        
        # Run facial landmark detection and classification
        facial_results, facial_viz = facial_pipeline.process_image(
            image_array,
            confidence_threshold=confidence_threshold,
            display=False
        )
        
        # Add tag names to facial results
        for result in facial_results:
            result['tag_name'] = tag_mapping.get(result['tag'], "Unknown")
        
        # Run anthropometric point detection
        point_results = point_detector.detect_points(
            image_array,
            confidence_threshold=confidence_threshold
        )
        
        # Create beautiful combined visualization
        if include_visualization:
            viz_path = await create_beautiful_visualization(
                image_array, facial_results, point_results
            )
        else:
            viz_path = None
        
        # Prepare response
        response = {
            "facial_landmarks": {
                "count": len(facial_results),
                "detections": facial_results
            },
            "anthropometric_points": {
                "count": len(point_results),
                "detections": point_results
            },
            "summary": {
                "total_detections": len(facial_results) + len(point_results),
                "confidence_threshold": confidence_threshold,
                "image_processed": True
            }
        }
        
        if viz_path:
            response["visualization_path"] = viz_path
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect-landmarks")
async def detect_landmarks(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    include_visualization: bool = True
):
    """Detect and classify facial landmarks only"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_array, temp_path = await process_uploaded_image(file)
        
        results, viz_image = facial_pipeline.process_image(
            image_array,
            confidence_threshold=confidence_threshold,
            display=False
        )
        
        # Add tag names
        for result in results:
            result['tag_name'] = tag_mapping.get(result['tag'], "Unknown")
        
        # Save visualization if requested
        viz_path = None
        if include_visualization:
            viz_filename = f"landmarks_{uuid.uuid4().hex}.jpg"
            viz_path = f"/app/results/{viz_filename}"
            cv2.imwrite(viz_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        response = {
            "landmarks": results,
            "count": len(results),
            "confidence_threshold": confidence_threshold
        }
        
        if viz_path:
            response["visualization_path"] = viz_path
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect-points")
async def detect_anthropometric_points(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    include_visualization: bool = True
):
    """Detect anthropometric points only"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_array, temp_path = await process_uploaded_image(file)
        
        results = point_detector.detect_points(
            image_array,
            confidence_threshold=confidence_threshold
        )
        
        # Create visualization if requested
        viz_path = None
        if include_visualization:
            viz_path = await point_detector.create_visualization(
                image_array, results
            )
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        response = {
            "points": results,
            "count": len(results),
            "confidence_threshold": confidence_threshold
        }
        
        if viz_path:
            response["visualization_path"] = viz_path
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    """Serve visualization images"""
    file_path = f"/app/results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Visualization not found")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Facial Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "analyze_face": "/analyze-face",
            "detect_landmarks": "/detect-landmarks", 
            "detect_points": "/detect-points",
            "health": "/health"
        },
        "models_loaded": facial_pipeline is not None and point_detector is not None
    }
