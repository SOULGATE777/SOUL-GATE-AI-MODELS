from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import cv2
import numpy as np
from typing import Optional
import uuid
from datetime import datetime

from app.models.espejo_pipeline import EspejoAnalyzer
from app.utils.visualization import create_mirror_visualization, create_analysis_dashboard
from app.utils.image_processing import process_uploaded_image
from app.utils.lazy_model_loader import LazyModelLoader

app = FastAPI(
    title="Espejo Analysis API",
    description="Advanced mirror face analysis with anthropometric measurements, decision tree classification, and hybrid splitting for facial diagnosis",
    version="1.0.0"
)

# Initialize lazy model loader
model_loader = LazyModelLoader(
    load_func=lambda: EspejoAnalyzer(),
    name="espejo_analyzer"
)

def get_analyzer():
    """Get espejo analyzer, loading it if necessary"""
    return model_loader.get_model()

@app.on_event("startup")
async def startup_event():
    """Register model for lazy loading"""
    print("🚀 Initializing Espejo Analysis API with lazy loading...")
    print("✅ Model registered for lazy loading. Will load on first request.")
    print("💾 RAM saved: Model will only load when needed!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "espejo",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "lazy_loading_enabled": True,
        "model_loaded": model_loader.is_loaded(),
        "features": [
            "mirror_face_generation",
            "anthropometric_analysis",
            "decision_tree_classification",
            "hybrid_class_splitting",
            "facial_diagnosis",
            "forehead_proportion_analysis",
            "temporal_proportion_analysis",
            "comprehensive_reporting"
        ]
    }

@app.post("/analyze-espejo")
async def analyze_espejo(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    include_visualization: bool = Form(True),
    include_dashboard: bool = Form(False)
):
    """
    Complete espejo analysis with mirror generation, anthropometric analysis, 
    decision tree classification, and hybrid splitting
    
    Args:
        file: Input image file
        confidence_threshold: Confidence threshold for model predictions (0.0-1.0)
        include_visualization: Whether to generate mirror visualization
        include_dashboard: Whether to generate analysis dashboard
    
    Returns:
        JSON response with comprehensive analysis results
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        # Process uploaded image
        image_array = await process_uploaded_image(file)
        
        # Perform complete analysis
        results = analyzer.analyze_complete(
            image_array, 
            confidence_threshold=confidence_threshold
        )
        
        if not results:
            return JSONResponse(content={
                "error": "No face detected in the image",
                "face_detected": False,
                "analysis_summary": {}
            })
        
        visualization_path = None
        dashboard_path = None
        
        if include_visualization:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"espejo_analysis_{timestamp}_{unique_id}.png"
            visualization_path = f"/app/results/{filename}"

            # Extract detected regions for visualization
            detected_regions_for_viz = {}
            if 'classification_results' in results:
                if 'right_mirrored' in results['classification_results'] and 'detected_regions' in results['classification_results']['right_mirrored']:
                    detected_regions_for_viz['right_mirrored'] = results['classification_results']['right_mirrored']['detected_regions']
                if 'left_mirrored' in results['classification_results'] and 'detected_regions' in results['classification_results']['left_mirrored']:
                    detected_regions_for_viz['left_mirrored'] = results['classification_results']['left_mirrored']['detected_regions']

            # Create visualization
            vis_image = create_mirror_visualization(
                image_array,
                results['mirror_images'],
                results['classification_results'],
                results['proportions'],
                detected_regions_for_viz
            )
            
            # Save visualization
            os.makedirs("/app/results", exist_ok=True)
            cv2.imwrite(visualization_path, vis_image)
        
        if include_dashboard:
            # Generate dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            dashboard_filename = f"espejo_dashboard_{timestamp}_{unique_id}.png"
            dashboard_path = f"/app/results/{dashboard_filename}"
            
            # Create analysis dashboard
            dashboard_image = create_analysis_dashboard(image_array, results, detected_regions_for_viz)
            
            # Save dashboard
            os.makedirs("/app/results", exist_ok=True)
            cv2.imwrite(dashboard_path, dashboard_image)
        
        # Prepare comprehensive response
        response = {
            "face_detected": True,
            "anthropometric_analysis": {
                "face_proportions": results['proportions']['face_proportions'],
                "forehead_proportions": results['proportions']['forehead_proportions'],
                "temporal_proportions": results['proportions']['temporal_proportions'],
                "custom_model_points": results['custom_model_points'],
                "landmarks_detected": results['landmarks_count']
            },
            "mirror_analysis": {
                "right_mirrored": results['classification_results']['right_mirrored'],
                "left_mirrored": results['classification_results']['left_mirrored']
            },
            "final_diagnosis": {
                "right_side": {
                    "frente_diagnosis": results['classification_results']['right_mirrored']['frente_split_diagnosis'],
                    "rostro_diagnosis": results['classification_results']['right_mirrored']['rostro_split_diagnosis'],
                    "confidence_scores": {
                        "frente": results['classification_results']['right_mirrored']['frente_probabilities'],
                        "rostro": results['classification_results']['right_mirrored']['rostro_probabilities']
                    }
                },
                "left_side": {
                    "frente_diagnosis": results['classification_results']['left_mirrored']['frente_split_diagnosis'],
                    "rostro_diagnosis": results['classification_results']['left_mirrored']['rostro_split_diagnosis'],
                    "confidence_scores": {
                        "frente": results['classification_results']['left_mirrored']['frente_probabilities'],
                        "rostro": results['classification_results']['left_mirrored']['rostro_probabilities']
                    }
                }
            },
            "decision_tree_analysis": {
                "right_side": {
                    "frente_applied_rules": results['classification_results']['right_mirrored']['frente_applied_rules'],
                    "rostro_applied_rules": results['classification_results']['right_mirrored']['rostro_applied_rules'],
                    "frente_split_rules": results['classification_results']['right_mirrored']['frente_split_rules'],
                    "rostro_split_rules": results['classification_results']['right_mirrored']['rostro_split_rules']
                },
                "left_side": {
                    "frente_applied_rules": results['classification_results']['left_mirrored']['frente_applied_rules'],
                    "rostro_applied_rules": results['classification_results']['left_mirrored']['rostro_applied_rules'],
                    "frente_split_rules": results['classification_results']['left_mirrored']['frente_split_rules'],
                    "rostro_split_rules": results['classification_results']['left_mirrored']['rostro_split_rules']
                }
            },
            "analysis_summary": results['analysis_summary'],
            "visualization_path": visualization_path,
            "dashboard_path": dashboard_path,
            "confidence_threshold": confidence_threshold
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/generate-mirrors")
async def generate_mirrors(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Generate mirror images only
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        results = analyzer.generate_mirror_images(image_array, confidence_threshold)
        
        if not results:
            return JSONResponse(content={"error": "No face detected"})
        
        return JSONResponse(content={
            "mirror_images_generated": True,
            "landmarks_detected": results['landmarks_count'],
            "custom_model_points": results['custom_model_points'],
            "face_aligned": results['face_aligned'],
            "confidence_threshold": confidence_threshold
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mirror generation failed: {str(e)}")

@app.post("/classify-regions")
async def classify_regions(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Classify facial regions with decision tree and hybrid splitting
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        results = analyzer.classify_regions(image_array, confidence_threshold)
        
        if not results:
            return JSONResponse(content={"error": "No face detected"})
        
        return JSONResponse(content={
            "classification_results": results['classification_results'],
            "proportions": results['proportions'],
            "decision_tree_applied": True,
            "hybrid_splitting_applied": True,
            "confidence_threshold": confidence_threshold
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/analyze-proportions")
async def analyze_proportions(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Analyze facial proportions only
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        results = analyzer.analyze_proportions(image_array, confidence_threshold)
        
        if not results:
            return JSONResponse(content={"error": "No face detected"})
        
        return JSONResponse(content={
            "proportions": results['proportions'],
            "custom_model_points": results['custom_model_points'],
            "landmarks_count": results['landmarks_count'],
            "proportion_calculations": {
                "face_proportions_calculated": results['proportions']['face_proportions']['right'] is not None,
                "forehead_proportions_calculated": results['proportions']['forehead_proportions']['right'] is not None,
                "temporal_proportions_calculated": results['proportions']['temporal_proportions']['right'] is not None
            },
            "confidence_threshold": confidence_threshold
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proportion analysis failed: {str(e)}")

@app.post("/get-diagnosis")
async def get_diagnosis(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    format: str = Form("json")  # "json" or "text"
):
    """
    Get final diagnosis with decision tree and hybrid splitting results
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        results = analyzer.get_diagnosis(image_array, confidence_threshold)
        
        if not results:
            return JSONResponse(content={"error": "No face detected"})
        
        if format == "json":
            return JSONResponse(content={
                "final_diagnosis": results['final_diagnosis'],
                "decision_tree_analysis": results['decision_tree_analysis'],
                "confidence_scores": results['confidence_scores'],
                "applied_rules": results['applied_rules'],
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence_threshold": confidence_threshold,
                    "face_detected": True,
                    "proportions_used": results['proportions_used']
                }
            })
        else:
            # Return text format
            report = analyzer.generate_text_report(results)
            return JSONResponse(content={"report": report})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")

@app.get("/results/{filename}")
async def get_result_image(filename: str):
    """
    Retrieve generated visualization or dashboard image
    """
    file_path = f"/app/results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/model-info")
async def get_model_info():
    """
    Get information about loaded models
    """
    return {
        "service": "Espejo Analysis API",
        "version": "1.0.0",
        "models": {
            "facial_points_detection": "Custom trained Faster R-CNN model",
            "binary_region_classifier": "Binary CNN for FRENTE/rostro_menton classification",
            "frente_classifier": "CNN for forehead region classification",
            "rostro_menton_classifier": "CNN for chin/jaw region classification",
            "landmark_detector": "dlib 68-point facial landmark detector"
        },
        "analysis_methods": {
            "anthropometric_analysis": "Face, forehead, and temporal proportion calculations",
            "mirror_generation": "Left and right mirrored face generation",
            "decision_tree_classification": "Excel-based decision tree rules",
            "hybrid_class_splitting": "Proportion-based class splitting"
        },
        "supported_formats": ["jpg", "jpeg", "png", "bmp"],
        "gpu_acceleration": "CUDA supported with CPU fallback"
    }

@app.get("/api-info")
async def get_api_info():
    """
    Get information about all available endpoints
    """
    return {
        "service": "Espejo Analysis API",
        "version": "1.0.0",
        "description": "Advanced mirror face analysis with decision tree classification",
        "endpoints": {
            "/analyze-espejo": "Complete espejo analysis with all features",
            "/generate-mirrors": "Generate mirror images only",
            "/classify-regions": "Classify facial regions with decision tree",
            "/analyze-proportions": "Analyze facial proportions only",
            "/get-diagnosis": "Get final diagnosis with rules applied",
            "/results/{filename}": "Retrieve generated images",
            "/model-info": "Get model information",
            "/health": "Health check",
            "/api-info": "This endpoint"
        },
        "features": {
            "mirror_analysis": "Generate left and right mirrored faces",
            "anthropometric_measurements": "Calculate face, forehead, and temporal proportions",
            "decision_tree_classification": "Apply Excel-based decision rules",
            "hybrid_splitting": "Proportion-based class splitting",
            "comprehensive_reporting": "Detailed analysis reports and visualizations"
        },
        "next_port": 8008
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)