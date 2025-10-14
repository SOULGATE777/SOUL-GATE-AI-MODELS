from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
import uvicorn
import os
import cv2
import numpy as np
from typing import Optional
import uuid
from datetime import datetime

from app.models.anthropometric_pipeline import AnthropometricAnalyzer
from app.models.eye_colorimetry_analyzer import EyeColorimetryAnalyzer
from app.utils.visualization import create_visualization, create_detailed_report_image
from app.utils.image_processing import process_uploaded_image
from app.utils.lazy_model_loader import MultiModelLoader

app = FastAPI(
    title="Antropometrico Analysis API",
    description="Advanced anthropometric facial analysis with custom point detection and comprehensive features",
    version="2.0.0"
)

# Initialize lazy model loader
model_loader = MultiModelLoader()

# Register models for lazy loading
def _load_anthropometric_analyzer():
    """Lazy load function for anthropometric analyzer"""
    print("🔄 Loading anthropometric analyzer...")
    analyzer = AnthropometricAnalyzer()
    print("✅ Anthropometric analyzer loaded successfully!")
    return analyzer

def _load_eye_colorimetry_analyzer():
    """Lazy load function for eye colorimetry analyzer"""
    print("🔄 Loading eye colorimetry analyzer...")
    analyzer = EyeColorimetryAnalyzer()
    print("✅ Eye colorimetry analyzer loaded successfully!")
    return analyzer

@app.on_event("startup")
async def startup_event():
    """Register models for lazy loading (no actual loading here)"""
    print("🚀 Initializing Antropometrico Analysis API with lazy loading...")

    # Register models without loading them
    model_loader.register_model("analyzer", _load_anthropometric_analyzer)
    model_loader.register_model("eye_analyzer", _load_eye_colorimetry_analyzer)

    print("✅ Models registered for lazy loading. They will load on first request.")
    print("💾 RAM saved: Models will only load when needed!")

# Helper functions to get models
def get_analyzer():
    """Get anthropometric analyzer, loading it if necessary"""
    return model_loader.get_model("analyzer")

def get_eye_analyzer():
    """Get eye colorimetry analyzer, loading it if necessary"""
    return model_loader.get_model("eye_analyzer")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = model_loader.get_loaded_models()
    return {
        "status": "healthy",
        "service": "antropometrico",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "lazy_loading_enabled": True,
        "models_registered": ["analyzer", "eye_analyzer"],
        "models_currently_loaded": loaded_models,
        "features": [
            "facial_landmarks_detection",
            "custom_model_points",
            "eyebrow_length_analysis",
            "eye_angle_analysis",
            "eye_face_area_proportions",
            "inner_outer_face_analysis",
            "comprehensive_reporting",
            "eye_colorimetry_analysis",
            "iris_color_classification",
            "rgb_hsv_color_systems",
            "lazy_loading"
        ]
    }

@app.post("/analyze-anthropometric")
async def analyze_anthropometric(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    include_visualization: bool = Form(True),
    include_detailed_report: bool = Form(False)
):
    """
    Complete anthropometric facial analysis with all features
    
    Args:
        file: Input image file
        confidence_threshold: Confidence threshold for model predictions (0.0-1.0)
        include_visualization: Whether to generate visualization
        include_detailed_report: Whether to generate detailed report image
    
    Returns:
        JSON response with comprehensive analysis results
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        # Process uploaded image
        image_array = await process_uploaded_image(file)
        
        # Perform analysis
        results = analyzer.analyze_face(
            image_array, 
            confidence_threshold=confidence_threshold
        )
        
        if not results:
            return JSONResponse(content={
                "error": "No face detected in the image",
                "facial_landmarks": {"count": 0},
                "model_predictions": {},
                "analysis_summary": {}
            })
        
        visualization_path = None
        detailed_report_path = None
        
        if include_visualization:
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
                results['slopes'],
                results.get('calculated_c1')
            )
            
            # Save visualization
            os.makedirs("/app/results", exist_ok=True)
            cv2.imwrite(visualization_path, vis_image)
        
        if include_detailed_report:
            # Generate detailed report image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            report_filename = f"detailed_report_{timestamp}_{unique_id}.jpg"
            detailed_report_path = f"/app/results/{report_filename}"
            
            # Create detailed report image
            report_image = create_detailed_report_image(image_array, results)
            
            # Save detailed report
            os.makedirs("/app/results", exist_ok=True)
            cv2.imwrite(detailed_report_path, report_image)
        
        # Prepare comprehensive response
        response = {
            "facial_landmarks": {
                "count": 68,
                "extended_points": len(results['extended_points']) if results else 0
            },
            "model_predictions": results['model_predictions'],
            "proportions": results['proportions'],
            "slopes": results['slopes'],
            "eyebrow_analysis": {
                "proportions": results['eyebrow_proportions'],
                "classifications": {
                    "left_eyebrow": analyzer._classify_eyebrow_length(results['eyebrow_proportions']['left_eyebrow_proportion']),
                    "right_eyebrow": analyzer._classify_eyebrow_length(results['eyebrow_proportions']['right_eyebrow_proportion'])
                }
            },
            "eye_analysis": {
                "angles": results['eye_angles'],
                "classifications": {
                    "left_eye_angle": analyzer._classify_eye_angle(results['eye_angles']['left_eye_angle']),
                    "right_eye_angle": analyzer._classify_eye_angle(results['eye_angles']['right_eye_angle'])
                },
                "eyebrow_eyelid_distances": results['eyebrow_eyelid_distances'],
                "face_proportions": results['eye_face_proportions']
            },
            "mouth_analysis": {
                "cupid_arches": {
                    "left_cupid_arch": results['mouth_measurements']['left_cupid_arch_proportion'],
                    "right_cupid_arch": results['mouth_measurements']['right_cupid_arch_proportion']
                },
                "lips_ratio": results['mouth_measurements']['lips_ratio'],
                "measurements": results['mouth_measurements']
            },
            "face_area_analysis": results['inner_outer_proportions'],
            "analysis_summary": results['summary'],
            "visualization_path": visualization_path,
            "detailed_report_path": detailed_report_path,
            "confidence_threshold": confidence_threshold
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ DETAILED ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-eyebrows")
async def analyze_eyebrows(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Analyze eyebrow proportions and characteristics
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        results = analyzer.analyze_face(image_array, confidence_threshold)
        
        if not results:
            return JSONResponse(content={"error": "No face detected"})
        
        eyebrow_analysis = {
            "eyebrow_proportions": results['eyebrow_proportions'],
            "classifications": {
                "left_eyebrow": analyzer._classify_eyebrow_length(results['eyebrow_proportions']['left_eyebrow_proportion']),
                "right_eyebrow": analyzer._classify_eyebrow_length(results['eyebrow_proportions']['right_eyebrow_proportion'])
            },
            "slopes": results['slopes'],
            "summary": {
                "length_analysis": results['summary']['eyebrow_analysis'],
                "slope_analysis": results['summary']['eyebrow_slope_analysis']
            }
        }
        
        return JSONResponse(content=eyebrow_analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eyebrow analysis failed: {str(e)}")

@app.post("/analyze-eyes")
async def analyze_eyes(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Analyze eye angles and proportions
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        results = analyzer.analyze_face(image_array, confidence_threshold)
        
        if not results:
            return JSONResponse(content={"error": "No face detected"})
        
        eye_analysis = {
            "eye_angles": results['eye_angles'],
            "angle_classifications": {
                "left_eye": analyzer._classify_eye_angle(results['eye_angles']['left_eye_angle']),
                "right_eye": analyzer._classify_eye_angle(results['eye_angles']['right_eye_angle'])
            },
            "eyebrow_eyelid_distances": results['eyebrow_eyelid_distances'],
            "eye_face_proportions": results['eye_face_proportions'],
            "internal_eye_proportion": results['proportions']['eye_distance_proportion'],
            "summary": results['summary']['eye_analysis']
        }
        
        return JSONResponse(content=eye_analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eye analysis failed: {str(e)}")

@app.post("/analyze-face-areas")
async def analyze_face_areas(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Analyze face area proportions (inner/outer)
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        results = analyzer.analyze_face(image_array, confidence_threshold)
        
        if not results:
            return JSONResponse(content={"error": "No face detected"})
        
        area_analysis = {
            "inner_outer_proportions": results['inner_outer_proportions'],
            "eye_face_proportions": results['eye_face_proportions'],
            "total_facial_measurements": {
                "head_width_proportion": results['proportions']['head_width_proportion'],
                "mouth_length_proportion": results['proportions']['mouth_length_proportion'],
                "chin_to_face_width_proportion": results['proportions']['chin_to_face_width_proportion']
            },
            "summary": results['summary']['face_area_analysis']
        }
        
        return JSONResponse(content=area_analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face area analysis failed: {str(e)}")

@app.post("/analyze-mouth")
async def analyze_mouth(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Analyze mouth measurements including cupid's bow arches and lips ratio
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")

        image_array = await process_uploaded_image(file)
        results = analyzer.analyze_face(image_array, confidence_threshold)

        if not results:
            return JSONResponse(content={"error": "No face detected"})

        mouth_analysis = {
            "integral_diagnosis": results['summary']['mouth_analysis']['integral_diagnosis'],
            "mouth_length": {
                "proportion": results['proportions']['mouth_length_proportion'],
                "percentage": results['proportions']['mouth_length_proportion'] * 100,
                "classification": analyzer._classify_mouth_length(results['proportions']['mouth_length_proportion'])
            },
            "mouth_to_eye": {
                "proportion": results['proportions']['mouth_to_eye_proportion'],
                "classification": results['summary']['mouth_analysis']['mouth_to_eye_relation']
            },
            "cupid_arches": {
                "left_cupid_arch_proportion": results['mouth_measurements']['left_cupid_arch_proportion'],
                "right_cupid_arch_proportion": results['mouth_measurements']['right_cupid_arch_proportion'],
                "left_cupid_arch_distance": results['mouth_measurements']['left_cupid_arch_distance'],
                "right_cupid_arch_distance": results['mouth_measurements']['right_cupid_arch_distance']
            },
            "lips_ratio": {
                "ratio": results['mouth_measurements']['lips_ratio'],
                "upper_lip_distance": results['mouth_measurements']['upper_lip_distance'],
                "lower_lip_distance": results['mouth_measurements']['lower_lip_distance']
            },
            "summary": results['summary']['mouth_analysis']
        }

        return JSONResponse(content=mouth_analysis)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mouth analysis failed: {str(e)}")

@app.post("/get-detailed-report")
async def get_detailed_report(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    format: str = Form("text")  # "text" or "json"
):
    """
    Generate a detailed analysis report
    
    Args:
        file: Input image file
        confidence_threshold: Confidence threshold for model predictions
        format: Output format ("text" or "json")
    """
    try:
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        results = analyzer.analyze_face(image_array, confidence_threshold)
        
        if not results:
            return JSONResponse(content={"error": "No face detected"})
        
        if format == "text":
            report_text = analyzer.get_detailed_analysis_report(results)
            return PlainTextResponse(content=report_text)
        else:
            # Return comprehensive JSON report
            detailed_report = {
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "confidence_threshold": confidence_threshold,
                    "model_points_detected": len(results['model_predictions']),
                    "extended_points_count": len(results['extended_points'])
                },
                "facial_thirds": {
                    "primer_tercio": {
                        "value": results['proportions']['distance_69_68_proportion'],
                        "classification": results['summary']['facial_thirds']['primer_tercio']
                    },
                    "segundo_tercio": {
                        "value": results['proportions']['distance_68_34_proportion'],
                        "classification": results['summary']['facial_thirds']['segundo_tercio']
                    },
                    "tercer_tercio": {
                        "value": results['proportions']['distance_34_9_proportion'],
                        "classification": results['summary']['facial_thirds']['tercer_tercio']
                    }
                },
                "eye_analysis": {
                    "eye_angles": results['eye_angles'],
                    "eyebrow_eyelid_distances": results['eyebrow_eyelid_distances']
                },
                "mouth_analysis": results['mouth_measurements'],
                "eyebrow_analysis": results['eyebrow_proportions'],
                "face_proportions": results['proportions'],
                "area_analysis": results['inner_outer_proportions'],
                "model_integration": results['model_predictions'],
                "summary": results['summary']
            }
            
            return JSONResponse(content=detailed_report)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/detect-landmarks")
async def detect_landmarks(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Detect facial landmarks only
    """
    try:
        analyzer = get_analyzer()
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
        analyzer = get_analyzer()
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        image_array = await process_uploaded_image(file)
        model_predictions = analyzer.detect_model_points_only(image_array, confidence_threshold)
        
        return JSONResponse(content={
            "model_predictions": model_predictions,
            "points_detected": len(model_predictions),
            "confidence_threshold": confidence_threshold,
            "point_descriptions": {
                "1": "Custom point 1",
                "2": "Between eyebrows", 
                "3": "Top of head",
                "4": "Custom point 4",
                "5": "Custom point 5",
                "6": "Custom point 6",
                "7": "Custom point 7",
                "8": "Custom point 8",
                "9": "Custom point 9",
                "10": "Custom point 10",
                "11": "Custom point 11",
                "12": "Custom point 12",
                "13": "Custom point 13"
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Point detection failed: {str(e)}")

@app.get("/results/{filename}")
async def get_result_image(filename: str):
    """
    Retrieve generated visualization or report image
    """
    file_path = f"/app/results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.post("/analyze-eye-colorimetry")
async def analyze_eye_colorimetry(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    include_visualization: bool = Form(False)
):
    """
    Perform comprehensive eye colorimetry analysis
    
    Args:
        file: Input image file
        confidence_threshold: Confidence threshold (kept for consistency)
        include_visualization: Whether to generate visualization
    
    Returns:
        JSON response with eye colorimetry analysis results
    """
    try:
        eye_analyzer = get_eye_analyzer()
        if eye_analyzer is None:
            raise HTTPException(status_code=500, detail="Eye colorimetry analyzer not initialized")
        
        # Process uploaded image
        image_array = await process_uploaded_image(file)
        
        # Perform eye colorimetry analysis
        results = eye_analyzer.analyze_eye_colorimetry(image_array, confidence_threshold)
        
        if "error" in results:
            return JSONResponse(content=results)
        
        # Add metadata
        results["timestamp"] = datetime.now().isoformat()
        results["confidence_threshold"] = confidence_threshold
        results["include_visualization"] = include_visualization
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eye colorimetry analysis failed: {str(e)}")

@app.post("/analyze-iris-color")
async def analyze_iris_color(
    file: UploadFile = File(...),
    use_dominant_color: bool = Form(False)
):
    """
    Focused iris color analysis with simplified output
    
    Args:
        file: Input image file
        use_dominant_color: Whether to use dominant color for classification
    
    Returns:
        JSON response with iris color classifications
    """
    try:
        eye_analyzer = get_eye_analyzer()
        if eye_analyzer is None:
            raise HTTPException(status_code=500, detail="Eye colorimetry analyzer not initialized")
        
        # Process uploaded image
        image_array = await process_uploaded_image(file)
        
        # Perform analysis
        results = eye_analyzer.analyze_eye_colorimetry(image_array)
        
        if "error" in results:
            return JSONResponse(content=results)
        
        # Extract simplified iris color results
        simplified_results = {
            "timestamp": datetime.now().isoformat(),
            "total_faces": results.get("total_faces", 0),
            "faces": []
        }
        
        for face in results.get("faces", []):
            face_result = {"face_index": face.get("face_index", 0)}
            
            for eye_side in ['left', 'right']:
                eye_key = f"{eye_side}_eye"
                if eye_key in face and "error" not in face[eye_key]:
                    eye_data = face[eye_key]
                    classifications = eye_data.get("classifications", {})
                    iris_analysis = eye_data.get("iris_color_analysis", {})
                    
                    face_result[f"{eye_side}_eye"] = {
                        "rgb_classification": classifications.get("iris_rgb_dominant" if use_dominant_color else "iris_rgb_average", "unknown"),
                        "hsv_classification": classifications.get("iris_hsv", "unknown"),
                        "average_rgb": iris_analysis.get("average_color_rgb", [0, 0, 0]),
                        "dominant_rgb": iris_analysis.get("dominant_colors", [[0, 0, 0]])[0][0] if iris_analysis.get("dominant_colors") else [0, 0, 0],
                        "pixels_analyzed": iris_analysis.get("total_pixels_analyzed", 0)
                    }
                else:
                    face_result[f"{eye_side}_eye"] = {"error": "Analysis failed"}
            
            simplified_results["faces"].append(face_result)
        
        return JSONResponse(content=simplified_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Iris color analysis failed: {str(e)}")

@app.post("/compare-eye-colors")
async def compare_eye_colors(
    file: UploadFile = File(...)
):
    """
    Compare eye color classifications across different methods
    
    Args:
        file: Input image file
    
    Returns:
        JSON response with comparison of classification methods
    """
    try:
        eye_analyzer = get_eye_analyzer()
        if eye_analyzer is None:
            raise HTTPException(status_code=500, detail="Eye colorimetry analyzer not initialized")
        
        # Process uploaded image
        image_array = await process_uploaded_image(file)
        
        # Perform analysis
        results = eye_analyzer.analyze_eye_colorimetry(image_array)
        
        if "error" in results:
            return JSONResponse(content=results)
        
        # Create comparison results
        comparison_results = {
            "timestamp": datetime.now().isoformat(),
            "total_faces": results.get("total_faces", 0),
            "classification_methods": {
                "hsv_system": "Original HSV-based classification",
                "rgb_average": "RGB ranges using average color",
                "rgb_dominant": "RGB ranges using dominant color"
            },
            "faces": []
        }
        
        for face in results.get("faces", []):
            face_result = {"face_index": face.get("face_index", 0)}
            
            for eye_side in ['left', 'right']:
                eye_key = f"{eye_side}_eye"
                if eye_key in face and "error" not in face[eye_key]:
                    eye_data = face[eye_key]
                    classifications = eye_data.get("classifications", {})
                    
                    face_result[f"{eye_side}_eye"] = {
                        "iris_classifications": {
                            "hsv_system": classifications.get("iris_hsv", "unknown"),
                            "rgb_average": classifications.get("iris_rgb_average", "unknown"),
                            "rgb_dominant": classifications.get("iris_rgb_dominant", "unknown")
                        },
                        "agreement": {
                            "rgb_methods_agree": classifications.get("iris_rgb_average") == classifications.get("iris_rgb_dominant"),
                            "all_methods_agree": len(set([
                                classifications.get("iris_hsv", "unknown"),
                                classifications.get("iris_rgb_average", "unknown"),
                                classifications.get("iris_rgb_dominant", "unknown")
                            ])) == 1
                        }
                    }
                else:
                    face_result[f"{eye_side}_eye"] = {"error": "Analysis failed"}
            
            comparison_results["faces"].append(face_result)
        
        return JSONResponse(content=comparison_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eye color comparison failed: {str(e)}")

@app.get("/api-info")
async def get_api_info():
    """
    Get information about all available endpoints
    """
    return {
        "service": "Antropometrico Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "/analyze-anthropometric": "Complete facial analysis with all features",
            "/analyze-eyebrows": "Focused eyebrow analysis",
            "/analyze-eyes": "Eye angle and proportion analysis",
            "/analyze-mouth": "Mouth measurements including cupid's bow arches and lips ratio",
            "/analyze-face-areas": "Face area proportion analysis",
            "/get-detailed-report": "Generate comprehensive report",
            "/detect-landmarks": "Detect facial landmarks only",
            "/detect-points": "Detect custom model points only",
            "/analyze-eye-colorimetry": "Complete eye colorimetry analysis",
            "/analyze-iris-color": "Focused iris color analysis",
            "/compare-eye-colors": "Compare eye color classification methods",
            "/results/{filename}": "Retrieve generated images",
            "/health": "Health check",
            "/api-info": "This endpoint"
        },
        "new_features": {
            "eyebrow_length_analysis": "Classifies eyebrow length relative to eye length",
            "eye_angle_analysis": "Measures and classifies eye angles",
            "eyebrow_eyelid_distances": "Measures proportional distances from eyebrow to eyelid (points 19-37, 24-44)",
            "mouth_measurements": "Analyzes cupid's bow arches (points 50-61, 52-63) and lips ratio (51-62/66-57)",
            "face_area_proportions": "Analyzes inner/outer face area ratios",
            "comprehensive_reporting": "Detailed text and JSON reports",
            "enhanced_model_integration": "Uses all 13 custom model points",
            "eye_colorimetry_analysis": "RGB and HSV-based iris color classification",
            "iris_color_classification": "Multiple classification systems for eye colors",
            "color_comparison_methods": "Compare different color analysis approaches"
        }
  }

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8001)


