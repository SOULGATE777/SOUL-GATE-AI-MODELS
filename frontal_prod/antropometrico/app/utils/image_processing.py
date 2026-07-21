import cv2
import numpy as np
from fastapi import UploadFile, HTTPException
import io
from PIL import Image

async def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """
    Process uploaded image file and convert to numpy array
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        numpy.ndarray: Image as numpy array in BGR format
        
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Validate image dimensions
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            raise HTTPException(status_code=400, detail="Image too small (minimum 100x100)")
        
        if height > 4000 or width > 4000:
            # Resize large images
            scale = min(4000/height, 4000/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

def validate_image_quality(image: np.ndarray) -> dict:
    """
    Validate image quality for facial analysis
    
    Args:
        image: Input image as numpy array
        
    Returns:
        dict: Quality assessment results
    """
    height, width = image.shape[:2]
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate image sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = gray.std()
    
    # Assess quality
    quality_score = 0
    warnings = []
    
    # Sharpness check
    if laplacian_var > 100:
        quality_score += 25
    elif laplacian_var < 50:
        warnings.append("Image may be blurry")
    else:
        quality_score += 15
    
    # Brightness check
    if 50 <= brightness <= 200:
        quality_score += 25
    elif brightness < 30:
        warnings.append("Image is too dark")
    elif brightness > 220:
        warnings.append("Image is too bright")
    else:
        quality_score += 15
    
    # Contrast check
    if contrast > 40:
        quality_score += 25
    elif contrast < 20:
        warnings.append("Image has low contrast")
    else:
        quality_score += 15
    
    # Size check
    if width >= 400 and height >= 400:
        quality_score += 25
    elif width >= 200 and height >= 200:
        quality_score += 15
        warnings.append("Image resolution is low")
    else:
        warnings.append("Image resolution is very low")
    
    return {
        "quality_score": quality_score,
        "sharpness": laplacian_var,
        "brightness": brightness,
        "contrast": contrast,
        "resolution": f"{width}x{height}",
        "warnings": warnings,
        "suitable_for_analysis": quality_score >= 60
    }

def preprocess_for_analysis(image: np.ndarray, target_size: tuple = None) -> tuple:
    """
    Preprocess image for facial analysis
    
    Args:
        image: Input image
        target_size: Optional target size (width, height)
        
    Returns:
        tuple: (processed_image, grayscale_image, scale_factor)
    """
    original_height, original_width = image.shape[:2]
    
    # Calculate scale factor if target size is specified
    scale_factor = 1.0
    processed_image = image.copy()
    
    if target_size:
        target_width, target_height = target_size
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale_factor = min(scale_x, scale_y)
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        processed_image = cv2.resize(image, (new_width, new_height))
    
    # Grayscale for detection internals — no CLAHE.
    # Illumination CLAHE is owned by frontal_prod/preprocesamiento (maybe_enhance_dark).
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    return processed_image, gray, scale_factor

def enhance_image_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Pass-through (no CLAHE).

    Illumination CLAHE is owned by frontal_prod/preprocesamiento
    (maybe_enhance_dark). Do not re-apply on preprocessed crops.
    """
    return image

def resize_image_for_model(image: np.ndarray, target_size: int = 224) -> tuple:
    """
    Resize image for model input while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target size for the largest dimension
        
    Returns:
        tuple: (resized_image, scale_factor)
    """
    height, width = image.shape[:2]
    
    # Calculate scale factor
    scale = target_size / max(height, width)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Pad to square if needed
    if new_width != new_height:
        # Create square canvas
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_size - new_width) // 2
        pad_y = (target_size - new_height) // 2
        
        # Place resized image on canvas
        canvas[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        
        return canvas, scale
    
    return resized, scale

def save_image_result(image: np.ndarray, filepath: str, quality: int = 95) -> bool:
    """
    Save image result with error handling
    
    Args:
        image: Image to save
        filepath: Output file path
        quality: JPEG quality (0-100)
        
    Returns:
        bool: Success status
    """
    try:
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save image
        if filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg'):
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(filepath, image)
        
        return True
        
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def create_thumbnail(image: np.ndarray, size: tuple = (150, 150)) -> np.ndarray:
    """
    Create thumbnail of the image
    
    Args:
        image: Input image
        size: Thumbnail size (width, height)
        
    Returns:
        numpy.ndarray: Thumbnail image
    """
    return cv2.resize(image, size)

def convert_to_base64(image: np.ndarray) -> str:
    """
    Convert image to base64 string
    
    Args:
        image: Input image
        
    Returns:
        str: Base64 encoded image
    """
    import base64
    
    # Encode image
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert to base64
    base64_string = base64.b64encode(buffer).decode('utf-8')
    
    return base64_string
