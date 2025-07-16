import cv2
import numpy as np
from PIL import Image
import io
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)

async def validate_image(file: UploadFile) -> np.ndarray:
    """
    Validate and load image from uploaded file
    
    Args:
        file: Uploaded image file
        
    Returns:
        np.ndarray: Image as numpy array in RGB format
        
    Raises:
        HTTPException: If image is invalid or cannot be processed
    """
    # Check file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(content))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Validate image dimensions
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        height, width = image_array.shape[:2]
        if height < 50 or width < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50 pixels)")
        
        if height > 4000 or width > 4000:
            raise HTTPException(status_code=400, detail="Image too large (maximum 4000x4000 pixels)")
        
        logger.info(f"Image validated: {width}x{height}, {file.content_type}")
        return image_array
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def preprocess_image(image: np.ndarray, target_size: tuple = None) -> np.ndarray:
    """
    Preprocess image for analysis
    
    Args:
        image: Input image as numpy array
        target_size: Optional target size (width, height) for resizing
        
    Returns:
        np.ndarray: Preprocessed image
    """
    try:
        # Convert RGB to BGR for OpenCV operations
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            processed_image = image.copy()
        
        # Resize if target size specified
        if target_size:
            processed_image = cv2.resize(processed_image, target_size, interpolation=cv2.INTER_AREA)
        
        # Basic noise reduction
        processed_image = cv2.bilateralFilter(processed_image, 9, 75, 75)
        
        return processed_image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image

def crop_image_with_bbox(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Crop image using bounding box coordinates
    
    Args:
        image: Input image as numpy array
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
        
    Returns:
        np.ndarray: Cropped image
    """
    try:
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x_min = max(0, min(int(x_min), width - 1))
        y_min = max(0, min(int(y_min), height - 1))
        x_max = max(x_min + 1, min(int(x_max), width))
        y_max = max(y_min + 1, min(int(y_max), height))
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        logger.info(f"Image cropped from {width}x{height} to {x_max-x_min}x{y_max-y_min}")
        return cropped
        
    except Exception as e:
        logger.error(f"Error cropping image: {e}")
        return image

def enhance_hand_region(image: np.ndarray) -> np.ndarray:
    """
    Enhance hand region for better analysis
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        np.ndarray: Enhanced image
    """
    try:
        # Convert to LAB color space for better color processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply slight gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing hand region: {e}")
        return image

def create_skin_mask_advanced(image: np.ndarray) -> np.ndarray:
    """
    Create advanced skin mask using multiple color spaces
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        np.ndarray: Binary mask where skin pixels are white (255)
    """
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # HSV skin range (more permissive)
        lower_hsv = np.array([0, 10, 60])
        upper_hsv = np.array([25, 180, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb skin range
        lower_ycrcb = np.array([0, 130, 80])
        upper_ycrcb = np.array([255, 185, 140])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # Morphological operations to clean up mask
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove noise
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill holes
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Remove small components
        skin_mask = cv2.medianBlur(skin_mask, 5)
        
        return skin_mask
        
    except Exception as e:
        logger.error(f"Error creating skin mask: {e}")
        # Return empty mask on error
        return np.zeros(image.shape[:2], dtype=np.uint8)

def validate_bbox_coordinates(bbox: list, image_shape: tuple) -> tuple:
    """
    Validate and adjust bounding box coordinates
    
    Args:
        bbox: Bounding box coordinates [x_min, y_min, x_max, y_max]
        image_shape: Image shape (height, width, channels)
        
    Returns:
        tuple: Validated bounding box coordinates
    """
    try:
        height, width = image_shape[:2]
        
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure coordinates are integers
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        # Ensure coordinates are within bounds
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(x_min + 1, min(x_max, width))
        y_max = max(y_min + 1, min(y_max, height))
        
        # Ensure minimum size
        min_size = 50
        if x_max - x_min < min_size:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - min_size // 2)
            x_max = min(width, x_min + min_size)
        
        if y_max - y_min < min_size:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - min_size // 2)
            y_max = min(height, y_min + min_size)
        
        return (x_min, y_min, x_max, y_max)
        
    except Exception as e:
        logger.error(f"Error validating bbox coordinates: {e}")
        # Return full image bounds on error
        height, width = image_shape[:2]
        return (0, 0, width, height)

def resize_image_maintaining_aspect(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image as numpy array
        max_size: Maximum dimension (width or height)
        
    Returns:
        np.ndarray: Resized image
    """
    try:
        height, width = image.shape[:2]
        
        if max(height, width) <= max_size:
            return image
        
        # Calculate new dimensions
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        return resized
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image