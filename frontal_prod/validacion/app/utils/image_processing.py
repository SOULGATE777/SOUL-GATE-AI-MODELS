import cv2
import numpy as np
import os
import tempfile
import shutil
from typing import Optional, Tuple
import logging
from fastapi import UploadFile

logger = logging.getLogger(__name__)

async def save_uploaded_file(file: UploadFile) -> str:
    """
    Save uploaded file to temporary location
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Path to saved temporary file
    """
    try:
        # Create temporary file with proper extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_extension)
        
        # Write file contents
        with os.fdopen(temp_fd, 'wb') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        logger.info(f"✅ File saved to temporary location: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise

def cleanup_temp_files(*file_paths: str):
    """
    Cleanup temporary files
    
    Args:
        file_paths: Paths to files to be deleted
    """
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not cleanup file {file_path}: {e}")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better detection
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image
    """
    try:
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR input from OpenCV
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            processed = image.copy()
        
        # Basic image enhancement
        # Adjust contrast and brightness slightly
        alpha = 1.1  # Contrast control
        beta = 10    # Brightness control
        processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
        
        return processed
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image

def validate_image_format(image_path: str) -> Tuple[bool, str]:
    """
    Validate image format and readability
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not os.path.exists(image_path):
            return False, f"Image file not found: {image_path}"
        
        # Try to load image
        image = cv2.imread(image_path)
        if image is None:
            return False, "Could not read image file"
        
        # Check image dimensions
        if len(image.shape) < 2:
            return False, "Invalid image dimensions"
        
        # Check minimum size
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return False, f"Image too small: {width}x{height} (minimum 100x100)"
        
        # Check maximum size (optional, for memory management)
        if height > 4000 or width > 4000:
            return False, f"Image too large: {width}x{height} (maximum 4000x4000)"
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def resize_image_if_needed(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """
    Resize image if it's too large
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image if needed, original otherwise
    """
    try:
        height, width = image.shape[:2]
        
        if max(height, width) <= max_size:
            return image
        
        # Calculate scaling factor
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
        return resized
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image
