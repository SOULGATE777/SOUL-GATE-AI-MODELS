import cv2
import numpy as np
import uuid
import os
import aiofiles
from fastapi import UploadFile
from typing import Tuple
from PIL import Image
import io

async def process_uploaded_image(file: UploadFile) -> Tuple[np.ndarray, str]:
    """
    Process uploaded image file and return numpy array and temp file path
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Tuple of (image_array, temp_file_path)
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "/app/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
        temp_filename = f"temp_{uuid.uuid4().hex}{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save uploaded file
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Load image with OpenCV
        image = cv2.imread(temp_path)
        if image is None:
            # Try with PIL if OpenCV fails
            pil_image = Image.open(temp_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb, temp_path
        
    except Exception as e:
        print(f"Error processing uploaded image: {e}")
        raise e

def resize_image_maintain_aspect(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image as numpy array
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_width, new_height = width, height
    else:
        if height > max_size:
            new_height = max_size
            new_width = int((width * max_size) / height)
        else:
            new_width, new_height = width, height
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Apply basic image enhancement for better model performance
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def validate_image(image: np.ndarray) -> bool:
    """
    Validate if the image is suitable for processing
    
    Args:
        image: Input image as numpy array
        
    Returns:
        True if valid, False otherwise
    """
    if image is None:
        return False
    
    # Check dimensions
    if len(image.shape) != 3:
        return False
    
    height, width, channels = image.shape
    
    # Check minimum size
    if height < 50 or width < 50:
        return False
    
    # Check if it's too large
    if height > 4000 or width > 4000:
        return False
    
    # Check channels
    if channels != 3:
        return False
    
    return True

def preprocess_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image specifically for detection models
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image
    """
    # Validate image
    if not validate_image(image):
        raise ValueError("Invalid image for processing")
    
    # Resize to reasonable size for processing speed
    processed = resize_image_maintain_aspect(image, max_size=800)
    
    # Enhance quality
    processed = enhance_image_quality(processed)
    
    # Normalize pixel values
    processed = processed.astype(np.float32) / 255.0
    processed = (processed * 255).astype(np.uint8)
    
    return processed

def create_image_thumbnail(image: np.ndarray, size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """
    Create a thumbnail version of the image
    
    Args:
        image: Input image as numpy array
        size: Thumbnail size as (width, height)
        
    Returns:
        Thumbnail image
    """
    thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return thumbnail

def save_image_with_metadata(image: np.ndarray, output_path: str, 
                           metadata: dict = None) -> str:
    """
    Save image with optional metadata
    
    Args:
        image: Image to save
        output_path: Output file path
        metadata: Optional metadata dictionary
        
    Returns:
        Path to saved file
    """
    try:
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Save image
        success = cv2.imwrite(output_path, image_bgr)
        
        if not success:
            raise Exception("Failed to save image with OpenCV")
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.replace(os.path.splitext(output_path)[1], '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return output_path
        
    except Exception as e:
        print(f"Error saving image: {e}")
        raise e

def cleanup_temp_files(temp_path: str) -> None:
    """
    Clean up temporary files
    
    Args:
        temp_path: Path to temporary file to remove
    """
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Warning: Could not remove temp file {temp_path}: {e}")

def batch_process_images(image_paths: list, processor_func, **kwargs) -> list:
    """
    Process multiple images with the same function
    
    Args:
        image_paths: List of image file paths
        processor_func: Function to apply to each image
        **kwargs: Additional arguments for processor_func
        
    Returns:
        List of processing results
    """
    results = []
    
    for image_path in image_paths:
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process image
                result = processor_func(image_rgb, **kwargs)
                results.append({
                    'image_path': image_path,
                    'result': result,
                    'status': 'success'
                })
            else:
                results.append({
                    'image_path': image_path,
                    'result': None,
                    'status': 'failed_to_load'
                })
                
        except Exception as e:
            results.append({
                'image_path': image_path,
                'result': None,
                'status': 'error',
                'error': str(e)
            })
    
    return results
