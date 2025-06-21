import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import Tuple, Optional, List
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

async def validate_image(file: UploadFile) -> np.ndarray:
    """
    Validate and load uploaded image file
    
    Args:
        file: Uploaded image file
    
    Returns:
        Image as numpy array in RGB format
    
    Raises:
        HTTPException: If image validation fails
    """
    try:
        # Check file extension
        if file.filename:
            file_ext = '.' + file.filename.lower().split('.')[-1]
            if file_ext not in SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
                )
        
        # Read file content
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Load image using PIL
        try:
            pil_image = Image.open(io.BytesIO(content))
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
        except Exception as e:
            # Fallback to OpenCV
            logger.warning(f"PIL failed, trying OpenCV: {e}")
            nparr = np.frombuffer(content, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image_array is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
            
            # Convert BGR to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Validate image dimensions
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise HTTPException(status_code=400, detail="Image must be in RGB format")
        
        height, width = image_array.shape[:2]
        if height < 50 or width < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50 pixels)")
        
        if height > 4000 or width > 4000:
            logger.warning(f"Large image detected ({width}x{height}), consider resizing for better performance")
        
        logger.info(f"Image validated successfully: {width}x{height} pixels")
        return image_array
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Image validation failed: {str(e)}")

def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Preprocess image for body analysis
    
    Args:
        image: Input image as numpy array
        target_size: Optional target size (width, height)
    
    Returns:
        Preprocessed image
    """
    try:
        # Ensure image is in correct format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB format")
        
        processed_image = image.copy()
        
        # Resize if target size specified
        if target_size:
            width, height = target_size
            processed_image = cv2.resize(processed_image, (width, height), interpolation=cv2.INTER_AREA)
        
        # Ensure values are in correct range
        if processed_image.dtype != np.uint8:
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
        
        return processed_image
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def crop_image_with_bbox(image: np.ndarray, bbox: List[int], padding: int = 10) -> np.ndarray:
    """
    Crop image using bounding box with optional padding
    
    Args:
        image: Input image as numpy array
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Additional padding around bbox
    
    Returns:
        Cropped image
    """
    try:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Validate crop region
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box coordinates")
        
        cropped = image[y1:y2, x1:x2]
        
        # Ensure minimum size
        if cropped.shape[0] < 50 or cropped.shape[1] < 50:
            logger.warning("Cropped region is very small, returning original image")
            return image
        
        return cropped
        
    except Exception as e:
        logger.error(f"Image cropping error: {e}")
        return image  # Return original if cropping fails

def enhance_image_quality(image: np.ndarray, enhance_contrast: bool = True, 
                         enhance_brightness: bool = True) -> np.ndarray:
    """
    Enhance image quality for better analysis
    
    Args:
        image: Input image as numpy array
        enhance_contrast: Whether to enhance contrast
        enhance_brightness: Whether to adjust brightness
    
    Returns:
        Enhanced image
    """
    try:
        enhanced = image.copy().astype(np.float32)
        
        if enhance_brightness:
            # Auto-adjust brightness
            mean_brightness = np.mean(enhanced)
            target_brightness = 128.0  # Target middle brightness
            brightness_factor = target_brightness / (mean_brightness + 1e-8)
            
            # Limit brightness adjustment
            brightness_factor = np.clip(brightness_factor, 0.7, 1.3)
            enhanced = enhanced * brightness_factor
        
        if enhance_contrast:
            # Enhance contrast using CLAHE
            enhanced_uint8 = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for i in range(3):  # RGB channels
                enhanced_uint8[:, :, i] = clahe.apply(enhanced_uint8[:, :, i])
            
            enhanced = enhanced_uint8.astype(np.float32)
        
        # Ensure values are in valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Image enhancement error: {e}")
        return image  # Return original if enhancement fails

def detect_image_quality_issues(image: np.ndarray) -> dict:
    """
    Detect potential image quality issues
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Dictionary with quality assessment
    """
    try:
        issues = {
            'blur_detected': False,
            'low_contrast': False,
            'too_dark': False,
            'too_bright': False,
            'quality_score': 0.0,
            'recommendations': []
        }
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            issues['blur_detected'] = True
            issues['recommendations'].append("Image appears blurry - use a sharper image")
        
        # Contrast detection
        contrast = gray.std()
        if contrast < 30:
            issues['low_contrast'] = True
            issues['recommendations'].append("Low contrast detected - improve lighting conditions")
        
        # Brightness analysis
        brightness = np.mean(gray)
        if brightness < 50:
            issues['too_dark'] = True
            issues['recommendations'].append("Image is too dark - increase brightness")
        elif brightness > 200:
            issues['too_bright'] = True
            issues['recommendations'].append("Image is too bright - reduce exposure")
        
        # Calculate overall quality score (0-100)
        blur_score = min(100, laplacian_var / 2)  # Normalize to 0-100
        contrast_score = min(100, contrast * 2)   # Normalize to 0-100
        brightness_score = 100 - abs(brightness - 128) / 128 * 100  # Best at 128, worst at extremes
        
        issues['quality_score'] = (blur_score + contrast_score + brightness_score) / 3
        
        # Add general recommendations
        if issues['quality_score'] < 60:
            issues['recommendations'].append("Overall image quality is low - consider retaking the photo")
        
        return issues
        
    except Exception as e:
        logger.error(f"Quality assessment error: {e}")
        return {
            'blur_detected': False,
            'low_contrast': False,
            'too_dark': False,
            'too_bright': False,
            'quality_score': 0.0,
            'recommendations': ["Quality assessment failed"]
        }

def resize_image_for_display(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Resize image for display purposes while maintaining aspect ratio
    
    Args:
        image: Input image as numpy array
        max_size: Maximum dimension size
    
    Returns:
        Resized image
    """
    try:
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_size / width, max_size / height)
        
        if scale < 1.0:  # Only resize if image is larger than max_size
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image
        
    except Exception as e:
        logger.error(f"Image resize error: {e}")
        return image

def normalize_image_orientation(image: np.ndarray) -> np.ndarray:
    """
    Normalize image orientation (ensure it's not rotated)
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Normalized image
    """
    try:
        # For body analysis, we expect images to be in portrait or landscape
        # This is a placeholder for more sophisticated orientation detection
        height, width = image.shape[:2]
        
        # If image is very wide compared to height, it might be rotated
        aspect_ratio = width / height
        
        if aspect_ratio > 2.0:  # Very wide image
            logger.info("Detected potentially rotated wide image")
            # Could add rotation logic here if needed
        
        return image
        
    except Exception as e:
        logger.error(f"Orientation normalization error: {e}")
        return image

def validate_body_region(image: np.ndarray, bbox: Optional[List[int]] = None) -> bool:
    """
    Validate that the image/region contains a suitable body for analysis
    
    Args:
        image: Input image as numpy array
        bbox: Optional bounding box to focus validation
    
    Returns:
        True if region is suitable for body analysis
    """
    try:
        # If bbox provided, crop to that region for validation
        if bbox:
            try:
                validation_image = crop_image_with_bbox(image, bbox)
            except:
                validation_image = image
        else:
            validation_image = image
        
        height, width = validation_image.shape[:2]
        
        # Basic size validation
        if height < 100 or width < 100:
            logger.warning("Region too small for reliable body analysis")
            return False
        
        # Check if image has reasonable aspect ratio for body analysis
        aspect_ratio = height / width
        if aspect_ratio < 0.5 or aspect_ratio > 3.0:
            logger.warning(f"Unusual aspect ratio ({aspect_ratio:.2f}) for body analysis")
            # Don't reject, just warn
        
        # Basic quality checks
        quality_issues = detect_image_quality_issues(validation_image)
        if quality_issues['quality_score'] < 30:
            logger.warning("Very low image quality detected")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Body region validation error: {e}")
        return True  # Default to True if validation fails

def create_image_thumbnail(image: np.ndarray, size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """
    Create a thumbnail of the image
    
    Args:
        image: Input image as numpy array
        size: Thumbnail size (width, height)
    
    Returns:
        Thumbnail image
    """
    try:
        thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return thumbnail
        
    except Exception as e:
        logger.error(f"Thumbnail creation error: {e}")
        return image

def convert_image_format(image: np.ndarray, target_format: str = 'RGB') -> np.ndarray:
    """
    Convert image to target format
    
    Args:
        image: Input image as numpy array
        target_format: Target color format ('RGB', 'BGR', 'GRAY')
    
    Returns:
        Converted image
    """
    try:
        if target_format == 'RGB':
            if len(image.shape) == 3:
                return image  # Already RGB
            else:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        elif target_format == 'BGR':
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        elif target_format == 'GRAY':
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                return image  # Already grayscale
        
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
        
    except Exception as e:
        logger.error(f"Format conversion error: {e}")
        return image

def apply_image_filters(image: np.ndarray, filters: List[str] = None) -> np.ndarray:
    """
    Apply various filters to enhance image for body analysis
    
    Args:
        image: Input image as numpy array
        filters: List of filter names to apply
    
    Returns:
        Filtered image
    """
    if filters is None:
        filters = ['denoise', 'sharpen']
    
    try:
        filtered_image = image.copy()
        
        for filter_name in filters:
            if filter_name == 'denoise':
                # Apply denoising
                filtered_image = cv2.bilateralFilter(filtered_image, 9, 75, 75)
            
            elif filter_name == 'sharpen':
                # Apply sharpening kernel
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                filtered_image = cv2.filter2D(filtered_image, -1, kernel)
            
            elif filter_name == 'smooth':
                # Apply Gaussian smoothing
                filtered_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)
            
            elif filter_name == 'edge_enhance':
                # Enhance edges
                gray = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Laplacian(gray, cv2.CV_64F)
                edges = np.uint8(np.absolute(edges))
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                filtered_image = cv2.addWeighted(filtered_image, 0.8, edges_colored, 0.2, 0)
        
        # Ensure values are in valid range
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        
        return filtered_image
        
    except Exception as e:
        logger.error(f"Filter application error: {e}")
        return image

def extract_image_metadata(image: np.ndarray) -> dict:
    """
    Extract metadata from image for analysis logging
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Dictionary with image metadata
    """
    try:
        height, width, channels = image.shape
        
        metadata = {
            'width': int(width),
            'height': int(height),
            'channels': int(channels),
            'aspect_ratio': float(width / height),
            'total_pixels': int(width * height),
            'data_type': str(image.dtype),
            'memory_usage_mb': float(image.nbytes / (1024 * 1024)),
            'mean_brightness': float(np.mean(image)),
            'std_brightness': float(np.std(image))
        }
        
        # Add quality assessment
        quality_info = detect_image_quality_issues(image)
        metadata['quality_score'] = quality_info['quality_score']
        metadata['quality_issues'] = {
            'blur_detected': quality_info['blur_detected'],
            'low_contrast': quality_info['low_contrast'],
            'too_dark': quality_info['too_dark'],
            'too_bright': quality_info['too_bright']
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Metadata extraction error: {e}")
        return {
            'width': 0,
            'height': 0,
            'channels': 0,
            'error': str(e)
        }

def prepare_image_for_model(image: np.ndarray, model_input_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Prepare image specifically for model inference
    
    Args:
        image: Input image as numpy array
        model_input_size: Required input size for the model
    
    Returns:
        Prepared image ready for model inference
    """
    try:
        # Resize to model requirements
        prepared = cv2.resize(image, model_input_size, interpolation=cv2.INTER_LINEAR)
        
        # Ensure correct data type
        prepared = prepared.astype(np.float32)
        
        # Normalize to 0-1 range (will be further normalized by model transforms)
        prepared = prepared / 255.0
        
        # Convert back to uint8 for consistency with transforms
        prepared = (prepared * 255).astype(np.uint8)
        
        return prepared
        
    except Exception as e:
        logger.error(f"Model preparation error: {e}")
        return image

def save_processed_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save processed image to disk
    
    Args:
        image: Image to save as numpy array
        output_path: Path to save the image
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        success = cv2.imwrite(output_path, image_bgr)
        
        if success:
            logger.info(f"Image saved successfully to: {output_path}")
        else:
            logger.error(f"Failed to save image to: {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Image save error: {e}")
        return False
