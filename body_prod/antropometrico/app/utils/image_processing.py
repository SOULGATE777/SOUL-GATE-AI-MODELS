# Use the same image processing utilities as the morfologico module
# since they handle the same basic image validation and preprocessing needs

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
        if height < 100 or width < 100:
            raise HTTPException(status_code=400, detail="Image too small for anthropometric analysis (minimum 100x100 pixels)")
        
        if height > 4000 or width > 4000:
            logger.warning(f"Large image detected ({width}x{height}), consider resizing for better performance")
        
        # Validate aspect ratio for anthropometric analysis
        aspect_ratio = height / width
        if aspect_ratio < 0.5 or aspect_ratio > 3.0:
            logger.warning(f"Unusual aspect ratio ({aspect_ratio:.2f}) for body anthropometric analysis")
        
        logger.info(f"Image validated successfully: {width}x{height} pixels")
        return image_array
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Image validation failed: {str(e)}")

def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Preprocess image for anthropometric analysis
    
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

def enhance_image_for_pose_detection(image: np.ndarray) -> np.ndarray:
    """
    Enhance image specifically for better pose detection
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Enhanced image
    """
    try:
        enhanced = image.copy()
        
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Ensure values are in valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Image enhancement error: {e}")
        return image

def detect_pose_suitability(image: np.ndarray) -> dict:
    """
    Assess image suitability for pose detection and anthropometric analysis
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Dictionary with suitability assessment
    """
    try:
        assessment = {
            'overall_suitable': True,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = image.shape[:2]
        
        # 1. Resolution check
        min_resolution = 300  # Minimum for reliable pose detection
        if min(height, width) < min_resolution:
            assessment['issues'].append('low_resolution')
            assessment['recommendations'].append(f"Image resolution too low ({width}x{height}). Minimum {min_resolution}px recommended.")
            assessment['overall_suitable'] = False
        
        # 2. Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:  # Lower threshold for anthropometric analysis
            assessment['issues'].append('blur_detected')
            assessment['recommendations'].append("Image appears blurry. Use a sharper image for better keypoint detection.")
        
        # 3. Contrast analysis
        contrast = gray.std()
        if contrast < 20:
            assessment['issues'].append('low_contrast')
            assessment['recommendations'].append("Low contrast detected. Improve lighting conditions.")
        
        # 4. Brightness analysis
        brightness = np.mean(gray)
        if brightness < 40:
            assessment['issues'].append('too_dark')
            assessment['recommendations'].append("Image is too dark. Increase brightness or improve lighting.")
        elif brightness > 220:
            assessment['issues'].append('too_bright')
            assessment['recommendations'].append("Image is overexposed. Reduce brightness or exposure.")
        
        # 5. Aspect ratio check for body analysis
        aspect_ratio = height / width
        if aspect_ratio < 1.0:  # Landscape orientation might be problematic for full body
            assessment['issues'].append('landscape_orientation')
            assessment['recommendations'].append("Landscape orientation detected. Portrait orientation preferred for body analysis.")
        
        # Calculate overall quality score
        resolution_score = min(100, (min(height, width) / min_resolution) * 100)
        blur_score = min(100, laplacian_var / 2)
        contrast_score = min(100, contrast * 3)
        brightness_score = 100 - abs(brightness - 128) / 128 * 100
        
        assessment['quality_score'] = (resolution_score + blur_score + contrast_score + brightness_score) / 4
        
        # Overall suitability
        if assessment['quality_score'] < 50:
            assessment['overall_suitable'] = False
            assessment['recommendations'].append("Overall image quality is too low for reliable anthropometric analysis.")
        
        return assessment
        
    except Exception as e:
        logger.error(f"Pose suitability assessment error: {e}")
        return {
            'overall_suitable': False,
            'quality_score': 0.0,
            'issues': ['assessment_failed'],
            'recommendations': ['Could not assess image suitability']
        }

def prepare_image_for_yolo(image: np.ndarray) -> np.ndarray:
    """
    Prepare image specifically for YOLO pose detection
    
    Args:
        image: Input image as numpy array (RGB)
    
    Returns:
        Image prepared for YOLO (still RGB, but optimized)
    """
    try:
        # YOLO expects RGB format, which we already have
        prepared = image.copy()
        
        # Ensure proper data type
        if prepared.dtype != np.uint8:
            prepared = np.clip(prepared, 0, 255).astype(np.uint8)
        
        # Optional enhancement for better detection
        # (YOLO is quite robust, so minimal preprocessing is often better)
        
        # Normalize brightness if extremely dark or bright
        mean_brightness = np.mean(prepared)
        if mean_brightness < 50:
            # Brighten dark images slightly
            prepared = cv2.convertScaleAbs(prepared, alpha=1.2, beta=20)
        elif mean_brightness > 200:
            # Darken bright images slightly
            prepared = cv2.convertScaleAbs(prepared, alpha=0.9, beta=-10)
        
        return prepared
        
    except Exception as e:
        logger.error(f"YOLO preparation error: {e}")
        return image

def extract_image_metadata_for_anthropometry(image: np.ndarray) -> dict:
    """
    Extract metadata specifically relevant for anthropometric analysis
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Dictionary with anthropometric-relevant metadata
    """
    try:
        height, width, channels = image.shape
        
        metadata = {
            'dimensions': {
                'width': int(width),
                'height': int(height),
                'channels': int(channels),
                'aspect_ratio': float(width / height),
                'total_pixels': int(width * height)
            },
            'quality_metrics': {
                'mean_brightness': float(np.mean(image)),
                'brightness_std': float(np.std(image)),
                'data_type': str(image.dtype),
                'memory_usage_mb': float(image.nbytes / (1024 * 1024))
            },
            'anthropometric_suitability': detect_pose_suitability(image)
        }
        
        # Add specific recommendations for anthropometric analysis
        recommendations = []
        
        # Resolution recommendations
        min_dim = min(width, height)
        if min_dim < 500:
            recommendations.append("Higher resolution recommended for detailed anthropometric measurements")
        elif min_dim > 2000:
            recommendations.append("Image resolution is excellent for detailed analysis")
        
        # Orientation recommendations
        if width > height:
            recommendations.append("Portrait orientation preferred for full body anthropometric analysis")
        
        # Quality recommendations
        if metadata['anthropometric_suitability']['quality_score'] > 80:
            recommendations.append("Excellent image quality for anthropometric analysis")
        elif metadata['anthropometric_suitability']['quality_score'] > 60:
            recommendations.append("Good image quality - suitable for analysis")
        else:
            recommendations.append("Image quality could be improved for better results")
        
        metadata['recommendations'] = recommendations
        
        return metadata
        
    except Exception as e:
        logger.error(f"Anthropometric metadata extraction error: {e}")
        return {
            'dimensions': {'width': 0, 'height': 0, 'channels': 0},
            'quality_metrics': {},
            'anthropometric_suitability': {'overall_suitable': False},
            'error': str(e)
        }

def validate_body_pose_region(image: np.ndarray, min_body_percentage: float = 0.3) -> bool:
    """
    Validate that the image contains a suitable body region for anthropometric analysis
    
    Args:
        image: Input image as numpy array
        min_body_percentage: Minimum percentage of image that should contain body
    
    Returns:
        True if suitable for body analysis
    """
    try:
        height, width = image.shape[:2]
        
        # Basic size validation
        if height < 200 or width < 150:
            logger.warning("Image too small for reliable body analysis")
            return False
        
        # Check if image has reasonable aspect ratio for body
        aspect_ratio = height / width
        if aspect_ratio < 0.8:  # Very wide image unlikely to contain full body
            logger.warning(f"Aspect ratio ({aspect_ratio:.2f}) suggests image may not contain full body")
            return False
        
        # Additional quality checks could be added here
        # For now, we rely on the pose detection to determine if body is present
        
        return True
        
    except Exception as e:
        logger.error(f"Body pose region validation error: {e}")
        return False

def save_anthropometric_image(image: np.ndarray, output_path: str, 
                             include_metadata: bool = True) -> bool:
    """
    Save image with optional anthropometric metadata
    
    Args:
        image: Image to save as numpy array
        output_path: Path to save the image
        include_metadata: Whether to include metadata in saved image
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Save image
        success = cv2.imwrite(output_path, image_bgr)
        
        if success:
            logger.info(f"Anthropometric image saved successfully to: {output_path}")
            
            if include_metadata:
                # Could add EXIF metadata here if needed
                pass
        else:
            logger.error(f"Failed to save anthropometric image to: {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Anthropometric image save error: {e}")
        return False

def resize_for_display(image: np.ndarray, max_dimension: int = 1200) -> np.ndarray:
    """
    Resize image for display while maintaining aspect ratio
    
    Args:
        image: Input image as numpy array
        max_dimension: Maximum dimension for display
    
    Returns:
        Resized image
    """
    try:
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_dimension / width, max_dimension / height)
        
        if scale < 1.0:  # Only resize if image is larger
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        
        return image
        
    except Exception as e:
        logger.error(f"Display resize error: {e}")
        return image

def create_image_thumbnail_for_anthropometry(image: np.ndarray, 
                                           size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Create thumbnail optimized for anthropometric preview
    
    Args:
        image: Input image as numpy array
        size: Thumbnail size (width, height)
    
    Returns:
        Thumbnail image
    """
    try:
        # Maintain aspect ratio while fitting in thumbnail size
        height, width = image.shape[:2]
        thumb_width, thumb_height = size
        
        # Calculate scaling to fit within thumbnail while maintaining aspect ratio
        scale = min(thumb_width / width, thumb_height / height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target size (pad if necessary)
        if new_width != thumb_width or new_height != thumb_height:
            canvas = np.ones((thumb_height, thumb_width, 3), dtype=np.uint8) * 255  # White background
            
            # Center the resized image on the canvas
            y_offset = (thumb_height - new_height) // 2
            x_offset = (thumb_width - new_width) // 2
            
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = thumbnail
            thumbnail = canvas
        
        return thumbnail
        
    except Exception as e:
        logger.error(f"Anthropometric thumbnail creation error: {e}")
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def apply_anthropometric_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Apply complete preprocessing pipeline for anthropometric analysis
    
    Args:
        image: Input image as numpy array
    
    Returns:
        Preprocessed image ready for pose detection
    """
    try:
        # Step 1: Basic validation and format check
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB format")
        
        # Step 2: Enhance for pose detection
        enhanced = enhance_image_for_pose_detection(image)
        
        # Step 3: Prepare for YOLO
        prepared = prepare_image_for_yolo(enhanced)
        
        # Step 4: Final validation
        if not validate_body_pose_region(prepared):
            logger.warning("Image may not be suitable for body pose analysis")
        
        return prepared
        
    except Exception as e:
        logger.error(f"Anthropometric preprocessing error: {e}")
        return image
