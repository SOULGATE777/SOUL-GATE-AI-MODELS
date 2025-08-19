import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Image processing utilities for frontal rotation analysis
    """
    
    def __init__(self):
        self.min_image_size = 224
        self.max_image_size = 2048
    
    def read_image_from_bytes(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Read image from bytes data
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image from bytes")
                return None
                
            return image
            
        except Exception as e:
            logger.error(f"Error reading image from bytes: {e}")
            return None
    
    def read_image_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Read image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return None
                
            return image
            
        except Exception as e:
            logger.error(f"Error reading image from path {image_path}: {e}")
            return None
    
    def validate_image_dimensions(self, image: np.ndarray) -> bool:
        """
        Validate image dimensions for processing
        
        Args:
            image: Input image
            
        Returns:
            True if dimensions are valid, False otherwise
        """
        if image is None:
            return False
            
        height, width = image.shape[:2]
        
        if height < self.min_image_size or width < self.min_image_size:
            logger.warning(f"Image too small: {width}x{height}, minimum: {self.min_image_size}x{self.min_image_size}")
            return False
            
        if height > self.max_image_size or width > self.max_image_size:
            logger.warning(f"Image too large: {width}x{height}, maximum: {self.max_image_size}x{self.max_image_size}")
            return False
            
        return True
    
    def resize_image_if_needed(self, image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """
        Resize image if it exceeds maximum size while maintaining aspect ratio
        
        Args:
            image: Input image
            max_size: Maximum dimension size
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if max(height, width) <= max_size:
            return image
        
        # Calculate scaling factor
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return resized_image
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Apply basic image enhancement for better analysis
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge([l, a, b])
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Error enhancing image quality: {e}, returning original")
            return image
    
    def assess_image_quality(self, image: np.ndarray) -> dict:
        """
        Assess basic image quality metrics for frontal faces
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            # Assess quality levels - adjusted for frontal faces
            sharpness_level = "good" if laplacian_var > 100 else "moderate" if laplacian_var > 50 else "poor"
            brightness_level = "good" if 80 <= brightness <= 180 else "moderate" if 40 <= brightness <= 220 else "poor"
            contrast_level = "good" if contrast > 50 else "moderate" if contrast > 25 else "poor"
            
            # Overall quality score
            quality_score = 0
            if sharpness_level == "good": quality_score += 40
            elif sharpness_level == "moderate": quality_score += 20
            
            if brightness_level == "good": quality_score += 30
            elif brightness_level == "moderate": quality_score += 15
            
            if contrast_level == "good": quality_score += 30
            elif contrast_level == "moderate": quality_score += 15
            
            quality_level = "excellent" if quality_score >= 80 else "good" if quality_score >= 60 else "moderate" if quality_score >= 40 else "poor"
            
            return {
                'sharpness': {
                    'value': float(laplacian_var),
                    'level': sharpness_level
                },
                'brightness': {
                    'value': float(brightness),
                    'level': brightness_level
                },
                'contrast': {
                    'value': float(contrast),
                    'level': contrast_level
                },
                'overall_quality': {
                    'score': quality_score,
                    'level': quality_level
                },
                'is_suitable': quality_score >= 40
            }
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return {
                'overall_quality': {
                    'score': 0,
                    'level': 'unknown'
                },
                'is_suitable': False,
                'error': str(e)
            }
    
    def prepare_image_for_analysis(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Prepare frontal face image for rotation analysis
        
        Args:
            image: Input image
            enhance: Whether to apply quality enhancement
            
        Returns:
            Processed image ready for analysis
        """
        # Resize if needed
        processed_image = self.resize_image_if_needed(image)
        
        # Enhance quality if requested
        if enhance:
            processed_image = self.enhance_image_quality(processed_image)
        
        return processed_image
    
    def save_image(self, image: np.ndarray, save_path: str, quality: int = 95) -> bool:
        """
        Save image to file
        
        Args:
            image: Image to save
            save_path: Output file path
            quality: JPEG quality (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set compression parameters
            if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif save_path.lower().endswith('.png'):
                params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            else:
                params = []
            
            success = cv2.imwrite(save_path, image, params)
            
            if success:
                logger.info(f"Image saved successfully to {save_path}")
            else:
                logger.error(f"Failed to save image to {save_path}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving image to {save_path}: {e}")
            return False