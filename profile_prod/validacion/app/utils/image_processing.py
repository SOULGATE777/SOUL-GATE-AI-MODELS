import cv2
import numpy as np
import io
from PIL import Image
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Image processing utilities for profile validation
    """
    
    def __init__(self):
        """Initialize the image processor"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.min_image_size = (224, 224)
        self.max_image_size = (4000, 4000)
    
    def read_image_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Read image from bytes data
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Image as numpy array in BGR format, or None if failed
        """
        try:
            # Convert bytes to PIL Image
            image_pil = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            # Convert PIL to numpy array
            image_np = np.array(image_pil)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            return image_bgr
            
        except Exception as e:
            logger.error(f"Failed to read image from bytes: {e}")
            return None
    
    def read_image_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Read image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array in BGR format, or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error reading image from {image_path}: {e}")
            return None
    
    def validate_image_dimensions(self, image: np.ndarray) -> bool:
        """
        Validate image dimensions
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if dimensions are valid, False otherwise
        """
        if image is None:
            return False
        
        height, width = image.shape[:2]
        
        # Check minimum size
        if width < self.min_image_size[0] or height < self.min_image_size[1]:
            logger.warning(f"Image too small: {width}x{height}, minimum: {self.min_image_size}")
            return False
        
        # Check maximum size
        if width > self.max_image_size[0] or height > self.max_image_size[1]:
            logger.warning(f"Image too large: {width}x{height}, maximum: {self.max_image_size}")
            return False
        
        return True
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            target_size: Target (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect_ratio:
            return self._resize_with_aspect_ratio(image, target_size)
        else:
            return cv2.resize(image, target_size)
    
    def _resize_with_aspect_ratio(self, image: np.ndarray, 
                                 target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized image with padding if necessary
        """
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_width - new_width) // 2
        pad_y = (target_height - new_height) // 2
        
        # Place resized image on canvas
        canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        return canvas
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.2, 
                        beta: int = 10) -> np.ndarray:
        """
        Enhance image contrast
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            Enhanced image
        """
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce image noise using bilateral filter
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def detect_edges(self, image: np.ndarray, low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges using Canny edge detector
        
        Args:
            image: Input image
            low_threshold: Low threshold for edge detection
            high_threshold: High threshold for edge detection
            
        Returns:
            Edge map
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate average brightness of image
        
        Args:
            image: Input image
            
        Returns:
            Average brightness value (0-255)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.mean(gray))
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate contrast of image using standard deviation
        
        Args:
            image: Input image
            
        Returns:
            Contrast value
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.std(gray))
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance
        
        Args:
            image: Input image
            
        Returns:
            Sharpness score (higher = sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def crop_to_square(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image to square aspect ratio (center crop)
        
        Args:
            image: Input image
            
        Returns:
            Square cropped image
        """
        height, width = image.shape[:2]
        size = min(height, width)
        
        # Calculate crop coordinates
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        
        return image[start_y:start_y + size, start_x:start_x + size]
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to improve contrast
        
        Args:
            image: Input image
            
        Returns:
            Equalized image
        """
        if len(image.shape) == 3:
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            # Apply histogram equalization to Y channel
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to BGR
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def validate_profile_orientation(self, image: np.ndarray) -> dict:
        """
        Basic validation of profile orientation
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with orientation analysis
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            height, width = gray.shape
            
            # Analyze left vs right brightness (simple heuristic)
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            
            left_brightness = np.mean(left_half)
            right_brightness = np.mean(right_half)
            
            # Basic face detection to validate profile
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            frontal_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            profile_faces = profile_cascade.detectMultiScale(gray, 1.1, 4)
            
            is_likely_profile = len(profile_faces) > 0 and len(frontal_faces) == 0
            orientation_confidence = abs(left_brightness - right_brightness) / max(left_brightness, right_brightness)
            
            return {
                'is_likely_profile': is_likely_profile,
                'orientation_confidence': float(orientation_confidence),
                'left_brightness': float(left_brightness),
                'right_brightness': float(right_brightness),
                'frontal_faces_detected': len(frontal_faces),
                'profile_faces_detected': len(profile_faces)
            }
            
        except Exception as e:
            logger.error(f"Error in profile orientation validation: {e}")
            return {
                'is_likely_profile': False,
                'orientation_confidence': 0.0,
                'error': str(e)
            }
