import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ProfileImageProcessor:
    """Image processing utilities for profile anthropometric analysis"""
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate if image is suitable for profile analysis
        
        Args:
            image: Input image array
            
        Returns:
            True if image is valid, False otherwise
        """
        if image is None:
            return False
        
        # Check dimensions
        if len(image.shape) not in [2, 3]:
            return False
        
        # Check size
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            logger.warning(f"Image too small: {w}x{h}")
            return False
        
        if h > 4000 or w > 4000:
            logger.warning(f"Image too large: {w}x{h}")
            return False
        
        return True
    
    @staticmethod
    def preprocess_for_analysis(image: np.ndarray, target_size: int = 224) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for anthropometric analysis
        
        Args:
            image: Input image (BGR or RGB)
            target_size: Target size for model input
            
        Returns:
            Tuple of (original_rgb, processed_image)
        """
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume it's already RGB from FastAPI processing
            original_rgb = image.copy()
        else:
            # Convert BGR to RGB if needed
            original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio with padding
        processed_image = ProfileImageProcessor.resize_with_padding(original_rgb, target_size)
        
        return original_rgb, processed_image
    
    @staticmethod
    def resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize image maintaining aspect ratio with padding
        
        Args:
            image: Input image
            target_size: Target square size
            
        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Enhance image contrast using CLAHE
        
        Args:
            image: Input image
            clip_limit: CLAHE clip limit
            
        Returns:
            Contrast enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_channel_clahe = clahe.apply(l_channel)
            
            # Merge channels and convert back
            lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
            enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    @staticmethod
    def detect_profile_orientation(image: np.ndarray) -> str:
        """
        Detect if profile is facing left or right using simple edge detection
        
        Args:
            image: Input profile image
            
        Returns:
            'left' or 'right' indicating profile direction
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Split image into left and right halves
        h, w = edges.shape
        left_half = edges[:, :w//2]
        right_half = edges[:, w//2:]
        
        # Count edge pixels in each half
        left_edges = np.sum(left_half > 0)
        right_edges = np.sum(right_half > 0)
        
        # More edges on the left typically means right-facing profile
        if left_edges > right_edges * 1.2:
            return 'right'
        elif right_edges > left_edges * 1.2:
            return 'left'
        else:
            return 'unknown'
    
    @staticmethod
    def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization for better contrast
        
        Args:
            image: Input image
            
        Returns:
            Histogram equalized image
        """
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            # Grayscale image
            equalized = cv2.equalizeHist(image)
        
        return equalized
    
    @staticmethod
    def remove_noise(image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Remove noise from image
        
        Args:
            image: Input image
            method: Noise removal method ('bilateral', 'gaussian', 'median')
            
        Returns:
            Denoised image
        """
        if method == 'bilateral':
            if len(image.shape) == 3:
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
            else:
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            denoised = cv2.medianBlur(image, 5)
        else:
            logger.warning(f"Unknown denoising method: {method}, using bilateral")
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        return denoised
    
    @staticmethod
    def crop_to_face_region(image: np.ndarray, face_cascade_path: Optional[str] = None) -> np.ndarray:
        """
        Crop image to focus on face region (if face detection is available)
        
        Args:
            image: Input image
            face_cascade_path: Path to face cascade classifier (optional)
            
        Returns:
            Cropped image or original if face not detected
        """
        try:
            if face_cascade_path and cv2.data.haarcascades:
                # Use OpenCV's built-in face cascade
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                
                # Convert to grayscale for detection
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Use the largest detected face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Add some padding around the face
                    padding = 50
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    
                    # Crop the image
                    if len(image.shape) == 3:
                        cropped = image[y:y+h, x:x+w]
                    else:
                        cropped = image[y:y+h, x:x+w]
                    
                    return cropped
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
        
        # Return original image if face detection fails
        return image
    
    @staticmethod
    def normalize_lighting(image: np.ndarray) -> np.ndarray:
        """
        Normalize lighting conditions in the image
        
        Args:
            image: Input image
            
        Returns:
            Lighting normalized image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply adaptive histogram equalization to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            normalized = clahe.apply(image)
        
        return normalized
    
    @staticmethod
    def calculate_image_quality_score(image: np.ndarray) -> float:
        """
        Calculate a quality score for the input image
        
        Args:
            image: Input image
            
        Returns:
            Quality score between 0 and 1 (higher is better)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (empirically determined thresholds)
        focus_score = min(laplacian_var / 1000.0, 1.0)
        
        # Calculate contrast using standard deviation
        contrast_score = min(gray.std() / 64.0, 1.0)
        
        # Calculate brightness distribution (penalize over/under exposure)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        
        # Penalize images with too many very dark or very bright pixels
        overexposed = hist_norm[240:].sum()
        underexposed = hist_norm[:15].sum()
        exposure_score = 1.0 - (overexposed + underexposed)
        
        # Calculate noise level (inverse of noise is quality)
        noise_level = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        noise_diff = cv2.absdiff(gray, noise_level)
        noise_score = 1.0 - min(noise_diff.mean() / 32.0, 1.0)
        
        # Combine scores with weights
        quality_score = (
            0.3 * focus_score +
            0.25 * contrast_score +
            0.25 * exposure_score +
            0.2 * noise_score
        )
        
        return min(quality_score, 1.0)
    
    @staticmethod
    def get_image_statistics(image: np.ndarray) -> dict:
        """
        Get comprehensive image statistics
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        stats = {
            'width': image.shape[1],
            'height': image.shape[0],
            'channels': len(image.shape) if len(image.shape) == 2 else image.shape[2],
            'mean_brightness': float(gray.mean()),
            'std_brightness': float(gray.std()),
            'min_brightness': int(gray.min()),
            'max_brightness': int(gray.max()),
            'quality_score': ProfileImageProcessor.calculate_image_quality_score(image),
            'estimated_orientation': ProfileImageProcessor.detect_profile_orientation(image)
        }
        
        return stats
