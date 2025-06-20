import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Union, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utilities for profile analysis"""
    
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            RGB image as numpy array or None if failed
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def preprocess_for_detection(self, image: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Preprocess image for object detection models
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Tuple of (original_image, preprocessed_tensor)
        """
        original_image = image.copy()
        
        # Resize image
        resized_image = cv2.resize(image, (self.target_size, self.target_size))
        
        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.from_numpy(resized_image.transpose((2, 0, 1))).float() / 255.0
        
        return original_image, image_tensor.unsqueeze(0)
    
    def preprocess_for_classification(self, image: np.ndarray, crop_size: int = 64) -> torch.Tensor:
        """
        Preprocess image crop for classification models
        
        Args:
            image: RGB image crop as numpy array
            crop_size: Target size for classification
            
        Returns:
            Preprocessed tensor with ImageNet normalization
        """
        # Resize crop
        resized_crop = cv2.resize(image, (crop_size, crop_size))
        
        # Convert to tensor
        crop_tensor = torch.from_numpy(resized_crop.transpose((2, 0, 1))).float() / 255.0
        
        # Apply ImageNet normalization
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        normalized_tensor = (crop_tensor - mean) / std
        
        return normalized_tensor.unsqueeze(0)
    
    def extract_bbox_crop(
        self, 
        image: np.ndarray, 
        bbox: List[float], 
        target_size: int = 224
    ) -> Optional[np.ndarray]:
        """
        Extract and resize crop from bounding box
        
        Args:
            image: Original RGB image
            bbox: Bounding box as [x1, y1, x2, y2]
            target_size: Target size for model input
            
        Returns:
            Cropped and resized image or None if invalid
        """
        try:
            h_orig, w_orig = image.shape[:2]
            
            # Scale bbox coordinates to original image size
            scale_x = w_orig / target_size
            scale_y = h_orig / target_size
            
            x1, y1, x2, y2 = bbox
            x1_orig = max(0, min(int(x1 * scale_x), w_orig))
            y1_orig = max(0, min(int(y1 * scale_y), h_orig))
            x2_orig = max(0, min(int(x2 * scale_x), w_orig))
            y2_orig = max(0, min(int(y2 * scale_y), h_orig))
            
            # Check if bbox is valid
            if x2_orig <= x1_orig or y2_orig <= y1_orig:
                logger.warning(f"Invalid bbox coordinates: {bbox}")
                return None
            
            # Extract crop
            crop = image[y1_orig:y2_orig, x1_orig:x2_orig]
            
            # Resize crop
            if crop.size > 0:
                return cv2.resize(crop, (64, 64))  # Standard size for classification
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting bbox crop: {str(e)}")
            return None
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image enhancement for better analysis
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Enhanced image
        """
        try:
            enhanced = image.copy()
            
            # Convert to LAB color space for better luminance control
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to luminance
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Slight gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image
    
    def detect_profile_orientation(self, image: np.ndarray) -> str:
        """
        Detect profile orientation using edge detection
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Orientation string: 'left', 'right', or 'unknown'
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Analyze edge distribution
            h, w = edges.shape
            left_edges = np.sum(edges[:, :w//2])
            right_edges = np.sum(edges[:, w//2:])
            
            # Determine orientation based on edge density
            if left_edges > right_edges * 1.2:
                return "left"
            elif right_edges > left_edges * 1.2:
                return "right"
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error detecting profile orientation: {str(e)}")
            return "unknown"
    
    def validate_image_quality(self, image: np.ndarray) -> dict:
        """
        Validate image quality for profile analysis
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Dictionary with quality metrics and recommendations
        """
        quality_report = {
            'is_suitable': True,
            'issues': [],
            'recommendations': [],
            'metrics': {}
        }
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            # Check image size
            if min(h, w) < 224:
                quality_report['is_suitable'] = False
                quality_report['issues'].append('Image resolution too low')
                quality_report['recommendations'].append('Use image with minimum 224x224 resolution')
            
            quality_report['metrics']['resolution'] = f"{w}x{h}"
            
            # Check brightness
            mean_brightness = np.mean(gray)
            quality_report['metrics']['brightness'] = float(mean_brightness)
            
            if mean_brightness < 50:
                quality_report['issues'].append('Image too dark')
                quality_report['recommendations'].append('Improve lighting conditions')
            elif mean_brightness > 200:
                quality_report['issues'].append('Image too bright')
                quality_report['recommendations'].append('Reduce exposure or lighting intensity')
            
            # Check contrast
            contrast = np.std(gray)
            quality_report['metrics']['contrast'] = float(contrast)
            
            if contrast < 20:
                quality_report['issues'].append('Low contrast')
                quality_report['recommendations'].append('Improve lighting contrast')
            
            # Check blur using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_report['metrics']['sharpness'] = float(blur_score)
            
            if blur_score < 100:
                quality_report['issues'].append('Image appears blurry')
                quality_report['recommendations'].append('Ensure camera focus is sharp')
            
            # Check for face-like features using basic edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            quality_report['metrics']['edge_density'] = float(edge_density)
            
            if edge_density < 0.05:
                quality_report['issues'].append('Insufficient facial features detected')
                quality_report['recommendations'].append('Ensure face is clearly visible and properly framed')
            
            # Overall suitability
            if len(quality_report['issues']) > 2:
                quality_report['is_suitable'] = False
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error validating image quality: {str(e)}")
            quality_report['is_suitable'] = False
            quality_report['issues'].append('Error during quality validation')
            return quality_report
    
    def resize_with_aspect_ratio(
        self, 
        image: np.ndarray, 
        target_size: int, 
        pad_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio with padding
        
        Args:
            image: RGB image as numpy array
            target_size: Target size for output
            pad_color: Color for padding (RGB)
            
        Returns:
            Resized image with padding
        """
        try:
            h, w = image.shape[:2]
            
            # Calculate scale factor
            scale = min(target_size / w, target_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create padded image
            padded = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
            
            # Calculate padding offsets
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            
            # Place resized image in center
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
            
        except Exception as e:
            logger.error(f"Error resizing with aspect ratio: {str(e)}")
            return cv2.resize(image, (target_size, target_size))
    
    def apply_data_augmentation(self, image: np.ndarray, augment_type: str = 'light') -> np.ndarray:
        """
        Apply data augmentation for better model robustness
        
        Args:
            image: RGB image as numpy array
            augment_type: Type of augmentation ('light', 'medium', 'heavy')
            
        Returns:
            Augmented image
        """
        try:
            augmented = image.copy()
            
            if augment_type == 'light':
                # Light brightness adjustment
                brightness_factor = np.random.uniform(0.9, 1.1)
                augmented = np.clip(augmented * brightness_factor, 0, 255).astype(np.uint8)
                
            elif augment_type == 'medium':
                # Medium augmentation: brightness + contrast
                brightness_factor = np.random.uniform(0.8, 1.2)
                contrast_factor = np.random.uniform(0.9, 1.1)
                
                augmented = np.clip(augmented * contrast_factor + 
                                  (brightness_factor - 1) * 128, 0, 255).astype(np.uint8)
                
            elif augment_type == 'heavy':
                # Heavy augmentation: multiple transformations
                brightness_factor = np.random.uniform(0.7, 1.3)
                contrast_factor = np.random.uniform(0.8, 1.2)
                
                # Apply transformations
                augmented = np.clip(augmented * contrast_factor + 
                                  (brightness_factor - 1) * 128, 0, 255).astype(np.uint8)
                
                # Add slight gaussian noise
                noise = np.random.normal(0, 5, augmented.shape)
                augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)
            
            return augmented
            
        except Exception as e:
            logger.error(f"Error applying augmentation: {str(e)}")
            return image
    
    def create_thumbnail(self, image: np.ndarray, size: int = 128) -> np.ndarray:
        """
        Create thumbnail of image for quick preview
        
        Args:
            image: RGB image as numpy array
            size: Thumbnail size
            
        Returns:
            Thumbnail image
        """
        try:
            return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            return image
