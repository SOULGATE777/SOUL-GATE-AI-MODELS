import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Image processing utilities for frontal preprocessing service"""

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate if image is suitable for processing

        Args:
            image: Input image array

        Returns:
            True if image is valid, False otherwise
        """
        if image is None:
            logger.warning("Image is None")
            return False

        # Check dimensions
        if len(image.shape) not in [2, 3]:
            logger.warning(f"Invalid image dimensions: {image.shape}")
            return False

        # Check size
        h, w = image.shape[:2]
        if h < 32 or w < 32:
            logger.warning(f"Image too small: {w}x{h}")
            return False

        if h > 8192 or w > 8192:
            logger.warning(f"Image too large: {w}x{h}")
            return False

        return True

    @staticmethod
    def decode_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode image from bytes

        Args:
            image_bytes: Raw image bytes

        Returns:
            Decoded image array in RGB format or None if failed
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)

            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.error("Failed to decode image")
                return None

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image_rgb

        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return None

    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray,
                                target_size: Tuple[int, int],
                                maintain_aspect: bool = True,
                                fill_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Resize image to target size while optionally maintaining aspect ratio

        Args:
            image: Input image array
            target_size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            fill_color: Fill color for letterboxing (RGB)

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        if not maintain_aspect:
            return cv2.resize(image, (target_w, target_h))

        # Calculate scale factor to fit within target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h))

        # Create canvas with fill color
        canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)

        # Center the resized image on canvas
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

        return canvas

    @staticmethod
    def crop_with_padding(image: np.ndarray,
                         bbox: List[float],
                         padding_factor: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        Crop image with padding around bounding box

        Args:
            image: Input image array
            bbox: Bounding box [x1, y1, x2, y2]
            padding_factor: Padding factor (0.1 = 10% padding)

        Returns:
            Tuple of (cropped_image, crop_info)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Calculate padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = box_w * padding_factor
        pad_h = box_h * padding_factor

        # Apply padding with bounds checking
        x1_pad = max(0, int(x1 - pad_w))
        y1_pad = max(0, int(y1 - pad_h))
        x2_pad = min(w, int(x2 + pad_w))
        y2_pad = min(h, int(y2 + pad_h))

        # Crop image
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]

        crop_info = {
            'original_bbox': bbox,
            'padded_bbox': [x1_pad, y1_pad, x2_pad, y2_pad],
            'padding_factor': padding_factor,
            'original_size': (w, h),
            'cropped_size': cropped.shape[:2]
        }

        return cropped, crop_info

    @staticmethod
    def enhance_image_quality(image: np.ndarray,
                             brightness: float = 0.0,
                             contrast: float = 1.0,
                             gamma: float = 1.0,
                             apply_clahe: bool = False) -> np.ndarray:
        """
        Enhance image quality with various adjustments

        Args:
            image: Input image array
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.5 to 3.0)
            gamma: Gamma correction (0.5 to 2.0)
            apply_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

        Returns:
            Enhanced image
        """
        enhanced = image.copy().astype(np.float32)

        # Apply brightness adjustment
        if brightness != 0.0:
            enhanced = enhanced + brightness

        # Apply contrast adjustment
        if contrast != 1.0:
            enhanced = enhanced * contrast

        # Apply gamma correction
        if gamma != 1.0:
            enhanced = np.power(enhanced / 255.0, gamma) * 255.0

        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        # Apply CLAHE if requested
        if apply_clahe:
            # Convert to LAB color space
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels and convert back to RGB
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return enhanced

    @staticmethod
    def normalize_image(image: np.ndarray,
                       method: str = 'minmax') -> np.ndarray:
        """
        Normalize image values

        Args:
            image: Input image array
            method: Normalization method ('minmax', 'zscore', 'robust')

        Returns:
            Normalized image
        """
        image_float = image.astype(np.float32)

        if method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(image_float)
            max_val = np.max(image_float)
            if max_val > min_val:
                normalized = (image_float - min_val) / (max_val - min_val)
            else:
                normalized = image_float

        elif method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(image_float)
            std_val = np.std(image_float)
            if std_val > 0:
                normalized = (image_float - mean_val) / std_val
                # Scale to [0, 1] range
                normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
            else:
                normalized = image_float / 255.0

        elif method == 'robust':
            # Robust normalization using percentiles
            p2, p98 = np.percentile(image_float, [2, 98])
            if p98 > p2:
                normalized = np.clip((image_float - p2) / (p98 - p2), 0, 1)
            else:
                normalized = image_float / 255.0

        else:
            # Default: simple division by 255
            normalized = image_float / 255.0

        return (normalized * 255).astype(np.uint8)

    @staticmethod
    def apply_noise_reduction(image: np.ndarray,
                             method: str = 'bilateral') -> np.ndarray:
        """
        Apply noise reduction to image

        Args:
            image: Input image array
            method: Noise reduction method ('bilateral', 'gaussian', 'median')

        Returns:
            Denoised image
        """
        if method == 'bilateral':
            # Bilateral filter preserves edges while reducing noise
            denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        elif method == 'gaussian':
            # Gaussian blur
            denoised = cv2.GaussianBlur(image, (5, 5), 0)

        elif method == 'median':
            # Median filter
            denoised = cv2.medianBlur(image, 5)

        else:
            logger.warning(f"Unknown noise reduction method: {method}")
            denoised = image

        return denoised

    @staticmethod
    def get_image_statistics(image: np.ndarray) -> Dict:
        """
        Get comprehensive image statistics

        Args:
            image: Input image array

        Returns:
            Dictionary with image statistics
        """
        h, w, c = image.shape if len(image.shape) == 3 else (*image.shape, 1)

        stats = {
            'dimensions': {
                'height': h,
                'width': w,
                'channels': c,
                'total_pixels': h * w
            },
            'data_type': str(image.dtype),
            'value_range': {
                'min': float(np.min(image)),
                'max': float(np.max(image)),
                'mean': float(np.mean(image)),
                'std': float(np.std(image))
            }
        }

        # Per-channel statistics for color images
        if c == 3:
            channel_names = ['red', 'green', 'blue']
            stats['channels_stats'] = {}

            for i, name in enumerate(channel_names):
                channel = image[:, :, i]
                stats['channels_stats'][name] = {
                    'min': float(np.min(channel)),
                    'max': float(np.max(channel)),
                    'mean': float(np.mean(channel)),
                    'std': float(np.std(channel))
                }

        return stats

    @staticmethod
    def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
        """
        Convert base64 string to image array

        Args:
            base64_string: Base64 encoded image string

        Returns:
            Image array in RGB format or None if failed
        """
        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)

            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert to numpy array
            image_array = np.array(pil_image)

            return image_array

        except Exception as e:
            logger.error(f"Error converting base64 to image: {str(e)}")
            return None

    @staticmethod
    def image_to_base64(image: np.ndarray,
                       format: str = 'JPEG',
                       quality: int = 95) -> str:
        """
        Convert image array to base64 string

        Args:
            image: Input image array in RGB format
            format: Output format ('JPEG', 'PNG')
            quality: JPEG quality (1-100, only for JPEG)

        Returns:
            Base64 encoded image string
        """
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))

            # Save to bytes buffer
            buffer = io.BytesIO()
            if format.upper() == 'JPEG':
                pil_image.save(buffer, format='JPEG', quality=quality)
            elif format.upper() == 'PNG':
                pil_image.save(buffer, format='PNG')
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Encode to base64
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            return img_base64

        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise e