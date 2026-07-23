import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)

WHITE_BG = (255, 255, 255)
MEAN_L_DARK_THRESHOLD = 90


def composite_on_white(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Soft-blend ``image_rgb`` onto a white background using ``mask``.

    Args:
        image_rgb: HxWx3 uint8 RGB image.
        mask: HxW (or HxWx1) float/uint8 alpha. Values in [0, 1] or [0, 255].

    Returns:
        HxWx3 uint8 RGB image composited on white.
    """
    return ImageProcessor.composite_on_white(image_rgb, mask)


def apply_white_background_grabcut(image_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Remove background via GrabCut and composite onto white.

    Args:
        image_rgb: HxWx3 uint8 RGB image.

    Returns:
        (composited_image, success). On failure returns (original, False).
    """
    return ImageProcessor.apply_white_background_grabcut(image_rgb)


def maybe_enhance_dark(image_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Conditionally CLAHE-enhance dark images (mean LAB L < threshold).

    Args:
        image_rgb: HxWx3 uint8 RGB image.

    Returns:
        (image, illumination_enhanced). Fail-open: returns (original, False).
    """
    return ImageProcessor.maybe_enhance_dark(image_rgb)


class ImageProcessor:
    """Image processing utilities for profile preprocessing service"""

    @staticmethod
    def composite_on_white(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Soft-blend ``image_rgb`` onto a white background using ``mask``.

        Args:
            image_rgb: HxWx3 uint8 RGB image.
            mask: HxW (or HxWx1) float/uint8 alpha. Values in [0, 1] or [0, 255].

        Returns:
            HxWx3 uint8 RGB image composited on white.
        """
        alpha = mask.astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)
        if alpha.ndim == 2:
            alpha = alpha[..., np.newaxis]

        image_f = image_rgb.astype(np.float32)
        white = np.full_like(image_f, WHITE_BG, dtype=np.float32)
        out = image_f * alpha + white * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_white_background_grabcut(image_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply OpenCV GrabCut and composite the foreground onto white.

        Mask init (tuned for profile heads / dark hair on busy BG):
        - Outer ~5% strip: definite GC_BGD
        - Everything else: GC_PR_BGD default (so frame corners can be removed)
        - Large centered ellipse (~40%w x 46%h): GC_PR_FGD (probable head+hair)
        - Inner ellipse (~26%w x 32%h): definite GC_FGD seed (face/hair core)

        The PR_FGD region is elliptical (not a full rectangle) so background
        pulled in by the crop padding at the top/side corners stays PR_BGD and
        can be cut. Post-GrabCut we keep only the largest connected FG component,
        which removes detached rectangular BG blocks left near the crown.

        Post: morph close (5x5) + open (3x3) + keep-largest-component + soft blur;
        FG-fraction quality gate fail-open.
        Fail-open: returns (image_rgb, False) on any error / bad mask.

        Args:
            image_rgb: HxWx3 uint8 RGB image.

        Returns:
            (result_rgb, white_bg_applied).
        """
        try:
            if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                return image_rgb, False

            h, w = image_rgb.shape[:2]
            if h < 16 or w < 16:
                return image_rgb, False

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            hard_y = max(1, int(round(h * 0.05)))
            hard_x = max(1, int(round(w * 0.05)))

            # Default = probable BG everywhere; hard outer strip = definite BG.
            mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
            mask[:hard_y, :] = cv2.GC_BGD
            mask[h - hard_y:, :] = cv2.GC_BGD
            mask[:, :hard_x] = cv2.GC_BGD
            mask[:, w - hard_x:] = cv2.GC_BGD

            cy, cx = h // 2, w // 2
            # Large elliptical PR_FGD leaves corners as PR_BGD (removable BG).
            pr_axes = (max(2, int(w * 0.40)), max(2, int(h * 0.46)))
            cv2.ellipse(mask, (cx, cy), pr_axes, 0, 0, 360, int(cv2.GC_PR_FGD), -1)
            # Inner definite FGD seed so face/hair core is not only probable.
            fgd_axes = (max(2, int(w * 0.26)), max(2, int(h * 0.32)))
            cv2.ellipse(mask, (cx, cy), fgd_axes, 0, 0, 360, int(cv2.GC_FGD), -1)

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                image_bgr,
                mask,
                None,
                bgd_model,
                fgd_model,
                8,
                cv2.GC_INIT_WITH_MASK,
            )

            fg = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                1.0,
                0.0,
            ).astype(np.float32)

            # Close small holes / reconnect hair strands before softening
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_u8 = (fg * 255).astype(np.uint8)
            fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            # Drop tiny floating FG islands (artifacts); 3x3 avoids eroding wispy crown hair
            fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel_open, iterations=1)

            # Keep only the largest connected FG component. This removes detached
            # rectangular BG blocks GrabCut leaves near the crown/side.
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                fg_u8, connectivity=8
            )
            if num > 2:
                # label 0 = background; pick largest FG label by area
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                fg_u8 = np.where(labels == largest, 255, 0).astype(np.uint8)

            fg = fg_u8.astype(np.float32) / 255.0

            fg_frac = float(fg.mean())
            # Quality gate: empty / full / near-full masks → fail-open (keep original)
            if fg_frac < 0.12 or fg_frac > 0.92:
                logger.warning(
                    "GrabCut FG fraction out of range (%.3f); fail-open", fg_frac
                )
                return image_rgb, False

            fg = cv2.GaussianBlur(fg, (7, 7), 0)

            result = ImageProcessor.composite_on_white(image_rgb, fg)
            return result, True

        except Exception as e:
            logger.warning(f"GrabCut white-background failed (fail-open): {e}")
            return image_rgb, False

    @staticmethod
    def maybe_enhance_dark(image_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
        """CLAHE-enhance when mean LAB L is below ``MEAN_L_DARK_THRESHOLD``.

        Fail-open: returns (image_rgb, False) on any error or invalid input.

        Args:
            image_rgb: HxWx3 uint8 RGB image.

        Returns:
            (result_rgb, illumination_enhanced).
        """
        try:
            if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                return image_rgb, False

            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            mean_l = float(np.mean(lab[:, :, 0]))
            if mean_l >= MEAN_L_DARK_THRESHOLD:
                return image_rgb, False

            enhanced = ImageProcessor.enhance_image_quality(
                image_rgb, apply_clahe=True
            )
            return enhanced, True

        except Exception as e:
            logger.warning(f"Dark illumination enhance failed (fail-open): {e}")
            return image_rgb, False

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
                                fill_color: Tuple[int, int, int] = WHITE_BG) -> np.ndarray:
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