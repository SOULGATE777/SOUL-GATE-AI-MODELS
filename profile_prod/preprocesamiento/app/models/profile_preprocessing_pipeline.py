import torch
import torch.nn as nn
import cv2
import numpy as np
import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import mediapipe as mp
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from app.utils.rotation_utils import FaceRotationAligner
from app.utils.image_processing import ImageProcessor, composite_on_white

logger = logging.getLogger(__name__)

class ProfilePreprocessingPipeline:
    """
    Profile face detection and preprocessing pipeline for preparing images 
    for downstream analysis services.
    """
    
    def __init__(self, model_path: str, device: str = 'auto', point_model_path: Optional[str] = None):
        """
        Initialize the preprocessing pipeline

        Args:
            model_path: Path to the trained Faster R-CNN model
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            point_model_path: Optional path to point detection model for face rotation alignment
        """
        self.device = self._setup_device(device)
        self.model = None
        self.model_path = model_path
        self.all_classes = []
        self.num_classes = 0

        # Default processing parameters
        self.default_confidence_threshold = 0.5
        self.default_target_size = (600, 600)
        self.default_padding_factor = 0.22

        # MediaPipe Selfie Segmentation for white-background cleaning.
        # Learned person/background segmentation — unlike GrabCut it separates
        # dark hair from busy/reflective backgrounds regardless of color overlap.
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmenter = None  # Lazy initialization

        # Face rotation aligner (optional)
        self.rotation_aligner = None
        if point_model_path and Path(point_model_path).exists():
            try:
                logger.info("Initializing face rotation aligner...")
                self.rotation_aligner = FaceRotationAligner(point_model_path, str(self.device))
                logger.info("Face rotation aligner initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize rotation aligner: {str(e)}")
                self.rotation_aligner = None

        logger.info(f"Initializing ProfilePreprocessingPipeline on {self.device}")
        self._load_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("CUDA not available, using CPU")
        
        return torch.device(device)
    
    def _get_model(self, num_classes: int):
        """Create Faster R-CNN model architecture"""
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model metadata
            self.all_classes = checkpoint.get('all_classes', [])
            self.num_classes = checkpoint.get('num_classes', len(self.all_classes) + 1)
            
            # Create and load model
            self.model = self._get_model(self.num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully with {self.num_classes} classes: {self.all_classes}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e

    def _initialize_selfie_segmenter(self):
        """Lazy initialization of Selfie Segmentation (reuse instance)."""
        if self.selfie_segmenter is None:
            self.selfie_segmenter = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
            logger.info("MediaPipe Selfie Segmentation initialized for white background")

    @staticmethod
    def _face_protect_rect(fx1: float, fy1: float, fx2: float, fy2: float,
                           crop_w: int, crop_h: int) -> Optional[Tuple[int, int, int, int]]:
        """Expand a crop-local face bbox into a protection rectangle.

        The face is never allowed to be clipped by the white-BG mask, so the
        detected face bbox is expanded (extra on top for the forehead/hairline,
        some on the sides and chin) and clamped to the crop. The white-BG
        refinement forces this rectangle fully opaque.

        Returns (px1, py1, px2, py2) or None when the bbox is degenerate.
        """
        fw = fx2 - fx1
        fh = fy2 - fy1
        if fw <= 0 or fh <= 0:
            return None

        px1 = int(round(fx1 - fw * 0.12))
        px2 = int(round(fx2 + fw * 0.12))
        py1 = int(round(fy1 - fh * 0.35))   # forehead / hairline
        py2 = int(round(fy2 + fh * 0.15))   # chin / jaw

        px1 = max(0, min(px1, crop_w))
        px2 = max(0, min(px2, crop_w))
        py1 = max(0, min(py1, crop_h))
        py2 = max(0, min(py2, crop_h))
        if px2 <= px1 or py2 <= py1:
            return None
        return px1, py1, px2, py2

    @staticmethod
    def _refine_segmentation_mask(mask: np.ndarray,
                                  fg_threshold: float = 0.5,
                                  feather_px: int = 2,
                                  protect_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Turn MediaPipe's soft probability mask into a clean alpha.

        The raw selfie-segmentation mask is a low-res soft probability. Composited
        directly it produces two artefacts on busy/reflective backgrounds:
        translucent "ghosting" of the real background (mid-range alpha) and a
        blocky/pixelated edge (coarse mask upscaled by the letterbox resize).

        This hardens the mask to remove both while keeping edges smooth:
        - threshold at ``fg_threshold`` → kills mid-alpha ghosting
        - morphological close → fills small holes inside the subject
        - keep largest connected component → drops detached background blobs
        - thin Gaussian feather → smooth (non-blocky) edge instead of a hard step
        - ``protect_rect`` → forced fully opaque so the face is NEVER clipped,
          regardless of how uncertain the segmenter is over those pixels

        Args:
            mask: HxW float/uint8 probability. Values in [0, 1] or [0, 255].
            fg_threshold: probability above which a pixel is foreground. Lower it
                (e.g. 0.35) to keep more wispy hair; raise it to cut more background.
            feather_px: half-width of the edge feather in pixels (0 = hard edge).
            protect_rect: (x1, y1, x2, y2) region forced to alpha 1 (face guard).

        Returns:
            HxW float32 alpha in [0, 1].
        """
        alpha = mask.astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)

        binary = (alpha >= fg_threshold).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = (labels == largest).astype(np.uint8)

        k = feather_px * 2 + 1
        if feather_px > 0:
            alpha_out = cv2.GaussianBlur(binary.astype(np.float32), (k, k), 0)
        else:
            alpha_out = binary.astype(np.float32)

        # Face guard: the protected rectangle must be EXACTLY opaque so the face is
        # never clipped. A soft halo just outside the rect blends the seam into the
        # mask; the rect interior is then hard-set to 1.0 (blur must not soften it).
        if protect_rect is not None:
            px1, py1, px2, py2 = protect_rect
            if feather_px > 0:
                halo = np.zeros_like(alpha_out, dtype=np.float32)
                halo[py1:py2, px1:px2] = 1.0
                halo = cv2.GaussianBlur(halo, (k, k), 0)
                alpha_out = np.maximum(alpha_out, halo)
            alpha_out[py1:py2, px1:px2] = 1.0

        return alpha_out

    def apply_white_background(self, image_rgb: np.ndarray,
                              protect_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, bool]:
        """Soft-composite subject onto white via MediaPipe Selfie Segmentation.

        Passes RGB directly to segmenter.process (not BGR). The raw soft mask is
        refined (threshold + keep-largest-component + thin feather) before
        compositing so busy/reflective backgrounds do not leave translucent
        ghosting or blocky edges. When ``protect_rect`` is given, that region is
        forced fully opaque so the face is NEVER clipped. Fail-open on errors /
        missing mask.

        Args:
            image_rgb: HxWx3 uint8 RGB image (a face/head crop).
            protect_rect: (x1, y1, x2, y2) crop-local face region to never clip.

        Returns:
            Tuple of (composited_or_original_image, white_bg_applied).
        """
        try:
            self._initialize_selfie_segmenter()
            results = self.selfie_segmenter.process(image_rgb)
            if results.segmentation_mask is None:
                logger.warning("Selfie segmentation returned no mask; skipping white BG")
                return image_rgb, False
            alpha = self._refine_segmentation_mask(
                results.segmentation_mask, protect_rect=protect_rect
            )
            composited = composite_on_white(image_rgb, alpha)
            # composite_on_white returns the same object on shape-guard no-op
            return composited, composited is not image_rgb
        except Exception as e:
            logger.warning(f"White background cleaning failed (fail-open): {e}")
            return image_rgb, False

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image in RGB format
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = None) -> List[Dict]:
        """
        Detect profile faces in the image
        
        Args:
            image: Input image in RGB format
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detection results with bounding boxes and confidence scores
        """
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process results
        detections = []
        prediction = predictions[0]
        
        for i, (box, label, score) in enumerate(zip(
            prediction['boxes'], prediction['labels'], prediction['scores']
        )):
            if score > confidence_threshold:
                bbox = box.cpu().numpy()
                class_name = self.all_classes[label.item() - 1] if label.item() - 1 < len(self.all_classes) else "unknown"
                
                detections.append({
                    'bbox': bbox.tolist(),
                    'confidence': score.item(),
                    'label': label.item(),
                    'class_name': class_name,
                    'detection_id': i
                })
        
        logger.info(f"Detected {len(detections)} faces with confidence > {confidence_threshold}")
        return detections
    
    def crop_face_with_padding(self, image: np.ndarray, bbox: List[float], 
                              target_size: Tuple[int, int] = None, 
                              padding_factor: float = None) -> Tuple[np.ndarray, bool, bool]:
        """
        Crop face from image with padding and resize to target size while preserving proportions
        
        Args:
            image: Input image in RGB format
            bbox: Bounding box [x1, y1, x2, y2]
            target_size: Target output size (width, height)
            padding_factor: Padding factor around the bounding box
            
        Returns:
            Tuple of (cropped and resized face image, white_bg_applied, illumination_enhanced)
        """
        if target_size is None:
            target_size = self.default_target_size
        if padding_factor is None:
            padding_factor = self.default_padding_factor
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding around detection (extra top pad frames the full hair crown
        # so segmentation keeps it in view)
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = box_w * padding_factor
        pad_h = box_h * padding_factor
        pad_top = pad_h * 1.45
        
        # Calculate padded coordinates
        x1_pad = max(0, int(x1 - pad_w))
        y1_pad = max(0, int(y1 - pad_top))
        x2_pad = min(w, int(x2 + pad_w))
        y2_pad = min(h, int(y2 + pad_h))
        
        # Crop the image
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]

        # Detected face rectangle in crop-local coordinates. The white-BG mask must
        # NEVER clip the face, so this rect (expanded to cover forehead/hairline and
        # jaw) is forced fully opaque during segmentation refinement — regardless of
        # how uncertain MediaPipe is over those pixels.
        crop_h0, crop_w0 = cropped.shape[:2]
        face_protect_rect = self._face_protect_rect(
            x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad, crop_w0, crop_h0
        )

        # Conditional dark CLAHE before white-BG.
        # CLAHE ownership: preprocess only — morph/antro must not re-apply.
        cropped, illumination_enhanced = ImageProcessor.maybe_enhance_dark(cropped)

        # White-background clean (MediaPipe selfie segmentation) on crop; fail-open.
        # face_protect_rect guarantees the face is never cut by the mask.
        cropped, white_bg_applied = self.apply_white_background(
            cropped, protect_rect=face_protect_rect
        )
        
        crop_h, crop_w = cropped.shape[:2]
        
        # Scale to fit within target size while preserving aspect ratio
        scale = min(target_size[0] / crop_w, target_size[1] / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        resized = cv2.resize(cropped, (new_w, new_h))
        
        # Center in target size canvas with white letterbox
        final_image = np.full((target_size[1], target_size[0], 3), 255, dtype=np.uint8)
        start_y = (target_size[1] - new_h) // 2
        start_x = (target_size[0] - new_w) // 2
        final_image[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return final_image, white_bg_applied, illumination_enhanced
    
    def image_to_base64(self, image: np.ndarray, format: str = 'JPEG', quality: int = 95) -> str:
        """
        Convert image to base64 string
        
        Args:
            image: Input image in RGB format
            format: Output format ('JPEG', 'PNG')
            quality: JPEG quality (1-100, only for JPEG)
            
        Returns:
            Base64 encoded image string
        """
        # Convert RGB to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Convert to bytes
        buffer = io.BytesIO()
        if format.upper() == 'JPEG':
            pil_image.save(buffer, format='JPEG', quality=quality)
        else:
            pil_image.save(buffer, format='PNG')
        
        # Encode to base64
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64
    
    def process_image(self, image: np.ndarray,
                     confidence_threshold: float = None,
                     target_size: Tuple[int, int] = None,
                     padding_factor: float = None,
                     output_format: str = 'JPEG',
                     quality: int = 95,
                     apply_rotation: bool = False) -> Dict:
        """
        Complete preprocessing pipeline: detect faces, crop, and convert to base64

        Args:
            image: Input image in RGB format
            confidence_threshold: Minimum confidence for face detection
            target_size: Target output size for cropped faces
            padding_factor: Padding factor around detected faces
            output_format: Output image format ('JPEG', 'PNG')
            quality: JPEG quality (1-100)
            apply_rotation: Whether to apply face rotation alignment using points 34 and 10

        Returns:
            Dictionary with detection results and base64 encoded cropped faces
        """
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        if target_size is None:
            target_size = self.default_target_size
        if padding_factor is None:
            padding_factor = self.default_padding_factor

        # Apply rotation alignment if requested and available
        rotation_metadata = None
        working_image = image

        if apply_rotation and self.rotation_aligner is not None:
            logger.info("Applying face rotation alignment...")
            rotated_image, rotation_metadata = self.rotation_aligner.align_face(image)

            if rotated_image is not None:
                working_image = rotated_image
                logger.info(f"Rotation applied: {rotation_metadata.get('rotation_angle', 0):.2f}°")
            else:
                logger.warning(f"Rotation failed: {rotation_metadata.get('error', 'Unknown error')}")
        elif apply_rotation and self.rotation_aligner is None:
            logger.warning("Rotation requested but rotation aligner not available")

        # Detect faces
        detections = self.detect_faces(working_image, confidence_threshold)

        # Process each detection
        processed_faces = []
        for detection in detections:
            # Crop face (dark enhance → segmentation white-bg → letterbox)
            cropped_face, white_bg_applied, illumination_enhanced = self.crop_face_with_padding(
                working_image, detection['bbox'], target_size, padding_factor
            )

            # Convert to base64
            face_base64 = self.image_to_base64(cropped_face, output_format, quality)

            processed_faces.append({
                'detection_id': detection['detection_id'],
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'class_name': detection['class_name'],
                'cropped_image_base64': face_base64,
                'target_size': target_size,
                'padding_factor': padding_factor,
                'white_bg_applied': white_bg_applied,
                'illumination_enhanced': illumination_enhanced,
            })

        result = {
            'total_detections': len(detections),
            'processed_faces': processed_faces,
            'original_image_size': image.shape[:2],
            'working_image': working_image,  # The image used for detection (rotated or original)
            'processing_parameters': {
                'confidence_threshold': confidence_threshold,
                'target_size': target_size,
                'padding_factor': padding_factor,
                'output_format': output_format,
                'quality': quality,
                'rotation_applied': apply_rotation
            }
        }

        # Add rotation metadata if rotation was attempted
        if rotation_metadata is not None:
            result['rotation_metadata'] = rotation_metadata
            if rotation_metadata.get('rotation_applied'):
                logger.info(f"Added rotation_metadata to result: {rotation_metadata}")
            else:
                logger.debug("Rotation not applied (e.g. required points not detected): %s", rotation_metadata.get('error', 'Unknown'))

        return result
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': 'Faster R-CNN Profile Face Detection',
            'device': str(self.device),
            'model_path': self.model_path,
            'num_classes': self.num_classes,
            'all_classes': self.all_classes,
            'default_confidence_threshold': self.default_confidence_threshold,
            'default_target_size': self.default_target_size,
            'default_padding_factor': self.default_padding_factor
        }