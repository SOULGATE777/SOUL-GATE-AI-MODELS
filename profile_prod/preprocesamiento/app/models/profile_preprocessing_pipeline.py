import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from rembg import remove as rembg_remove, new_session as rembg_new_session
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
        # Generous pad so nose/crown/back-of-head stay inside the crop frame
        # (gateway previously forced 0.15 and clipped anatomy).
        self.default_padding_factor = 0.40
        # White border around crop before rembg so the subject is never at the
        # tensor edge (rembg softens edge pixels → “cut” hair / soft neck).
        self.rembg_edge_margin_frac = 0.08
        self.rembg_edge_margin_min_px = 12

        # rembg person matting for white-background cleaning (REMBG_MODEL). A
        # dedicated matting model: unlike MediaPipe selfie segmentation it does
        # not confidently mis-classify reflective glass / walls / fences adjacent
        # to the head as foreground, so busy real-world profile backgrounds are
        # removed cleanly. Model overridable via REMBG_MODEL.
        self.rembg_model_name = os.getenv("REMBG_MODEL", "isnet-general-use")
        self.rembg_session = None  # Lazy initialization

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

    def _get_rembg_session(self):
        """Lazy initialization of the rembg matting session (reuse instance)."""
        if self.rembg_session is None:
            self.rembg_session = rembg_new_session(self.rembg_model_name)
            logger.info(f"rembg session initialized (model={self.rembg_model_name}) for white background")
        return self.rembg_session

    @staticmethod
    def _face_protect_rect(fx1: float, fy1: float, fx2: float, fy2: float,
                           crop_w: int, crop_h: int) -> Optional[Tuple[int, int, int, int]]:
        """Shrink a crop-local face bbox into a small central protection core.

        rembg's person matte (REMBG_MODEL) is a reliable, high-confidence alpha
        that keeps the whole face/head on its own, so this guard is only a
        catastrophic-failure backstop — and it MUST stay strictly INSIDE the subject.

        The profile detector's bbox is ~the whole head and the crop is tight
        around it (crop = bbox + padding), so an OUTWARD-expanded guard clamps to
        the entire crop and forces the background fully opaque, defeating
        background removal entirely. Instead we keep the central ~50% of the
        detected face bbox: that core always lands on the subject (cheek / ear /
        hair mass), so it guarantees the face core is never whitened WITHOUT
        re-adding the surrounding background. rembg's matte protects the actual
        face edges (forehead / nose / jaw), which it does cleanly.

        Returns (px1, py1, px2, py2) or None when the bbox is degenerate.
        """
        fw = fx2 - fx1
        fh = fy2 - fy1
        if fw <= 0 or fh <= 0:
            return None

        # Central core (~50% of the bbox), centered on the face bbox. Shrinking
        # inward keeps the guard on-subject so it can never force background
        # opaque, unlike the previous outward expansion.
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        half_w = fw * 0.25
        half_h = fh * 0.25

        px1 = int(round(cx - half_w))
        px2 = int(round(cx + half_w))
        py1 = int(round(cy - half_h))
        py2 = int(round(cy + half_h))

        px1 = max(0, min(px1, crop_w))
        px2 = max(0, min(px2, crop_w))
        py1 = max(0, min(py1, crop_h))
        py2 = max(0, min(py2, crop_h))
        if px2 <= px1 or py2 <= py1:
            return None
        return px1, py1, px2, py2

    @staticmethod
    def _refine_matte(mask: np.ndarray,
                      protect_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Clean a rembg person matte and enforce the face guard.

        See ``app.utils.matte_refine.refine_person_matte`` for behaviour.
        """
        from ..utils.matte_refine import refine_person_matte
        return refine_person_matte(mask, protect_rect=protect_rect)

    def apply_white_background(self, image_rgb: np.ndarray,
                              protect_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, bool]:
        """Soft-composite subject onto white via rembg person matting (REMBG_MODEL).

        A dedicated matting model replaces MediaPipe selfie segmentation because
        MediaPipe confidently mis-classified reflective glass / walls / fences
        adjacent to the head as foreground, leaving large background regions on
        real-world profile photos. rembg's ``only_mask`` output is a clean soft
        matte; it is lightly refined (keep-largest to drop specks) and a small
        on-subject face core is forced opaque as a backstop (rembg owns the face
        edges). Fail-open on errors / missing mask (returns the original crop untouched).

        Args:
            image_rgb: HxWx3 uint8 RGB image (a face/head crop).
            protect_rect: (x1, y1, x2, y2) crop-local face region to never clip.

        Returns:
            Tuple of (composited_or_original_image, white_bg_applied).
        """
        try:
            session = self._get_rembg_session()
            mask = rembg_remove(image_rgb, session=session, only_mask=True)
            if mask is None or getattr(mask, "size", 0) == 0:
                logger.warning("rembg returned no mask; skipping white BG")
                return image_rgb, False
            alpha = self._refine_matte(mask, protect_rect=protect_rect)
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
        
        # Asymmetric pad: extra horizontal for nose + hair silhouette; extra top
        # for crown; slightly more bottom for neck. Locked coeffs (critic).
        box_w = x2 - x1
        box_h = y2 - y1
        pad_side = box_w * padding_factor * 1.35
        pad_h = box_h * padding_factor
        pad_top = pad_h * 1.65
        pad_bottom = pad_h * 1.20
        
        # Calculate padded coordinates
        x1_pad = max(0, int(x1 - pad_side))
        y1_pad = max(0, int(y1 - pad_top))
        x2_pad = min(w, int(x2 + pad_side))
        y2_pad = min(h, int(y2 + pad_bottom))
        
        # Crop the image
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]

        # Detected face rectangle in crop-local coordinates. A small central core
        # of this bbox (kept strictly on-subject) is forced fully opaque during
        # matte refinement as a face-clip backstop; rembg's matte handles the
        # actual face edges and the surrounding background is removed normally.
        crop_h0, crop_w0 = cropped.shape[:2]
        face_protect_rect = self._face_protect_rect(
            x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad, crop_w0, crop_h0
        )

        # Conditional dark CLAHE before white-BG.
        # CLAHE ownership: preprocess only — morph/antro must not re-apply.
        cropped, illumination_enhanced = ImageProcessor.maybe_enhance_dark(cropped)

        # White margin ring before rembg so the subject is never at the tensor
        # edge (otherwise rembg soft-fades hair/neck into white). Offset the
        # face-protect rect by the same margin.
        margin = max(
            self.rembg_edge_margin_min_px,
            int(min(cropped.shape[0], cropped.shape[1]) * self.rembg_edge_margin_frac),
        )
        cropped_for_matte = cv2.copyMakeBorder(
            cropped, margin, margin, margin, margin,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        protect_for_matte = None
        if face_protect_rect is not None:
            px1, py1, px2, py2 = face_protect_rect
            protect_for_matte = (
                px1 + margin, py1 + margin, px2 + margin, py2 + margin
            )

        # White-background clean (rembg person matting / REMBG_MODEL); fail-open.
        matted, white_bg_applied = self.apply_white_background(
            cropped_for_matte, protect_rect=protect_for_matte
        )
        # Keep the margin (becomes letterbox whitespace) — do not trim back to
        # the pre-ring crop, or edge pixels would again sit on the frame.
        
        crop_h, crop_w = matted.shape[:2]
        
        # Scale to fit within target size while preserving aspect ratio
        scale = min(target_size[0] / crop_w, target_size[1] / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        resized = cv2.resize(matted, (new_w, new_h))
        
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