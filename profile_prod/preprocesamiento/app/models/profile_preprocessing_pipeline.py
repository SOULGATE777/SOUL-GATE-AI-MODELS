import torch
import cv2
import numpy as np
import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from app.utils.rotation_utils import FaceRotationAligner
from app.utils.image_processing import ImageProcessor
from app.utils import photoroom_client

logger = logging.getLogger(__name__)

# Log once that rembg-era kwargs on apply_white_background are ignored.
_REMBG_OVERRIDES_IGNORED_LOGGED = False

# Locked asymmetric pad coeffs relative to base padding (side / top / bottom).
PADDING_ASYMMETRIC = {"side": 1.35, "top": 1.65, "bottom": 1.20}


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
        # White border around crop before white-BG so the subject is never at
        # the tensor edge. Calibrated 2026-08-01 against /Downloads/sample
        # (75 imgs): edge 0.16/20 beats 0.08/12 on residual BG mass.
        self.rembg_edge_margin_frac = 0.16
        self.rembg_edge_margin_min_px = 20
        # Face-protect core half-extent as fraction of bbox (0.25 → ~50% central core).
        self.face_protect_core_frac = 0.25

        # White-BG provider meta (Photoroom). Legacy rembg_model Form/kwargs kept
        # for call-site compat but ignored by apply_white_background.
        self.rembg_model_name = "photoroom"

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

    @staticmethod
    def _face_protect_rect(fx1: float, fy1: float, fx2: float, fy2: float,
                           crop_w: int, crop_h: int,
                           core_frac: float = 0.25) -> Optional[Tuple[int, int, int, int]]:
        """Shrink a crop-local face bbox into a small central protection core.

        rembg's person matte (REMBG_MODEL) is a reliable, high-confidence alpha
        that keeps the whole face/head on its own, so this guard is only a
        catastrophic-failure backstop — and it MUST stay strictly INSIDE the subject.

        The profile detector's bbox is ~the whole head and the crop is tight
        around it (crop = bbox + padding), so an OUTWARD-expanded guard clamps to
        the entire crop and forces the background fully opaque, defeating
        background removal entirely. Instead we keep the central core of the
        detected face bbox (default core_frac=0.25 → ~50% of each side): that
        core always lands on the subject (cheek / ear / hair mass), so it
        guarantees the face core is never whitened WITHOUT re-adding the
        surrounding background. rembg's matte protects the actual face edges
        (forehead / nose / jaw), which it does cleanly.

        Returns (px1, py1, px2, py2) or None when the bbox is degenerate.
        """
        fw = fx2 - fx1
        fh = fy2 - fy1
        if fw <= 0 or fh <= 0:
            return None

        # Central core, centered on the face bbox. Shrinking inward keeps the
        # guard on-subject so it can never force background opaque.
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        half_w = fw * core_frac
        half_h = fh * core_frac

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
    def _face_silhouette_rect(fx1: float, fy1: float, fx2: float, fy2: float,
                              crop_w: int, crop_h: int,
                              expand_frac: float = 0.18,
                              ) -> Optional[Tuple[int, int, int, int]]:
        """Expanded face bbox for subject-aware silhouette protect.

        Profile detectors put nose/lips on the bbox edge; a small outward expand
        (clamped to the crop) lets subject-aware protect restore those pixels
        without using a hard opaque rectangle that would lock light BG inside
        the head box (the failure mode of raising face_protect_core_frac).

        Horizontal expand is asymmetric toward the side closer to the crop edge
        (profile front usually hugs one side).
        """
        fw = fx2 - fx1
        fh = fy2 - fy1
        if fw <= 0 or fh <= 0:
            return None
        gap_left = fx1
        gap_right = crop_w - fx2
        if gap_right <= gap_left:
            # Front toward right edge
            ex_right = fw * expand_frac
            ex_left = fw * expand_frac * 0.5
        else:
            # Front toward left edge
            ex_left = fw * expand_frac
            ex_right = fw * expand_frac * 0.5
        ey = fh * expand_frac
        sx1 = int(round(fx1 - ex_left))
        sx2 = int(round(fx2 + ex_right))
        sy1 = int(round(fy1 - ey * 0.5))  # less top (hair/BG)
        sy2 = int(round(fy2 + ey))        # more bottom (chin)
        sx1 = max(0, min(sx1, crop_w))
        sx2 = max(0, min(sx2, crop_w))
        sy1 = max(0, min(sy1, crop_h))
        sy2 = max(0, min(sy2, crop_h))
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        return sx1, sy1, sx2, sy2

    @staticmethod
    def _refine_matte(mask: np.ndarray,
                      protect_rect: Optional[Tuple[int, int, int, int]] = None,
                      image_rgb: Optional[np.ndarray] = None,
                      silhouette_rect: Optional[Tuple[int, int, int, int]] = None,
                      ) -> np.ndarray:
        """Clean a rembg person matte and enforce the face guards.

        See ``app.utils.matte_refine.refine_person_matte`` for behaviour.
        """
        from ..utils.matte_refine import refine_person_matte
        return refine_person_matte(
            mask,
            protect_rect=protect_rect,
            image_rgb=image_rgb,
            silhouette_rect=silhouette_rect,
        )

    def apply_white_background(self, image_rgb: np.ndarray,
                              protect_rect: Optional[Tuple[int, int, int, int]] = None,
                              rembg_model: Optional[str] = None,
                              silhouette_rect: Optional[Tuple[int, int, int, int]] = None,
                              use_photoroom: bool = False,
                              photoroom_bg_color: str = "white",
                              ) -> Tuple[np.ndarray, bool]:
        """Composite subject onto bg via Photoroom Remove Background API.

        Signature keeps protect_rect / rembg_model / silhouette_rect for call-site
        compatibility; those rembg-era overrides are ignored (Photoroom returns a
        finished RGB image). Fail-open on missing key / API errors.

        Args:
            image_rgb: HxWx3 uint8 RGB image (a face/head crop).
            protect_rect: Ignored (legacy rembg face-core guard).
            rembg_model: Ignored (legacy rembg model override).
            silhouette_rect: Ignored (legacy rembg silhouette restore).
            use_photoroom: When False (default), skip Photoroom and return image unchanged.
            photoroom_bg_color: Photoroom bg_color (white or #a6a6a6; unknown → white).

        Returns:
            Tuple of (composited_or_original_image, white_bg_applied).
        """
        if not use_photoroom:
            return image_rgb, False

        global _REMBG_OVERRIDES_IGNORED_LOGGED
        if not _REMBG_OVERRIDES_IGNORED_LOGGED:
            logger.debug(
                "rembg overrides (protect_rect/silhouette_rect/rembg_model) "
                "ignored; white-BG uses Photoroom"
            )
            _REMBG_OVERRIDES_IGNORED_LOGGED = True

        try:
            composited = photoroom_client.remove_background_white(
                image_rgb, bg_color=photoroom_bg_color
            )
            if composited is None:
                return image_rgb, False
            return composited, True
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
                              padding_factor: float = None,
                              apply_white_bg: bool = True,
                              rembg_model: Optional[str] = None,
                              rembg_edge_margin_frac: Optional[float] = None,
                              rembg_edge_margin_min_px: Optional[int] = None,
                              face_protect_core_frac: Optional[float] = None,
                              use_photoroom: bool = False,
                              photoroom_bg_color: str = "white",
                              ) -> Tuple[np.ndarray, bool, bool]:
        """
        Crop face from image with padding and resize to target size while preserving proportions
        
        Args:
            image: Input image in RGB format
            bbox: Bounding box [x1, y1, x2, y2]
            target_size: Target output size (width, height)
            padding_factor: Padding factor around the bounding box
            apply_white_bg: When False, skip Photoroom white-BG and return crop after illumination
            rembg_model: Legacy override (echoed in meta; ignored by Photoroom path)
            rembg_edge_margin_frac: Optional edge margin fraction override
            rembg_edge_margin_min_px: Optional edge margin min px override
            face_protect_core_frac: Optional face-protect core half-extent override
            use_photoroom: When False (default), skip Photoroom API even if apply_white_bg
            photoroom_bg_color: Photoroom bg_color (white or #a6a6a6; unknown → white)
            
        Returns:
            Tuple of (cropped and resized face image, white_bg_applied, illumination_enhanced)
        """
        if target_size is None:
            target_size = self.default_target_size
        if padding_factor is None:
            padding_factor = self.default_padding_factor

        resolved_rembg_model = rembg_model if rembg_model is not None else self.rembg_model_name
        resolved_margin_frac = (
            rembg_edge_margin_frac
            if rembg_edge_margin_frac is not None
            else self.rembg_edge_margin_frac
        )
        resolved_margin_min = (
            rembg_edge_margin_min_px
            if rembg_edge_margin_min_px is not None
            else self.rembg_edge_margin_min_px
        )
        resolved_core_frac = (
            face_protect_core_frac
            if face_protect_core_frac is not None
            else self.face_protect_core_frac
        )
        resolved_bg = photoroom_client.normalize_photoroom_bg_color(photoroom_bg_color)
        fill_rgb = (
            photoroom_client.bg_color_to_rgb(resolved_bg)
            if (apply_white_bg and use_photoroom)
            else (255, 255, 255)
        )
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Asymmetric pad: extra horizontal for nose + hair silhouette; extra top
        # for crown; slightly more bottom for neck. Locked coeffs (critic).
        box_w = x2 - x1
        box_h = y2 - y1
        pad_side = box_w * padding_factor * PADDING_ASYMMETRIC["side"]
        pad_h = box_h * padding_factor
        pad_top = pad_h * PADDING_ASYMMETRIC["top"]
        pad_bottom = pad_h * PADDING_ASYMMETRIC["bottom"]
        
        # Calculate padded coordinates
        x1_pad = max(0, int(x1 - pad_side))
        y1_pad = max(0, int(y1 - pad_top))
        x2_pad = min(w, int(x2 + pad_side))
        y2_pad = min(h, int(y2 + pad_bottom))

        # Pin crop to image border when face bbox hugs that edge so profile
        # nose/mouth at the frame edge are never left outside the crop.
        edge_tol_x = max(8, box_w * 0.12)
        edge_tol_y = max(8, box_h * 0.12)
        if (w - x2) <= edge_tol_x:
            x2_pad = w
        if x1 <= edge_tol_x:
            x1_pad = 0
        if (h - y2) <= edge_tol_y:
            y2_pad = h
        if y1 <= edge_tol_y:
            y1_pad = 0
        
        # Crop the image
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]

        # Detected face rectangle in crop-local coordinates.
        # Hard core: small central protect (catastrophic backstop).
        # Silhouette shell: slightly expanded bbox + subject-aware restore so
        # profile nose/lips/chin at the detector edge are not whitened.
        crop_h0, crop_w0 = cropped.shape[:2]
        fx1_c = x1 - x1_pad
        fy1_c = y1 - y1_pad
        fx2_c = x2 - x1_pad
        fy2_c = y2 - y1_pad
        face_protect_rect = self._face_protect_rect(
            fx1_c, fy1_c, fx2_c, fy2_c, crop_w0, crop_h0,
            core_frac=resolved_core_frac,
        )
        face_silhouette_rect = self._face_silhouette_rect(
            fx1_c, fy1_c, fx2_c, fy2_c, crop_w0, crop_h0,
        )

        # Conditional dark CLAHE before white-BG.
        # CLAHE ownership: preprocess only — morph/antro must not re-apply.
        cropped, illumination_enhanced = ImageProcessor.maybe_enhance_dark(cropped)

        white_bg_applied = False
        if apply_white_bg and use_photoroom:
            # Color margin ring before Photoroom so the subject is never at the
            # tensor edge. Offset face-protect / silhouette rects by the same
            # margin (kwargs kept for call-site compat; Photoroom ignores them).
            margin = max(
                resolved_margin_min,
                int(min(cropped.shape[0], cropped.shape[1]) * resolved_margin_frac),
            )
            cropped_for_matte = cv2.copyMakeBorder(
                cropped, margin, margin, margin, margin,
                cv2.BORDER_CONSTANT, value=fill_rgb,
            )
            protect_for_matte = None
            if face_protect_rect is not None:
                px1, py1, px2, py2 = face_protect_rect
                protect_for_matte = (
                    px1 + margin, py1 + margin, px2 + margin, py2 + margin
                )
            silhouette_for_matte = None
            if face_silhouette_rect is not None:
                sx1, sy1, sx2, sy2 = face_silhouette_rect
                silhouette_for_matte = (
                    sx1 + margin, sy1 + margin, sx2 + margin, sy2 + margin
                )

            # Background clean (Photoroom); fail-open.
            matted, white_bg_applied = self.apply_white_background(
                cropped_for_matte,
                protect_rect=protect_for_matte,
                rembg_model=resolved_rembg_model,
                silhouette_rect=silhouette_for_matte,
                use_photoroom=use_photoroom,
                photoroom_bg_color=resolved_bg,
            )
            # Keep the margin (becomes letterbox whitespace) — do not trim back to
            # the pre-ring crop, or edge pixels would again sit on the frame.
        else:
            matted = cropped
        
        crop_h, crop_w = matted.shape[:2]
        
        # Scale to fit within target size while preserving aspect ratio
        scale = min(target_size[0] / crop_w, target_size[1] / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        resized = cv2.resize(matted, (new_w, new_h))
        
        # Center in target size canvas (match Photoroom bg when enabled)
        final_image = np.full(
            (target_size[1], target_size[0], 3), fill_rgb, dtype=np.uint8
        )
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
                     apply_rotation: bool = False,
                     apply_white_bg: bool = True,
                     rembg_model: Optional[str] = None,
                     rembg_edge_margin_frac: Optional[float] = None,
                     rembg_edge_margin_min_px: Optional[int] = None,
                     face_protect_core_frac: Optional[float] = None,
                     use_photoroom: bool = False,
                     photoroom_bg_color: str = "white") -> Dict:
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
            apply_white_bg: When False, skip Photoroom white-background cleaning
            rembg_model: Legacy override (echoed in meta; ignored by Photoroom path)
            rembg_edge_margin_frac: Optional edge margin fraction override
            rembg_edge_margin_min_px: Optional edge margin min px override
            face_protect_core_frac: Optional face-protect core half-extent override
            use_photoroom: Admin testing — call Photoroom when True (default False)
            photoroom_bg_color: Photoroom bg_color (white or #a6a6a6; unknown → white)

        Returns:
            Dictionary with detection results and base64 encoded cropped faces
        """
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        if target_size is None:
            target_size = self.default_target_size
        if padding_factor is None:
            padding_factor = self.default_padding_factor

        resolved_rembg_model = rembg_model if rembg_model is not None else self.rembg_model_name
        resolved_margin_frac = (
            rembg_edge_margin_frac
            if rembg_edge_margin_frac is not None
            else self.rembg_edge_margin_frac
        )
        resolved_margin_min = (
            rembg_edge_margin_min_px
            if rembg_edge_margin_min_px is not None
            else self.rembg_edge_margin_min_px
        )
        resolved_core_frac = (
            face_protect_core_frac
            if face_protect_core_frac is not None
            else self.face_protect_core_frac
        )
        resolved_bg = photoroom_client.normalize_photoroom_bg_color(photoroom_bg_color)

        effective_pipeline = {
            "rembg_model": resolved_rembg_model,
            "apply_white_bg": apply_white_bg,
            "use_photoroom": use_photoroom,
            "photoroom_bg_color": resolved_bg,
            "rembg_edge_margin_frac": resolved_margin_frac,
            "rembg_edge_margin_min_px": resolved_margin_min,
            "face_protect_core_frac": resolved_core_frac,
            "padding_asymmetric": dict(PADDING_ASYMMETRIC),
            "default_padding_factor": self.default_padding_factor,
        }

        # Apply rotation alignment if requested and available
        rotation_metadata = None
        working_image = image

        if apply_rotation and self.rotation_aligner is not None:
            logger.info("Applying face rotation alignment...")
            rotated_image, rotation_metadata = self.rotation_aligner.align_face(
                image,
                border_value=photoroom_client.bg_color_to_rgb(resolved_bg)
                if use_photoroom
                else (255, 255, 255),
            )

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
                working_image,
                detection['bbox'],
                target_size,
                padding_factor,
                apply_white_bg=apply_white_bg,
                rembg_model=resolved_rembg_model,
                rembg_edge_margin_frac=resolved_margin_frac,
                rembg_edge_margin_min_px=resolved_margin_min,
                face_protect_core_frac=resolved_core_frac,
                use_photoroom=use_photoroom,
                photoroom_bg_color=resolved_bg,
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
            'effective_pipeline': effective_pipeline,
            'processing_parameters': {
                'confidence_threshold': confidence_threshold,
                'target_size': target_size,
                'padding_factor': padding_factor,
                'output_format': output_format,
                'quality': quality,
                'rotation_applied': apply_rotation,
                'effective_pipeline': effective_pipeline,
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
            'default_padding_factor': self.default_padding_factor,
            'rembg_model_name': self.rembg_model_name,
            'rembg_edge_margin_frac': self.rembg_edge_margin_frac,
            'rembg_edge_margin_min_px': self.rembg_edge_margin_min_px,
            'face_protect_core_frac': self.face_protect_core_frac,
            'padding_asymmetric': dict(PADDING_ASYMMETRIC),
        }