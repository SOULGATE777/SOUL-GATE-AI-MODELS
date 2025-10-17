import torch
import torch.nn as nn
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
        self.default_padding_factor = 0.15

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
                              padding_factor: float = None) -> np.ndarray:
        """
        Crop face from image with padding and resize to target size while preserving proportions
        
        Args:
            image: Input image in RGB format
            bbox: Bounding box [x1, y1, x2, y2]
            target_size: Target output size (width, height)
            padding_factor: Padding factor around the bounding box
            
        Returns:
            Cropped and resized face image
        """
        if target_size is None:
            target_size = self.default_target_size
        if padding_factor is None:
            padding_factor = self.default_padding_factor
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding around detection
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = box_w * padding_factor
        pad_h = box_h * padding_factor
        
        # Calculate padded coordinates
        x1_pad = max(0, int(x1 - pad_w))
        y1_pad = max(0, int(y1 - pad_h))
        x2_pad = min(w, int(x2 + pad_w))
        y2_pad = min(h, int(y2 + pad_h))
        
        # Crop the image
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        crop_h, crop_w = cropped.shape[:2]
        
        # Scale to fit within target size while preserving aspect ratio
        scale = min(target_size[0] / crop_w, target_size[1] / crop_h)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        resized = cv2.resize(cropped, (new_w, new_h))
        
        # Center in target size canvas with black background
        final_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        start_y = (target_size[1] - new_h) // 2
        start_x = (target_size[0] - new_w) // 2
        final_image[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return final_image
    
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
            # Crop face
            cropped_face = self.crop_face_with_padding(
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
                'padding_factor': padding_factor
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
            logger.info(f"Added rotation_metadata to result: {rotation_metadata}")

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