import cv2
import numpy as np
import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import mediapipe as mp
from PIL import Image

logger = logging.getLogger(__name__)

class FrontalPreprocessingPipeline:
    """
    Frontal cranium detection and preprocessing pipeline for preparing images
    for downstream analysis services using MediaPipe Face Detection.
    """

    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize the preprocessing pipeline

        Args:
            model_path: Not used for MediaPipe (kept for compatibility)
            device: Device preference (MediaPipe runs on CPU)
        """
        self.device = 'cpu'  # MediaPipe runs on CPU
        self.model_path = model_path

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        # Default processing parameters
        self.default_confidence_threshold = 0.5
        self.default_target_size = (600, 600)
        self.default_padding_factor = 0.15

        # Cranium expansion factors
        self.cranium_height_multiplier = 1.8  # Expand face height by 80% for cranium
        self.cranium_width_multiplier = 1.4   # Expand face width by 40% for cranium

        logger.info(f"Initializing FrontalPreprocessingPipeline with MediaPipe")
        self._load_model()

    def _setup_device(self, device: str) -> str:
        """Setup computation device - MediaPipe uses CPU"""
        logger.info("MediaPipe Face Detection runs on CPU")
        return 'cpu'

    def _load_model(self):
        """Initialize MediaPipe Face Detection"""
        try:
            # Initialize face detection with specified confidence
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range detection (better for close faces)
                min_detection_confidence=self.default_confidence_threshold
            )

            logger.info("MediaPipe Face Detection initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {str(e)}")
            raise e

    def detect_heads(self, image: np.ndarray, confidence_threshold: float = None) -> List[Dict]:
        """
        Detect cranium regions in the frontal image using MediaPipe Face Detection

        Args:
            image: Input image in RGB format
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of detection results with cranium bounding boxes and confidence scores
        """
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold

        # Convert RGB to BGR for MediaPipe
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]

        # Run MediaPipe face detection
        results = self.face_detection.process(image_bgr)

        detections = []

        if results.detections:
            for i, detection in enumerate(results.detections):
                # Get confidence score
                confidence = detection.score[0]

                if confidence >= confidence_threshold:
                    # Get face bounding box in relative coordinates
                    bbox = detection.location_data.relative_bounding_box

                    # Convert to absolute coordinates
                    face_x = int(bbox.xmin * w)
                    face_y = int(bbox.ymin * h)
                    face_w = int(bbox.width * w)
                    face_h = int(bbox.height * h)

                    face_x2 = face_x + face_w
                    face_y2 = face_y + face_h

                    # Expand face region to estimate cranium
                    cranium_w = int(face_w * self.cranium_width_multiplier)
                    cranium_h = int(face_h * self.cranium_height_multiplier)

                    # Calculate cranium center based on face center
                    face_center_x = face_x + face_w // 2
                    face_center_y = face_y + face_h // 2

                    # Position cranium box (move center up slightly to capture more forehead/hair)
                    cranium_center_x = face_center_x
                    cranium_center_y = face_center_y - int(face_h * 0.2)  # Move up 20% of face height

                    # Calculate cranium bounds
                    cranium_x1 = max(0, cranium_center_x - cranium_w // 2)
                    cranium_y1 = max(0, cranium_center_y - cranium_h // 2)
                    cranium_x2 = min(w, cranium_center_x + cranium_w // 2)
                    cranium_y2 = min(h, cranium_center_y + cranium_h // 2)

                    cranium_bbox = [cranium_x1, cranium_y1, cranium_x2, cranium_y2]

                    detections.append({
                        'bbox': cranium_bbox,
                        'confidence': confidence,
                        'label': 0,
                        'class_name': 'cranium',
                        'detection_id': i,
                        'original_face_bbox': [face_x, face_y, face_x2, face_y2],
                        'detection_type': 'cranium_from_face',
                        'expansion_factors': {
                            'width_multiplier': self.cranium_width_multiplier,
                            'height_multiplier': self.cranium_height_multiplier
                        }
                    })

        logger.info(f"Detected {len(detections)} cranium regions with confidence > {confidence_threshold}")
        return detections

    def crop_head_with_padding(self, image: np.ndarray, bbox: List[float],
                              target_size: Tuple[int, int] = None,
                              padding_factor: float = None) -> np.ndarray:
        """
        Crop cranium from image with padding and resize to target size while preserving proportions

        Args:
            image: Input image in RGB format
            bbox: Bounding box [x1, y1, x2, y2]
            target_size: Target output size (width, height)
            padding_factor: Padding factor around the bounding box

        Returns:
            Cropped and resized cranium image
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
                     quality: int = 95) -> Dict:
        """
        Complete preprocessing pipeline: detect cranium, crop, and convert to base64

        Args:
            image: Input image in RGB format
            confidence_threshold: Minimum confidence for cranium detection
            target_size: Target output size for cropped cranium
            padding_factor: Padding factor around detected cranium
            output_format: Output image format ('JPEG', 'PNG')
            quality: JPEG quality (1-100)

        Returns:
            Dictionary with detection results and base64 encoded cropped cranium
        """
        if confidence_threshold is None:
            confidence_threshold = self.default_confidence_threshold
        if target_size is None:
            target_size = self.default_target_size
        if padding_factor is None:
            padding_factor = self.default_padding_factor

        # Detect cranium regions
        detections = self.detect_heads(image, confidence_threshold)

        # Process each detection
        processed_heads = []
        for detection in detections:
            # Crop cranium
            cropped_cranium = self.crop_head_with_padding(
                image, detection['bbox'], target_size, padding_factor
            )

            # Convert to base64
            cranium_base64 = self.image_to_base64(cropped_cranium, output_format, quality)

            processed_heads.append({
                'detection_id': detection['detection_id'],
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'class_name': detection['class_name'],
                'cropped_image_base64': cranium_base64,
                'target_size': target_size,
                'padding_factor': padding_factor,
                'detection_type': detection['detection_type'],
                'original_face_bbox': detection.get('original_face_bbox'),
                'expansion_factors': detection.get('expansion_factors')
            })

        return {
            'total_detections': len(detections),
            'processed_heads': processed_heads,
            'original_image_size': image.shape[:2],
            'processing_parameters': {
                'confidence_threshold': confidence_threshold,
                'target_size': target_size,
                'padding_factor': padding_factor,
                'output_format': output_format,
                'quality': quality
            }
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': 'MediaPipe Frontal Cranium Detection',
            'device': str(self.device),
            'model_path': 'MediaPipe Face Detection (built-in)',
            'detection_classes': ['cranium'],
            'default_confidence_threshold': self.default_confidence_threshold,
            'default_target_size': self.default_target_size,
            'default_padding_factor': self.default_padding_factor,
            'cranium_expansion': {
                'height_multiplier': self.cranium_height_multiplier,
                'width_multiplier': self.cranium_width_multiplier
            }
        }