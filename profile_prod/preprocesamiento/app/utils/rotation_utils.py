import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MinimalPointModel(nn.Module):
    """Minimal point detection model for detecting anthropometric points"""
    def __init__(self, num_keypoints):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.profile_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 2), nn.Softmax(dim=1)
        )

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(2048, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 32, 1), nn.ReLU(),
                nn.Conv2d(32, 512, 1), nn.Sigmoid()
            ),
            nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
                         nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU()),
            nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                         nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()),
            nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                         nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        ])

        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, num_keypoints, 1), nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        profile_logits = self.profile_classifier(features)

        decoded = features
        stage1 = self.decoder[0]
        main_layers, attention_layers = stage1[:4], stage1[4:]
        decoded = main_layers(decoded)
        attention = attention_layers(decoded)
        decoded = decoded * attention

        for i in range(1, len(self.decoder)):
            decoded = self.decoder[i](decoded)

        heatmaps = self.final_layer(decoded)
        return heatmaps, profile_logits


class FaceRotationAligner:
    """
    Face alignment using point detection to create vertical alignment
    based on specific anthropometric points (34 and 10).
    """

    def __init__(self, point_model_path: str, device: str = 'cuda'):
        """
        Initialize the face rotation aligner

        Args:
            point_model_path: Path to the point detection model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.point_model = None
        self.point_classes = []
        self.heatmap_size = 112

        logger.info(f"Initializing FaceRotationAligner on {self.device}")
        self._load_point_model(point_model_path)

    def _load_point_model(self, model_path: str):
        """Load the point detection model"""
        try:
            logger.info(f"Loading point detection model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            self.point_classes = checkpoint['all_classes']
            # Handle both 'num_keypoints' and inferring from all_classes length
            if 'num_keypoints' in checkpoint:
                num_keypoints = checkpoint['num_keypoints']
            else:
                num_keypoints = len(checkpoint['all_classes'])
                logger.warning(f"'num_keypoints' not in checkpoint, inferring from all_classes: {num_keypoints}")

            self.heatmap_size = checkpoint.get('heatmap_size', 112)

            # Create and load model
            self.point_model = MinimalPointModel(num_keypoints)
            self.point_model.load_state_dict(checkpoint['model_state_dict'])
            self.point_model.to(self.device)
            self.point_model.eval()

            logger.info(f"Point detection model loaded with {len(self.point_classes)} point classes")

        except Exception as e:
            logger.error(f"Failed to load point detection model: {str(e)}")
            raise e

    def detect_points(self, image: np.ndarray, target_size: int = 224) -> Dict[str, Tuple[float, float]]:
        """
        Detect anthropometric points in the image

        Args:
            image: Input image in RGB format
            target_size: Size to resize image for point detection

        Returns:
            Dictionary mapping point names to coordinates
        """
        # Resize for model
        original_h, original_w = image.shape[:2]
        resized_image = cv2.resize(image, (target_size, target_size))

        # Convert to tensor
        image_tensor = torch.from_numpy(resized_image.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Detect points
        with torch.no_grad():
            heatmaps, profile_logits = self.point_model(image_tensor)

        batch_size, num_keypoints, hm_height, hm_width = heatmaps.shape

        # Simple smoothing
        smoothed_list = []
        kernel = torch.ones(1, 1, 3, 3, device=heatmaps.device) / 9

        for i in range(num_keypoints):
            channel = heatmaps[:, i:i+1, :, :]
            smoothed_channel = F.conv2d(channel, kernel, padding=1)
            smoothed_list.append(smoothed_channel)

        smoothed = torch.cat(smoothed_list, dim=1)
        heatmaps_flat = smoothed.view(batch_size, num_keypoints, -1)
        max_vals, max_indices = torch.max(heatmaps_flat, dim=2)

        max_x = (max_indices % hm_width).float()
        max_y = (max_indices // hm_width).float()

        scale_x = target_size / hm_width
        scale_y = target_size / hm_height

        keypoints = torch.stack([max_x * scale_x, max_y * scale_y], dim=2)
        confidences = max_vals

        # Extract points and scale back to original image size
        points_dict = {}
        keypoints_np = keypoints[0].cpu().numpy()
        confidences_np = confidences[0].cpu().numpy()

        # Use adaptive threshold
        threshold = 0.15 if any(conf > 0.15 for conf in confidences_np) else 0.05

        # Scale factors to convert from target_size to original image size
        scale_to_original_x = original_w / target_size
        scale_to_original_y = original_h / target_size

        for i, (point, conf) in enumerate(zip(keypoints_np, confidences_np)):
            if conf > threshold and i < len(self.point_classes):
                class_name = self.point_classes[i]
                # Remove _i and _d suffixes
                clean_class = class_name.replace('_i', '').replace('_d', '')

                # Scale coordinates back to original image size
                original_x = point[0] * scale_to_original_x
                original_y = point[1] * scale_to_original_y

                points_dict[clean_class] = (original_x, original_y)

        logger.info(f"Detected {len(points_dict)} anthropometric points")
        return points_dict

    def calculate_rotation_angle(self, point_34: Tuple[float, float],
                                 point_10: Tuple[float, float]) -> float:
        """
        Calculate rotation angle needed to make the vector from point 34 to point 10 vertical

        Args:
            point_34: Coordinates of point 34 (x, y)
            point_10: Coordinates of point 10 (x, y)

        Returns:
            Rotation angle in degrees
        """
        # Create vector from point 34 to point 10
        dx = point_10[0] - point_34[0]
        dy = point_10[1] - point_34[1]

        # Calculate angle of this vector relative to horizontal
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # To make this vector vertical (parallel to Y-axis, pointing down at 90°),
        # we need to rotate by (angle_deg - 90) degrees
        # OpenCV rotates counter-clockwise for positive angles
        rotation_angle = angle_deg - 90

        logger.info(f"Vector 34->10: dx={dx:.2f}, dy={dy:.2f}, angle={angle_deg:.2f}°, rotation={rotation_angle:.2f}°")

        return rotation_angle

    def rotate_image(self, image: np.ndarray, angle: float,
                    center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Rotate image around a center point

        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)
            center: Center of rotation (if None, uses image center)

        Returns:
            Rotated image
        """
        h, w = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image size to prevent cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix to account for new image size
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))

        return rotated

    def align_face(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Align face by detecting points 34 and 10 and rotating to make their vector vertical

        Args:
            image: Input image in RGB format

        Returns:
            Tuple of (rotated_image, metadata_dict)
            - rotated_image: Aligned image (None if points not found)
            - metadata_dict: Information about the rotation process
        """
        metadata = {
            'rotation_applied': False,
            'rotation_angle': None,
            'points_detected': {},
            'error': None
        }

        try:
            # Detect anthropometric points
            points = self.detect_points(image)

            # Check if points 34 and 10 are detected
            if '34' not in points or '10' not in points:
                missing_points = []
                if '34' not in points:
                    missing_points.append('34')
                if '10' not in points:
                    missing_points.append('10')

                error_msg = f"Required points not detected: {', '.join(missing_points)}"
                logger.warning(error_msg)
                metadata['error'] = error_msg
                metadata['points_detected'] = list(points.keys())
                return None, metadata

            point_34 = points['34']
            point_10 = points['10']

            metadata['points_detected'] = {
                '34': list(point_34),  # Convert tuple to list for JSON serialization
                '10': list(point_10)
            }

            # Calculate rotation angle
            rotation_angle = self.calculate_rotation_angle(point_34, point_10)

            # Calculate center of rotation (midpoint between the two points)
            center = (
                (point_34[0] + point_10[0]) / 2,
                (point_34[1] + point_10[1]) / 2
            )

            # Rotate image
            rotated_image = self.rotate_image(image, rotation_angle, center)

            metadata['rotation_applied'] = True
            metadata['rotation_angle'] = float(rotation_angle)  # Ensure it's JSON serializable
            metadata['rotation_center'] = list(center)  # Convert tuple to list

            logger.info(f"Successfully aligned face with rotation angle: {rotation_angle:.2f}°")

            return rotated_image, metadata

        except Exception as e:
            error_msg = f"Face alignment failed: {str(e)}"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata
