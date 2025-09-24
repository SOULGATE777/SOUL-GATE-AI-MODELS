import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class LightweightHierarchicalModel(nn.Module):
    """Lightweight model optimized for limited GPU memory"""
    
    def __init__(self, num_body_types=7, num_coarse_types=4):
        super(LightweightHierarchicalModel, self).__init__()
        
        # Use ResNet18 instead of EfficientNet-B4 (much smaller)
        self.backbone = models.resnet18(pretrained=True)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Smaller feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Simple attention
        self.attention = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.Sigmoid()
        )
        
        # Coarse classifier
        self.coarse_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_coarse_types)
        )
        
        # Fine classifier
        self.fine_head = nn.Sequential(
            nn.Linear(128 + num_coarse_types, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_body_types)
        )
        
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        processed_features = self.feature_extractor(features)
        
        # Apply attention
        attention_weights = self.attention(processed_features)
        attended_features = processed_features * attention_weights
        
        # Coarse classification
        coarse_logits = self.coarse_head(attended_features)
        coarse_probs = F.softmax(coarse_logits, dim=1)
        
        # Fine classification
        fine_input = torch.cat([attended_features, coarse_probs], dim=1)
        fine_logits = self.fine_head(fine_input)
        
        
        return {
            'body_type': fine_logits,
            'coarse_type': coarse_logits
        }


class ComprehensiveAnatomicalClassifierResNet34(nn.Module):
    """CNN classifier using ResNet-34 backbone for comprehensive anatomical data."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ComprehensiveAnatomicalClassifierResNet34, self).__init__()

        # Load pre-trained ResNet-34
        if pretrained:
            self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet34(weights=None)

        # Replace classifier head with enhanced regularization for ensemble readiness
        in_features = self.backbone.fc.in_features  # 512 for ResNet-34
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.7),  # Strong regularization
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),  # Added batch normalization
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class AnatomicalPartClassifier(nn.Module):
    """CNN classifier for individual anatomical parts using ResNet-18 backbone - LEGACY."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(AnatomicalPartClassifier, self).__init__()

        # Load pre-trained ResNet-18
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)

        # Replace classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class BodyAnalysisPipeline:
    """Production pipeline for anatomical parts-based body morphological analysis - IMPROVED VERSION"""

    def __init__(self, model_path: str = "/app/models/best_comprehensive_ensemble_resnet34_fixed.pth"):
        self.model_path = model_path
        self.yolo_model_path = "/app/models/yolov8n-pose.pt"
        self.device = self._get_device()
        self.part_size = (128, 128)

        # Updated body type mapping (6 classes)
        self.body_type_mapping = {
            'delgado': 0,
            'gordo': 1,
            'gordograsacuelga': 2,
            'musculoso': 3,
            'musculosogordo': 4,
            'normalpocagrasa': 5
        }

        # Reverse mapping for index to body type
        self.idx_to_body_type = {v: k for k, v in self.body_type_mapping.items()}

        # For backward compatibility, map to old format for API
        self.body_type_classes = list(self.body_type_mapping.keys())
        self.body_type_simple = list(self.body_type_mapping.keys())
        
        
        # Anatomical parts configuration (7 body parts)
        self.ANATOMICAL_PARTS = {
            'torso': {'keypoints': [5, 6, 11, 12], 'names': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']},
            'left_arm': {'keypoints': [5, 7, 9], 'names': ['left_shoulder', 'left_elbow', 'left_wrist']},
            'right_arm': {'keypoints': [6, 8, 10], 'names': ['right_shoulder', 'right_elbow', 'right_wrist']},
            'left_leg': {'keypoints': [11, 13, 15], 'names': ['left_hip', 'left_knee', 'left_ankle']},
            'right_leg': {'keypoints': [12, 14, 16], 'names': ['right_hip', 'right_knee', 'right_ankle']},
            'torso_upper_legs': {'keypoints': [5, 6, 11, 12, 13, 14], 'names': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee']},
            'full_body': {'keypoints': [5, 6, 9, 10, 15, 16], 'names': ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']}
        }

        # Weights for ensemble prediction based on body part reliability
        self.part_weights = {
            'torso': 1.5,
            'full_body': 1.4,
            'torso_upper_legs': 1.3,
            'left_arm': 1.0,
            'right_arm': 1.0,
            'left_leg': 1.2,  # Increased weight since legs now have better crops
            'right_leg': 1.2,  # Increased weight since legs now have better crops
        }
        
        # Models (will be loaded later)
        self.yolo_model = None
        self.model = None  # This will be the AnatomicalPartClassifier
        
        # NEW: Image preprocessing transforms (128x128 with aspect ratio preservation)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"BodyAnalysisPipeline (Anatomical Parts - NO CABEZA) initialized on device: {self.device}")
        logger.info(f"Body types: {list(self.body_type_mapping.values())}")
        logger.info(f"Anatomical parts: {list(self.ANATOMICAL_PARTS.keys())} (NO CABEZA)")
    
    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transforms - REMOVED: Not used in anatomical parts version""" 
        return self.transform
    
    def load_model(self) -> bool:
        """Load YOLO pose model and comprehensive anatomical parts classifier"""
        try:
            # Load YOLO pose detection model
            if not os.path.exists(self.yolo_model_path):
                logger.info("Downloading YOLOv8n-pose model...")
                self.yolo_model = YOLO('yolov8n-pose.pt')
            else:
                self.yolo_model = YOLO(self.yolo_model_path)

            # Load trained comprehensive anatomical parts classifier
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Comprehensive model not found: {self.model_path}")

            # Determine number of classes from model file first
            logger.info(f"Analyzing model structure...")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Get number of classes from the final layer
            if 'backbone.fc.9.weight' in checkpoint:
                num_classes_from_model = checkpoint['backbone.fc.9.weight'].shape[0]
                logger.info(f"Model has {num_classes_from_model} output classes")
            else:
                num_classes_from_model = len(self.body_type_mapping)
                logger.info(f"Using mapping size: {num_classes_from_model} classes")

            # Update body_type_mapping to match model classes
            if num_classes_from_model != len(self.body_type_mapping):
                logger.info(f"Adjusting body_type_mapping from {len(self.body_type_mapping)} to {num_classes_from_model} classes")
                if num_classes_from_model < len(self.body_type_mapping):
                    # Trim mapping
                    keys_to_keep = list(self.body_type_mapping.keys())[:num_classes_from_model]
                    self.body_type_mapping = {k: i for i, k in enumerate(keys_to_keep)}
                else:
                    # Add placeholder classes
                    for i in range(len(self.body_type_mapping), num_classes_from_model):
                        placeholder_name = f"unknown_class_{i}"
                        self.body_type_mapping[placeholder_name] = i
                        logger.info(f"Added placeholder: {placeholder_name} -> {i}")

                # Update reverse mapping
                self.idx_to_body_type = {v: k for k, v in self.body_type_mapping.items()}
                self.body_type_classes = list(self.body_type_mapping.keys())

            # Create comprehensive anatomical parts classifier
            self.model = ComprehensiveAnatomicalClassifierResNet34(
                num_classes=num_classes_from_model,
                pretrained=False
            )

            # Load weights (handle different checkpoint formats)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                best_acc = checkpoint.get('best_acc', 'Unknown')
            else:
                self.model.load_state_dict(checkpoint)
                best_acc = 'Unknown'

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Comprehensive Anatomical Parts Model loaded successfully!")
            logger.info(f"Model type: ComprehensiveAnatomicalClassifierResNet34")
            logger.info(f"Best training accuracy: {best_acc}")
            logger.info(f"Body type classes: {len(self.body_type_classes)}")
            logger.info(f"Anatomical parts: {list(self.ANATOMICAL_PARTS.keys())}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

            return True

        except Exception as e:
            logger.error(f"Failed to load comprehensive anatomical parts models: {e}")
            return False
    
    def get_model_parameters(self) -> int:
        """Get total number of model parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def detect_pose_keypoints(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """Extract pose keypoints using YOLO"""
        try:
            results = self.yolo_model(image, verbose=False)
            
            all_detections = []
            for result in results:
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    
                    for person_keypoints in keypoints_data:
                        person_detection = {
                            'keypoints': {},
                            'confidence_scores': {}
                        }
                        
                        for i, (x, y, conf) in enumerate(person_keypoints):
                            if conf > confidence_threshold and x > 0 and y > 0:
                                person_detection['keypoints'][i] = (float(x), float(y))
                                person_detection['confidence_scores'][i] = float(conf)
                        
                        if len(person_detection['keypoints']) >= 5:
                            all_detections.append(person_detection)
            
            return all_detections
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return []
    
    def create_anatomical_bbox_from_joints(self, keypoints: Dict, part_config: Dict,
                                         torso_width_estimate: Optional[float] = None,
                                         expansion_factor: float = 1.4, min_size: int = 60) -> Optional[Dict]:
        """Create bounding box for anatomical part from keypoints with intelligent leg cropping"""
        part_keypoints = part_config['keypoints']
        part_names = part_config['names']

        valid_joints = []
        used_joints = []

        for i, keypoint_idx in enumerate(part_keypoints):
            if keypoint_idx in keypoints:
                x, y = keypoints[keypoint_idx]
                valid_joints.append([x, y])
                used_joints.append(part_names[i])

        min_joints_required = 2 if len(part_keypoints) <= 3 else 3
        if len(valid_joints) < min_joints_required:
            return None

        valid_joints = np.array(valid_joints)

        x_coords = valid_joints[:, 0]
        y_coords = valid_joints[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Check if this is a leg part and apply intelligent cropping
        part_type = part_names[0] if part_names else ''
        is_leg_part = any('leg' in name.lower() or 'knee' in name.lower() or 'ankle' in name.lower() for name in part_names)

        if is_leg_part and torso_width_estimate is not None:
            # Apply intelligent leg cropping based on torso width
            min_leg_width_ratio = 0.4  # Leg should be at least 40% of torso width
            min_leg_width = torso_width_estimate * min_leg_width_ratio

            # Ensure minimum width for proper muscle capture
            if width < min_leg_width:
                # Center the expansion around the current leg center
                half_new_width = min_leg_width / 2
                x_min = center_x - half_new_width
                x_max = center_x + half_new_width
                width = min_leg_width

            # Apply generous padding for legs
            pad_x = width * 0.4  # Use generous padding to ensure full leg capture
            pad_y = height * 0.15  # Moderate vertical padding

            expanded_width = width + 2 * pad_x
            expanded_height = height + 2 * pad_y
        else:
            # Standard expansion for non-leg parts
            expanded_width = max(width * expansion_factor, min_size)
            expanded_height = max(height * expansion_factor, min_size)

        final_x_min = max(0, int(center_x - expanded_width / 2))
        final_y_min = max(0, int(center_y - expanded_height / 2))
        final_x_max = int(center_x + expanded_width / 2)
        final_y_max = int(center_y + expanded_height / 2)

        final_width = final_x_max - final_x_min
        final_height = final_y_max - final_y_min

        aspect_ratio = final_height / final_width if final_width > 0 else 0
        if final_width < 30 or final_height < 30:
            return None

        # Relax aspect ratio constraints for legs since they can be very tall and narrow
        if is_leg_part:
            if aspect_ratio < 0.2 or aspect_ratio > 6.0:
                return None
        else:
            if aspect_ratio < 0.3 or aspect_ratio > 4.0:
                return None

        return {
            'bbox': [final_x_min, final_y_min, final_x_max, final_y_max],
            'joints_used': used_joints,
            'joint_count': len(valid_joints),
            'aspect_ratio': aspect_ratio,
            'is_leg_part': is_leg_part,
            'applied_intelligent_sizing': is_leg_part and torso_width_estimate is not None
        }
    
    def crop_part_preserve_aspect_ratio(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crop anatomical part with 128x128 aspect ratio preservation"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Ensure bbox is within image boundaries
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))  
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Crop the part
        part_img = image[y1:y2, x1:x2]
        
        if part_img.size == 0:
            return np.zeros((self.part_size[1], self.part_size[0], 3), dtype=np.uint8)
        
        # Calculate current aspect ratio
        crop_h, crop_w = part_img.shape[:2]
        crop_aspect_ratio = crop_h / crop_w
        
        # Target aspect ratio (square canvas but preserving proportions)
        target_h, target_w = self.part_size
        target_aspect_ratio = target_h / target_w
        
        # Create a square canvas with padding to preserve aspect ratio
        if crop_aspect_ratio > target_aspect_ratio:
            # Part is taller than target - fit by height, pad width
            new_h = target_h
            new_w = int(target_h / crop_aspect_ratio)
            
            # Resize maintaining aspect ratio
            resized_part = cv2.resize(part_img, (new_w, new_h))
            
            # Create target-sized canvas and center the resized part
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            canvas[:, x_offset:x_offset+new_w] = resized_part
            
        else:
            # Part is wider than target - fit by width, pad height
            new_w = target_w
            new_h = int(target_w * crop_aspect_ratio)
            
            # Resize maintaining aspect ratio
            resized_part = cv2.resize(part_img, (new_w, new_h))
            
            # Create target-sized canvas and center the resized part
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, :] = resized_part
            
        return canvas
    
    def extract_anatomical_parts(self, image: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """Extract all anatomical parts from an image with intelligent leg cropping"""
        # Detect pose keypoints
        pose_detections = self.detect_pose_keypoints(image)

        if not pose_detections:
            return None, image

        # Use the first (best) detection
        person_data = pose_detections[0]
        keypoints = person_data['keypoints']

        # Calculate torso width for intelligent leg sizing
        torso_width_estimate = self._calculate_torso_width(keypoints)

        extracted_parts = {}

        # Extract each anatomical part with improved logic
        for part_name, part_config in self.ANATOMICAL_PARTS.items():
            bbox_info = self.create_anatomical_bbox_from_joints(
                keypoints, part_config, torso_width_estimate=torso_width_estimate
            )

            if bbox_info is not None:
                part_keypoint_indices = part_config['keypoints']
                confidences = [person_data['confidence_scores'][idx]
                             for idx in part_keypoint_indices if idx in person_data['confidence_scores']]
                avg_confidence = np.mean(confidences) if confidences else 0.0

                # Crop the part WITH 128x128 ASPECT RATIO PRESERVATION
                part_image = self.crop_part_preserve_aspect_ratio(image, bbox_info['bbox'])

                extracted_parts[part_name] = {
                    'image': part_image,
                    'bbox': bbox_info['bbox'],
                    'confidence': avg_confidence,
                    'joints_used': bbox_info['joints_used'],
                    'aspect_ratio': bbox_info['aspect_ratio'],
                    'applied_intelligent_sizing': bbox_info.get('applied_intelligent_sizing', False)
                }

        return extracted_parts, image

    def _calculate_torso_width(self, keypoints: Dict) -> Optional[float]:
        """Calculate torso width estimate for intelligent leg cropping"""
        try:
            # Get shoulder keypoints
            left_shoulder = keypoints.get(5)  # left_shoulder
            right_shoulder = keypoints.get(6)  # right_shoulder
            left_hip = keypoints.get(11)  # left_hip
            right_hip = keypoints.get(12)  # right_hip

            torso_width_estimate = None

            if left_shoulder is not None and right_shoulder is not None:
                shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
                if left_hip is not None and right_hip is not None:
                    hip_width = abs(right_hip[0] - left_hip[0])
                    torso_width_estimate = max(shoulder_width, hip_width)  # Use wider measurement
                else:
                    torso_width_estimate = shoulder_width
            elif left_hip is not None and right_hip is not None:
                torso_width_estimate = abs(right_hip[0] - left_hip[0])

            return torso_width_estimate

        except Exception as e:
            logger.warning(f"Failed to calculate torso width: {e}")
            return None
    
    def predict_single_part(self, part_image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Predict body type for a single anatomical part"""
        # Preprocess image
        input_tensor = self.transform(part_image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()

        # Convert index to body type name
        predicted_body_type = self.idx_to_body_type[predicted_idx]

        return predicted_body_type, confidence, probabilities[0].cpu().numpy()
    
    def analyze_body_type(self, image: np.ndarray, bbox: Optional[List[int]] = None, 
                         confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Complete body type analysis using anatomical parts - NO CABEZA VERSION"""
        if self.model is None or self.yolo_model is None:
            raise RuntimeError("Models not loaded. Call load_model() first.")
        
        try:
            # Apply bbox crop if provided (crop the full image first)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    image = image[y1:y2, x1:x2]
            
            # Extract anatomical parts
            parts, original_image = self.extract_anatomical_parts(image)
            
            if parts is None or len(parts) == 0:
                return {
                    'body_type_analysis': {
                        'predicted_class': 'Unknown',
                        'predicted_class_simple': 'Unknown',
                        'confidence': 0.0,
                        'meets_threshold': False,
                        'all_probabilities': {}
                    },
                    'anatomical_parts_analysis': {
                        'parts_detected': [],
                        'total_parts': 0,
                        'part_predictions': {},
                        'error': 'No anatomical parts detected'
                    },
                    'analysis_summary': {
                        'timestamp': datetime.now().isoformat(),
                        'processing_successful': False,
                        'error': 'No anatomical parts detected',
                        'device_used': str(self.device),
                        'model_type': 'anatomical_parts_no_cabeza'
                    }
                }
            
            # Predict for each part
            part_predictions = {}
            part_predictions_for_ensemble = {}

            for part_name, part_data in parts.items():
                part_image = part_data['image']
                predicted_body_type, confidence, probabilities = self.predict_single_part(part_image)

                part_predictions[part_name] = {
                    'predicted_body_type': predicted_body_type,
                    'confidence': confidence,
                    'probabilities': probabilities.tolist(),
                    'pose_confidence': part_data['confidence'],
                    'original_aspect_ratio': part_data['aspect_ratio'],
                    'bbox': part_data['bbox'],
                    'joints_used': part_data['joints_used'],
                    'applied_intelligent_sizing': part_data.get('applied_intelligent_sizing', False)
                }

                # Prepare for ensemble prediction
                if part_name not in part_predictions_for_ensemble:
                    part_predictions_for_ensemble[part_name] = []

                part_predictions_for_ensemble[part_name].append({
                    'body_type': predicted_body_type,
                    'confidence': confidence,
                    'probabilities': probabilities
                })

            # Generate ensemble prediction using weighted voting
            final_body_type, final_confidence, ensemble_probs = self._ensemble_prediction(part_predictions_for_ensemble)

            # Create all_probabilities mapping for API compatibility
            all_probabilities_dict = {
                body_type: float(ensemble_probs.get(body_type, 0.0))
                for body_type in self.body_type_classes
            }
            
            
            # Calculate analysis metrics
            analysis_metrics = {
                'overall_confidence': final_confidence,
                'parts_detected_count': len(parts),
                'voting_strategy': 'enhanced_weighted_ensemble',
                'intelligent_leg_cropping_applied': any(p.get('applied_intelligent_sizing', False) for p in part_predictions.values())
            }
            
            result = {
                'body_type_analysis': {
                    'predicted_class': predicted_body_type,
                    'predicted_class_simple': predicted_body_type,
                    'confidence': final_confidence,
                    'meets_threshold': final_confidence >= confidence_threshold,
                    'all_probabilities': all_probabilities_dict
                },
                'anatomical_parts_analysis': {
                    'parts_detected': list(parts.keys()),
                    'total_parts': len(parts),
                    'part_predictions': part_predictions,
                    'voting_strategy': 'confidence_weighted',
                    'aggregated_probabilities': list(ensemble_probs.values()) if ensemble_probs else []
                },
                'analysis_metrics': analysis_metrics,
                'analysis_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_successful': True,
                    'confidence_threshold_used': confidence_threshold,
                    'device_used': str(self.device),
                    'model_type': 'anatomical_parts_no_cabeza',
                    'image_preprocessed': True,
                    'bbox_applied': bbox is not None,
                    'analysis_type': 'anatomical_parts_morphological_analysis'
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in anatomical parts body analysis: {e}")
            return {
                'body_type_analysis': {
                    'predicted_class': 'Unknown',
                    'predicted_class_simple': 'Unknown',
                    'confidence': 0.0,
                    'meets_threshold': False,
                    'all_probabilities': {}
                },
                'anatomical_parts_analysis': {
                    'parts_detected': [],
                    'total_parts': 0,
                    'part_predictions': {},
                    'error': str(e)
                },
                'analysis_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_successful': False,
                    'error': str(e),
                    'device_used': str(self.device),
                    'model_type': 'anatomical_parts_no_cabeza'
                }
            }
    
    def _ensemble_prediction(self, part_predictions: Dict) -> Tuple[str, float, Dict]:
        """Combine predictions from multiple body parts using optimized weighted voting."""
        if not part_predictions:
            return "unknown", 0.0, {}

        # Aggregate weighted probabilities
        aggregated_probs = defaultdict(float)
        total_weight = 0.0
        diagnostic_info = {}  # Track contribution of each part

        for part_name, predictions in part_predictions.items():
            weight = self.part_weights.get(part_name, 1.0)

            for pred in predictions:
                body_type = pred['body_type']
                confidence = pred['confidence']
                weighted_contribution = confidence * weight

                aggregated_probs[body_type] += weighted_contribution
                total_weight += weight

                # Track diagnostic info
                if body_type not in diagnostic_info:
                    diagnostic_info[body_type] = []
                diagnostic_info[body_type].append({
                    'part': part_name,
                    'confidence': confidence,
                    'weight': weight,
                    'contribution': weighted_contribution
                })

        # Normalize probabilities
        if total_weight > 0:
            for body_type in aggregated_probs:
                aggregated_probs[body_type] /= total_weight

        # Get final prediction
        if aggregated_probs:
            final_body_type = max(aggregated_probs, key=aggregated_probs.get)
            final_confidence = aggregated_probs[final_body_type]
        else:
            final_body_type = "unknown"
            final_confidence = 0.0

        # Add diagnostic reasoning
        result_dict = dict(aggregated_probs)
        result_dict['_diagnostic_info'] = diagnostic_info
        result_dict['_total_weight'] = total_weight

        return final_body_type, final_confidence, result_dict

    def classify_body_type_only(self, image: np.ndarray, bbox: Optional[List[int]] = None,
                               confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Run body type classification only using anatomical parts"""
        return self.analyze_body_type(image, bbox, confidence_threshold)
    
