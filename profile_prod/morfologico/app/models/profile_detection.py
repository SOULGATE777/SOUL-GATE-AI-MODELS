import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ProfileDetectionModel(nn.Module):
    """
    Specialized model for profile detection and classification
    Based on the MinimalModel architecture from the pipeline
    """
    
    def __init__(self, num_keypoints: int):
        super().__init__()
        
        # ResNet50 backbone
        backbone = torchvision.models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Profile classifier (left vs right)
        self.profile_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(2048, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(512, 2), 
            nn.Softmax(dim=1)
        )
        
        # Decoder for heatmap generation
        self.decoder = nn.ModuleList([
            # Stage 1 with attention mechanism
            nn.Sequential(
                nn.ConvTranspose2d(2048, 512, 4, 2, 1), 
                nn.BatchNorm2d(512), 
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), 
                nn.BatchNorm2d(512), 
                nn.ReLU(),
                # Attention layers
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(512, 32, 1), 
                nn.ReLU(),
                nn.Conv2d(32, 512, 1), 
                nn.Sigmoid()
            ),
            # Stage 2
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1), 
                nn.BatchNorm2d(256), 
                nn.ReLU(), 
                nn.Conv2d(256, 256, 3, 1, 1), 
                nn.BatchNorm2d(256), 
                nn.ReLU()
            ),
            # Stage 3
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1), 
                nn.BatchNorm2d(128), 
                nn.ReLU(), 
                nn.Conv2d(128, 128, 3, 1, 1), 
                nn.BatchNorm2d(128), 
                nn.ReLU()
            ),
            # Stage 4
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1), 
                nn.BatchNorm2d(64), 
                nn.ReLU(), 
                nn.Conv2d(64, 64, 3, 1, 1), 
                nn.BatchNorm2d(64), 
                nn.ReLU()
            )
        ])
        
        # Final layer for keypoint heatmaps
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.Conv2d(16, num_keypoints, 1), 
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of (heatmaps, profile_logits)
        """
        # Extract features
        features = self.backbone(x)
        
        # Profile classification
        profile_logits = self.profile_classifier(features)
        
        # Decode features
        decoded = features
        
        # Stage 1 with attention
        stage1 = self.decoder[0]
        main_layers = stage1[:4]  # First 4 layers (conv, bn, relu, conv, bn, relu)
        attention_layers = stage1[4:]  # Attention layers
        
        decoded = main_layers(decoded)
        attention = attention_layers(decoded)
        decoded = decoded * attention
        
        # Remaining stages
        for i in range(1, len(self.decoder)):
            decoded = self.decoder[i](decoded)
        
        # Generate final heatmaps
        heatmaps = self.final_layer(decoded)
        
        return heatmaps, profile_logits


class ProfileDetector:
    """
    Profile detection and analysis utility class
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"ProfileDetector initialized on device: {self.device}")
    
    def load_model(self, model_path: str) -> ProfileDetectionModel:
        """
        Load a trained profile detection model
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Loaded ProfileDetectionModel
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model parameters
            num_keypoints = checkpoint['num_keypoints']
            all_classes = checkpoint['all_classes']
            heatmap_size = checkpoint.get('heatmap_size', 112)
            
            # Create and load model
            model = ProfileDetectionModel(num_keypoints)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info(f"Profile model loaded: {len(all_classes)} classes, {num_keypoints} keypoints")
            
            return model, all_classes, heatmap_size
            
        except Exception as e:
            logger.error(f"Failed to load profile model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 224) -> torch.Tensor:
        """
        Preprocess image for profile detection
        
        Args:
            image: Input image as numpy array (RGB)
            target_size: Target size for resizing
            
        Returns:
            Preprocessed tensor
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            resized = cv2.resize(image, (target_size, target_size))
        else:
            raise ValueError(f"Expected RGB image, got shape {image.shape}")
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized.transpose((2, 0, 1))).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)
    
    def detect_profile_side(self, model: ProfileDetectionModel, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Detect profile side (left/right)
        
        Args:
            model: Loaded profile detection model
            image_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (profile_side, confidence)
        """
        with torch.no_grad():
            _, profile_logits = model(image_tensor)
        
        profile_prob = profile_logits[0].cpu().numpy()
        profile_side = "left" if profile_prob[0] > profile_prob[1] else "right"
        confidence = max(profile_prob)
        
        return profile_side, confidence
    
    def extract_keypoints_from_heatmaps(
        self, 
        heatmaps: torch.Tensor, 
        confidence_threshold: float = 0.15,
        target_size: int = 224
    ) -> List[Dict[str, Any]]:
        """
        Extract keypoints from heatmaps
        
        Args:
            heatmaps: Model output heatmaps [B, K, H, W]
            confidence_threshold: Minimum confidence threshold
            target_size: Target image size for coordinate scaling
            
        Returns:
            List of detected keypoints with coordinates and confidences
        """
        batch_size, num_keypoints, hm_height, hm_width = heatmaps.shape
        
        # Apply smoothing
        kernel = torch.ones(1, 1, 3, 3, device=heatmaps.device) / 9
        smoothed_heatmaps = torch.zeros_like(heatmaps)
        
        for i in range(num_keypoints):
            channel = heatmaps[:, i:i+1, :, :]
            smoothed_channel = torch.nn.functional.conv2d(channel, kernel, padding=1)
            smoothed_heatmaps[:, i:i+1, :, :] = smoothed_channel
        
        # Find maximum points
        heatmaps_flat = smoothed_heatmaps.view(batch_size, num_keypoints, -1)
        max_vals, max_indices = torch.max(heatmaps_flat, dim=2)
        
        # Convert to coordinates
        max_x = (max_indices % hm_width).float()
        max_y = (max_indices // hm_width).float()
        
        # Scale to target size
        scale_x = target_size / hm_width
        scale_y = target_size / hm_height
        
        keypoints = torch.stack([max_x * scale_x, max_y * scale_y], dim=2)
        confidences = max_vals
        
        # Extract valid keypoints
        detected_keypoints = []
        keypoints_np = keypoints[0].cpu().numpy()
        confidences_np = confidences[0].cpu().numpy()
        
        for i, (point, conf) in enumerate(zip(keypoints_np, confidences_np)):
            if conf > confidence_threshold:
                detected_keypoints.append({
                    'keypoint_id': i,
                    'coordinates': point.tolist(),
                    'confidence': float(conf)
                })
        
        return detected_keypoints
    
    def analyze_profile_features(
        self, 
        keypoints: List[Dict[str, Any]], 
        point_classes: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze profile features based on detected keypoints
        
        Args:
            keypoints: List of detected keypoints
            point_classes: List of point class names
            
        Returns:
            Analysis results with feature classifications
        """
        analysis = {
            'total_keypoints': len(keypoints),
            'feature_analysis': {},
            'quality_assessment': {}
        }
        
        if not keypoints:
            analysis['quality_assessment']['sufficient_points'] = False
            return analysis
        
        # Map keypoints to anatomical features
        feature_groups = {
            'nasal': [],
            'frontal': [],
            'mandibular': [],
            'auricular': []
        }
        
        for kp in keypoints:
            kp_id = kp['keypoint_id']
            if kp_id < len(point_classes):
                class_name = point_classes[kp_id].lower()
                
                # Categorize by anatomical region
                if any(nasal_term in class_name for nasal_term in ['nariz', 'nose', 'nasal']):
                    feature_groups['nasal'].append(kp)
                elif any(frontal_term in class_name for frontal_term in ['frente', 'front', 'forehead']):
                    feature_groups['frontal'].append(kp)
                elif any(mandib_term in class_name for mandib_term in ['menton', 'chin', 'mandib']):
                    feature_groups['mandibular'].append(kp)
                elif any(ear_term in class_name for ear_term in ['oreja', 'ear', 'auricular']):
                    feature_groups['auricular'].append(kp)
        
        # Analyze each feature group
        for feature_name, feature_points in feature_groups.items():
            if feature_points:
                avg_confidence = np.mean([p['confidence'] for p in feature_points])
                analysis['feature_analysis'][feature_name] = {
                    'detected_points': len(feature_points),
                    'average_confidence': float(avg_confidence),
                    'points': feature_points
                }
        
        # Quality assessment
        total_detected = len(keypoints)
        avg_confidence = np.mean([kp['confidence'] for kp in keypoints])
        
        analysis['quality_assessment'] = {
            'sufficient_points': total_detected >= 5,
            'high_confidence': avg_confidence > 0.3,
            'average_confidence': float(avg_confidence),
            'feature_coverage': len([fg for fg in feature_groups.values() if fg]) / len(feature_groups)
        }
        
        return analysis
    
    def classify_profile_type(self, keypoints: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> str:
        """
        Classify profile type based on keypoint distribution
        
        Args:
            keypoints: Detected keypoints
            image_shape: Original image shape (height, width)
            
        Returns:
            Profile type classification
        """
        if not keypoints:
            return "insufficient_data"
        
        # Analyze horizontal distribution of points
        x_coords = [kp['coordinates'][0] for kp in keypoints]
        image_width = image_shape[1] if len(image_shape) > 1 else 224
        
        # Calculate center bias
        center_x = image_width / 2
        left_points = sum(1 for x in x_coords if x < center_x)
        right_points = sum(1 for x in x_coords if x >= center_x)
        
        # Determine profile orientation
        if left_points > right_points * 1.5:
            return "left_profile"
        elif right_points > left_points * 1.5:
            return "right_profile"
        else:
            return "frontal_or_mixed"
    
    def get_profile_quality_score(self, analysis_results: Dict[str, Any]) -> float:
        """
        Calculate overall profile quality score
        
        Args:
            analysis_results: Results from analyze_profile_features
            
        Returns:
            Quality score between 0 and 1
        """
        quality = analysis_results.get('quality_assessment', {})
        
        # Base score from point detection
        sufficient_points = quality.get('sufficient_points', False)
        avg_confidence = quality.get('average_confidence', 0.0)
        feature_coverage = quality.get('feature_coverage', 0.0)
        
        # Calculate weighted score
        score = 0.0
        
        if sufficient_points:
            score += 0.3
        
        score += avg_confidence * 0.4  # Weight confidence heavily
        score += feature_coverage * 0.3  # Weight feature coverage
        
        return min(1.0, score)  # Cap at 1.0
