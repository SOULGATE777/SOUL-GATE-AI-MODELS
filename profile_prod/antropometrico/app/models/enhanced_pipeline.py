import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import base64
import io
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ProfileAwareHeatmapModel(nn.Module):
    """Profile-aware point detection model - matches training architecture exactly"""
    def __init__(self, num_classes, image_size=224, heatmap_size=112):
        super(ProfileAwareHeatmapModel, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        
        # Backbone - ResNet50 with pretrained weights (match training)
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Freeze early layers (match training - 60 layers frozen)
        for param in list(self.backbone.parameters())[:60]:
            param.requires_grad = False
        
        backbone_features = 2048
        
        # Profile classification branch (match actual training script)
        self.profile_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),  # 2 classes: izquierdo (0), derecho (1)
            nn.Softmax(dim=1)
        )
        
        # Enhanced decoder with profile-aware processing (match training script)
        self.decoder = nn.ModuleList([
            # Stage 1: 7x7 -> 14x14
            nn.Sequential(
                nn.ConvTranspose2d(backbone_features, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            
            # Stage 2: 14x14 -> 28x28
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            
            # Stage 3: 28x28 -> 56x56
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            
            # Stage 4: 56x56 -> 112x112
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Final prediction layers with higher precision
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
        
    def forward(self, x):
        # Extract features with backbone
        features = self.backbone(x)
        
        # Profile classification (auxiliary task to help learn left/right differences)
        profile_logits = self.profile_classifier(features)
        
        # Progressive upsampling through decoder stages
        decoded_features = features
        
        # Apply decoder stages
        for decoder_stage in self.decoder:
            decoded_features = decoder_stage(decoded_features)
        
        # Final prediction with higher precision
        heatmaps = self.final_layer(decoded_features)
        
        return heatmaps, profile_logits

class FacialLandmarkGNN(nn.Module):
    """Facial Landmark GNN - matches the actual trained model"""
    def __init__(self, num_landmarks, hidden_dim=64):
        super(FacialLandmarkGNN, self).__init__()
        
        # Node encoder: (x, y, confidence) -> hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers
        self.coord_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # (dx, dy) refinement
        )
        
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def graph_conv(self, x, adj_matrix):
        """Simple graph convolution: normalize adjacency and multiply"""
        degree = adj_matrix.sum(dim=1, keepdim=True) + 1e-6
        norm_adj = adj_matrix / degree
        return torch.matmul(norm_adj, x)
    
    def forward(self, node_features, adj_matrix):
        # Encode nodes
        x = self.node_encoder(node_features)
        
        # Three graph convolution layers with residuals
        x1 = torch.relu(self.conv1(self.graph_conv(x, adj_matrix)))
        x = x + x1
        
        x2 = torch.relu(self.conv2(self.graph_conv(x, adj_matrix)))
        x = x + x2
        
        x3 = torch.relu(self.conv3(self.graph_conv(x, adj_matrix)))
        x = x + x3
        
        # Output refinements and confidence
        coord_refinement = self.coord_refiner(x)
        confidence = self.confidence_scorer(x)
        
        return coord_refinement, confidence

class EnhancedProfileAnthropometricPipeline:
    """Enhanced profile anthropometric analysis pipeline with GNN validation"""
    
    def __init__(self, point_model_path: str, gnn_model_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Enhanced pipeline using device: {self.device}")
        
        # Model components
        self.point_model = None
        self.gnn_model = None
        self.adjacency_matrix = None
        self.point_classes = []
        self.heatmap_size = 112
        
        # Load models
        self._load_models(point_model_path, gnn_model_path)
        
    def _load_models(self, point_model_path: str, gnn_model_path: Optional[str] = None):
        """Load the enhanced models with proper error handling"""
        try:
            logger.info(f"Loading enhanced point detection model from: {point_model_path}")
            
            # Load point detection model checkpoint
            checkpoint = torch.load(point_model_path, map_location=self.device, weights_only=False)
            
            # Get model configuration
            if 'all_classes' in checkpoint:
                self.point_classes = checkpoint['all_classes']
                num_classes = len(self.point_classes)
                logger.info(f"Model trained with {num_classes} classes")
            else:
                # Fallback: try to infer from model weights
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                    
                # Get number of classes from final layer
                final_layer_key = 'final_layer.6.weight'
                if final_layer_key in state_dict:
                    num_classes = state_dict[final_layer_key].shape[0]
                    # Generate numbered class names with suffixes for compatibility
                    base_numbers = list(range(1, (num_classes // 2) + 1))
                    self.point_classes = []
                    for num in base_numbers:
                        self.point_classes.extend([f"{num}_d", f"{num}_i"])
                    logger.info(f"Inferred {num_classes} classes from model weights")
                else:
                    raise ValueError("Could not determine number of classes from model")
            
            # Initialize point detection model
            self.point_model = ProfileAwareHeatmapModel(
                num_classes=num_classes,
                image_size=224,
                heatmap_size=112
            ).to(self.device)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.point_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.point_model.load_state_dict(checkpoint)
            
            self.point_model.eval()
            logger.info("✅ Enhanced point detection model loaded successfully")
            
            # Create adjacency matrix for GNN
            self.adjacency_matrix = self._create_adjacency_matrix()
            logger.info("✅ Adjacency matrix created")
            
            # Load GNN model if provided
            if gnn_model_path:
                try:
                    logger.info(f"Loading GNN model from: {gnn_model_path}")
                    
                    self.gnn_model = FacialLandmarkGNN(
                        num_landmarks=len(self.point_classes),
                        hidden_dim=64
                    ).to(self.device)
                    
                    gnn_checkpoint = torch.load(gnn_model_path, map_location=self.device, weights_only=False)
                    if 'model_state_dict' in gnn_checkpoint:
                        self.gnn_model.load_state_dict(gnn_checkpoint['model_state_dict'])
                    else:
                        self.gnn_model.load_state_dict(gnn_checkpoint)
                    
                    self.gnn_model.eval()
                    logger.info("✅ GNN model loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load GNN model ({e}), using point detection only")
                    self.gnn_model = None
            else:
                logger.info("⚠️ No GNN model path provided, using point detection only")
                self.gnn_model = None
                
        except Exception as e:
            logger.error(f"❌ Error loading enhanced models: {str(e)}")
            raise e
    
    def _create_adjacency_matrix(self):
        """Create adjacency matrix based on facial landmark relationships"""
        num_classes = len(self.point_classes)
        adjacency = torch.zeros(num_classes, num_classes, device=self.device)
        
        # Build relationships based on anatomical knowledge
        # 1. Bilateral connections (_d <-> _i pairs)
        bilateral_pairs = []
        base_classes = set()
        
        for cls in self.point_classes:
            if '_d' in cls:
                base = cls.replace('_d', '')
                partner = f"{base}_i"
                if partner in self.point_classes:
                    bilateral_pairs.append((cls, partner))
                base_classes.add(base)
        
        # Add bilateral connections
        for cls1, cls2 in bilateral_pairs:
            idx1 = self.point_classes.index(cls1)
            idx2 = self.point_classes.index(cls2)
            adjacency[idx1, idx2] = 1.0
            adjacency[idx2, idx1] = 1.0
        
        # 2. Sequential connections within regions (approximate)
        # Group points by number ranges for facial regions
        regions = {
            'face_contour': range(1, 18),
            'eyebrows': range(18, 28),
            'nose': range(28, 37),
            'eyes': range(37, 49),
            'mouth': range(49, 69)
        }
        
        for region_points in regions.values():
            for side in ['_d', '_i']:
                region_classes = [f"{i}{side}" for i in region_points if f"{i}{side}" in self.point_classes]
                # Connect sequential points in region
                for j in range(len(region_classes) - 1):
                    cls1, cls2 = region_classes[j], region_classes[j + 1]
                    if cls1 in self.point_classes and cls2 in self.point_classes:
                        idx1 = self.point_classes.index(cls1)
                        idx2 = self.point_classes.index(cls2)
                        adjacency[idx1, idx2] = 0.8
                        adjacency[idx2, idx1] = 0.8
        
        # 3. Cross-region structural connections (key facial relationships)
        structural_connections = [
            # Eye-eyebrow connections
            ('37_d', '19_d'), ('37_i', '19_i'),  # Inner eye to inner eyebrow
            ('42_d', '22_d'), ('42_i', '22_i'),  # Outer eye to outer eyebrow
            # Nose-mouth connections
            ('33_d', '51_d'), ('33_i', '51_i'),  # Nose base to mouth
            # Eye-nose connections
            ('39_d', '31_d'), ('39_i', '31_i'),  # Eye to nose side
        ]
        
        for cls1, cls2 in structural_connections:
            if cls1 in self.point_classes and cls2 in self.point_classes:
                idx1 = self.point_classes.index(cls1)
                idx2 = self.point_classes.index(cls2)
                adjacency[idx1, idx2] = 0.6
                adjacency[idx2, idx1] = 0.6
        
        # Add self-connections
        adjacency.fill_diagonal_(1.0)
        
        return adjacency
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 224) -> Tuple[np.ndarray, torch.Tensor]:
        """Preprocess image for analysis (match training preprocessing exactly)"""
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            original_image = image.copy()
        else:
            # Convert if needed
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions for scaling later
        original_height, original_width = original_image.shape[:2]
        
        # Resize to training size (224x224) - exact match to training
        resized_image = cv2.resize(original_image, (target_size, target_size))
        
        # Convert to tensor and normalize (match training preprocessing exactly)
        image_tensor = torch.from_numpy(resized_image.transpose((2, 0, 1))).float() / 255.0
        
        return original_image, image_tensor.unsqueeze(0).to(self.device)
    
    def _extract_keypoints_from_heatmaps(self, heatmaps, image_size):
        """Extract keypoint coordinates from heatmaps with sub-pixel accuracy"""
        batch_size, num_classes, heatmap_h, heatmap_w = heatmaps.shape
        
        # Find peak locations
        heatmaps_flat = heatmaps.view(batch_size, num_classes, -1)
        max_vals, max_indices = torch.max(heatmaps_flat, dim=2)
        
        # Convert flat indices to 2D coordinates
        max_y = (max_indices // heatmap_w).float()
        max_x = (max_indices % heatmap_w).float()
        
        # Sub-pixel refinement using weighted centroid
        for b in range(batch_size):
            for k in range(num_classes):
                if max_vals[b, k] > 0.1:
                    x_int, y_int = int(max_x[b, k]), int(max_y[b, k])
                    
                    # Extract 3x3 neighborhood around the maximum
                    x_start = max(0, x_int - 1)
                    x_end = min(heatmap_w, x_int + 2)
                    y_start = max(0, y_int - 1)
                    y_end = min(heatmap_h, y_int + 2)
                    
                    neighborhood = heatmaps[b, k, y_start:y_end, x_start:x_end]
                    
                    if neighborhood.numel() > 0:
                        # Calculate weighted centroid
                        y_coords, x_coords = torch.meshgrid(
                            torch.arange(y_start, y_end, dtype=torch.float32, device=heatmaps.device),
                            torch.arange(x_start, x_end, dtype=torch.float32, device=heatmaps.device),
                            indexing='ij'
                        )
                        
                        total_weight = neighborhood.sum()
                        if total_weight > 0:
                            weighted_x = (x_coords * neighborhood).sum() / total_weight
                            weighted_y = (y_coords * neighborhood).sum() / total_weight
                            
                            max_x[b, k] = weighted_x
                            max_y[b, k] = weighted_y
        
        # Stack coordinates
        coords = torch.stack([max_x, max_y], dim=2)
        
        # Scale coordinates to image size
        scale_x = image_size / heatmap_w
        scale_y = image_size / heatmap_h
        coords[:, :, 0] *= scale_x
        coords[:, :, 1] *= scale_y
        
        # Normalize confidence scores
        confidences = torch.sigmoid(max_vals)
        
        return coords, confidences
    
    def _apply_profile_filtering(self, coords, confidences, profile_type):
        """Apply profile-aware filtering (reduce wrong suffix confidence by 90%)"""
        filtered_confidences = confidences.clone()
        
        expected_suffix = '_i' if profile_type == 'left' else '_d'
        wrong_suffix = '_d' if profile_type == 'left' else '_i'
        
        removed_count = 0
        for i, class_name in enumerate(self.point_classes):
            if wrong_suffix in class_name:
                # Significantly reduce confidence for wrong suffix points (90% reduction)
                filtered_confidences[i] *= 0.1
                removed_count += 1
        
        logger.info(f"Profile filtering: {profile_type} profile → reduced confidence for {removed_count} {wrong_suffix} points")
        return filtered_confidences
    
    def detect_points(self, image_tensor: torch.Tensor) -> List[Dict]:
        """Detect anthropometric points using enhanced pipeline"""
        with torch.no_grad():
            # Step 1: Point detection and profile classification
            heatmaps, profile_logits = self.point_model(image_tensor)
            
            # Determine profile type with stronger confidence
            profile_probs = torch.softmax(profile_logits[0], dim=0)
            profile_idx = profile_probs.argmax().item()
            profile_type = 'left' if profile_idx == 0 else 'right'  # izquierdo=0, derecho=1
            profile_confidence = profile_probs.max().item()
            
            logger.info(f"Profile prediction: {profile_type} (confidence: {profile_confidence:.3f})")
            
            # Step 2: Extract coordinates from heatmaps
            coords, confidences = self._extract_keypoints_from_heatmaps(heatmaps, 224)
            coords = coords[0]  # Remove batch dimension
            confidences = confidences[0]
            
            # Step 3: Apply profile-aware filtering
            confidences = self._apply_profile_filtering(coords, confidences, profile_type)
            
            # Step 4: GNN-based confidence refinement (if available)
            if self.gnn_model is not None:
                # Normalize coordinates for GNN (0-1 range)
                norm_coords = coords / 224.0
                node_features = torch.cat([norm_coords, confidences.unsqueeze(1)], dim=1)
                
                # Get GNN confidence scores (ignore coordinate refinement as per training)
                _, gnn_confidence = self.gnn_model(node_features, self.adjacency_matrix)
                
                # Use GNN ONLY for filtering - keep original coordinates
                refined_coords = coords  # Preserve original coordinates
                final_confidences = confidences * gnn_confidence.squeeze()  # Only adjust confidence
                
                logger.info("GNN filtering applied (coordinates preserved)")
            else:
                refined_coords = coords
                final_confidences = confidences
        
        # Extract detected points with confidence threshold
        detected_points = []
        confidence_threshold = 0.15
        
        for i, class_name in enumerate(self.point_classes):
            conf = final_confidences[i].item()
            x, y = refined_coords[i].cpu().numpy()
            
            if conf > confidence_threshold:
                detected_points.append({
                    'class': class_name,
                    'coordinates': [float(x), float(y)],
                    'confidence': float(conf)
                })
        
        return detected_points
    
    def filter_spurious_predictions(self, detected_points: List[Dict]) -> Tuple[List[Dict], str]:
        """Filter spurious predictions and determine profile side"""
        # Count points by suffix
        left_count = sum(1 for p in detected_points if p['class'].endswith('_i'))
        right_count = sum(1 for p in detected_points if p['class'].endswith('_d'))
        
        logger.info(f"Point count - Left (_i): {left_count}, Right (_d): {right_count}")
        
        # Determine the dominant side
        if left_count > right_count:
            dominant_suffix = '_i'
            actual_profile = 'left'
        elif right_count > left_count:
            dominant_suffix = '_d'
            actual_profile = 'right'
        else:
            # Equal count - use profile classification from CNN
            dominant_suffix = None
            actual_profile = 'unknown'
        
        logger.info(f"Dominant side: {actual_profile}")
        
        # Filter points to keep only the dominant side and remove suffixes for compatibility
        filtered_points = []
        removed_points = []
        
        for point in detected_points:
            class_name = point['class']
            
            if dominant_suffix is None:
                # If we can't determine dominant side, keep all points
                clean_class = class_name.replace('_i', '').replace('_d', '')
                point_copy = point.copy()
                point_copy['class'] = clean_class
                filtered_points.append(point_copy)
            else:
                # Keep points that have the dominant suffix or no suffix
                if class_name.endswith(dominant_suffix) or not (class_name.endswith('_i') or class_name.endswith('_d')):
                    # Remove suffix for compatibility with existing measurement system
                    clean_class = class_name.replace('_i', '').replace('_d', '')
                    point_copy = point.copy()
                    point_copy['class'] = clean_class
                    filtered_points.append(point_copy)
                else:
                    removed_points.append(point)
        
        if removed_points:
            logger.info(f"Removed {len(removed_points)} spurious predictions from minority side")
            for p in removed_points[:3]:  # Log first 3 for debugging
                logger.info(f"  - {p['class']} (conf: {p['confidence']:.3f})")
        
        return filtered_points, actual_profile
    
    def create_points_dict(self, filtered_points: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """Create a dictionary mapping point names to coordinates"""
        points = {}
        for point in filtered_points:
            points[point['class']] = tuple(point['coordinates'])
        return points
    
    def dist(self, p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]) -> Optional[float]:
        """Calculate Euclidean distance between two points"""
        if p1 and p2:
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return None

# Import existing measurement functions from the original pipeline
# These will be used directly without modification for compatibility
try:
    from .profile_anthropometric_pipeline import ProfileAnthropometricPipeline
    ORIGINAL_PIPELINE_AVAILABLE = True
except ImportError:
    try:
        from app.models.profile_anthropometric_pipeline import ProfileAnthropometricPipeline
        ORIGINAL_PIPELINE_AVAILABLE = True
    except ImportError:
        ORIGINAL_PIPELINE_AVAILABLE = False
        logger.warning("Original pipeline not available, using basic compatibility mode")

class EnhancedCompatibilityPipeline(EnhancedProfileAnthropometricPipeline):
    """Enhanced pipeline with full compatibility layer for existing API"""
    
    def __init__(self, point_model_path: str, gnn_model_path: Optional[str] = None, device: str = 'cuda'):
        super().__init__(point_model_path, gnn_model_path, device)
        
        # Create a reference to original pipeline for measurement functions
        # We'll use this for all measurement calculations to ensure compatibility
        self._measurement_pipeline = None
        
    def _get_measurement_pipeline(self):
        """Create a measurement pipeline instance for compatibility"""
        if self._measurement_pipeline is None and ORIGINAL_PIPELINE_AVAILABLE:
            # Create a dummy pipeline just for accessing measurement functions
            # We'll override the key methods but use the measurement logic
            class MeasurementPipeline(ProfileAnthropometricPipeline):
                def __init__(self):
                    # Skip the normal initialization
                    self.device = torch.device('cpu')  # Dummy device
                    pass
                    
                def _load_model(self, model_path):
                    # Skip model loading
                    pass
            
            self._measurement_pipeline = MeasurementPipeline()
            
            # Set required attributes for measurements
            self._measurement_pipeline.point_classes = [str(i) for i in range(1, 101)]  # Dummy classes
            self._measurement_pipeline.heatmap_size = 112
            
        return self._measurement_pipeline
    
    def perform_anthropometric_analysis(self, points: Dict[str, Tuple[float, float]]) -> Dict:
        """Perform anthropometric measurements using original system"""
        if ORIGINAL_PIPELINE_AVAILABLE:
            measurement_pipeline = self._get_measurement_pipeline()
            return measurement_pipeline.perform_anthropometric_analysis(points)
        else:
            # Basic fallback measurements
            return self._basic_measurements_fallback(points)
    
    def calculate_angular_measurements(self, points: Dict[str, Tuple[float, float]], measurements: Dict):
        """Calculate angular measurements using original system"""
        if ORIGINAL_PIPELINE_AVAILABLE:
            measurement_pipeline = self._get_measurement_pipeline()
            return measurement_pipeline.calculate_angular_measurements(points, measurements)
    
    def calculate_implantation_angles(self, point_24, point_18, point_4, point_5, point_1, point_3, head_direction=None):
        """Calculate implantation angles using original system"""
        if ORIGINAL_PIPELINE_AVAILABLE:
            measurement_pipeline = self._get_measurement_pipeline()
            return measurement_pipeline.calculate_implantation_angles(
                point_24, point_18, point_4, point_5, point_1, point_3, head_direction
            )
        return None
    
    def create_visualization(self, original_image: np.ndarray, detected_points: List[Dict], 
                           actual_profile: str, measurements: Dict) -> str:
        """Create visualization using original system"""
        if ORIGINAL_PIPELINE_AVAILABLE:
            measurement_pipeline = self._get_measurement_pipeline()
            return measurement_pipeline.create_visualization(
                original_image, detected_points, actual_profile, measurements
            )
        else:
            return self._basic_visualization_fallback(original_image, detected_points)
    
    def _create_comprehensive_measurement_summary(self, measurements: Dict) -> str:
        """Create measurement summary using original system"""
        if ORIGINAL_PIPELINE_AVAILABLE:
            measurement_pipeline = self._get_measurement_pipeline()
            return measurement_pipeline._create_comprehensive_measurement_summary(measurements)
        else:
            return self._basic_summary_fallback(measurements)
    
    def print_final_summary(self, measurements: Dict):
        """Print final summary using original system"""
        if ORIGINAL_PIPELINE_AVAILABLE:
            measurement_pipeline = self._get_measurement_pipeline()
            return measurement_pipeline.print_final_summary(measurements)
        else:
            logger.info(f"Basic measurements: {len(measurements)} calculated")
    
    def _basic_measurements_fallback(self, points: Dict[str, Tuple[float, float]]) -> Dict:
        """Basic measurements fallback when original pipeline not available"""
        measurements = {}
        
        # Calculate basic reference distance if points available
        if '24' in points and '10' in points:
            ref_distance = self.dist(points['24'], points['10'])
            measurements['reference_distance'] = ref_distance
        
        logger.info("Using basic measurements fallback")
        return measurements
    
    def _basic_visualization_fallback(self, original_image: np.ndarray, detected_points: List[Dict]) -> str:
        """Basic visualization fallback"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.imshow(original_image)
        ax.set_title("Enhanced Profile Point Detection", fontsize=14, fontweight='bold')
        
        # Plot detected points
        for i, point in enumerate(detected_points):
            x, y = point['coordinates']
            color = plt.cm.rainbow(i / max(1, len(detected_points)))
            ax.plot(x, y, 'o', color=color, markersize=8)
            ax.text(x+3, y-3, f"{point['class']}", color=color, fontsize=8)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _basic_summary_fallback(self, measurements: Dict) -> str:
        """Basic summary fallback"""
        summary_lines = ["=== ENHANCED PIPELINE BASIC SUMMARY ==="]
        for key, value in measurements.items():
            summary_lines.append(f"{key}: {value}")
        return '\n'.join(summary_lines)
    
    def analyze_image(self, image: np.ndarray, include_visualization: bool = True) -> Dict:
        """Complete analysis pipeline with enhanced detection and original measurements"""
        logger.info("Analyzing profile image with enhanced pipeline...")
        
        # Preprocess image
        original_image, image_tensor = self.preprocess_image(image)
        original_height, original_width = original_image.shape[:2]
        
        # Detect anthropometric points using enhanced pipeline (returns coords in 224x224 space)
        logger.info("Detecting anthropometric points with enhanced CNN + GNN...")
        detected_points = self.detect_points(image_tensor)
        
        # Filter spurious predictions (working in 224x224 space, same as original)
        logger.info("Filtering spurious predictions...")
        filtered_points, actual_profile = self.filter_spurious_predictions(detected_points)
        
        # Scale coordinates to original image dimensions for measurements
        scale_x = original_width / 224.0
        scale_y = original_height / 224.0
        
        # Create points dictionary for measurements (scale to original image size)
        scaled_points = {}
        for point in filtered_points:
            x, y = point['coordinates']  # These are in 224x224 space
            scaled_x = x * scale_x
            scaled_y = y * scale_y
            scaled_points[point['class']] = (scaled_x, scaled_y)
        
        # Perform comprehensive anthropometric analysis using original system
        measurements = self.perform_anthropometric_analysis(scaled_points)
        
        # Print final summary
        self.print_final_summary(measurements)
        
        # Create visualization if requested
        # Note: The original visualization system expects points in 224x224 space and scales them internally
        visualization_base64 = None
        if include_visualization and filtered_points:
            visualization_base64 = self.create_visualization(
                original_image, filtered_points, actual_profile, measurements
            )
        
        # Compile results in the expected format
        # Note: Keep anthropometric_points in 224x224 space for API compatibility
        results = {
            'profile_side': actual_profile,
            'total_detected_points': len(detected_points),
            'filtered_points': len(filtered_points),
            'anthropometric_points': filtered_points,  # Keep in 224x224 space for consistency
            'measurements': measurements,
            'visualization': visualization_base64 if include_visualization else None
        }
        
        return results