import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import logging
from collections import defaultdict
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import centralized threshold validation from common module
# This ensures consistent threshold application per Umbrales_Rasgos.txt
try:
    from common import ThresholdValidator, ModuleType
    from common.threshold_config import PROFILE_MORFOLOGICO_RULES
    THRESHOLD_VALIDATOR_AVAILABLE = True
except ImportError:
    # Fallback if common module not available (development mode)
    THRESHOLD_VALIDATOR_AVAILABLE = False
    PROFILE_MORFOLOGICO_RULES = {}
    logger.warning("common.ThresholdValidator not available, using fallback thresholds")

# Excluded classes (same across all models)
EXCLUDED_CLASSES = ['cabello_tapando_frente', 'cabello_tapando_oreja', 'objeto']

class ProfileLandmarkClassifier(nn.Module):
    """Landmark classification model from model 2"""
    def __init__(self, num_classes):
        super(ProfileLandmarkClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

class ProfileAnalysisPipeline:
    """Complete pipeline combining all three models with improved filtering"""
    
    def __init__(self, bbox_model_path, classifier_model_path, point_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.bbox_model = self._load_bbox_model(bbox_model_path)
        self.classifier_model = self._load_classifier_model(classifier_model_path)
        self.point_model = self._load_point_model(point_model_path)
        logger.info("All models loaded successfully!")
        
        # Initialize centralized threshold validator
        # Uses thresholds from common/threshold_config.py (Single Source of Truth)
        if THRESHOLD_VALIDATOR_AVAILABLE:
            self.threshold_validator = ThresholdValidator()
            logger.info("✓ ThresholdValidator initialized from common module")
        else:
            self.threshold_validator = None
            logger.warning("⚠ ThresholdValidator not available, thresholds not applied")
    
    def _load_bbox_model(self, model_path):
        """Load the bounding box detection model (model 1)"""
        logger.info("Loading bbox detection model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        all_classes = checkpoint['all_classes']
        num_classes = len(all_classes) + 1  # +1 for background
        
        # Create model
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.bbox_classes = all_classes
        logger.info(f"Bbox model loaded with {len(all_classes)} classes")
        return model
    
    def _load_classifier_model(self, model_path):
        """Load the landmark classifier model (model 2)"""
        logger.info("Loading landmark classifier model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        tags = checkpoint['tags']
        num_classes = checkpoint['num_classes']
        
        # Create model
        model = ProfileLandmarkClassifier(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.classifier_tags = tags
        logger.info(f"Classifier model loaded with {len(tags)} tags")
        return model
    
    def _load_point_model(self, model_path):
        """Load the point detection model (model 3)"""
        logger.info("Loading point detection model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        all_classes = checkpoint['all_classes']
        num_keypoints = checkpoint['num_keypoints']
        heatmap_size = checkpoint.get('heatmap_size', 112)
        
        class MinimalModel(nn.Module):
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
                    nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU()),
                    nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()),
                    nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
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
        
        # Create model
        model = MinimalModel(num_keypoints)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.point_classes = all_classes
        self.heatmap_size = heatmap_size
        logger.info(f"Point model loaded with {len(all_classes)} point classes")
        return model
    
    def preprocess_image(self, image_path, target_size=224):
        """Load and preprocess image for analysis"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(str(image_path))
        else:
            image = image_path
            
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # Resize for models
        resized_image = cv2.resize(image, (target_size, target_size))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(resized_image.transpose((2, 0, 1))).float() / 255.0
        
        return original_image, image_tensor.unsqueeze(0).to(self.device)
    
    def _remove_duplicate_bbox_classes(self, detected_objects):
        """Remove duplicate bbox classes, keeping only the one with highest confidence"""
        if not detected_objects:
            return detected_objects
        
        # Group by class name
        class_groups = defaultdict(list)
        for obj in detected_objects:
            class_groups[obj['class']].append(obj)
        
        # Keep only highest confidence for each class
        filtered_objects = []
        for class_name, objects in class_groups.items():
            # Sort by confidence and take the highest
            best_object = max(objects, key=lambda x: x['confidence'])
            filtered_objects.append(best_object)
        
        logger.info(f"Removed {len(detected_objects) - len(filtered_objects)} duplicate bbox classes")
        return filtered_objects
    
    def _filter_spurious_points_by_suffix(self, detected_points):
        """Filter spurious points based on suffix majority (_i vs _d)"""
        if not detected_points:
            return detected_points
        
        # Count suffixes
        left_count = 0  # _i suffix
        right_count = 0  # _d suffix
        no_suffix_count = 0
        
        for point in detected_points:
            class_name = point['class']
            if class_name.endswith('_i'):
                left_count += 1
            elif class_name.endswith('_d'):
                right_count += 1
            else:
                no_suffix_count += 1
        
        # Determine majority side
        if left_count == 0 and right_count == 0:
            # No sided points, return all
            return detected_points
        
        majority_side = '_i' if left_count >= right_count else '_d'
        minority_side = '_d' if majority_side == '_i' else '_i'
        
        # Filter out minority side points
        filtered_points = []
        removed_count = 0
        
        for point in detected_points:
            class_name = point['class']
            
            # Keep points that don't have suffixes
            if not (class_name.endswith('_i') or class_name.endswith('_d')):
                filtered_points.append(point)
            # Keep points from majority side
            elif class_name.endswith(majority_side):
                filtered_points.append(point)
            # Remove points from minority side
            else:
                removed_count += 1
                logger.debug(f"Removed spurious point: {class_name} (minority side)")
        
        if removed_count > 0:
            logger.info(f"Filtered {removed_count} spurious points. Majority side: {majority_side}")
        
        return filtered_points
    
    def detect_bboxes(self, image_tensor, confidence_threshold=0.5):
        """Detect bounding boxes using model 1 with duplicate removal"""
        with torch.no_grad():
            predictions = self.bbox_model(image_tensor)
        
        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter by confidence
        valid_detections = scores >= confidence_threshold
        boxes = boxes[valid_detections]
        labels = labels[valid_detections]
        scores = scores[valid_detections]
        
        # Convert labels to class names and filter excluded classes
        detected_objects = []
        for box, label, score in zip(boxes, labels, scores):
            if label > 0 and (label - 1) < len(self.bbox_classes):
                class_name = self.bbox_classes[label - 1]
                # Skip excluded classes
                if class_name not in EXCLUDED_CLASSES:
                    detected_objects.append({
                        'bbox': box,
                        'class': class_name,
                        'confidence': score
                    })
        
        # Remove duplicate classes (keep highest confidence)
        detected_objects = self._remove_duplicate_bbox_classes(detected_objects)
        
        return detected_objects
    
    def _validate_diagnosis_threshold(self, landmark_class: str, tag_name: str, confidence: float) -> Dict[str, Any]:
        """
        Validate diagnosis against thresholds from common/threshold_config.py
        
        Thresholds are per Umbrales_Rasgos.txt document (lines 293-373).
        The diagnosis key is constructed as: landmark_class + '_' + tag_name
        Example: 'lobulo_izquierdo' + 'pegado' = 'lobulo_izquierdo_pegado' (threshold 60%)
        
        Args:
            landmark_class: The detected landmark (e.g., 'lobulo_izquierdo', 'submenton', 'frente')
            tag_name: The classification tag (e.g., 'pegado', 'visible', 'f_rd_incl')
            confidence: The classification confidence (0.0 - 1.0)
            
        Returns:
            Dict with 'passes', 'threshold', and 'rule' keys
        """
        # Construct the full diagnosis key
        diagnosis_key = f"{landmark_class}_{tag_name}"
        
        # Use centralized ThresholdValidator if available
        if self.threshold_validator is not None and PROFILE_MORFOLOGICO_RULES:
            try:
                if diagnosis_key in PROFILE_MORFOLOGICO_RULES:
                    rule = PROFILE_MORFOLOGICO_RULES[diagnosis_key]
                    passes = confidence >= rule.threshold
                    return {
                        'passes': passes,
                        'threshold': rule.threshold,
                        'rule': f"{'Accepted' if passes else 'Rejected'} {diagnosis_key} ({confidence:.1%} {'≥' if passes else '<'} {rule.threshold:.1%})"
                    }
                else:
                    # No specific threshold for this diagnosis
                    return {
                        'passes': True,  # No threshold = passes by default
                        'threshold': None,
                        'rule': f"No threshold defined for {diagnosis_key}, accepted by default"
                    }
                    
            except Exception as e:
                logger.error(f"ThresholdValidator error for {diagnosis_key}: {e}")
        
        # Fallback: No threshold validation available
        return {
            'passes': True,
            'threshold': None,
            'rule': f"Threshold validation not available for {diagnosis_key}"
        }
    
    def classify_landmarks(self, original_image, detected_objects, image_size=224):
        """Classify landmark tags using model 2 - returns top 2 predictions"""
        classifications = []
        h_orig, w_orig = original_image.shape[:2]
        scale_x = w_orig / image_size
        scale_y = h_orig / image_size
        
        for obj in detected_objects:
            bbox = obj['bbox']
            
            # Scale bbox back to original image size
            x1, y1, x2, y2 = bbox
            x1_orig = max(0, min(int(x1 * scale_x), w_orig))
            y1_orig = max(0, min(int(y1 * scale_y), h_orig))
            x2_orig = max(0, min(int(x2 * scale_x), w_orig))
            y2_orig = max(0, min(int(y2 * scale_y), h_orig))
            
            # Extract and preprocess crop
            if x2_orig > x1_orig and y2_orig > y1_orig:
                crop = original_image[y1_orig:y2_orig, x1_orig:x2_orig]
                crop_resized = cv2.resize(crop, (64, 64))
                
                # Convert to tensor and normalize
                crop_tensor = torch.from_numpy(crop_resized.transpose((2, 0, 1))).float() / 255.0
                
                # Apply ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                crop_tensor = (crop_tensor - mean) / std
                crop_tensor = crop_tensor.unsqueeze(0).to(self.device)
                
                # Classify
                with torch.no_grad():
                    outputs = self.classifier_model(crop_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Get top 2 predictions
                    top2_values, top2_indices = torch.topk(probabilities[0], k=2)
                    
                    # Prepare top 2 tags
                    top_tags = []
                    for i in range(min(2, len(top2_indices))):
                        class_idx = top2_indices[i].item()
                        confidence = top2_values[i].item()
                        
                        if class_idx < len(self.classifier_tags):
                            tag = self.classifier_tags[class_idx]
                            top_tags.append({
                                'tag': tag,
                                'confidence': confidence,
                                'rank': i + 1  # 1 for first, 2 for second
                            })
                    
                    if top_tags:  # Only add if we have valid tags
                        # Apply threshold validation for top tag
                        threshold_result = self._validate_diagnosis_threshold(
                            obj['class'], top_tags[0]['tag'], top_tags[0]['confidence']
                        )
                        
                        classifications.append({
                            'bbox': bbox,
                            'original_class': obj['class'],
                            'top_tags': top_tags,  # Changed from single 'tag' to 'top_tags' list
                            'bbox_confidence': obj['confidence'],
                            'passes_threshold': threshold_result['passes'],
                            'threshold_applied': threshold_result['threshold'],
                            'threshold_rule': threshold_result['rule']
                        })
        
        return classifications
    
    def detect_points(self, image_tensor):
        """Detect anthropometric points with spurious point filtering"""
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
        
        scale_x = 224 / hm_width
        scale_y = 224 / hm_height
        
        keypoints = torch.stack([max_x * scale_x, max_y * scale_y], dim=2)
        confidences = max_vals
        
        # Extract points
        detected_points = []
        keypoints_np = keypoints[0].cpu().numpy()
        confidences_np = confidences[0].cpu().numpy()
        
        # Use adaptive threshold
        threshold = 0.15 if any(conf > 0.15 for conf in confidences_np) else 0.05
        
        for i, (point, conf) in enumerate(zip(keypoints_np, confidences_np)):
            if conf > threshold and i < len(self.point_classes):
                detected_points.append({
                    'class': self.point_classes[i],
                    'coordinates': point,
                    'confidence': conf
                })
        
        # Filter spurious points by suffix
        detected_points = self._filter_spurious_points_by_suffix(detected_points)
        
        return detected_points
    
    def analyze_image(self, image_path, bbox_threshold=0.5, save_result=True):
        """Complete analysis pipeline with improved filtering - NO PROFILE PREDICTION"""
        logger.info(f"Analyzing image: {image_path}")
        
        # Preprocess image
        original_image, image_tensor = self.preprocess_image(image_path)
        
        # Step 1: Detect bounding boxes (with duplicate removal)
        logger.info("Detecting bounding boxes...")
        detected_objects = self.detect_bboxes(image_tensor, bbox_threshold)
        
        # Step 2: Classify landmarks
        logger.info("Classifying landmarks...")
        classifications = self.classify_landmarks(original_image, detected_objects)
        
        # Step 3: Detect anthropometric points (with spurious point filtering)
        logger.info("Detecting anthropometric points...")
        detected_points = self.detect_points(image_tensor)
        
        # Step 4: Infer profile side from point suffixes
        profile_side = ""
        if detected_points:
            left_count = sum(1 for p in detected_points if p["class"].endswith("_i"))
            right_count = sum(1 for p in detected_points if p["class"].endswith("_d"))
            if left_count > right_count:
                profile_side = "left"
            elif right_count > left_count:
                profile_side = "right"

        # Compile results (WITHOUT profile type and confidence)
        results = {
            'image_path': str(image_path),
            'profile_side': profile_side,
            'detected_objects': detected_objects,
            'landmark_classifications': classifications,
            'anthropometric_points': detected_points
        }
        
        # Visualize results
        if save_result:
            self.visualize_results(original_image, results, image_path)
        
        return results
    
    def visualize_results(self, original_image, results, image_path):
        """Visualize all analysis results with top 2 tags"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Bounding boxes + classifications
            axes[1].imshow(original_image)
            axes[1].set_title(f"Detected Objects + Tags ({len(results['detected_objects'])})")
            
            image_h, image_w = original_image.shape[:2]
            scale_x = image_w / 224
            scale_y = image_h / 224
            
            # Draw bounding boxes with classifications
            for i, obj in enumerate(results['detected_objects']):
                bbox = obj['bbox']
                x1, y1, x2, y2 = bbox
                x1_orig = x1 * scale_x
                y1_orig = y1 * scale_y
                x2_orig = x2 * scale_x
                y2_orig = y2 * scale_y
                
                width = x2_orig - x1_orig
                height = y2_orig - y1_orig
                
                # Find corresponding classification with top 2 tags
                tag_info = ""
                for cls in results['landmark_classifications']:
                    if np.array_equal(cls['bbox'], bbox):
                        if 'top_tags' in cls and cls['top_tags']:
                            # Format top 2 tags
                            tag_strings = []
                            for tag_data in cls['top_tags']:
                                tag_strings.append(f"{tag_data['tag']} ({tag_data['confidence']:.2f})")
                            tag_info = f" -> {' | '.join(tag_strings)}"
                        break
                
                color = plt.cm.tab10(i % 10)
                rect = plt.Rectangle((x1_orig, y1_orig), width, height,
                                   fill=False, edgecolor=color, linewidth=2)
                axes[1].add_patch(rect)
                axes[1].text(x1_orig, y1_orig-5, f"{obj['class']}: {obj['confidence']:.2f}{tag_info}",
                            color=color, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
            axes[1].axis('off')
            
            # Anthropometric points
            axes[2].imshow(original_image)
            axes[2].set_title(f"Anthropometric Points ({len(results['anthropometric_points'])})")
            
            for i, point in enumerate(results['anthropometric_points']):
                x, y = point['coordinates']
                x_orig = x * scale_x
                y_orig = y * scale_y
                
                color = plt.cm.rainbow(i / max(1, len(results['anthropometric_points'])))
                axes[2].plot(x_orig, y_orig, 'o', color=color, markersize=6)
                axes[2].text(x_orig+3, y_orig-3, f"{point['class']}", 
                            color=color, fontsize=7, bbox=dict(facecolor='white', alpha=0.7))
            
            if len(results['anthropometric_points']) == 0:
                axes[2].text(image_w//2, image_h//2, "No points detected\nabove threshold", 
                            ha='center', va='center', fontsize=12, 
                            bbox=dict(facecolor='yellow', alpha=0.8))
            
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            if isinstance(image_path, str):
                output_path = Path(image_path).stem + "_analysis_result.png"
            else:
                output_path = "analysis_result.png"
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to prevent memory leaks
            logger.info(f"Visualization saved as: {output_path}")
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            # Don't raise exception for visualization errors
