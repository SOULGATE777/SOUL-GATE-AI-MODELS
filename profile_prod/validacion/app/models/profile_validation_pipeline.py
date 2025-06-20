import torch
import cv2
import numpy as np
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileValidationPipeline:
    """
    Profile validation pipeline for detecting occlusions and quality issues in profile images.
    Based on trained Faster R-CNN model for profile-specific validation.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the profile validation pipeline
        
        Args:
            model_path: Path to the trained model file (.pth)
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            logger.info(f"Successfully loaded model checkpoint from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
        
        # Extract model information
        self.all_classes = checkpoint['all_classes']
        self.included_classes = checkpoint.get('included_classes', 
                                              ['objeto', 'cabello_tapando_oreja', 'cabello_tapando_frente'])
        self.num_classes = checkpoint['num_classes']
        
        logger.info(f"Loaded model with {self.num_classes} classes: {self.all_classes}")
        
        # Initialize model architecture
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Set detection thresholds for each class
        self.class_thresholds = {
            'objeto': 0.6,
            'cabello_tapando_oreja': 0.5,
            'cabello_tapando_frente': 0.5,
            'default': 0.5
        }
        
        # Colors for visualization
        self.class_colors = {
            'objeto': 'red',
            'cabello_tapando_oreja': 'yellow', 
            'cabello_tapando_frente': 'orange'
        }
        
        # Profile quality assessment criteria
        self.quality_criteria = {
            'min_image_size': (224, 224),
            'max_blur_threshold': 100,
            'min_brightness': 50,
            'max_brightness': 200,
            'contrast_threshold': 20
        }
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (preprocessed tensor, resized image for visualization)
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to model input size (224x224)
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
        
        return image_tensor, image_resized
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1, box2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        # Calculate intersection area
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _apply_nms_per_class(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """
        Apply Non-Maximum Suppression per class to remove overlapping detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for considering boxes as overlapping
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return detections
        
        # Group detections by class
        class_detections = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in class_detections:
                class_detections[class_name] = []
            class_detections[class_name].append(detection)
        
        # Apply NMS per class
        filtered_detections = []
        
        for class_name, class_dets in class_detections.items():
            if len(class_dets) == 1:
                filtered_detections.extend(class_dets)
                continue
            
            # Sort by confidence (highest first)
            class_dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS
            keep = []
            for i, detection in enumerate(class_dets):
                should_keep = True
                for kept_detection in keep:
                    iou = self._calculate_iou(detection['bbox'], kept_detection['bbox'])
                    if iou > iou_threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    keep.append(detection)
            
            filtered_detections.extend(keep)
        
        return filtered_detections
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess basic image quality metrics
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing quality metrics
        """
        # Convert to grayscale for quality analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Image dimensions
        height, width = gray.shape[:2]
        
        # Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness analysis
        brightness = np.mean(gray)
        
        # Contrast analysis
        contrast = np.std(gray)
        
        # Assess quality
        quality_issues = []
        quality_score = 100  # Start with perfect score
        
        # Check image size
        if width < self.quality_criteria['min_image_size'][0] or height < self.quality_criteria['min_image_size'][1]:
            quality_issues.append("Image resolution too low")
            quality_score -= 20
        
        # Check blur
        if blur_score < self.quality_criteria['max_blur_threshold']:
            quality_issues.append("Image appears blurry")
            quality_score -= 25
        
        # Check brightness
        if brightness < self.quality_criteria['min_brightness']:
            quality_issues.append("Image too dark")
            quality_score -= 15
        elif brightness > self.quality_criteria['max_brightness']:
            quality_issues.append("Image too bright")
            quality_score -= 15
        
        # Check contrast
        if contrast < self.quality_criteria['contrast_threshold']:
            quality_issues.append("Low contrast")
            quality_score -= 10
        
        return {
            'quality_score': max(0, quality_score),
            'blur_score': float(blur_score),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'resolution': (width, height),
            'quality_issues': quality_issues,
            'is_suitable': len(quality_issues) == 0 and quality_score >= 70
        }
    
    def _generate_recommendations(self, quality_assessment: Dict, detections: List[Dict]) -> List[str]:
        """
        Generate actionable recommendations for improving image quality
        
        Args:
            quality_assessment: Quality assessment results
            detections: Detected occlusions
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Quality-based recommendations
        for issue in quality_assessment['quality_issues']:
            if "resolution too low" in issue:
                recommendations.append("Use higher resolution camera or get closer to subject")
            elif "blurry" in issue:
                recommendations.append("Ensure camera is stable and subject is still during capture")
            elif "too dark" in issue:
                recommendations.append("Improve lighting conditions or increase exposure")
            elif "too bright" in issue:
                recommendations.append("Reduce lighting intensity or decrease exposure")
            elif "Low contrast" in issue:
                recommendations.append("Improve lighting setup to create better contrast")
        
        # Occlusion-based recommendations
        occlusion_types = [d['class'] for d in detections]
        
        if 'cabello_tapando_oreja' in occlusion_types:
            recommendations.append("Move hair away from ear area for clear profile view")
        if 'cabello_tapando_frente' in occlusion_types:
            recommendations.append("Ensure forehead area is visible and not covered by hair")
        if 'objeto' in occlusion_types:
            recommendations.append("Remove any objects or accessories that might obstruct the profile")
        
        # General profile recommendations
        if detections:
            recommendations.append("Ensure true profile view (90-degree angle from camera)")
            recommendations.append("Use plain background without distracting elements")
        
        return recommendations
    
    def analyze_profile_validation(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Complete profile validation analysis
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence threshold for detections
            
        Returns:
            Dictionary containing complete validation results
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            # Preprocess image
            image_tensor, image_resized = self.preprocess_image(image)
            
            # Run occlusion detection
            with torch.no_grad():
                x = image_tensor.unsqueeze(0).to(self.device)
                predictions = self.model(x)
            
            # Process predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            
            # Filter predictions by confidence threshold
            detections = []
            
            for box, label, score in zip(boxes, labels, scores):
                class_name = self.all_classes[label - 1]
                threshold = self.class_thresholds.get(class_name, confidence_threshold)
                
                if score > threshold:
                    detections.append({
                        'class': class_name,
                        'confidence': float(score),
                        'bbox': box.tolist(),
                        'color': self.class_colors.get(class_name, 'blue')
                    })
            
            # Apply NMS per class
            detections = self._apply_nms_per_class(detections, iou_threshold=0.3)
            
            # Assess image quality
            quality_assessment = self._assess_image_quality(image)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(quality_assessment, detections)
            
            # Determine overall validation status
            has_occlusions = len(detections) > 0
            is_suitable = quality_assessment['is_suitable'] and not has_occlusions
            
            # Calculate overall score
            overall_score = quality_assessment['quality_score']
            if has_occlusions:
                overall_score -= len(detections) * 10  # Penalize for each occlusion
            overall_score = max(0, min(100, overall_score))
            
            return {
                'analysis_id': analysis_id,
                'validation_status': {
                    'is_suitable': is_suitable,
                    'overall_score': overall_score,
                    'has_occlusions': has_occlusions,
                    'quality_passed': quality_assessment['is_suitable']
                },
                'occlusion_analysis': {
                    'total_detections': len(detections),
                    'confidence_threshold_used': confidence_threshold,
                    'detections': detections
                },
                'quality_assessment': quality_assessment,
                'recommendations': recommendations,
                'analysis_summary': {
                    'timestamp': None,  # Will be set by the API
                    'processing_successful': True,
                    'model_classes': self.all_classes,
                    'device_used': str(self.device)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in profile validation analysis: {e}")
            return {
                'analysis_id': analysis_id,
                'validation_status': {
                    'is_suitable': False,
                    'overall_score': 0,
                    'has_occlusions': True,
                    'quality_passed': False
                },
                'error': str(e),
                'analysis_summary': {
                    'timestamp': None,
                    'processing_successful': False,
                    'error_message': str(e)
                }
            }
    
    def detect_occlusions_only(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect only occlusions without full quality assessment
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence threshold for detections
            
        Returns:
            Dictionary containing occlusion detection results
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            # Preprocess image
            image_tensor, image_resized = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                x = image_tensor.unsqueeze(0).to(self.device)
                predictions = self.model(x)
            
            # Process predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            
            # Filter predictions by confidence threshold
            detections = []
            
            for box, label, score in zip(boxes, labels, scores):
                class_name = self.all_classes[label - 1]
                threshold = self.class_thresholds.get(class_name, confidence_threshold)
                
                if score > threshold:
                    detections.append({
                        'class': class_name,
                        'confidence': float(score),
                        'bbox': box.tolist()
                    })
            
            # Apply NMS per class
            detections = self._apply_nms_per_class(detections, iou_threshold=0.3)
            
            has_occlusions = len(detections) > 0
            
            return {
                'analysis_id': analysis_id,
                'has_occlusions': has_occlusions,
                'total_detections': len(detections),
                'detections': detections,
                'confidence_threshold_used': confidence_threshold,
                'processing_successful': True
            }
            
        except Exception as e:
            logger.error(f"Error in occlusion detection: {e}")
            return {
                'analysis_id': analysis_id,
                'has_occlusions': True,
                'total_detections': 0,
                'detections': [],
                'error': str(e),
                'processing_successful': False
            }
