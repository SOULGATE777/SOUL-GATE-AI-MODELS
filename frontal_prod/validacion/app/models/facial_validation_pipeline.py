import torch
import cv2
import numpy as np
import os
import asyncio
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import uuid

from ultralytics import YOLO
from ..utils.visualization import create_validation_visualization
from ..utils.image_processing import preprocess_image

logger = logging.getLogger(__name__)

class FacialValidationPipeline:
    """
    Facial validation pipeline using YOLO for feature detection
    """
    
    def __init__(self, model_path: str = "/app/models/best.pt"):
        """
        Initialize the facial validation pipeline
        
        Args:
            model_path: Path to YOLO model weights
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Class names mapping
        self.class_names = {
            0: 'piercing', 1: 'cabello_tapando_i', 2: 'cabello_tapando_derecho',
            3: 'cabello_tapando_central', 4: 'tatuaje', 5: 'barba',
            6: 'p_d_g_iz', 7: 'p_d_g_d', 8: 'p_d_v',
            9: 'l_ej_i', 10: 'l_ej_d', 11: 'calvo',
            12: 'lentes', 13: 'objeto_frente', 14: 'bc_abierta',
            15: 'bc_bigote', 16: 'bc_sonriendo'
        }
        
        # Feature categories
        self.feature_categories = {
            'hair_coverage': ['cabello_tapando_i', 'cabello_tapando_derecho', 'cabello_tapando_central'],
            'facial_hair': ['barba', 'bc_bigote'],
            'facial_expression': ['bc_abierta', 'bc_sonriendo'],
            'accessories': ['piercing', 'lentes', 'objeto_frente'],
            'body_modifications': ['tatuaje'],
            'head_characteristics': ['calvo'],
            'eye_features': ['l_ej_i', 'l_ej_d'],
            'facial_points': ['p_d_g_iz', 'p_d_g_d', 'p_d_v']
        }
        
        self.results_dir = "/app/results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load model
        self._load_model()
        
        logger.info(f"✅ Facial validation pipeline initialized on {self.device}")
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = YOLO(self.model_path)
            logger.info(f"✅ YOLO model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    async def detect_features(
        self, 
        image_path: str, 
        confidence_threshold: float = 0.20
    ) -> Dict:
        """
        Detect facial features in image
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Run detection
            results = self.model(image_path, conf=confidence_threshold, verbose=False)
            
            # Process results
            detections = []
            class_counts = {}
            total_detections = 0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        class_name = self.class_names.get(cls, f'class_{cls}')
                        
                        detection = {
                            'class_id': cls,
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                            'area': float((x2 - x1) * (y2 - y1))
                        }
                        
                        detections.append(detection)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        total_detections += 1
            
            # Calculate statistics
            avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0.0
            high_conf_count = len([d for d in detections if d['confidence'] > 0.7])
            
            # Categorize features
            categorized_features = self._categorize_features(detections)
            
            return {
                'detection_results': {
                    'total_detections': total_detections,
                    'detections': detections,
                    'class_counts': class_counts,
                    'average_confidence': float(avg_confidence),
                    'high_confidence_count': high_conf_count
                },
                'feature_analysis': {
                    'categorized_features': categorized_features,
                    'feature_summary': self._generate_feature_summary(categorized_features)
                },
                'image_info': {
                    'dimensions': [image.shape[1], image.shape[0]],
                    'channels': image.shape[2] if len(image.shape) > 2 else 1
                },
                'processing_info': {
                    'confidence_threshold': confidence_threshold,
                    'model_path': self.model_path,
                    'device': str(self.device),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in feature detection: {e}")
            raise
    
    async def analyze_complete(
        self,
        image_path: str,
        confidence_threshold: float = 0.20,
        include_visualization: bool = True
    ) -> Dict:
        """
        Complete facial validation analysis with optional visualization
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
            include_visualization: Whether to create visualization
            
        Returns:
            Complete analysis results
        """
        try:
            # Get detection results
            detection_results = await self.detect_features(image_path, confidence_threshold)
            
            # Create visualization if requested
            visualization_path = None
            if include_visualization:
                visualization_path = await self._create_visualization(
                    image_path, 
                    detection_results['detection_results']['detections'],
                    confidence_threshold
                )
            
            # Combine results
            analysis_results = {
                **detection_results,
                'validation_summary': {
                    'image_suitable': self._assess_image_suitability(detection_results),
                    'quality_issues': self._identify_quality_issues(detection_results),
                    'recommendations': self._generate_recommendations(detection_results)
                }
            }
            
            if visualization_path:
                analysis_results['visualization_path'] = visualization_path
                analysis_results['visualization_url'] = f"/visualization/{os.path.basename(visualization_path)}"
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise
    
    def _categorize_features(self, detections: List[Dict]) -> Dict:
        """Categorize detected features by type"""
        categorized = {}
        
        for category, feature_names in self.feature_categories.items():
            category_detections = [
                d for d in detections 
                if d['class_name'] in feature_names
            ]
            categorized[category] = {
                'count': len(category_detections),
                'detections': category_detections,
                'max_confidence': max([d['confidence'] for d in category_detections]) if category_detections else 0.0
            }
        
        return categorized
    
    def _generate_feature_summary(self, categorized_features: Dict) -> Dict:
        """Generate summary of detected features"""
        summary = {}
        
        for category, data in categorized_features.items():
            if data['count'] > 0:
                summary[category] = {
                    'present': True,
                    'count': data['count'],
                    'confidence': data['max_confidence'],
                    'features': list(set([d['class_name'] for d in data['detections']]))
                }
            else:
                summary[category] = {'present': False}
        
        return summary
    
    def _assess_image_suitability(self, results: Dict) -> Dict:
        """Assess if image is suitable for analysis"""
        issues = []
        score = 100
        
        # Check for obstructions
        hair_coverage = results['feature_analysis']['categorized_features']['hair_coverage']
        if hair_coverage['count'] > 0:
            issues.append("Hair covering facial features")
            score -= 20
        
        # Check for accessories that might interfere
        accessories = results['feature_analysis']['categorized_features']['accessories']
        if accessories['count'] > 0:
            for detection in accessories['detections']:
                if detection['class_name'] in ['lentes', 'objeto_frente']:
                    issues.append(f"Accessory detected: {detection['class_name']}")
                    score -= 15
        
        # Check image quality indicators
        if results['detection_results']['average_confidence'] < 0.4:
            issues.append("Low detection confidence - possible image quality issues")
            score -= 25
        
        return {
            'suitable': len(issues) == 0,
            'suitability_score': max(0, score),
            'issues': issues
        }
    
    def _identify_quality_issues(self, results: Dict) -> List[str]:
        """Identify potential quality issues"""
        issues = []
        
        # Low confidence detections
        low_conf_detections = [
            d for d in results['detection_results']['detections'] 
            if d['confidence'] < 0.3
        ]
        if low_conf_detections:
            issues.append(f"Low confidence detections: {len(low_conf_detections)}")
        
        # Check for common problematic features
        problematic_features = ['objeto_frente', 'cabello_tapando_central']
        for detection in results['detection_results']['detections']:
            if detection['class_name'] in problematic_features:
                issues.append(f"Problematic feature: {detection['class_name']}")
        
        return issues
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Hair coverage recommendations
        hair_issues = [d for d in results['detection_results']['detections'] 
                      if 'cabello_tapando' in d['class_name']]
        if hair_issues:
            recommendations.append("Consider moving hair away from face for better analysis")
        
        # Accessory recommendations
        if any(d['class_name'] == 'lentes' for d in results['detection_results']['detections']):
            recommendations.append("Remove glasses if possible for more accurate measurements")
        
        # Expression recommendations
        if any(d['class_name'] == 'bc_abierta' for d in results['detection_results']['detections']):
            recommendations.append("Close mouth for standard facial measurements")
        
        # Quality recommendations
        if results['detection_results']['average_confidence'] < 0.5:
            recommendations.append("Improve image quality - better lighting or resolution")
        
        return recommendations
    
    async def _create_visualization(
        self, 
        image_path: str, 
        detections: List[Dict], 
        confidence_threshold: float
    ) -> str:
        """Create visualization of detections"""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            output_filename = f"validation_{timestamp}_{unique_id}.jpg"
            output_path = os.path.join(self.results_dir, output_filename)
            
            # Create visualization
            visualization_path = create_validation_visualization(
                image_path=image_path,
                detections=detections,
                class_names=self.class_names,
                output_path=output_path,
                confidence_threshold=confidence_threshold
            )
            
            return visualization_path
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
