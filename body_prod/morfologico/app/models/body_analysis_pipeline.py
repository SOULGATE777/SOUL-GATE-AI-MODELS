import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class LightweightHierarchicalModel(nn.Module):
    """Lightweight model optimized for limited GPU memory"""
    
    def __init__(self, num_body_types=7, num_coarse_types=4, num_genders=2):
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
        
        # Gender head
        self.gender_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_genders)
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
        
        # Gender classification
        gender_logits = self.gender_head(attended_features)
        
        return {
            'body_type': fine_logits,
            'coarse_type': coarse_logits,
            'gender': gender_logits
        }

class BodyAnalysisPipeline:
    """Production pipeline for body morphological analysis"""
    
    def __init__(self, model_path: str = "/app/models/lightweight_body_classifier.pth"):
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        self.transform = self._get_transform()
        
        # Default class names (will be loaded from checkpoint if available)
        self.body_type_classes = [
            'Bilioso/NormalPocaGrasa',
            'Nervioso/Delgado', 
            'SanguineoLinfatico/MusculosoGordo',
            'Sanguineo/Musculoso',
            'Flematico/Gordograsacuelga',
            'Linfatico/Gordo',
            'BiliosoSanguineo/NormalMusculoso'
        ]
        
        self.gender_classes = ['Hombre', 'Mujer']
        
        # Simplified class names for better display
        self.body_type_simple = [
            'Normal Poca Grasa',
            'Delgado',
            'Musculoso Gordo',
            'Musculoso',
            'Gordo Grasa Cuelga',
            'Gordo',
            'Normal Musculoso'
        ]
        
        logger.info(f"Pipeline initialized on device: {self.device}")
    
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
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            logger.info(f"Loaded checkpoint from: {self.model_path}")
            
            # Create model
            self.model = LightweightHierarchicalModel(
                num_body_types=7,
                num_coarse_types=4,
                num_genders=2
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Update class names if available in checkpoint
            if 'body_type_classes' in checkpoint:
                self.body_type_classes = checkpoint['body_type_classes']
            if 'gender_classes' in checkpoint:
                self.gender_classes = checkpoint['gender_classes']
            
            # Log model info
            best_acc = checkpoint.get('best_body_acc', 'Unknown')
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"Best training accuracy: {best_acc}")
            logger.info(f"Body type classes: {len(self.body_type_classes)}")
            logger.info(f"Gender classes: {len(self.gender_classes)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_parameters(self) -> int:
        """Get total number of model parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _preprocess_image(self, image: np.ndarray, bbox: Optional[List[int]] = None) -> torch.Tensor:
        """Preprocess image for inference"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume it's already RGB from the API processing
            pass
        
        # Crop to bbox if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 > x1 and y2 > y1:
                image = image[y1:y2, x1:x2]
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def _extract_predictions(self, predictions: Dict[str, torch.Tensor], confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Extract and format predictions"""
        # Get probabilities
        body_probs = F.softmax(predictions['body_type'], dim=1).cpu().numpy()[0]
        gender_probs = F.softmax(predictions['gender'], dim=1).cpu().numpy()[0]
        coarse_probs = F.softmax(predictions['coarse_type'], dim=1).cpu().numpy()[0]
        
        # Get top predictions
        body_idx = np.argmax(body_probs)
        gender_idx = np.argmax(gender_probs)
        coarse_idx = np.argmax(coarse_probs)
        
        # Format results
        results = {
            # Fine-grained body type prediction
            'body_type_analysis': {
                'predicted_class': self.body_type_classes[body_idx],
                'predicted_class_simple': self.body_type_simple[body_idx],
                'confidence': float(body_probs[body_idx]),
                'meets_threshold': float(body_probs[body_idx]) >= confidence_threshold,
                'all_probabilities': {
                    self.body_type_classes[i]: float(prob) 
                    for i, prob in enumerate(body_probs)
                },
                'top_3_predictions': self._get_top_k_predictions(body_probs, self.body_type_classes, k=3)
            },
            
            # Gender prediction
            'gender_analysis': {
                'predicted_gender': self.gender_classes[gender_idx],
                'confidence': float(gender_probs[gender_idx]),
                'meets_threshold': float(gender_probs[gender_idx]) >= confidence_threshold,
                'all_probabilities': {
                    self.gender_classes[i]: float(prob) 
                    for i, prob in enumerate(gender_probs)
                }
            },
            
            # Coarse classification (additional info)
            'coarse_analysis': {
                'predicted_coarse': f"Coarse_Type_{coarse_idx}",
                'confidence': float(coarse_probs[coarse_idx]),
                'all_probabilities': {
                    f"Coarse_Type_{i}": float(prob) 
                    for i, prob in enumerate(coarse_probs)
                }
            }
        }
        
        return results
    
    def _get_top_k_predictions(self, probabilities: np.ndarray, class_names: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """Get top-k predictions sorted by confidence"""
        sorted_indices = np.argsort(probabilities)[::-1][:k]
        return [
            {
                'class': class_names[idx],
                'confidence': float(probabilities[idx]),
                'rank': rank + 1
            }
            for rank, idx in enumerate(sorted_indices)
        ]
    
    def classify_body_type_only(self, image: np.ndarray, bbox: Optional[List[int]] = None, 
                               confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Run body type classification only (no additional analysis)"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image, bbox)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Extract predictions
            results = self._extract_predictions(predictions, confidence_threshold)
            
            # Add metadata
            results.update({
                'analysis_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_successful': True,
                    'confidence_threshold_used': confidence_threshold,
                    'device_used': str(self.device),
                    'image_preprocessed': True,
                    'bbox_applied': bbox is not None
                }
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in body type classification: {e}")
            return {
                'analysis_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_successful': False,
                    'error': str(e),
                    'device_used': str(self.device)
                }
            }
    
    def analyze_body_type(self, image: np.ndarray, bbox: Optional[List[int]] = None, 
                         confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Complete body type analysis with additional metrics"""
        # Get base classification
        results = self.classify_body_type_only(image, bbox, confidence_threshold)
        
        if not results.get('analysis_summary', {}).get('processing_successful', False):
            return results
        
        try:
            # Add morphological analysis
            body_analysis = results.get('body_type_analysis', {})
            gender_analysis = results.get('gender_analysis', {})
            
            # Calculate analysis metrics
            analysis_metrics = {
                'overall_confidence': (
                    body_analysis.get('confidence', 0) + 
                    gender_analysis.get('confidence', 0)
                ) / 2,
                'prediction_certainty': 'high' if body_analysis.get('confidence', 0) > 0.8 else 
                                      'medium' if body_analysis.get('confidence', 0) > 0.6 else 'low',
                'gender_body_consistency': self._assess_gender_body_consistency(
                    gender_analysis.get('predicted_gender', ''),
                    body_analysis.get('predicted_class', '')
                )
            }
            
            # Add morphological insights
            morphological_insights = self._generate_morphological_insights(
                body_analysis.get('predicted_class_simple', ''),
                gender_analysis.get('predicted_gender', ''),
                body_analysis.get('confidence', 0)
            )
            
            # Update results
            results.update({
                'analysis_metrics': analysis_metrics,
                'morphological_insights': morphological_insights,
                'classification_summary': {
                    'primary_classification': body_analysis.get('predicted_class_simple', 'Unknown'),
                    'gender': gender_analysis.get('predicted_gender', 'Unknown'),
                    'confidence_level': analysis_metrics['prediction_certainty'],
                    'recommended_action': self._get_recommendation(analysis_metrics['prediction_certainty'])
                }
            })
            
            # Update processing status
            results['analysis_summary']['analysis_type'] = 'complete_morphological_analysis'
            
            return results
            
        except Exception as e:
            logger.error(f"Error in complete body analysis: {e}")
            results['analysis_summary'].update({
                'processing_successful': False,
                'error': f"Analysis error: {str(e)}"
            })
            return results
    
    def _assess_gender_body_consistency(self, gender: str, body_type: str) -> str:
        """Assess consistency between gender and body type predictions"""
        # This is a simplified consistency check
        # In a real application, you might have more sophisticated rules
        if not gender or not body_type:
            return 'unknown'
        
        # Basic consistency rules (these would be refined based on domain knowledge)
        muscular_types = ['Musculoso', 'Normal Musculoso']
        if gender == 'Hombre' and any(mt in body_type for mt in muscular_types):
            return 'high'
        elif gender == 'Mujer' and 'Delgado' in body_type:
            return 'high'
        else:
            return 'medium'
    
    def _generate_morphological_insights(self, body_type: str, gender: str, confidence: float) -> Dict[str, Any]:
        """Generate morphological insights based on classification"""
        insights = {
            'body_composition': '',
            'metabolic_tendency': '',
            'physical_characteristics': '',
            'health_considerations': ''
        }
        
        # Body type specific insights
        if 'Delgado' in body_type:
            insights.update({
                'body_composition': 'Ectomorphic build with low body fat and muscle mass',
                'metabolic_tendency': 'Fast metabolism, difficulty gaining weight',
                'physical_characteristics': 'Narrow frame, lean appearance, long limbs',
                'health_considerations': 'Focus on strength training and adequate nutrition'
            })
        elif 'Musculoso' in body_type:
            insights.update({
                'body_composition': 'Mesomorphic build with well-developed musculature',
                'metabolic_tendency': 'Efficient metabolism, responds well to exercise',
                'physical_characteristics': 'Athletic build, defined muscle structure',
                'health_considerations': 'Maintain balanced training and nutrition'
            })
        elif 'Gordo' in body_type:
            insights.update({
                'body_composition': 'Endomorphic build with higher body fat percentage',
                'metabolic_tendency': 'Slower metabolism, tendency to store fat',
                'physical_characteristics': 'Rounder physique, softer muscle definition',
                'health_considerations': 'Focus on cardiovascular exercise and dietary management'
            })
        else:
            insights.update({
                'body_composition': 'Balanced body composition',
                'metabolic_tendency': 'Moderate metabolic rate',
                'physical_characteristics': 'Proportioned build',
                'health_considerations': 'Maintain active lifestyle and balanced nutrition'
            })
        
        # Add confidence-based insights
        if confidence < 0.6:
            insights['analysis_note'] = 'Low confidence prediction - consider multiple assessments'
        elif confidence > 0.8:
            insights['analysis_note'] = 'High confidence prediction - reliable classification'
        
        return insights
    
    def _get_recommendation(self, certainty_level: str) -> str:
        """Get recommendation based on prediction certainty"""
        recommendations = {
            'high': 'Classification is reliable - proceed with analysis',
            'medium': 'Classification is moderately reliable - consider additional assessment',
            'low': 'Low confidence classification - recommend manual review or better image quality'
        }
        return recommendations.get(certainty_level, 'Unknown certainty level')
