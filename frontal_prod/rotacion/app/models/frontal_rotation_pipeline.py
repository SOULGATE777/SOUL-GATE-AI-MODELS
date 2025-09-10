import torch
import cv2
import numpy as np
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrontalMultiLabelRotationClassifier(nn.Module):
    """
    Multi-label CNN model for frontal face rotation classification
    Based on EfficientNet-B0 architecture
    """
    def __init__(self, num_classes):
        super(FrontalMultiLabelRotationClassifier, self).__init__()
        # Load pretrained weights with hash checking disabled
        import torch.hub
        original_download = torch.hub.download_url_to_file
        def download_no_hash(url, dst, hash_prefix=None, progress=True):
            return original_download(url, dst, None, progress)
        
        torch.hub.download_url_to_file = download_no_hash
        try:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        finally:
            torch.hub.download_url_to_file = original_download
        
        # Freeze most layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze classifier and last features
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        for param in self.backbone.features[-3:].parameters():
            param.requires_grad = True
            
        # Multi-label classifier for frontal rotations
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class FrontalRotationPipeline:
    """
    Frontal face rotation assessment pipeline for determining if frontal images
    are suitable for anthropometric and morphological analysis.
    
    Uses a multi-label CNN to classify frontal face orientation and quality.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the frontal rotation pipeline
        
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
        
        # Extract model information from frontal face model
        self.class_names = checkpoint.get('class_names', [
            'hacia_arriba_o_tomadao_desde_abajo',
            'horizontal', 
            'diagonal',
            'hacia_abajo_o_tomado_desde_arriba',
            'aceptable'
        ])
        self.num_classes = len(self.class_names)
        
        # Find indices for pattern analysis
        self.aceptable_idx = None
        self.rotation_indices = []
        for i, class_name in enumerate(self.class_names):
            if class_name == 'aceptable':
                self.aceptable_idx = i
            else:
                self.rotation_indices.append(i)
        
        # Initialize model
        self.model = FrontalMultiLabelRotationClassifier(self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Configuration
        self.default_threshold = 0.5
        
        logger.info(f"Frontal rotation pipeline initialized successfully")
        logger.info(f"Model classes: {self.class_names}")
        logger.info(f"Number of classes: {self.num_classes}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor
    
    def predict_rotation_pattern_aware(self, image: np.ndarray, threshold: float = None) -> Dict[str, Any]:
        """
        Pattern-aware multi-label prediction for frontal rotation assessment
        
        Args:
            image: Input image as numpy array
            threshold: Confidence threshold for predictions (default: 0.5)
            
        Returns:
            Dictionary containing rotation assessment results
        """
        if threshold is None:
            threshold = self.default_threshold
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                outputs = self.model(image_tensor)
                probabilities = torch.sigmoid(outputs)[0]
            
            # Get predictions above threshold
            predictions = probabilities > threshold
            
            # Apply pattern constraints for frontal faces
            aceptable_pred = predictions[self.aceptable_idx] if self.aceptable_idx is not None else False
            aceptable_confidence = probabilities[self.aceptable_idx].item() if self.aceptable_idx is not None else 0.0
            
            predicted_tags = []
            confidences = {}
            rotation_issues = []
            
            # Get all probabilities for analysis
            all_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                all_probabilities[class_name] = probabilities[i].item()
                confidences[class_name] = probabilities[i].item()
            
            if aceptable_pred:
                # If aceptable is predicted, frontal face is suitable for analysis
                predicted_tags = ['aceptable']
                is_suitable = True
                suitability_reason = "Frontal face orientation is acceptable for analysis"
            else:
                # Return rotation issues (can be single or multiple)
                for i in self.rotation_indices:
                    if predictions[i]:
                        predicted_tags.append(self.class_names[i])
                        rotation_issues.append(self.class_names[i])
                
                is_suitable = False
                if rotation_issues:
                    suitability_reason = f"Frontal face has rotation issues: {', '.join(rotation_issues)}"
                else:
                    suitability_reason = "Frontal face quality assessment unclear - low confidence predictions"
            
            # Overall confidence assessment
            max_confidence = max(all_probabilities.values())
            prediction_certainty = self._assess_prediction_certainty(all_probabilities, threshold)
            
            return {
                'predicted_tags': predicted_tags,
                'rotation_issues': rotation_issues,
                'is_suitable': is_suitable,
                'suitability_reason': suitability_reason,
                'aceptable_confidence': aceptable_confidence,
                'max_confidence': max_confidence,
                'prediction_certainty': prediction_certainty,
                'all_probabilities': all_probabilities,
                'confidences': confidences,
                'threshold_used': threshold
            }
            
        except Exception as e:
            logger.error(f"Error in frontal rotation prediction: {e}")
            raise
    
    def _assess_prediction_certainty(self, probabilities: Dict[str, float], threshold: float) -> str:
        """
        Assess the certainty of the prediction based on probability distribution
        
        Args:
            probabilities: Dictionary of class probabilities
            threshold: Threshold used for predictions
            
        Returns:
            Certainty level as string
        """
        max_prob = max(probabilities.values())
        above_threshold_count = sum(1 for p in probabilities.values() if p > threshold)
        
        if max_prob > 0.8:
            return "high"
        elif max_prob > 0.6:
            return "medium"
        elif max_prob > threshold:
            return "low"
        else:
            return "very_low"
    
    def analyze_frontal_rotation(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Complete frontal rotation analysis with detailed assessment
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Complete rotation analysis results
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            # Get basic image information
            height, width = image.shape[:2]
            
            # Run rotation prediction
            rotation_results = self.predict_rotation_pattern_aware(image, confidence_threshold)
            
            # Create comprehensive analysis
            analysis_results = {
                'analysis_id': analysis_id,
                'image_info': {
                    'width': int(width),
                    'height': int(height),
                    'channels': int(image.shape[2]) if len(image.shape) == 3 else 1
                },
                'rotation_assessment': rotation_results,
                'viability_for_analysis': {
                    'suitable_for_anthropometric': rotation_results['is_suitable'],
                    'suitable_for_morphological': rotation_results['is_suitable'],
                    'recommendation': self._generate_recommendation(rotation_results),
                    'confidence_level': rotation_results['prediction_certainty']
                },
                'analysis_summary': {
                    'processing_successful': True,
                    'main_finding': rotation_results['suitability_reason'],
                    'predicted_orientation': rotation_results['predicted_tags'][0] if rotation_results['predicted_tags'] else 'unclear',
                    'max_confidence': rotation_results['max_confidence'],
                    'threshold_used': confidence_threshold
                }
            }
            
            logger.info(f"Frontal rotation analysis completed - ID: {analysis_id}, "
                       f"Suitable: {rotation_results['is_suitable']}, "
                       f"Tags: {rotation_results['predicted_tags']}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in frontal rotation analysis: {e}")
            return {
                'analysis_id': analysis_id,
                'analysis_summary': {
                    'processing_successful': False,
                    'error': str(e),
                    'main_finding': 'Analysis failed due to processing error'
                }
            }
    
    def _generate_recommendation(self, rotation_results: Dict[str, Any]) -> str:
        """
        Generate actionable recommendations based on frontal rotation assessment
        
        Args:
            rotation_results: Results from rotation prediction
            
        Returns:
            Recommendation string
        """
        if rotation_results['is_suitable']:
            return "Frontal face orientation is acceptable. Proceed with anthropometric and morphological analysis."
        
        if rotation_results['rotation_issues']:
            issues = rotation_results['rotation_issues']
            recommendations = []
            
            for issue in issues:
                if 'hacia_arriba' in issue or 'desde_abajo' in issue:
                    recommendations.append("Lower camera angle - face is tilted up or camera positioned too low")
                elif 'hacia_abajo' in issue or 'desde_arriba' in issue:
                    recommendations.append("Raise camera angle - face is tilted down or camera positioned too high")
                elif 'horizontal' in issue:
                    recommendations.append("Adjust face positioning - detected horizontal orientation, ensure frontal view")
                elif 'diagonal' in issue:
                    recommendations.append("Straighten face orientation - diagonal tilt detected, align face vertically")
            
            if recommendations:
                return f"Rotation issues detected. Recommendations: {'; '.join(recommendations)}"
        
        return "Frontal face orientation unclear. Ensure proper lighting and positioning for clear frontal face capture."
    
    def classify_rotation_only(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Simple rotation classification without full analysis
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Simple rotation classification results
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            rotation_results = self.predict_rotation_pattern_aware(image, confidence_threshold)
            
            return {
                'analysis_id': analysis_id,
                'predicted_orientation': rotation_results['predicted_tags'][0] if rotation_results['predicted_tags'] else 'unclear',
                'is_acceptable': rotation_results['is_suitable'],
                'confidence': rotation_results['max_confidence'],
                'all_predictions': rotation_results['predicted_tags'],
                'rotation_issues': rotation_results['rotation_issues'],
                'processing_successful': True
            }
            
        except Exception as e:
            logger.error(f"Error in frontal rotation classification: {e}")
            return {
                'analysis_id': analysis_id,
                'processing_successful': False,
                'error': str(e)
            }
    
    def assess_viability_only(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Assess viability for anthropometric/morphological analysis only
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Viability assessment results
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            rotation_results = self.predict_rotation_pattern_aware(image, confidence_threshold)
            
            return {
                'analysis_id': analysis_id,
                'viable_for_analysis': rotation_results['is_suitable'],
                'viability_reason': rotation_results['suitability_reason'],
                'confidence_level': rotation_results['prediction_certainty'],
                'recommendation': self._generate_recommendation(rotation_results),
                'aceptable_confidence': rotation_results['aceptable_confidence'],
                'processing_successful': True
            }
            
        except Exception as e:
            logger.error(f"Error in frontal viability assessment: {e}")
            return {
                'analysis_id': analysis_id,
                'processing_successful': False,
                'error': str(e)
            }