import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from collections import Counter
import colorsys
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class HandClassifier(nn.Module):
    """
    ResNet-based binary classifier for hand orientation
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(HandClassifier, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        # Replace final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class PalmColorimetryAnalyzer:
    """
    Colorimetry analyzer for palm regions
    """
    
    def __init__(self):
        pass
    
    def create_skin_mask(self, palm_image):
        """Create skin mask using color thresholds."""
        hsv = cv2.cvtColor(palm_image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(palm_image, cv2.COLOR_BGR2YCrCb)
        
        # HSV skin range
        lower_hsv = np.array([0, 15, 60])
        upper_hsv = np.array([20, 170, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb skin range
        lower_ycrcb = np.array([0, 135, 85])
        upper_ycrcb = np.array([255, 180, 135])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine and clean up
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask
    
    def analyze_colors(self, palm_image, mask, n_clusters=5):
        """Perform color analysis on palm region."""
        masked_pixels = palm_image[mask > 0]
        
        if len(masked_pixels) == 0:
            return None
        
        # Convert to RGB
        masked_pixels_rgb = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB)
        masked_pixels_rgb = masked_pixels_rgb.reshape(-1, 3)
        
        # K-means clustering
        n_clusters = min(n_clusters, len(masked_pixels_rgb))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(masked_pixels_rgb)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        color_percentages = []
        for i, color in enumerate(colors):
            percentage = (label_counts[i] / total_pixels) * 100
            color_percentages.append((color, percentage))
        
        color_percentages.sort(key=lambda x: x[1], reverse=True)
        
        # Average color
        avg_color = np.mean(masked_pixels_rgb, axis=0).astype(int)
        
        # HSV conversion
        avg_color_hsv = colorsys.rgb_to_hsv(avg_color[0]/255, avg_color[1]/255, avg_color[2]/255)
        
        # Hue analysis
        hues = []
        for pixel in masked_pixels_rgb:
            h, s, v = colorsys.rgb_to_hsv(pixel[0]/255, pixel[1]/255, pixel[2]/255)
            hues.append(h * 360)
        
        return {
            "average_color_rgb": avg_color.tolist(),
            "average_color_hsv": [avg_color_hsv[0] * 360, avg_color_hsv[1] * 100, avg_color_hsv[2] * 100],
            "dominant_colors": [[color.tolist(), percentage] for color, percentage in color_percentages],
            "hue_mean": float(np.mean(hues)),
            "hue_std": float(np.std(hues)),
            "total_pixels": int(total_pixels)
        }

class HandAnalysisPipeline:
    """
    Complete hand analysis pipeline combining CNN classification and colorimetry
    """
    
    def __init__(self, model_path='/app/models/dorso_palma_classifier.pth'):
        """
        Initialize the hand analysis pipeline
        
        Args:
            model_path (str): Path to the trained CNN model
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.colorimetry_analyzer = PalmColorimetryAnalyzer()
        
        # Transform for CNN model
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Define palm color classification ranges
        self.palm_color_ranges = {
            'rosa/sanguineo-linfatico oscuro': {
                'r_range': (185, 255),
                'g_range': (130, 185),
                'b_range': (130, 185),
                'condition': 'R mayor a G por mas de 20%'
            },
            'rojo/sanguineo': {
                'r_range': (185, 255),
                'g_range': (0, 145),
                'b_range': (0, 145),
                'condition': 'R mayor que G y B por mas de 20%'
            },
            'amarillo/nervioso': {
                'r_range': (0, 245),
                'g_range': (80, 255),
                'b_range': (0, 160),
                'condition': 'R y G dentro del 35% de cada uno, B mas del 20% menor que el mayor de los 2 de R o G'
            },
            'blanco/linfatico': {
                'r_range': (180, 255),
                'g_range': (150, 255),
                'b_range': (105, 255),
                'condition': 'mínimo 2 de los 3 arriba de 150'
            },
            'bilioso/cafe_o_oscuro': {
                'r_range': (0, 210),
                'g_range': (0, 180),
                'b_range': (0, 255),
                'condition': 'mínimo 2 de los 3 valores menor de 175'
            }
        }
        
        self.class_names = ['Dorso', 'Palma']
    
    def load_model(self):
        """Load the CNN model for prediction"""
        try:
            if os.path.exists(self.model_path):
                # Initialize model
                self.model = HandClassifier(num_classes=2, pretrained=True).to(self.device)
                
                # Load checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                logger.info(f"CNN model loaded successfully from {self.model_path}")
                return True
            else:
                logger.warning(f"CNN model file not found: {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
            return False
    
    def predict_hand_side(self, image_path, bbox=None):
        """Predict hand side using CNN model"""
        if self.model is None:
            return None
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Crop if bbox provided
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        # Resize and transform
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def classify_color_type(self, rgb_color):
        """
        Classify RGB color according to palm color ranges with inclusive criteria
        
        Args:
            rgb_color (tuple): RGB color values (r, g, b)
            
        Returns:
            dict: Classification results with percentages
        """
        r, g, b = rgb_color
        matches = []
        
        for color_type, ranges in self.palm_color_ranges.items():
            # Check if color falls within basic ranges
            r_in_range = ranges['r_range'][0] <= r <= ranges['r_range'][1]
            g_in_range = ranges['g_range'][0] <= g <= ranges['g_range'][1]
            b_in_range = ranges['b_range'][0] <= b <= ranges['b_range'][1]
            
            # Apply specific conditions
            condition_met = False
            
            if color_type == 'rosa/sanguineo-linfatico oscuro':
                # R mayor a G por mas de 20%
                if r_in_range and g_in_range and b_in_range:
                    condition_met = r > g * 1.2
                    
            elif color_type == 'rojo/sanguineo':
                # R mayor que G y B por mas de 20%
                if r_in_range and g_in_range and b_in_range:
                    condition_met = r > g * 1.2 and r > b * 1.2
                    
            elif color_type == 'amarillo/nervioso':
                # R y G dentro del 35% de cada uno, B mas del 20% menor que el mayor de los 2 de R o G
                if r_in_range and g_in_range and b_in_range:
                    rg_close = abs(r - g) <= max(r, g) * 0.35
                    max_rg = max(r, g)
                    b_lower = b < max_rg * 0.8
                    condition_met = rg_close and b_lower
                    
            elif color_type == 'blanco/linfatico':
                # mínimo 2 de los 3 arriba de 150
                if r_in_range and g_in_range and b_in_range:
                    high_values = sum([r > 150, g > 150, b > 150])
                    condition_met = high_values >= 2
                    
            elif color_type == 'bilioso/cafe_o_oscuro':
                # mínimo 2 de los 3 valores menor de 175
                if r_in_range and g_in_range and b_in_range:
                    low_values = sum([r < 175, g < 175, b < 175])
                    condition_met = low_values >= 2
            
            if condition_met:
                matches.append(color_type)
        
        # Calculate percentages (inclusive criterion)
        if len(matches) == 0:
            return {'no_match': 100.0}
        elif len(matches) == 1:
            return {matches[0]: 100.0}
        else:
            # Split equally between matches
            percentage = 100.0 / len(matches)
            return {match: percentage for match in matches}
    
    def analyze_hand_comprehensive(self, image_path, bbox=None, confidence_threshold=0.5, 
                                 include_colorimetry=True):
        """
        Complete analysis of a hand image: CNN classification + colorimetry
        
        Args:
            image_path (str): Path to the image
            bbox (tuple, optional): Bounding box (x_min, y_min, x_max, y_max)
            confidence_threshold (float): Minimum confidence for CNN prediction
            include_colorimetry (bool): Whether to include colorimetry analysis
            
        Returns:
            dict: Complete analysis results
        """
        logger.info(f"Analyzing hand image: {image_path}")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found at {image_path}")
        
        results = {
            'image_path': image_path,
            'bbox': bbox,
            'analysis_type': 'comprehensive_hand_analysis'
        }
        
        # 1. CNN Hand Side Classification
        try:
            if self.model is not None:
                prediction_result = self.predict_hand_side(image_path, bbox)
                if prediction_result:
                    predicted_class, confidence, probabilities = prediction_result
                    results['cnn_prediction'] = {
                        'predicted_class': predicted_class,
                        'confidence': float(confidence),
                        'probabilities': {
                            'Dorso': float(probabilities[0]),
                            'Palma': float(probabilities[1])
                        },
                        'meets_threshold': confidence >= confidence_threshold
                    }
                    logger.info(f"CNN prediction: {predicted_class} (confidence: {confidence:.2%})")
                else:
                    results['cnn_prediction'] = None
                    logger.warning("CNN prediction failed")
            else:
                results['cnn_prediction'] = None
                logger.warning("CNN model not available")
        except Exception as e:
            logger.error(f"Error in CNN prediction: {e}")
            results['cnn_prediction'] = None
        
        # 2. Colorimetry Analysis
        if include_colorimetry:
            try:
                # Load and process image
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                
                # If bbox provided, crop to that region
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    palm_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                else:
                    palm_image = image
                
                # Create skin mask
                skin_mask = self.colorimetry_analyzer.create_skin_mask(palm_image)
                
                # Analyze colors
                color_analysis = self.colorimetry_analyzer.analyze_colors(palm_image, skin_mask)
                
                if color_analysis is not None:
                    results['colorimetry'] = color_analysis
                    
                    # Extract top 3 dominant colors for classification (exclude average)
                    top_3_colors = color_analysis['dominant_colors'][:3]
                    
                    # Color Classification for each of the top 3 dominant colors
                    color_classifications = {}
                    for i, (color, percentage) in enumerate(top_3_colors):
                        classification = self.classify_color_type(color)
                        color_classifications[f'dominant_color_{i+1}'] = {
                            'rgb': color,
                            'percentage': percentage,
                            'classification': classification
                        }
                    
                    results['color_classification'] = color_classifications
                    
                    logger.info(f"Colorimetry analysis completed: {color_analysis['total_pixels']} pixels analyzed")
                else:
                    results['colorimetry'] = None
                    results['color_classification'] = None
                    logger.warning("No skin pixels found for colorimetry analysis")
                    
            except Exception as e:
                logger.error(f"Error in colorimetry analysis: {e}")
                results['colorimetry'] = None
                results['color_classification'] = None
        
        logger.info("Hand analysis completed")
        return results
    
    def classify_hand_side_only(self, image_path, bbox=None, confidence_threshold=0.5):
        """
        Hand side classification only (no colorimetry)
        
        Args:
            image_path (str): Path to the image
            bbox (tuple, optional): Bounding box coordinates
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            dict: CNN classification results only
        """
        logger.info(f"Classifying hand side: {image_path}")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found at {image_path}")
        
        results = {
            'image_path': image_path,
            'bbox': bbox,
            'analysis_type': 'hand_side_classification_only'
        }
        
        try:
            if self.model is not None:
                prediction_result = self.predict_hand_side(image_path, bbox)
                if prediction_result:
                    predicted_class, confidence, probabilities = prediction_result
                    results['cnn_prediction'] = {
                        'predicted_class': predicted_class,
                        'confidence': float(confidence),
                        'probabilities': {
                            'Dorso': float(probabilities[0]),
                            'Palma': float(probabilities[1])
                        },
                        'meets_threshold': confidence >= confidence_threshold
                    }
                    logger.info(f"Classification: {predicted_class} (confidence: {confidence:.2%})")
                else:
                    results['cnn_prediction'] = None
                    logger.warning("Classification failed")
            else:
                results['cnn_prediction'] = None
                logger.warning("CNN model not available")
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            results['cnn_prediction'] = None
        
        return results
    
    def analyze_colorimetry_only(self, image_path, bbox=None):
        """
        Colorimetry analysis only (no CNN classification)
        
        Args:
            image_path (str): Path to the image
            bbox (tuple, optional): Bounding box coordinates
            
        Returns:
            dict: Colorimetry analysis results only
        """
        logger.info(f"Analyzing colorimetry: {image_path}")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found at {image_path}")
        
        results = {
            'image_path': image_path,
            'bbox': bbox,
            'analysis_type': 'colorimetry_only'
        }
        
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # If bbox provided, crop to that region
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                palm_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            else:
                palm_image = image
            
            # Create skin mask
            skin_mask = self.colorimetry_analyzer.create_skin_mask(palm_image)
            
            # Analyze colors
            color_analysis = self.colorimetry_analyzer.analyze_colors(palm_image, skin_mask)
            
            if color_analysis is not None:
                results['colorimetry'] = color_analysis
                
                # Extract top 3 dominant colors for classification (exclude average)
                top_3_colors = color_analysis['dominant_colors'][:3]
                
                # Color Classification for each of the top 3 dominant colors
                color_classifications = {}
                for i, (color, percentage) in enumerate(top_3_colors):
                    classification = self.classify_color_type(color)
                    color_classifications[f'dominant_color_{i+1}'] = {
                        'rgb': color,
                        'percentage': percentage,
                        'classification': classification
                    }
                
                results['color_classification'] = color_classifications
                
                logger.info(f"Colorimetry analysis completed: {color_analysis['total_pixels']} pixels analyzed")
            else:
                results['colorimetry'] = None
                results['color_classification'] = None
                logger.warning("No skin pixels found for colorimetry analysis")
                
        except Exception as e:
            logger.error(f"Error in colorimetry analysis: {e}")
            results['colorimetry'] = None
            results['color_classification'] = None
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded models"""
        return {
            'cnn_model': {
                'architecture': 'ResNet50',
                'classes': self.class_names,
                'model_loaded': self.model is not None,
                'model_path': self.model_path,
                'device': str(self.device)
            },
            'colorimetry': {
                'color_types': list(self.palm_color_ranges.keys()),
                'clustering_algorithm': 'KMeans',
                'skin_detection': 'HSV + YCrCb color space filtering'
            }
        }