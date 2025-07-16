import cv2
import numpy as np
import dlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from scipy.spatial import ConvexHull
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Any

class BinaryRegionClassifier(nn.Module):
    """Binary classifier for FRENTE/rostro_menton regions"""
    def __init__(self, input_size=224):
        super(BinaryRegionClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.feature_size = self._get_feature_size(input_size)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.feature_size, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(inplace=True), 
            nn.Linear(128, 2)
        )
    
    def _get_feature_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size, input_size)
            x = self.features(x)
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FrenteShapeClassifier(nn.Module):
    """Classifier for FRENTE region shapes"""
    def __init__(self, num_classes, input_size=224):
        super(FrenteShapeClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
        )
        self.feature_size = self._get_feature_size(input_size)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.feature_size, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(inplace=True), 
            nn.Linear(256, num_classes)
        )
    
    def _get_feature_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size, input_size)
            x = self.features(x)
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class RostroMentonShapeClassifier(nn.Module):
    """Classifier for rostro_menton region shapes"""
    def __init__(self, num_classes, input_size=224):
        super(RostroMentonShapeClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),
        )
        self.feature_size = self._get_feature_size(input_size)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.feature_size, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(inplace=True), 
            nn.Linear(256, num_classes)
        )
    
    def _get_feature_size(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size, input_size)
            x = self.features(x)
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EspejoAnalyzer:
    """
    Complete espejo analysis pipeline including:
    - Anthropometric facial analysis
    - Mirror face generation
    - Decision tree classification
    - Hybrid class splitting
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Espejo Analyzer initialized on device: {self.device}")
        
        # Initialize models
        self.facial_points_model = None
        self.binary_model = None
        self.frente_model = None
        self.rostro_model = None
        self.detector = None
        self.predictor = None
        
        # Initialize encoders
        self.binary_encoder = None
        self.frente_encoder = None
        self.rostro_encoder = None
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all required models"""
        try:
            # Load facial points detection model
            self.facial_points_model = self._load_facial_points_model()
            
            # Load classification models
            self._load_classification_models()
            
            # Load dlib models
            self.detector = dlib.get_frontal_face_detector()
            predictor_path = "/app/models/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                self.predictor = dlib.shape_predictor(predictor_path)
            else:
                print(f"Warning: dlib predictor not found at {predictor_path}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _load_facial_points_model(self):
        """Load facial points detection model"""
        try:
            model_path = "/app/models/facial_points_detection_model.pth"
            if not os.path.exists(model_path):
                print(f"Warning: Facial points model not found at {model_path}")
                return None
                
            # Create model
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            num_classes = 14
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            print("✓ Facial points detection model loaded successfully")
            return model
            
        except Exception as e:
            print(f"✗ Error loading facial points model: {e}")
            return None
    
    def _load_classification_models(self):
        """Load all classification models"""
        try:
            # Load binary classification model
            binary_model_path = "/app/models/binary_region_classifier_best.pth"
            if os.path.exists(binary_model_path):
                self.binary_model = BinaryRegionClassifier()
                self.binary_model.load_state_dict(torch.load(binary_model_path, map_location=self.device, weights_only=True))
                self.binary_model.to(self.device)
                self.binary_model.eval()
                
                # Binary encoder
                self.binary_encoder = LabelEncoder()
                self.binary_encoder.classes_ = np.array(['FRENTE', 'rostro_menton'])
                print("✓ Binary classification model loaded successfully")
            
            # Load FRENTE classification model
            frente_model_path = "/app/models/frente_best_model.pth"
            if os.path.exists(frente_model_path):
                self.frente_encoder = LabelEncoder()
                self.frente_encoder.classes_ = np.array([
                    'jupiter_aplio_base_ancha', 'marte_rectangular', 'mercurio_triangulo',
                    'neptuno_ovalo/capsula', 'solar/lunar_redonda', 'tierra_cuadrada',
                    'venus_corazon_o_trapezoide_angosto'
                ])
                
                self.frente_model = FrenteShapeClassifier(num_classes=len(self.frente_encoder.classes_))
                self.frente_model.load_state_dict(torch.load(frente_model_path, map_location=self.device, weights_only=True))
                self.frente_model.to(self.device)
                self.frente_model.eval()
                print("✓ FRENTE classification model loaded successfully")
            
            # Load rostro_menton classification model
            rostro_model_path = "/app/models/rostro_menton_best_model.pth"
            if os.path.exists(rostro_model_path):
                self.rostro_encoder = LabelEncoder()
                self.rostro_encoder.classes_ = np.array([
                    'jupiter/luna_redondo_ancho', 'marte/tierra_rectangulo',
                    'mercurio_triangular', 'pluton-venus', 'pluton_hexagonal',
                    'saturno_trapezoide_base_angosta', 'sol_neptuno_ovalo', 'venus_corazon'
                ])
                
                self.rostro_model = RostroMentonShapeClassifier(num_classes=len(self.rostro_encoder.classes_))
                self.rostro_model.load_state_dict(torch.load(rostro_model_path, map_location=self.device, weights_only=True))
                self.rostro_model.to(self.device)
                self.rostro_model.eval()
                print("✓ rostro_menton classification model loaded successfully")
                
        except Exception as e:
            print(f"✗ Error loading classification models: {e}")
    
    def _preprocess_image(self, img):
        """Preprocess image for analysis"""
        img_resized = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        return gray, img_resized
    
    def _detect_faces(self, gray_img):
        """Detect faces in image"""
        if self.detector is None:
            return []
        return self.detector(gray_img)
    
    def _detect_landmarks(self, gray_img, face):
        """Detect facial landmarks"""
        if self.predictor is None:
            return None
        return self.predictor(gray_img, face)
    
    def _predict_facial_points(self, image, confidence_threshold=0.5):
        """Predict facial points using custom model"""
        if self.facial_points_model is None:
            return {}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.facial_points_model(image_tensor)
        
        detected_points = {}
        
        if len(predictions) > 0:
            prediction = predictions[0]
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    center_x = center_x * (width / 224)
                    center_y = center_y * (height / 224)
                    
                    detected_points[label] = (int(center_x), int(center_y))
        
        return detected_points
    
    def _calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        """Calculate intersection of two lines"""
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        intersection_x = x1 + t*(x2-x1)
        intersection_y = y1 + t*(y2-y1)
        
        return (int(intersection_x), int(intersection_y))
    
    def _calculate_face_proportions(self, landmarks, custom_model_points):
        """Calculate face proportions"""
        if 2 not in custom_model_points:
            return None, None, None
        
        dlib_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        
        point_m2 = custom_model_points[2]
        point_dlib_8 = tuple(dlib_points[8])
        point_dlib_1 = tuple(dlib_points[1])
        point_dlib_15 = tuple(dlib_points[15])
        
        intersection_point = self._line_intersection(point_m2, point_dlib_8, point_dlib_1, point_dlib_15)
        
        if intersection_point is None:
            return None, None, None
        
        distance_m2_to_8 = self._calculate_distance(point_m2, point_dlib_8)
        distance_1_to_intersection = self._calculate_distance(point_dlib_1, intersection_point)
        distance_15_to_intersection = self._calculate_distance(point_dlib_15, intersection_point)
        
        right_face_proportion = distance_m2_to_8 / (distance_1_to_intersection * 2) if distance_1_to_intersection > 0 else 0
        left_face_proportion = distance_m2_to_8 / (distance_15_to_intersection * 2) if distance_15_to_intersection > 0 else 0
        
        return right_face_proportion, left_face_proportion, intersection_point
    
    def _calculate_forehead_proportions(self, custom_model_points):
        """Calculate forehead proportions"""
        right_forehead = None
        left_forehead = None
        
        required_points_right = [2, 3, 13]
        required_points_left = [2, 3, 8]
        
        if all(point in custom_model_points for point in required_points_right):
            point_m3 = custom_model_points[3]
            point_m2 = custom_model_points[2]
            point_m13 = custom_model_points[13]
            
            distance_m3_m2 = self._calculate_distance(point_m3, point_m2)
            distance_m13_m2 = self._calculate_distance(point_m13, point_m2)
            
            right_forehead = distance_m3_m2 / (distance_m13_m2 * 2) if distance_m13_m2 > 0 else 0
        
        if all(point in custom_model_points for point in required_points_left):
            point_m3 = custom_model_points[3]
            point_m2 = custom_model_points[2]
            point_m8 = custom_model_points[8]
            
            distance_m3_m2 = self._calculate_distance(point_m3, point_m2)
            distance_m2_m8 = self._calculate_distance(point_m2, point_m8)
            
            left_forehead = distance_m3_m2 / (distance_m2_m8 * 2) if distance_m2_m8 > 0 else 0
        
        return right_forehead, left_forehead
    
    def _calculate_temporal_proportions(self, landmarks, custom_model_points):
        """Calculate temporal proportions"""
        if 2 not in custom_model_points:
            return None, None, None
        
        dlib_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        
        point_m2 = custom_model_points[2]
        point_dlib_8 = tuple(dlib_points[8])
        point_dlib_0 = tuple(dlib_points[0])
        point_dlib_16 = tuple(dlib_points[16])
        
        intersection_point = self._line_intersection(point_m2, point_dlib_8, point_dlib_0, point_dlib_16)
        
        if intersection_point is None:
            return None, None, None
        
        distance_m2_to_8 = self._calculate_distance(point_m2, point_dlib_8)
        distance_0_to_intersection = self._calculate_distance(point_dlib_0, intersection_point)
        distance_16_to_intersection = self._calculate_distance(point_dlib_16, intersection_point)
        
        temporal_right = distance_m2_to_8 / (distance_0_to_intersection * 2) if distance_0_to_intersection > 0 else 0
        temporal_left = distance_m2_to_8 / (distance_16_to_intersection * 2) if distance_16_to_intersection > 0 else 0
        
        return temporal_right, temporal_left, intersection_point
    
    def _get_midline_x(self, landmarks):
        """Get midline x coordinate"""
        return landmarks.part(28).x
    
    def _rotate_image(self, img, angle, center=None):
        """Rotate image by angle"""
        if center is None:
            center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return rotated_img
    
    def _align_face_to_x_axis(self, landmarks, img):
        """Align face to x-axis"""
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        rotated_img = self._rotate_image(img, angle)
        return rotated_img
    
    def _create_mirrored_face(self, img, midline_x, side="right"):
        """Create mirrored face"""
        if side == "right":
            right_half = img[:, midline_x:]
            mirrored_right = np.flip(right_half, axis=1)
            mirrored_face = np.hstack((mirrored_right, right_half))
        elif side == "left":
            left_half = img[:, :midline_x]
            mirrored_left = np.flip(left_half, axis=1)
            mirrored_face = np.hstack((left_half, mirrored_left))
        return mirrored_face
    
    def _preprocess_image_for_classification(self, image, target_size=224):
        """Preprocess image for classification"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize while preserving aspect ratio
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create square canvas
        square_image = Image.new('RGB', (target_size, target_size), color=(0, 0, 0))
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        square_image.paste(image, (paste_x, paste_y))
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(square_image)
    
    def _apply_frente_decision_tree(self, predictions, probabilities, face_proportion):
        """Apply decision tree rules for FRENTE region"""
        applied_rules = []
        pred_dict = {pred: prob for pred, prob in zip(predictions, probabilities)}
        
        # Apply exclusion rules
        excluded_preds = []
        
        # Always exclude mercurio_triangulo
        if 'mercurio_triangulo' in pred_dict:
            excluded_preds.append('mercurio_triangulo')
            applied_rules.append("mercurio_triangulo always excluded")
        
        # Confidence-based exclusions
        exclusion_rules = {
            'solar/lunar_redonda': 0.19,
            'neptuno_ovalo/capsula': 0.20,
            'jupiter_aplio_base_ancha': 0.20,
            'venus_corazon_o_trapezoide_angosto': 0.50
        }
        
        for pred, threshold in exclusion_rules.items():
            if pred in pred_dict and pred_dict[pred] < threshold:
                excluded_preds.append(pred)
                applied_rules.append(f"Excluded {pred} (confidence {pred_dict[pred]:.1%} < {threshold:.1%})")
        
        # Apply proportion-based rules
        if 'neptuno_ovalo/capsula' in pred_dict and 'neptuno_ovalo/capsula' not in excluded_preds:
            if face_proportion < 0.25:
                excluded_preds.append('neptuno_ovalo/capsula')
                applied_rules.append(f"neptuno + proportion {face_proportion:.3f} < 0.25 → excluded")
            elif face_proportion <= 0.4:
                excluded_preds.append('neptuno_ovalo/capsula')
                applied_rules.append(f"neptuno + proportion {face_proportion:.3f} ≤ 0.4 → excluded")
        
        # Return highest confidence remaining prediction
        remaining_preds = {pred: prob for pred, prob in pred_dict.items() if pred not in excluded_preds}
        
        if not remaining_preds:
            applied_rules.append("All predictions excluded, returning highest confidence")
            return predictions[0], applied_rules
        
        top_remaining = max(remaining_preds.items(), key=lambda x: x[1])
        top_pred, top_prob = top_remaining
        
        applied_rules.append(f"Returning highest confidence remaining prediction: {top_pred}")
        return top_pred, applied_rules
    
    def _apply_rostro_menton_decision_tree(self, predictions, probabilities, face_proportion):
        """Apply decision tree rules for rostro_menton region"""
        applied_rules = []
        pred_dict = {pred: prob for pred, prob in zip(predictions, probabilities)}
        
        # Apply exclusion rules
        excluded_preds = []
        
        exclusion_rules = {
            'venus_corazon': 0.15,
            'pluton-venus': 0.10,
            'pluton_hexagonal': 0.17,
            'jupiter/luna_redondo_ancho': 0.10,
            'saturno_trapezoide_base_angosta': 0.17,
            'mercurio_triangular': 0.19,
            'sol_neptuno_ovalo': 0.11
        }
        
        for pred, threshold in exclusion_rules.items():
            if pred in pred_dict and pred_dict[pred] < threshold:
                excluded_preds.append(pred)
                applied_rules.append(f"Excluded {pred} (confidence {pred_dict[pred]:.1%} < {threshold:.1%})")
        
        # Apply complex logic rules
        if all(pred in pred_dict for pred in ['venus_corazon', 'pluton-venus', 'pluton_hexagonal']):
            applied_rules.append("All three (venus_corazon, pluton-venus, pluton_hexagonal) present → choosing pluton-venus")
            return 'pluton-venus', applied_rules
        
        if 'pluton-venus' in pred_dict and 'venus_corazon' in pred_dict:
            if pred_dict['venus_corazon'] > pred_dict['pluton-venus'] + 0.06:
                applied_rules.append(f"venus_corazon ({pred_dict['venus_corazon']:.1%}) > pluton-venus ({pred_dict['pluton-venus']:.1%}) by >6% → choosing venus_corazon")
                return 'venus_corazon', applied_rules
        
        # Get remaining predictions after exclusions
        remaining_preds = {pred: prob for pred, prob in pred_dict.items() if pred not in excluded_preds}
        
        if not remaining_preds:
            applied_rules.append("All predictions excluded, returning highest confidence")
            return predictions[0], applied_rules
        
        # Apply anthropometric rules
        if 'sol_neptuno_ovalo' in remaining_preds:
            if face_proportion >= 1.17:
                applied_rules.append(f"sol_neptuno_ovalo + proportion {face_proportion:.3f} ≥ 1.17 → neptuno")
                return 'neptuno', applied_rules
            elif face_proportion < 1.0:
                applied_rules.append(f"sol_neptuno_ovalo + proportion {face_proportion:.3f} < 1.0 → jupiter")
                return 'jupiter', applied_rules
            else:
                applied_rules.append(f"sol_neptuno_ovalo + proportion {face_proportion:.3f} < 1.17 → sol")
                return 'sol', applied_rules
        
        # Find highest confidence remaining prediction
        top_remaining = max(remaining_preds.items(), key=lambda x: x[1])
        top_pred, top_prob = top_remaining
        
        applied_rules.append(f"Returning highest confidence remaining prediction: {top_pred}")
        return top_pred, applied_rules
    
    def _apply_hybrid_class_splitting(self, final_diagnosis, face_proportion, forehead_proportion, region_type):
        """Apply hybrid class splitting"""
        applied_rules = []
        
        if region_type == 'FRENTE':
            # Check for solar/lunar_redonda specifically
            if final_diagnosis.lower() == 'solar/lunar_redonda':
                if forehead_proportion is not None:
                    if forehead_proportion < 0.35:
                        applied_rules.append(f"solar/lunar_redonda + forehead proportion {forehead_proportion:.3f} < 0.35 → luna")
                        return 'luna', applied_rules
                    else:
                        applied_rules.append(f"solar/lunar_redonda + forehead proportion {forehead_proportion:.3f} ≥ 0.35 → solar")
                        return 'solar', applied_rules
                else:
                    applied_rules.append("solar/lunar_redonda detected but forehead proportion N/A → no splitting")
                    return final_diagnosis, applied_rules
            
            applied_rules.append("No hybrid splitting needed for FRENTE")
            return final_diagnosis, applied_rules
        
        elif region_type == 'rostro_menton':
            # Check for jupiter/luna_redondo_ancho specifically
            if final_diagnosis.lower() == 'jupiter/luna_redondo_ancho':
                if face_proportion is not None:
                    if face_proportion >= 1.17:
                        applied_rules.append(f"jupiter/luna_redondo_ancho + face proportion {face_proportion:.3f} ≥ 1.17 → neptuno")
                        return 'neptuno', applied_rules
                    elif face_proportion < 0.99:
                        applied_rules.append(f"jupiter/luna_redondo_ancho + face proportion {face_proportion:.3f} < 0.99 → luna")
                        return 'luna', applied_rules
                    else:
                        applied_rules.append(f"jupiter/luna_redondo_ancho + face proportion {face_proportion:.3f} between 0.99-1.17 → jupiter")
                        return 'jupiter', applied_rules
                else:
                    applied_rules.append("jupiter/luna_redondo_ancho detected but face proportion N/A → no splitting")
                    return final_diagnosis, applied_rules
            
            # Check for sol_neptuno_ovalo specifically
            if final_diagnosis.lower() == 'sol_neptuno_ovalo':
                if face_proportion is not None:
                    if face_proportion >= 1.17:
                        applied_rules.append(f"sol_neptuno_ovalo + face proportion {face_proportion:.3f} ≥ 1.17 → neptuno")
                        return 'neptuno', applied_rules
                    elif face_proportion < 1.0:
                        applied_rules.append(f"sol_neptuno_ovalo + face proportion {face_proportion:.3f} < 1.0 → jupiter")
                        return 'jupiter', applied_rules
                    else:
                        applied_rules.append(f"sol_neptuno_ovalo + face proportion {face_proportion:.3f} between 1.0-1.17 → sol")
                        return 'sol', applied_rules
                else:
                    applied_rules.append("sol_neptuno_ovalo detected but face proportion N/A → no splitting")
                    return final_diagnosis, applied_rules
            
            applied_rules.append("No hybrid splitting needed for rostro_menton")
            return final_diagnosis, applied_rules
        
        return final_diagnosis, applied_rules
    
    def _classify_mirror_images(self, right_mirrored_face, left_mirrored_face, right_face_prop, left_face_prop, right_forehead_prop, left_forehead_prop):
        """Classify mirror images with decision tree and hybrid splitting"""
        results = {
            'right_mirrored': {
                'frente_predictions': [], 'frente_probabilities': [],
                'rostro_predictions': [], 'rostro_probabilities': [],
                'frente_final_diagnosis': None, 'frente_applied_rules': [],
                'rostro_final_diagnosis': None, 'rostro_applied_rules': [],
                'frente_split_diagnosis': None, 'frente_split_rules': [],
                'rostro_split_diagnosis': None, 'rostro_split_rules': []
            },
            'left_mirrored': {
                'frente_predictions': [], 'frente_probabilities': [],
                'rostro_predictions': [], 'rostro_probabilities': [],
                'frente_final_diagnosis': None, 'frente_applied_rules': [],
                'rostro_final_diagnosis': None, 'rostro_applied_rules': [],
                'frente_split_diagnosis': None, 'frente_split_rules': [],
                'rostro_split_diagnosis': None, 'rostro_split_rules': []
            }
        }
        
        if self.frente_model is None or self.rostro_model is None:
            return results
        
        for side, image in [('right_mirrored', right_mirrored_face), ('left_mirrored', left_mirrored_face)]:
            try:
                # Calculate face proportion for decision tree
                face_proportion = image.shape[0] / image.shape[1] if image.shape[1] > 0 else 1.0
                
                # Get correct proportions for this side
                current_face_prop = right_face_prop if side == 'right_mirrored' else left_face_prop
                current_forehead_prop = right_forehead_prop if side == 'right_mirrored' else left_forehead_prop
                
                # Preprocess image
                image_tensor = self._preprocess_image_for_classification(image)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # FRENTE classification
                with torch.no_grad():
                    frente_output = self.frente_model(image_tensor)
                    frente_prob = torch.softmax(frente_output, dim=1)
                    frente_top3_probs, frente_top3_indices = torch.topk(frente_prob[0], min(3, len(self.frente_encoder.classes_)))
                    
                    frente_predictions = []
                    frente_probabilities = []
                    
                    for i in range(len(frente_top3_indices)):
                        class_name = self.frente_encoder.classes_[frente_top3_indices[i].item()]
                        probability = frente_top3_probs[i].item()
                        frente_predictions.append(class_name)
                        frente_probabilities.append(probability)
                    
                    results[side]['frente_predictions'] = frente_predictions
                    results[side]['frente_probabilities'] = frente_probabilities
                    
                    # Apply decision tree to FRENTE
                    frente_diagnosis, frente_rules = self._apply_frente_decision_tree(
                        frente_predictions, frente_probabilities, face_proportion
                    )
                    results[side]['frente_final_diagnosis'] = frente_diagnosis
                    results[side]['frente_applied_rules'] = frente_rules
                    
                    # Apply hybrid class splitting to FRENTE
                    frente_split_diagnosis, frente_split_rules = self._apply_hybrid_class_splitting(
                        frente_diagnosis, current_face_prop, current_forehead_prop, 'FRENTE'
                    )
                    results[side]['frente_split_diagnosis'] = frente_split_diagnosis
                    results[side]['frente_split_rules'] = frente_split_rules
                    
                    # rostro_menton classification
                    rostro_output = self.rostro_model(image_tensor)
                    rostro_prob = torch.softmax(rostro_output, dim=1)
                    rostro_top3_probs, rostro_top3_indices = torch.topk(rostro_prob[0], min(3, len(self.rostro_encoder.classes_)))
                    
                    rostro_predictions = []
                    rostro_probabilities = []
                    
                    for i in range(len(rostro_top3_indices)):
                        class_name = self.rostro_encoder.classes_[rostro_top3_indices[i].item()]
                        probability = rostro_top3_probs[i].item()
                        rostro_predictions.append(class_name)
                        rostro_probabilities.append(probability)
                    
                    results[side]['rostro_predictions'] = rostro_predictions
                    results[side]['rostro_probabilities'] = rostro_probabilities
                    
                    # Apply decision tree to rostro_menton
                    rostro_diagnosis, rostro_rules = self._apply_rostro_menton_decision_tree(
                        rostro_predictions, rostro_probabilities, current_face_prop
                    )
                    results[side]['rostro_final_diagnosis'] = rostro_diagnosis
                    results[side]['rostro_applied_rules'] = rostro_rules
                    
                    # Apply hybrid class splitting to rostro_menton
                    rostro_split_diagnosis, rostro_split_rules = self._apply_hybrid_class_splitting(
                        rostro_diagnosis, current_face_prop, current_forehead_prop, 'rostro_menton'
                    )
                    results[side]['rostro_split_diagnosis'] = rostro_split_diagnosis
                    results[side]['rostro_split_rules'] = rostro_split_rules
                    
            except Exception as e:
                print(f"Error classifying {side}: {e}")
        
        return results
    
    def analyze_complete(self, image_array, confidence_threshold=0.5):
        """Complete espejo analysis"""
        try:
            # Preprocess image
            gray, img_resized = self._preprocess_image(image_array)
            
            # Detect faces
            faces = self._detect_faces(gray)
            if len(faces) == 0:
                return None
            
            # Use first face
            face = faces[0]
            landmarks = self._detect_landmarks(gray, face)
            if landmarks is None:
                return None
            
            # Get custom model predictions
            custom_model_points = self._predict_facial_points(img_resized, confidence_threshold)
            
            # Calculate proportions
            right_face_prop, left_face_prop, face_intersection_point = self._calculate_face_proportions(landmarks, custom_model_points)
            right_forehead_prop, left_forehead_prop = self._calculate_forehead_proportions(custom_model_points)
            temporal_right_prop, temporal_left_prop, temporal_intersection_point = self._calculate_temporal_proportions(landmarks, custom_model_points)
            
            # Generate mirror images
            img_aligned = self._align_face_to_x_axis(landmarks, img_resized)
            midline_x = self._get_midline_x(landmarks)
            
            right_mirrored_face = self._create_mirrored_face(img_aligned, midline_x, side="right")
            left_mirrored_face = self._create_mirrored_face(img_aligned, midline_x, side="left")
            
            # Classify mirror images
            classification_results = self._classify_mirror_images(
                right_mirrored_face, left_mirrored_face, 
                right_face_prop, left_face_prop, 
                right_forehead_prop, left_forehead_prop
            )
            
            # Prepare results
            results = {
                'landmarks_count': 68,
                'custom_model_points': custom_model_points,
                'proportions': {
                    'face_proportions': {
                        'right': right_face_prop,
                        'left': left_face_prop,
                        'intersection_point': face_intersection_point
                    },
                    'forehead_proportions': {
                        'right': right_forehead_prop,
                        'left': left_forehead_prop
                    },
                    'temporal_proportions': {
                        'right': temporal_right_prop,
                        'left': temporal_left_prop,
                        'intersection_point': temporal_intersection_point
                    }
                },
                'mirror_images': {
                    'right_mirrored': right_mirrored_face,
                    'left_mirrored': left_mirrored_face,
                    'aligned_face': img_aligned,
                    'midline_x': midline_x
                },
                'classification_results': classification_results,
                'analysis_summary': {
                    'face_detected': True,
                    'landmarks_detected': 68,
                    'custom_points_detected': len(custom_model_points),
                    'mirror_images_generated': True,
                    'classification_completed': True,
                    'decision_tree_applied': True,
                    'hybrid_splitting_applied': True
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error in complete analysis: {e}")
            return None
    
    def generate_mirror_images(self, image_array, confidence_threshold=0.5):
        """Generate mirror images only"""
        try:
            gray, img_resized = self._preprocess_image(image_array)
            faces = self._detect_faces(gray)
            
            if len(faces) == 0:
                return None
            
            face = faces[0]
            landmarks = self._detect_landmarks(gray, face)
            if landmarks is None:
                return None
            
            custom_model_points = self._predict_facial_points(img_resized, confidence_threshold)
            
            img_aligned = self._align_face_to_x_axis(landmarks, img_resized)
            midline_x = self._get_midline_x(landmarks)
            
            right_mirrored_face = self._create_mirrored_face(img_aligned, midline_x, side="right")
            left_mirrored_face = self._create_mirrored_face(img_aligned, midline_x, side="left")
            
            return {
                'landmarks_count': 68,
                'custom_model_points': custom_model_points,
                'face_aligned': True,
                'right_mirrored': right_mirrored_face,
                'left_mirrored': left_mirrored_face,
                'midline_x': midline_x
            }
            
        except Exception as e:
            print(f"Error generating mirror images: {e}")
            return None
    
    def classify_regions(self, image_array, confidence_threshold=0.5):
        """Classify regions only"""
        try:
            complete_results = self.analyze_complete(image_array, confidence_threshold)
            if complete_results is None:
                return None
            
            return {
                'classification_results': complete_results['classification_results'],
                'proportions': complete_results['proportions']
            }
            
        except Exception as e:
            print(f"Error classifying regions: {e}")
            return None
    
    def analyze_proportions(self, image_array, confidence_threshold=0.5):
        """Analyze proportions only"""
        try:
            gray, img_resized = self._preprocess_image(image_array)
            faces = self._detect_faces(gray)
            
            if len(faces) == 0:
                return None
            
            face = faces[0]
            landmarks = self._detect_landmarks(gray, face)
            if landmarks is None:
                return None
            
            custom_model_points = self._predict_facial_points(img_resized, confidence_threshold)
            
            right_face_prop, left_face_prop, face_intersection_point = self._calculate_face_proportions(landmarks, custom_model_points)
            right_forehead_prop, left_forehead_prop = self._calculate_forehead_proportions(custom_model_points)
            temporal_right_prop, temporal_left_prop, temporal_intersection_point = self._calculate_temporal_proportions(landmarks, custom_model_points)
            
            return {
                'landmarks_count': 68,
                'custom_model_points': custom_model_points,
                'proportions': {
                    'face_proportions': {
                        'right': right_face_prop,
                        'left': left_face_prop,
                        'intersection_point': face_intersection_point
                    },
                    'forehead_proportions': {
                        'right': right_forehead_prop,
                        'left': left_forehead_prop
                    },
                    'temporal_proportions': {
                        'right': temporal_right_prop,
                        'left': temporal_left_prop,
                        'intersection_point': temporal_intersection_point
                    }
                }
            }
            
        except Exception as e:
            print(f"Error analyzing proportions: {e}")
            return None
    
    def get_diagnosis(self, image_array, confidence_threshold=0.5):
        """Get diagnosis only"""
        try:
            complete_results = self.analyze_complete(image_array, confidence_threshold)
            if complete_results is None:
                return None
            
            return {
                'final_diagnosis': {
                    'right_side': {
                        'frente': complete_results['classification_results']['right_mirrored']['frente_split_diagnosis'],
                        'rostro': complete_results['classification_results']['right_mirrored']['rostro_split_diagnosis']
                    },
                    'left_side': {
                        'frente': complete_results['classification_results']['left_mirrored']['frente_split_diagnosis'],
                        'rostro': complete_results['classification_results']['left_mirrored']['rostro_split_diagnosis']
                    }
                },
                'decision_tree_analysis': complete_results['classification_results'],
                'confidence_scores': {
                    'right_side': {
                        'frente': complete_results['classification_results']['right_mirrored']['frente_probabilities'],
                        'rostro': complete_results['classification_results']['right_mirrored']['rostro_probabilities']
                    },
                    'left_side': {
                        'frente': complete_results['classification_results']['left_mirrored']['frente_probabilities'],
                        'rostro': complete_results['classification_results']['left_mirrored']['rostro_probabilities']
                    }
                },
                'applied_rules': complete_results['classification_results'],
                'proportions_used': complete_results['proportions']
            }
            
        except Exception as e:
            print(f"Error getting diagnosis: {e}")
            return None
    
    def generate_text_report(self, diagnosis_results):
        """Generate text report"""
        try:
            report = "ESPEJO ANALYSIS REPORT\n"
            report += "=" * 50 + "\n\n"
            
            report += "FINAL DIAGNOSIS:\n"
            report += f"Right Side - FRENTE: {diagnosis_results['final_diagnosis']['right_side']['frente']}\n"
            report += f"Right Side - ROSTRO: {diagnosis_results['final_diagnosis']['right_side']['rostro']}\n"
            report += f"Left Side - FRENTE: {diagnosis_results['final_diagnosis']['left_side']['frente']}\n"
            report += f"Left Side - ROSTRO: {diagnosis_results['final_diagnosis']['left_side']['rostro']}\n\n"
            
            report += "DECISION TREE RULES APPLIED:\n"
            
            # Add right side rules
            right_rules = diagnosis_results['applied_rules']['right_mirrored']
            report += "\nRight Side FRENTE Rules:\n"
            for rule in right_rules.get('frente_applied_rules', []):
                report += f"  - {rule}\n"
            
            report += "\nRight Side ROSTRO Rules:\n"
            for rule in right_rules.get('rostro_applied_rules', []):
                report += f"  - {rule}\n"
            
            # Add left side rules
            left_rules = diagnosis_results['applied_rules']['left_mirrored']
            report += "\nLeft Side FRENTE Rules:\n"
            for rule in left_rules.get('frente_applied_rules', []):
                report += f"  - {rule}\n"
            
            report += "\nLeft Side ROSTRO Rules:\n"
            for rule in left_rules.get('rostro_applied_rules', []):
                report += f"  - {rule}\n"
            
            return report
            
        except Exception as e:
            return f"Error generating text report: {e}"