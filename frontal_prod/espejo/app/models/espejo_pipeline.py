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
import timm  # For EfficientNet

class FasterRCNNDetector:
    """Faster R-CNN detector for FRENTE and rostro_menton regions"""
    def __init__(self, model_path, device):
        self.device = device
        self.model = None
        self.detection_classes = ['rostro_menton', 'FRENTE']
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained Faster R-CNN model"""
        try:
            if os.path.exists(model_path):
                # Load model checkpoint
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                # Create Faster R-CNN model
                num_classes = len(self.detection_classes) + 1  # +1 for background
                self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

                # Replace the classifier head
                in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

                # Load trained weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()

                print("✓ Faster R-CNN detection model loaded successfully")
            else:
                print(f"✗ Faster R-CNN model not found at {model_path}")
        except Exception as e:
            print(f"✗ Error loading Faster R-CNN model: {e}")
            self.model = None

    def detect_regions(self, image, confidence_threshold=0.5):
        """Detect FRENTE and rostro_menton regions in image"""
        if self.model is None:
            return []

        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)

            orig_height, orig_width = image_rgb.shape[:2]

            # Resize for detection (maintain aspect ratio)
            target_size = 416
            scale = min(target_size / orig_width, target_size / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)

            resized_image = cv2.resize(image_rgb, (new_width, new_height))

            # Pad to target size
            padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            padded_image[:new_height, :new_width] = resized_image

            # Convert to tensor
            image_tensor = torch.FloatTensor(padded_image / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Run detection
            with torch.no_grad():
                predictions = self.model(image_tensor)

            # Extract predictions
            prediction = predictions[0]
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            # Scale boxes back to original image size
            detections = []
            for box, label, score in zip(boxes, labels, scores):
                if score > confidence_threshold and label > 0:  # Skip background class
                    # Scale bbox back to original image size
                    x1, y1, x2, y2 = box

                    # Adjust for padding and scaling
                    x1 = max(0, x1 / scale)
                    y1 = max(0, y1 / scale)
                    x2 = min(orig_width, x2 / scale)
                    y2 = min(orig_height, y2 / scale)

                    # Ensure valid bbox
                    if x2 > x1 and y2 > y1:
                        class_name = self.detection_classes[label - 1]  # -1 for background
                        detections.append({
                            'class': class_name,
                            'confidence': float(score),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })

            return detections

        except Exception as e:
            print(f"Error in region detection: {e}")
            return []

class FrenteShapeClassifier(nn.Module):
    """EfficientNet-B3 based classifier for FRENTE region shapes"""
    def __init__(self, num_classes):
        super(FrenteShapeClassifier, self).__init__()

        # Load pretrained EfficientNet-B3 from timm
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True)

        # Get the actual number of features from the classifier
        backbone_features = self.backbone.classifier.in_features

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Enhanced classifier matching the training architecture
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(backbone_features),
            nn.Dropout(0.4),

            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features from EfficientNet backbone
        features = self.backbone(x)

        # Classification
        output = self.classifier(features)
        return output

class RostroMentonShapeClassifier(nn.Module):
    """Enhanced ResNet-based classifier for rostro_menton region shapes"""
    def __init__(self, num_classes):
        super(RostroMentonShapeClassifier, self).__init__()

        # Load pretrained ResNet-50
        from torchvision.models import resnet50, ResNet50_Weights
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove original classifier
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add batch normalization after features (anti-overfitting)
        self.bn_features = nn.BatchNorm1d(2048)

        # HEAVILY REGULARIZED classifier with progressive dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.7),  # Strong dropout at input

            # First layer - larger to capture more features
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),  # Medium dropout

            # Second layer - intermediate
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),  # Light dropout

            # Final layer
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features using ResNet-50 backbone
        features = self.features(x)

        # Apply feature batch normalization (anti-overfitting)
        features = features.view(features.size(0), -1)  # Flatten
        features = self.bn_features(features)

        # Classification with heavy regularization
        output = self.classifier(features)

        return output

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
        self.detection_model = None
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
            # Load Faster R-CNN detection model
            detection_model_path = "/app/models/faster_rcnn_detection_model.pth"
            if os.path.exists(detection_model_path):
                self.detection_model = FasterRCNNDetector(detection_model_path, self.device)
            else:
                print(f"✗ Detection model not found at {detection_model_path}")
            
            # Load FRENTE classification model
            frente_model_path = "/app/models/frente_enhanced_classifier_3datasets.pth"
            if os.path.exists(frente_model_path):
                self.frente_encoder = LabelEncoder()
                self.frente_encoder.classes_ = np.array([
                    'neptuno_combined', 'solar_lunar_combined', 'mercurio_triangulo',
                    'marte_rectangular', 'tierra_cuadrada', 'jupiter_aplio_base_ancha',
                    'venus_corazon_o_trapezoide_angosto'
                ])
                
                self.frente_model = FrenteShapeClassifier(num_classes=len(self.frente_encoder.classes_))
                self.frente_model.load_state_dict(torch.load(frente_model_path, map_location=self.device, weights_only=True))
                self.frente_model.to(self.device)
                self.frente_model.eval()
                print("✓ FRENTE classification model loaded successfully")
            
            # Load rostro_menton classification model
            rostro_model_path = "/app/models/rostro_classifier_mono.pth"
            if os.path.exists(rostro_model_path):
                self.rostro_encoder = LabelEncoder()
                self.rostro_encoder.classes_ = np.array([
                    'sol_neptuno_combined', 'luna_jupiter_combined', 'venus_corazon',
                    'pluton_hexagonal', 'mercurio_triangular', 'marte_tierra_rectangulo',
                    'saturno_trapezoide_base_angosta'
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
                    
                    detected_points[int(label)] = (int(center_x), int(center_y))
        
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
    
    def _detect_facial_regions(self, image, confidence_threshold=0.5):
        """Detect FRENTE and rostro_menton regions using Faster R-CNN"""
        if self.detection_model is None:
            return {'frente': None, 'rostro_menton': None}

        try:
            detections = self.detection_model.detect_regions(image, confidence_threshold)

            regions = {'frente': None, 'rostro_menton': None}

            # Group detections by class and select best confidence
            for detection in detections:
                class_name = detection['class']
                if class_name == 'FRENTE':
                    if regions['frente'] is None or detection['confidence'] > regions['frente']['confidence']:
                        regions['frente'] = detection
                elif class_name == 'rostro_menton':
                    if regions['rostro_menton'] is None or detection['confidence'] > regions['rostro_menton']['confidence']:
                        regions['rostro_menton'] = detection

            return regions

        except Exception as e:
            print(f"Error detecting facial regions: {e}")
            return {'frente': None, 'rostro_menton': None}

    def _extract_region_from_image(self, image, bbox, padding=10):
        """Extract region from image using bounding box with padding"""
        try:
            if bbox is None:
                return None

            x1, y1, x2, y2 = bbox

            # Add padding
            height, width = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            # Extract region
            region = image[y1:y2, x1:x2]

            # Ensure minimum size
            if region.shape[0] < 50 or region.shape[1] < 50:
                return None

            return region

        except Exception as e:
            print(f"Error extracting region: {e}")
            return None

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
            'solar_lunar_combined': 0.19,
            'neptuno_combined': 0.20,
            'jupiter_aplio_base_ancha': 0.20,
            'venus_corazon_o_trapezoide_angosto': 0.50
        }
        
        for pred, threshold in exclusion_rules.items():
            if pred in pred_dict and pred_dict[pred] < threshold:
                excluded_preds.append(pred)
                applied_rules.append(f"Excluded {pred} (confidence {pred_dict[pred]:.1%} < {threshold:.1%})")
        
        # Apply proportion-based rules
        if 'neptuno_combined' in pred_dict and 'neptuno_combined' not in excluded_preds:
            if face_proportion < 0.25:
                excluded_preds.append('neptuno_combined')
                applied_rules.append(f"neptuno + proportion {face_proportion:.3f} < 0.25 → excluded")
            elif face_proportion <= 0.4:
                excluded_preds.append('neptuno_combined')
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
        """Apply decision tree rules for rostro_menton region with updated calibration"""
        applied_rules = []
        pred_dict = {pred: prob for pred, prob in zip(predictions, probabilities)}

        # Check for solo diagnosis rules first (high confidence = exclusive diagnosis)
        solo_diagnosis_rules = {
            'saturno_trapezoide_base_angosta': 0.60,
            'venus_corazon': 0.65,
            'luna_jupiter_combined': 0.10,
            'mercurio_triangular': 0.35,
            'pluton_hexagonal': 0.45,
            'marte_tierra_rectangulo': 0.88,
            'sol_neptuno_combined': 0.10
        }

        for pred, threshold in solo_diagnosis_rules.items():
            if pred in pred_dict and pred_dict[pred] >= threshold:
                applied_rules.append(f"Solo diagnosis: {pred} (confidence {pred_dict[pred]:.1%} ≥ {threshold:.1%})")

                # Apply proportion-based splitting for solo diagnosis
                final_diagnosis = self._apply_proportion_based_splitting(pred, face_proportion, applied_rules)
                return final_diagnosis, applied_rules

        # Apply exclusion rules (updated thresholds)
        excluded_preds = []

        exclusion_rules = {
            'venus_corazon': 0.35,
            'pluton_hexagonal': 0.15,
            'luna_jupiter_combined': 0.03,
            'saturno_trapezoide_base_angosta': 0.23,
            'mercurio_triangular': 0.17,
            'sol_neptuno_combined': 0.04,
            'marte_tierra_rectangulo': 0.30
        }

        for pred, threshold in exclusion_rules.items():
            if pred in pred_dict and pred_dict[pred] < threshold:
                excluded_preds.append(pred)
                applied_rules.append(f"Excluded {pred} (confidence {pred_dict[pred]:.1%} < {threshold:.1%})")

        # Get remaining predictions after exclusions
        remaining_preds = {pred: prob for pred, prob in pred_dict.items() if pred not in excluded_preds}

        if not remaining_preds:
            applied_rules.append("All predictions excluded, returning highest confidence")
            final_diagnosis = self._apply_proportion_based_splitting(predictions[0], face_proportion, applied_rules)
            return final_diagnosis, applied_rules

        # Apply inclusive criteria - if multiple diagnoses meet thresholds, use highest confidence
        # But first check for proportion-based overrides
        top_remaining = max(remaining_preds.items(), key=lambda x: x[1])
        top_pred, top_prob = top_remaining

        # Apply proportion-based splitting
        final_diagnosis = self._apply_proportion_based_splitting(top_pred, face_proportion, applied_rules)

        applied_rules.append(f"Final diagnosis after decision tree: {final_diagnosis}")
        return final_diagnosis, applied_rules

    def _apply_proportion_based_splitting(self, prediction, face_proportion, applied_rules):
        """Apply proportion-based diagnosis splitting based on face height/width ratio"""
        if face_proportion is None:
            applied_rules.append(f"No proportion available, keeping original diagnosis: {prediction}")
            return prediction

        # Sol_neptuno_combined proportion rules
        if prediction == 'sol_neptuno_combined':
            if face_proportion >= 1.17:
                applied_rules.append(f"sol_neptuno_combined + proportion {face_proportion:.3f} ≥ 1.17 → neptuno")
                return 'neptuno'
            elif face_proportion < 1.0:
                applied_rules.append(f"sol_neptuno_combined + proportion {face_proportion:.3f} < 1.0 → jupiter")
                return 'jupiter'
            else:
                applied_rules.append(f"sol_neptuno_combined + proportion {face_proportion:.3f} < 1.17 → sol")
                return 'sol'

        # Mercurio_triangular proportion rules
        elif prediction == 'mercurio_triangular':
            if face_proportion >= 1.17:
                applied_rules.append(f"mercurio_triangular + proportion {face_proportion:.3f} ≥ 1.17 → mercurio_largo")
                return 'mercurio_largo'
            else:
                applied_rules.append(f"mercurio_triangular + proportion {face_proportion:.3f} < 1.17 → mercurio")
                return 'mercurio'

        # Luna_jupiter_combined proportion rules
        elif prediction == 'luna_jupiter_combined':
            if face_proportion >= 1.17:
                applied_rules.append(f"luna_jupiter_combined + proportion {face_proportion:.3f} ≥ 1.17 → neptuno")
                return 'neptuno'
            elif face_proportion < 0.99:
                applied_rules.append(f"luna_jupiter_combined + proportion {face_proportion:.3f} < 0.99 → luna")
                return 'luna'
            else:
                applied_rules.append(f"luna_jupiter_combined + proportion {face_proportion:.3f} between 0.99-1.17 → jupiter")
                return 'jupiter'

        # Marte_tierra_rectangulo proportion rules
        elif prediction == 'marte_tierra_rectangulo':
            if face_proportion < 0.99:
                applied_rules.append(f"marte_tierra_rectangulo + proportion {face_proportion:.3f} < 0.99 → tierra")
                return 'tierra'
            else:
                applied_rules.append(f"marte_tierra_rectangulo + proportion {face_proportion:.3f} ≥ 0.99 → marte")
                return 'marte'

        # Saturno_trapezoide_base_angosta proportion rules
        elif prediction == 'saturno_trapezoide_base_angosta':
            if face_proportion < 0.99:
                applied_rules.append(f"saturno_trapezoide_base_angosta + proportion {face_proportion:.3f} < 0.99 → saturno_tierra")
                return 'saturno_tierra'
            elif face_proportion > 1.17:
                applied_rules.append(f"saturno_trapezoide_base_angosta + proportion {face_proportion:.3f} > 1.17 → urano")
                return 'urano'
            else:
                applied_rules.append(f"saturno_trapezoide_base_angosta + proportion {face_proportion:.3f} between 0.99-1.17 → saturno_trapezoide_base_angosta")
                return 'saturno_trapezoide_base_angosta'

        # Venus_corazon proportion rules
        elif prediction == 'venus_corazon':
            if face_proportion < 0.99:
                applied_rules.append(f"venus_corazon + proportion {face_proportion:.3f} < 0.99 → luna")
                return 'luna'
            else:
                applied_rules.append(f"venus_corazon + proportion {face_proportion:.3f} ≥ 0.99 → venus_corazon")
                return 'venus_corazon'

        # Default case - no proportion rule applies
        applied_rules.append(f"No proportion rule for {prediction}, keeping original diagnosis")
        return prediction
    
    def _apply_hybrid_class_splitting(self, final_diagnosis, face_proportion, forehead_proportion, region_type):
        """Apply hybrid class splitting"""
        applied_rules = []
        
        if region_type == 'FRENTE':
            # Check for solar_lunar_combined specifically
            if final_diagnosis.lower() == 'solar_lunar_combined':
                if forehead_proportion is not None:
                    if forehead_proportion < 0.35:
                        applied_rules.append(f"solar_lunar_combined + forehead proportion {forehead_proportion:.3f} < 0.35 → luna")
                        return 'luna', applied_rules
                    else:
                        applied_rules.append(f"solar_lunar_combined + forehead proportion {forehead_proportion:.3f} ≥ 0.35 → solar")
                        return 'solar', applied_rules
                else:
                    applied_rules.append("solar_lunar_combined detected but forehead proportion N/A → no splitting")
                    return final_diagnosis, applied_rules
            
            applied_rules.append("No hybrid splitting needed for FRENTE")
            return final_diagnosis, applied_rules
        
        elif region_type == 'rostro_menton':
            # Use the new proportion-based splitting function for all rostro_menton diagnoses
            # This handles all the complex proportion rules including the new calibrations
            split_diagnosis = self._apply_proportion_based_splitting(final_diagnosis, face_proportion, applied_rules)

            # If the diagnosis changed, it means splitting was applied
            if split_diagnosis != final_diagnosis:
                applied_rules.append(f"Hybrid splitting applied: {final_diagnosis} → {split_diagnosis}")
                return split_diagnosis, applied_rules

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
                'rostro_split_diagnosis': None, 'rostro_split_rules': [],
                'detected_regions': None
            },
            'left_mirrored': {
                'frente_predictions': [], 'frente_probabilities': [],
                'rostro_predictions': [], 'rostro_probabilities': [],
                'frente_final_diagnosis': None, 'frente_applied_rules': [],
                'rostro_final_diagnosis': None, 'rostro_applied_rules': [],
                'frente_split_diagnosis': None, 'frente_split_rules': [],
                'rostro_split_diagnosis': None, 'rostro_split_rules': [],
                'detected_regions': None
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
                
                # Detect facial regions using Faster R-CNN
                detected_regions = self._detect_facial_regions(image, confidence_threshold=0.5)

                # Store detected regions for visualization
                results[side]['detected_regions'] = detected_regions

                # FRENTE classification
                frente_predictions = []
                frente_probabilities = []

                if detected_regions['frente'] is not None:
                    # Extract FRENTE region
                    frente_bbox = detected_regions['frente']['bbox']
                    frente_region = self._extract_region_from_image(image, frente_bbox)

                    if frente_region is not None:
                        # Preprocess FRENTE region for classification
                        frente_tensor = self._preprocess_image_for_classification(frente_region)
                        frente_tensor = frente_tensor.unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            frente_output = self.frente_model(frente_tensor)
                            frente_prob = torch.softmax(frente_output, dim=1)
                            frente_top3_probs, frente_top3_indices = torch.topk(frente_prob[0], min(3, len(self.frente_encoder.classes_)))

                            for i in range(len(frente_top3_indices)):
                                class_name = self.frente_encoder.classes_[frente_top3_indices[i].item()]
                                probability = frente_top3_probs[i].item()
                                frente_predictions.append(class_name)
                                frente_probabilities.append(probability)

                # Fallback to full image if no FRENTE region detected
                if not frente_predictions:
                    image_tensor = self._preprocess_image_for_classification(image)
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        frente_output = self.frente_model(image_tensor)
                        frente_prob = torch.softmax(frente_output, dim=1)
                        frente_top3_probs, frente_top3_indices = torch.topk(frente_prob[0], min(3, len(self.frente_encoder.classes_)))

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
                rostro_predictions = []
                rostro_probabilities = []

                if detected_regions['rostro_menton'] is not None:
                    # Extract rostro_menton region
                    rostro_bbox = detected_regions['rostro_menton']['bbox']
                    rostro_region = self._extract_region_from_image(image, rostro_bbox)

                    if rostro_region is not None:
                        # Preprocess rostro_menton region for classification
                        rostro_tensor = self._preprocess_image_for_classification(rostro_region)
                        rostro_tensor = rostro_tensor.unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            rostro_output = self.rostro_model(rostro_tensor)
                            rostro_prob = torch.softmax(rostro_output, dim=1)
                            rostro_top3_probs, rostro_top3_indices = torch.topk(rostro_prob[0], min(3, len(self.rostro_encoder.classes_)))

                            for i in range(len(rostro_top3_indices)):
                                class_name = self.rostro_encoder.classes_[rostro_top3_indices[i].item()]
                                probability = rostro_top3_probs[i].item()
                                rostro_predictions.append(class_name)
                                rostro_probabilities.append(probability)

                # Fallback to full image if no rostro_menton region detected
                if not rostro_predictions:
                    if 'image_tensor' not in locals():
                        image_tensor = self._preprocess_image_for_classification(image)
                        image_tensor = image_tensor.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        rostro_output = self.rostro_model(image_tensor)
                        rostro_prob = torch.softmax(rostro_output, dim=1)
                        rostro_top3_probs, rostro_top3_indices = torch.topk(rostro_prob[0], min(3, len(self.rostro_encoder.classes_)))

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