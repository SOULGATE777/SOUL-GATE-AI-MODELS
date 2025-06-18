import cv2
import numpy as np
import dlib
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import os
import math

class AnthropometricAnalyzer:
    def __init__(self):
        """Initialize the anthropometric analyzer with models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Initialize dlib components - REQUIRED for the system to work
        self.detector = dlib.get_frontal_face_detector()
        
        # Load dlib predictor (you'll need to mount this file)
        predictor_path = "/app/models/shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
            print("✅ Dlib facial landmark predictor loaded")
        else:
            print("❌ ERROR: Dlib predictor not found - system cannot function without it")
            print("Please ensure shape_predictor_68_face_landmarks.dat is in /app/models/")
            self.predictor = None
        
        # Load custom trained model
        model_path = "/app/models/facial_points_detection_model.pth"
        self.trained_model = self._load_trained_model(model_path)
        
    def _get_object_detection_model(self, num_classes):
        """Load the trained model architecture"""
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _load_trained_model(self, model_path):
        """Load the trained point detection model"""
        # num_classes = 13 point classes + 1 background
        num_classes = 14
        model = self._get_object_detection_model(num_classes)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.to(self.device)
                model.eval()
                print(f"✅ Trained model loaded from {model_path}")
                return model
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return None
        else:
            print(f"❌ Model file {model_path} not found")
            return None

    def _preprocess_image(self, img):
        """Process image without resizing to maintain coordinate consistency"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray, img

    def _detect_faces(self, img):
        """Detect faces using dlib"""
        faces = self.detector(img)
        print(f"👤 Number of faces detected: {len(faces)}")
        return faces

    def _detect_landmarks(self, img, face):
        """Detect landmarks for a face - ESSENTIAL for anthropometric analysis"""
        if self.predictor is None:
            raise RuntimeError("Cannot perform analysis: dlib predictor not loaded. Please ensure shape_predictor_68_face_landmarks.dat is available.")
        landmarks = self.predictor(img, face)
        return landmarks

    def _predict_facial_points(self, image, confidence_threshold=0.5):
        """Predict facial points using the trained model"""
        if self.trained_model is None:
            return {}
        
        # Prepare image for model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        print(f"🔍 Input image shape: {image.shape}")
        print(f"🔍 Image dimensions: {width}x{height}")
        
        image_resized = cv2.resize(image_rgb, (224, 224))
        print(f"🔍 Model input size: 224x224")
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.trained_model(image_tensor)
        
        # Extract points of interest (classes 1, 2, 3)
        detected_points = {}
        
        if len(predictions) > 0:
            prediction = predictions[0]
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if score > confidence_threshold and label in [1, 2, 3]:
                    # Calculate center point of bounding box
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Scale back to original image size
                    center_x = center_x * (width / 224)
                    center_y = center_y * (height / 224)
                    
                    print(f"🎯 Model prediction - Label: {label}, Score: {score:.3f}, Coords: ({center_x:.1f}, {center_y:.1f})")
                    detected_points[int(label)] = (int(center_x), int(center_y))
        
        return detected_points

    def _extend_landmarks_with_model(self, landmarks, img_shape, model_predictions):
        """Extend facial landmarks using both inferred points and model predictions"""
        # Convert landmarks to numpy array
        points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                           for i in range(68)])
        
        # Calculate extended points using original method as fallback
        face_height = max(points[:, 1]) - min(points[:, 1])
        
        # Find highest points of eyebrows for fallback
        left_eyebrow = points[17:22]  # Points 17-21
        right_eyebrow = points[22:27]  # Points 22-26
        
        # Get coordinates of highest points
        left_highest = left_eyebrow[np.argmin(left_eyebrow[:, 1])]
        right_highest = right_eyebrow[np.argmin(right_eyebrow[:, 1])]
        
        # Point 68: Between eyebrows - use model prediction (point 2) if available
        if 2 in model_predictions:
            between_eyebrows = model_predictions[2]
            print(f"✅ Using model point 2 for point 68 (between eyebrows): {between_eyebrows}")
        else:
            between_eyebrows = (
                int((left_highest[0] + right_highest[0]) // 2),
                int((left_highest[1] + right_highest[1]) // 2)
            )
            print("⚠️ Using inferred calculation for point 68 (between eyebrows)")
        
        # Point 69: Top of head - use model prediction (point 3) if available
        if 3 in model_predictions:
            top_of_head = model_predictions[3]
            print(f"✅ Using model point 3 for point 69 (top of head): {top_of_head}")
        else:
            top_of_head = (
                int(between_eyebrows[0]),
                int(between_eyebrows[1] - (face_height * 0.4))
            )
            print("⚠️ Using inferred calculation for point 69 (top of head)")
        
        # Pupil points: midpoint between eye landmarks
        left_pupil = (
            int((points[37][0] + points[40][0]) // 2),
            int((points[37][1] + points[40][1]) // 2)
        )
        right_pupil = (
            int((points[43][0] + points[46][0]) // 2),
            int((points[43][1] + points[46][1]) // 2)
        )
        
        # Combine all points
        extended_points = np.vstack([
            points,  # Original 68 dlib points (0-67)
            [between_eyebrows],  # Point 68 
            [top_of_head],  # Point 69 
            [left_pupil],  # Point 70
            [right_pupil]  # Point 71
        ])
        
        # Add model point 1 if available (point 72)
        point_1 = model_predictions.get(1, None)
        if point_1:
            extended_points = np.vstack([extended_points, [point_1]])
            print(f"✅ Model point 1 added as point 72: {point_1}")
        
        return extended_points, point_1 is not None

    def _calculate_proportions(self, extended_points):
        """Calculate facial proportions"""
        # Get key landmarks
        left_inner_eye = extended_points[39]  
        right_inner_eye = extended_points[42]  
        left_outter_eye = extended_points[36] 
        right_outter_eye = extended_points[44]
        
        # Calculate distances
        point_69 = extended_points[69]  # top of head
        point_68 = extended_points[68]  # between eyebrows
        point_34 = extended_points[33]  # nose base
        point_9 = extended_points[8]  # chin
        point_2 = extended_points[1] # left side of face
        point_16 = extended_points[15] # right side of face
        point_49 = extended_points[48] # left mouth
        point_54 = extended_points[53] # right mouth

        # Eyebrow landmarks
        right_eyebrow_points = extended_points[17:22]
        left_eyebrow_points = extended_points[22:27]

        # Pupils and chin
        right_pupil = extended_points[71]
        left_pupil = extended_points[70]
        right_chin = extended_points[7]
        left_chin = extended_points[9]

        # Calculate distances - WRAP ALL np.linalg.norm() with float()
        eye_distance = float(np.linalg.norm(right_inner_eye - left_inner_eye))
        outter_eye_distance = float(np.linalg.norm(right_outter_eye - left_outter_eye))
        distance_69_68 = float(np.linalg.norm(point_69 - point_68))
        distance_68_34 = float(np.linalg.norm(point_68 - point_34))
        distance_34_9 = float(np.linalg.norm(point_34 - point_9))
        head_height = float(np.linalg.norm(point_69 - point_9))
        head_width = float(np.linalg.norm(point_2 - point_16))
        mouth_length = float(np.linalg.norm(point_49 - point_54))
        pupil_distance = float(np.linalg.norm(left_pupil - right_pupil))
        whole_chin = float(np.linalg.norm(left_chin - right_chin))

        # Eyebrow segments - WRAP ALL np.linalg.norm() with float()
        first_third_r_eyebrow = float(np.linalg.norm(right_eyebrow_points[0] - right_eyebrow_points[2]))
        second_third_r_eyebrow = float(np.linalg.norm(right_eyebrow_points[2] - right_eyebrow_points[3]))
        third_third_r_eyebrow = float(np.linalg.norm(right_eyebrow_points[3] - right_eyebrow_points[4]))

        first_third_l_eyebrow = float(np.linalg.norm(left_eyebrow_points[0] - left_eyebrow_points[2]))
        second_third_l_eyebrow = float(np.linalg.norm(left_eyebrow_points[2] - left_eyebrow_points[3]))
        third_third_l_eyebrow = float(np.linalg.norm(left_eyebrow_points[3] - left_eyebrow_points[4]))

        # Debug key measurements
        print(f"📏 Head height: {head_height:.1f}, Distance 69-68: {distance_69_68:.1f}")
        print(f"📏 Point 69 (top): {point_69}, Point 68 (eyebrows): {point_68}")

        # Calculate proportions
        proportions = {
            "eye_distance_proportion": eye_distance / outter_eye_distance,
            "outter_eye_distance_proportion": outter_eye_distance / head_height, 
            "distance_69_68_proportion": distance_69_68 / head_height,
            "distance_68_34_proportion": distance_68_34 / head_height,
            "distance_34_9_proportion": distance_34_9 / head_height,  
            "head_width_proportion": head_width / head_height,
            "mouth_length_proportion": mouth_length / head_height,
            "1_right_eyebrow": first_third_r_eyebrow / head_height,
            "2_right_eyebrow": second_third_r_eyebrow / head_height,
            "3_right_eyebrow": third_third_r_eyebrow / head_height,
            "1_left_eyebrow": first_third_l_eyebrow / head_height,
            "2_left_eyebrow": second_third_l_eyebrow / head_height,
            "3_left_eyebrow": third_third_l_eyebrow / head_height,
            "mouth_to_eye_proportion": mouth_length / pupil_distance,
            "chin_to_face_width_proportion": whole_chin / head_width
        }
        
        return proportions

    def _calculate_eyebrow_slopes(self, extended_points):
        """Calculate eyebrow slopes"""
        # Point 1 and Point 17 (Across the face)
        point_1 = extended_points[0]
        point_17 = extended_points[16]
        
        # Calculate face line slope
        face_line_slope = (point_17[1] - point_1[1]) / (point_17[0] - point_1[0]) if point_17[0] != point_1[0] else float('inf')
        
        # Eyebrow points
        right_eyebrow_points = extended_points[17:22]
        left_eyebrow_points = extended_points[22:27]
        
        def get_slope(p1, p2):
            return (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float('inf')

        # Calculate slopes
        right_eyebrow_slopes = {
            "portion_1": get_slope(right_eyebrow_points[0], right_eyebrow_points[1]),
            "portion_2": get_slope(right_eyebrow_points[1], right_eyebrow_points[3]),
            "portion_3": get_slope(right_eyebrow_points[3], right_eyebrow_points[4])
        }

        left_eyebrow_slopes = {
            "portion_1": get_slope(left_eyebrow_points[0], left_eyebrow_points[1]),
            "portion_2": get_slope(left_eyebrow_points[1], left_eyebrow_points[3]),
            "portion_3": get_slope(left_eyebrow_points[3], left_eyebrow_points[4])
        }
        
        return {
            "face_line_slope": face_line_slope,
            "right_eyebrow": right_eyebrow_slopes,
            "left_eyebrow": left_eyebrow_slopes
        }

    def analyze_face(self, image, confidence_threshold=0.5):
        """
        Complete facial analysis
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Confidence threshold for model predictions
            
        Returns:
            dict: Analysis results
        """
        try:
            # Preprocess image (no resizing now)
            gray, img_processed = self._preprocess_image(image)
            
            # Detect faces
            faces = self._detect_faces(gray)
            
            if len(faces) == 0:
                return None
            
            # Process first face
            face = faces[0]
            
            # Detect landmarks
            landmarks = self._detect_landmarks(gray, face)
            if landmarks is None:
                return None
            
            # Get model predictions
            model_predictions = self._predict_facial_points(img_processed, confidence_threshold)
            
            # Extend landmarks with model predictions
            extended_points, has_point_1 = self._extend_landmarks_with_model(
                landmarks, img_processed.shape, model_predictions
            )
            
            # Calculate proportions and slopes
            proportions = self._calculate_proportions(extended_points)
            slopes = self._calculate_eyebrow_slopes(extended_points)
            
            # Create summary
            summary = self._create_analysis_summary(proportions, slopes, model_predictions)
            
            return {
                "landmarks": landmarks,
                "model_predictions": model_predictions,
                "extended_points": extended_points,
                "proportions": proportions,
                "slopes": slopes,
                "summary": summary,
                "image_processed": img_processed
            }
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None

    def _create_analysis_summary(self, proportions, slopes, model_predictions):
        """Create analysis summary with labels"""
        summary = {
            "facial_thirds": {
                "primer_tercio": self._label_proportion(proportions['distance_69_68_proportion'], 'primer tercio'),
                "segundo_tercio": self._label_proportion(proportions['distance_68_34_proportion'], 'segundo tercio'),
                "tercer_tercio": self._label_proportion(proportions['distance_34_9_proportion'], 'tercer tercio')
            },
            "eye_analysis": {
                "internal_proportion": self._label_proportion(proportions['eye_distance_proportion'], 'proporcion interna ojos')
            },
            "mouth_analysis": {
                "mouth_to_eye_relation": self._label_proportion(proportions['mouth_to_eye_proportion'], 'relacion boca - pupilas')
            },
            "model_integration": {
                "point_2_used": 2 in model_predictions,
                "point_3_used": 3 in model_predictions,
                "point_1_detected": 1 in model_predictions
            }
        }
        return summary

    def _label_proportion(self, proportion, section_name):
        """Label proportions based on thresholds"""
        if section_name == "primer tercio":
            if proportion > 0.38:
                return 'tercio superior largo'
            elif proportion < 0.27:
                return 'tercio superior corto'
            else:
                return 'tercio superior standard'
        elif section_name == "segundo tercio":
            if proportion > 0.38:
                return 'tercio medio largo'
            elif proportion < 0.27:
                return 'tercio medio corto'
            else:
                return 'tercio medio standard'
        elif section_name == "tercer tercio":
            if proportion > 0.38:
                return 'tercio inferior largo'
            elif proportion < 0.27:
                return 'tercio inferior corto'
            else:
                return 'tercio inferior standard'
        elif section_name == "relacion boca - pupilas":
            if proportion > 1.0:
                return 'boca grande en relación a las pupilas'
            elif proportion < 0.7:
                return 'boca pequeña en relación a las pupilas'
            else:
                return 'relación boca-pupilas estándar'
        elif section_name == "proporcion interna ojos":
            if proportion < 0.3:
                return 'Cercanos'
            elif 0.3 <= proportion <= 0.37:
                return 'Standard'
            elif proportion > 0.37:
                return 'Lejanos'
        return 'Unknown'

    def detect_landmarks_only(self, image):
        """Detect only facial landmarks"""
        gray, img_processed = self._preprocess_image(image)
        faces = self._detect_faces(gray)
        
        if len(faces) > 0:
            landmarks = self._detect_landmarks(gray, faces[0])
            if landmarks is not None:
                # Convert landmarks to a list of points for JSON serialization
                landmark_points = [(int(landmarks.part(i).x), int(landmarks.part(i).y)) for i in range(68)]
                return {"landmarks": landmark_points, "count": 68}
        return {"landmarks": [], "count": 0}

    def detect_model_points_only(self, image, confidence_threshold=0.5):
        """Detect only model points"""
        gray, img_processed = self._preprocess_image(image)
        return self._predict_facial_points(img_processed, confidence_threshold)
