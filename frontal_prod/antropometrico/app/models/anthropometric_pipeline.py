import cv2
import numpy as np
import dlib
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from scipy.spatial import ConvexHull
import os
import math

class AnthropometricAnalyzer:
    def __init__(self):
        """Initialize the anthropometric analyzer with models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize dlib components - REQUIRED for the system to work
        self.detector = dlib.get_frontal_face_detector()
        
        # Load dlib predictor (you'll need to mount this file)
        predictor_path = "/app/models/shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
            print("SUCCESS: Dlib facial landmark predictor loaded")
        else:
            print("ERROR: Dlib predictor not found - system cannot function without it")
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
                print(f"SUCCESS: Trained model loaded from {model_path}")
                return model
            except Exception as e:
                print(f"ERROR: Error loading model: {e}")
                return None
        else:
            print(f"ERROR: Model file {model_path} not found")
            return None

    def _preprocess_image(self, img):
        """Process image without resizing to maintain coordinate consistency"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray, img

    def _detect_faces(self, img):
        """Detect faces using dlib"""
        faces = self.detector(img)
        print(f"Number of faces detected: {len(faces)}")
        return faces

    def _detect_landmarks(self, img, face):
        """Detect landmarks for a face - ESSENTIAL for anthropometric analysis"""
        if self.predictor is None:
            raise RuntimeError("Cannot perform analysis: dlib predictor not loaded. Please ensure shape_predictor_68_face_landmarks.dat is available.")
        landmarks = self.predictor(img, face)
        return landmarks

    def _predict_facial_points(self, image, confidence_threshold=0.5):
        """Predict ALL facial points using the trained model (classes 1-13)"""
        if self.trained_model is None:
            return {}
        
        # Prepare image for model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        print(f"Input image shape: {image.shape}")
        print(f"Image dimensions: {width}x{height}")

        image_resized = cv2.resize(image_rgb, (224, 224))
        print(f"Model input size: 224x224")
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.trained_model(image_tensor)
        
        # Extract ALL detected points (classes 1-13)
        detected_points = {}
        
        if len(predictions) > 0:
            prediction = predictions[0]
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if score > confidence_threshold:
                    # Calculate center point of bounding box
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Scale back to original image size
                    center_x = center_x * (width / 224)
                    center_y = center_y * (height / 224)

                    print(f"Model prediction - Label: {label}, Score: {score:.3f}, Coords: ({center_x:.1f}, {center_y:.1f})")
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
        # RIGHT eyebrow: dlib points 18-22 (Python indices 17-21)
        right_eyebrow = points[17:22]  # Points 17-21
        # LEFT eyebrow: dlib points 23-27 (Python indices 22-26)
        left_eyebrow = points[22:27]  # Points 22-26

        # Get coordinates of highest points (lowest Y value = highest on screen)
        left_highest = left_eyebrow[np.argmin(left_eyebrow[:, 1])]
        right_highest = right_eyebrow[np.argmin(right_eyebrow[:, 1])]
        
        # Point 68: Between eyebrows - use model prediction (point 2) if available
        if 2 in model_predictions:
            between_eyebrows = model_predictions[2]
            print(f"SUCCESS: Using model point 2 for point 68 (between eyebrows): {between_eyebrows}")
        else:
            between_eyebrows = (
                int((left_highest[0] + right_highest[0]) // 2),
                int((left_highest[1] + right_highest[1]) // 2)
            )
            print("WARNING: Using inferred calculation for point 68 (between eyebrows)")
        
        # Point 69: Top of head - use calculated point C1 (M9 Y + M2 X) to avoid widow's peak issues
        calculated_c1 = None
        try:
            if 9 in model_predictions and 2 in model_predictions:
                # Create calculated point C1: X from model point 2, Y from model point 9
                calculated_c1 = (
                    int(model_predictions[2][0]),  # X from point 2 (between eyebrows)
                    int(model_predictions[9][1])   # Y from model point 9 (M9)
                )
                top_of_head = calculated_c1
                print(f"SUCCESS: Using calculated point C1 for point 69: X from M2({model_predictions[2][0]:.1f}) + Y from M9({model_predictions[9][1]:.1f}) = {top_of_head}")
            elif 3 in model_predictions:
                # Fallback to model point 3 if M9 or M2 not available
                top_of_head = model_predictions[3]
                print(f"WARNING: Fallback: Using model point 3 for point 69 (top of head): {top_of_head}")
            else:
                # Final fallback to calculated estimate
                top_of_head = (
                    int(between_eyebrows[0]),
                    int(between_eyebrows[1] - (face_height * 0.4))
                )
                print("WARNING: Using inferred calculation for point 69 (top of head)")
        except Exception as e:
            print(f"ERROR: Error calculating C1: {e}")
            # Safe fallback
            top_of_head = (
                int(between_eyebrows[0]),
                int(between_eyebrows[1] - (face_height * 0.4))
            )
            calculated_c1 = None
        
        # Pupil points: midpoint between eye landmarks
        # RIGHT pupil: midpoint of RIGHT eye (dlib points 38 and 41, indices 37 and 40)
        right_pupil = (
            int((points[37][0] + points[40][0]) // 2),
            int((points[37][1] + points[40][1]) // 2)
        )
        # LEFT pupil: midpoint of LEFT eye (dlib points 44 and 47, indices 43 and 46)
        left_pupil = (
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
            print(f"SUCCESS: Model point 1 added as point 72: {point_1}")
        
        return extended_points, point_1 is not None, calculated_c1

    def _calculate_proportions(self, extended_points):
        """Calculate facial proportions"""
        # Get key landmarks
        right_inner_eye = extended_points[39]  # Point 40 - RIGHT eye inner corner
        left_inner_eye = extended_points[42]   # Point 43 - LEFT eye inner corner
        right_outter_eye = extended_points[36] # Point 37 - RIGHT eye outer corner
        left_outter_eye = extended_points[45]  # Point 46 - LEFT eye outer corner
        
        # Calculate distances
        point_69 = extended_points[69]  # top of head
        point_68 = extended_points[68]  # between eyebrows
        point_34 = extended_points[33]  # nose base
        point_9 = extended_points[8]  # chin
        point_2 = extended_points[1] # left side of face
        point_16 = extended_points[15] # right side of face
        point_49 = extended_points[48] # left mouth corner
        point_55 = extended_points[54] # right mouth corner

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
        mouth_length = float(np.linalg.norm(point_49 - point_55))
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
        print(f"DEBUG: Head height: {head_height:.1f}, Distance 69-68: {distance_69_68:.1f}")
        print(f"DEBUG: Point 69 (top): {point_69}, Point 68 (eyebrows): {point_68}")

        # Calculate proportions
        proportions = {
            "eye_distance_proportion": eye_distance / outter_eye_distance,
            "outter_eye_distance_proportion": outter_eye_distance / head_height, 
            "distance_69_68_proportion": distance_69_68 / head_height,
            "distance_68_34_proportion": distance_68_34 / head_height,
            "distance_34_9_proportion": distance_34_9 / head_height,  
            "head_width_proportion": head_width / head_height,
            "mouth_length_proportion": mouth_length / head_width,
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

    def _calculate_eyebrow_slopes(self, extended_points, model_predictions):
        """Calculate eyebrow slopes relative to vertical midline reference (point 9 to M3)"""
        # Get vertical midline reference vector (point 9 to M3)
        point_9 = extended_points[8]  # chin

        # M3 should always be used for this measurement (not C1)
        if 3 not in model_predictions:
            print("WARNING: Model point 3 (M3) not available for vertical reference. Angles may be inaccurate.")
            # Fallback: use point 69 if M3 not available
            point_m3 = extended_points[69]
        else:
            point_m3 = model_predictions[3]

        # Calculate vertical midline reference angle (from point 9 to M3)
        vertical_ref_angle = math.atan2(point_m3[1] - point_9[1], point_m3[0] - point_9[0])

        # Calculate perpendicular to vertical midline (this is our "horizontal" reference)
        perpendicular_ref_angle = vertical_ref_angle + math.pi / 2

        # Eyebrow points - BOTH analyzed MEDIAL→LATERAL for consistency
        # RIGHT eyebrow (dlib 18-22): natural order is lateral→medial, so REVERSE to medial→lateral
        # LEFT eyebrow (dlib 23-27): natural order is already medial→lateral
        right_eyebrow_points = extended_points[17:22][::-1]  # Reversed: [22, 21, 20, 19, 18] medial→lateral
        left_eyebrow_points = extended_points[22:27]  # Natural: [23, 24, 25, 26, 27] medial→lateral

        def get_angle_relative_to_perpendicular(p1, p2):
            """Calculate angle relative to perpendicular of vertical midline (for LEFT eyebrow)"""
            # Calculate angle of segment from p1 to p2
            segment_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            # Calculate relative to perpendicular reference
            relative_angle = segment_angle - perpendicular_ref_angle
            # Normalize to [-pi, pi]
            while relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            while relative_angle < -math.pi:
                relative_angle += 2 * math.pi
            # Convert to degrees
            return math.degrees(relative_angle)

        def get_angle_relative_to_perpendicular_mirrored(p1, p2):
            """Calculate angle relative to perpendicular for RIGHT eyebrow (mirrored for symmetry)"""
            # For right eyebrow: reverse direction (p2→p1) so symmetrical eyebrows produce same angles
            segment_angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
            # Calculate relative to perpendicular reference
            relative_angle = segment_angle - perpendicular_ref_angle
            # Normalize to [-pi, pi]
            while relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            while relative_angle < -math.pi:
                relative_angle += 2 * math.pi
            # Convert to degrees
            return math.degrees(relative_angle)

        # Calculate angles
        # RIGHT eyebrow: medial→lateral anatomically [22→21→20→19→18]
        # Uses MIRRORED calculation so symmetrical eyebrows produce same angles as left
        # Portions: 1st(22-21 medial), 2nd(21-19 middle), 3rd(19-18 lateral)
        right_eyebrow_angles = {
            "portion_1": get_angle_relative_to_perpendicular_mirrored(right_eyebrow_points[0], right_eyebrow_points[1]),
            "portion_2": get_angle_relative_to_perpendicular_mirrored(right_eyebrow_points[1], right_eyebrow_points[3]),
            "portion_3": get_angle_relative_to_perpendicular_mirrored(right_eyebrow_points[3], right_eyebrow_points[4])
        }

        # LEFT eyebrow: medial→lateral anatomically [23→24→25→26→27]
        # Uses standard calculation (left-to-right on screen matches medial→lateral)
        # Portions: 1st(23-24 medial), 2nd(24-26 middle), 3rd(26-27 lateral)
        left_eyebrow_angles = {
            "portion_1": -get_angle_relative_to_perpendicular(left_eyebrow_points[0], left_eyebrow_points[1]),
            "portion_2": -get_angle_relative_to_perpendicular(left_eyebrow_points[1], left_eyebrow_points[3]),
            "portion_3": -get_angle_relative_to_perpendicular(left_eyebrow_points[3], left_eyebrow_points[4])
        }

        return {
            "vertical_reference_used": 3 in model_predictions,
            "vertical_reference_angle_deg": math.degrees(vertical_ref_angle),
            "perpendicular_reference_angle_deg": math.degrees(perpendicular_ref_angle),
            "right_eyebrow": right_eyebrow_angles,
            "left_eyebrow": left_eyebrow_angles
        }

    def _calculate_eyebrow_proportions(self, extended_points):
        """Calculate eyebrow length proportional to middle third of face and eye length ratio"""
        # Right eyebrow points (18-22 dlib): outer (temporal) to inner (nasal)
        right_eyebrow_outer = extended_points[17]  # Point 18 - outer/temporal end
        right_eyebrow_inner = extended_points[21]  # Point 22 - inner/nasal end
        right_eyebrow_length = float(np.linalg.norm(right_eyebrow_outer - right_eyebrow_inner))

        # Left eyebrow points (23-27 dlib): inner (nasal) to outer (temporal)
        left_eyebrow_inner = extended_points[22]  # Point 23 - inner/nasal end
        left_eyebrow_outer = extended_points[26]  # Point 27 - outer/temporal end
        left_eyebrow_length = float(np.linalg.norm(left_eyebrow_outer - left_eyebrow_inner))

        # Right eye length (36-39)
        right_eye_inner = extended_points[39]  # Inner corner
        right_eye_outer = extended_points[36]  # Outer corner
        right_eye_length = float(np.linalg.norm(right_eye_outer - right_eye_inner))

        # Left eye length (42-45)
        left_eye_inner = extended_points[42]  # Inner corner
        left_eye_outer = extended_points[45]  # Outer corner
        left_eye_length = float(np.linalg.norm(left_eye_outer - left_eye_inner))

        # Calculate middle third of face (point 68 to point 34)
        point_68 = extended_points[68]  # between eyebrows
        point_34 = extended_points[33]  # nose base
        middle_third_length = float(np.linalg.norm(point_68 - point_34))

        # Calculate proportions relative to middle third
        right_eyebrow_proportion = right_eyebrow_length / middle_third_length if middle_third_length > 0 else 0
        left_eyebrow_proportion = left_eyebrow_length / middle_third_length if middle_third_length > 0 else 0

        return {
            'right_eyebrow_proportion': right_eyebrow_proportion,
            'left_eyebrow_proportion': left_eyebrow_proportion,
            'right_eyebrow_length': right_eyebrow_length,
            'left_eyebrow_length': left_eyebrow_length,
            'right_eye_length': right_eye_length,
            'left_eye_length': left_eye_length,
            'middle_third_length': middle_third_length
        }

    def _calculate_eye_angles(self, extended_points, model_predictions):
        """Calculate the angle of both eyes relative to vertical midline reference (point 9 to M3)"""
        # Get vertical midline reference vector (point 9 to M3)
        point_9 = extended_points[8]  # chin

        # M3 should always be used for this measurement (not C1)
        if 3 not in model_predictions:
            print("WARNING: Model point 3 (M3) not available for vertical reference. Eye angles may be inaccurate.")
            # Fallback: use point 69 if M3 not available
            point_m3 = extended_points[69]
        else:
            point_m3 = model_predictions[3]

        # Calculate vertical midline reference angle (from point 9 to M3)
        vertical_ref_angle = math.atan2(point_m3[1] - point_9[1], point_m3[0] - point_9[0])

        # Calculate perpendicular to vertical midline (this is our "horizontal" reference)
        perpendicular_ref_angle = vertical_ref_angle + math.pi / 2

        # Right eye: inner corner (39) to outer corner (36)
        right_inner = extended_points[39]
        right_outer = extended_points[36]

        # Left eye: inner corner (42) to outer corner (45)
        left_inner = extended_points[42]
        left_outer = extended_points[45]

        # Calculate eye angles
        # Right eye: REVERSE direction (outer to inner) - goes rightward anatomically
        right_eye_angle = math.atan2(right_inner[1] - right_outer[1], right_inner[0] - right_outer[0])
        # Left eye: calculate from inner to outer (goes rightward in image)
        left_eye_angle = math.atan2(left_outer[1] - left_inner[1], left_outer[0] - left_inner[0])

        # Calculate relative to perpendicular reference
        right_relative_angle = right_eye_angle - perpendicular_ref_angle
        left_relative_angle = left_eye_angle - perpendicular_ref_angle

        # Normalize to [-pi, pi]
        while right_relative_angle > math.pi:
            right_relative_angle -= 2 * math.pi
        while right_relative_angle < -math.pi:
            right_relative_angle += 2 * math.pi

        while left_relative_angle > math.pi:
            left_relative_angle -= 2 * math.pi
        while left_relative_angle < -math.pi:
            left_relative_angle += 2 * math.pi

        # Convert to degrees
        right_angle_deg = math.degrees(right_relative_angle)
        # NEGATE left eye angle to ensure symmetrical eyes produce same sign
        # This mirrors the left eye angle to match anatomical expectations
        left_angle_deg = -math.degrees(left_relative_angle)

        return {
            'left_eye_angle': left_angle_deg,
            'right_eye_angle': right_angle_deg,
            'vertical_reference_used': 3 in model_predictions,
            'vertical_reference_angle_deg': math.degrees(vertical_ref_angle),
            'perpendicular_reference_angle_deg': math.degrees(perpendicular_ref_angle)
        }

    def _calculate_eyebrow_eyelid_distances(self, extended_points):
        """Calculate eyebrow to eyelid distances as proportion to middle third of face"""
        # Get key landmarks for measurements
        # RIGHT eye: point 20 (eyebrow) to point 38 (eyelid) - dlib points 20 and 38
        right_eyebrow_point = extended_points[19]  # dlib point 20, Python index 19 - RIGHT eyebrow
        right_eyelid_point = extended_points[37]   # dlib point 38, Python index 37 - RIGHT upper eyelid

        # LEFT eye: point 25 (eyebrow) to point 45 (eyelid) - dlib points 25 and 45
        left_eyebrow_point = extended_points[24]  # dlib point 25, Python index 24 - LEFT eyebrow
        left_eyelid_point = extended_points[44]   # dlib point 45, Python index 44 - LEFT upper eyelid

        # Calculate middle third of face (point 68 to 34) for proportional measurements
        point_68 = extended_points[68]  # between eyebrows
        point_34 = extended_points[33]  # nose base
        middle_third_length = float(np.linalg.norm(point_68 - point_34))

        # Calculate distances
        left_eyebrow_eyelid_distance = float(np.linalg.norm(left_eyebrow_point - left_eyelid_point))
        right_eyebrow_eyelid_distance = float(np.linalg.norm(right_eyebrow_point - right_eyelid_point))

        # Calculate proportions relative to middle third of face
        left_eyebrow_eyelid_proportion = left_eyebrow_eyelid_distance / middle_third_length if middle_third_length > 0 else 0
        right_eyebrow_eyelid_proportion = right_eyebrow_eyelid_distance / middle_third_length if middle_third_length > 0 else 0

        print(f"DEBUG: Eyebrow-Eyelid Distances - Left: {left_eyebrow_eyelid_distance:.1f}px ({left_eyebrow_eyelid_proportion:.4f}), Right: {right_eyebrow_eyelid_distance:.1f}px ({right_eyebrow_eyelid_proportion:.4f})")

        return {
            'left_eyebrow_eyelid_distance': left_eyebrow_eyelid_distance,
            'right_eyebrow_eyelid_distance': right_eyebrow_eyelid_distance,
            'left_eyebrow_eyelid_proportion': left_eyebrow_eyelid_proportion,
            'right_eyebrow_eyelid_proportion': right_eyebrow_eyelid_proportion,
            'middle_third_length': middle_third_length
        }

    def _calculate_mouth_measurements(self, extended_points):
        """Calculate mouth measurements: cupid's bow arches, lips ratio, and lip thickness"""
        # Calculate bottom third of face (point 34 to point 9)
        point_34 = extended_points[33]  # nose base
        point_9 = extended_points[8]    # chin
        bottom_third_length = float(np.linalg.norm(point_34 - point_9))

        # Cupid's bow arch measurements as proportions
        # Denominator (midline reference): dlib points 52→63 (upper lip center vertical)
        midline_point_52 = extended_points[51]  # dlib point 52, Python index 51
        midline_point_63 = extended_points[62]  # dlib point 63, Python index 62
        midline_distance = float(np.linalg.norm(midline_point_52 - midline_point_63))

        # RIGHT cupid's arch: dlib points 51→62 / midline (52→63)
        # Subject's RIGHT side of face (left side of image)
        right_cupid_point_51 = extended_points[50]  # dlib point 51, Python index 50
        right_cupid_point_62 = extended_points[61]  # dlib point 62, Python index 61
        right_cupid_arch_distance = float(np.linalg.norm(right_cupid_point_51 - right_cupid_point_62))
        right_cupid_arch_proportion = right_cupid_arch_distance / midline_distance if midline_distance > 0 else 0

        # LEFT cupid's arch: dlib points 53→64 / midline (52→63)
        # Subject's LEFT side of face (right side of image)
        left_cupid_point_53 = extended_points[52]  # dlib point 53, Python index 52
        left_cupid_point_64 = extended_points[63]  # dlib point 64, Python index 63
        left_cupid_arch_distance = float(np.linalg.norm(left_cupid_point_53 - left_cupid_point_64))
        left_cupid_arch_proportion = left_cupid_arch_distance / midline_distance if midline_distance > 0 else 0

        # Lip thickness: dlib point 52 to 58 (top center of upper lip to bottom center of lower lip)
        lip_thickness_point_52 = extended_points[51]  # dlib point 52, Python index 51
        lip_thickness_point_58 = extended_points[57]  # dlib point 58, Python index 57
        lip_thickness_distance = float(np.linalg.norm(lip_thickness_point_52 - lip_thickness_point_58))
        lip_thickness_proportion = lip_thickness_distance / bottom_third_length if bottom_third_length > 0 else 0

        # Lips ratio calculation
        # Upper lip: dlib point 52 to 63 (Python index 51 to 62)
        upper_lip_point_52 = extended_points[51]  # dlib point 52
        upper_lip_point_63 = extended_points[62]  # dlib point 63
        upper_lip_distance = float(np.linalg.norm(upper_lip_point_52 - upper_lip_point_63))

        # Lower lip: dlib point 67 to 58 (Python index 66 to 57)
        lower_lip_point_67 = extended_points[66]  # dlib point 67
        lower_lip_point_58 = extended_points[57]  # dlib point 58
        lower_lip_distance = float(np.linalg.norm(lower_lip_point_67 - lower_lip_point_58))

        # Calculate lips ratio: upper lip / lower lip
        lips_ratio = upper_lip_distance / lower_lip_distance if lower_lip_distance > 0 else 0

        print(f"DEBUG: Mouth Measurements:")
        print(f"   Midline reference (52-63): {midline_distance:.1f}px")
        print(f"   Right cupid's arch (51-62): {right_cupid_arch_distance:.1f}px (proportion: {right_cupid_arch_proportion:.4f})")
        print(f"   Left cupid's arch (53-64): {left_cupid_arch_distance:.1f}px (proportion: {left_cupid_arch_proportion:.4f})")
        print(f"   Lip thickness (52-58): {lip_thickness_distance:.1f}px (proportion: {lip_thickness_proportion:.4f})")
        print(f"   Upper lip distance (52-63): {upper_lip_distance:.1f}px")
        print(f"   Lower lip distance (67-58): {lower_lip_distance:.1f}px")
        print(f"   Lips ratio (upper/lower): {lips_ratio:.4f}")

        return {
            'left_cupid_arch_distance': left_cupid_arch_distance,
            'left_cupid_arch_proportion': left_cupid_arch_proportion,
            'right_cupid_arch_distance': right_cupid_arch_distance,
            'right_cupid_arch_proportion': right_cupid_arch_proportion,
            'lip_thickness_distance': lip_thickness_distance,
            'lip_thickness_proportion': lip_thickness_proportion,
            'upper_lip_distance': upper_lip_distance,
            'lower_lip_distance': lower_lip_distance,
            'lips_ratio': lips_ratio,
            'bottom_third_length': bottom_third_length
        }

    def _get_eye_area_points(self, landmarks, eye_side):
        """Get points that define the eye area boundary"""
        dlib_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                               for i in range(68)])
        
        if eye_side == 'right':
            # Right eye points (36-41)
            eye_points = dlib_points[36:42]
        else:
            # Left eye points (42-47)
            eye_points = dlib_points[42:48]
        
        return eye_points

    def _get_face_area_points(self, landmarks, model_predictions):
        """Get points that define the whole face area using dlib and model points"""
        dlib_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                               for i in range(68)])
        
        # Start with face contour points (0-16)
        face_points = list(dlib_points[0:17])
        
        # Add all available model points to better define face boundary
        for label, point in model_predictions.items():
            face_points.append(np.array([point[0], point[1]]))
        
        return np.array(face_points)

    def _calculate_polygon_area(self, points):
        """Calculate the area of a polygon using the Shoelace formula"""
        if len(points) < 3:
            return 0
        
        # Use ConvexHull to get the actual boundary
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            # Shoelace formula
            x = hull_points[:, 0]
            y = hull_points[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        except:
            # Fallback: simple polygon area calculation
            x = points[:, 0]
            y = points[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _get_outer_face_points(self, landmarks, model_predictions):
        """Get points that define the outer face perimeter (ear to ear)"""
        # Convert landmarks to numpy array
        dlib_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                               for i in range(68)])
        
        # Face contour points from dlib (points 0-16: jaw line only)
        face_contour = dlib_points[0:17]  # Points 0-16
        
        # Combine dlib perimeter points
        outer_points = []
        
        # Add jaw line points (0-16)
        for point in face_contour:
            outer_points.append(point)
        
        # Add ALL model points to better define outer face boundary
        for label, point in model_predictions.items():
            outer_points.append(np.array([point[0], point[1]]))
        
        return np.array(outer_points)

    def _get_inner_face_points(self, landmarks):
        """Get points that define the inner face boundary (eyebrows to mouth)"""
        # Convert landmarks to numpy array
        dlib_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                               for i in range(68)])
        
        inner_points = []
        
        # Eyebrow points (17-26) for upper boundary
        eyebrow_points = dlib_points[17:27]
        for point in eyebrow_points:
            inner_points.append(point)
        
        # Mouth outer points (48-67) for lower boundary
        mouth_points = dlib_points[48:68]
        for point in mouth_points:
            inner_points.append(point)
        
        return np.array(inner_points)

    def _calculate_eye_face_proportions(self, landmarks, model_predictions):
        """Calculate left and right eye area proportion to face area"""
        # Get face area points
        face_points = self._get_face_area_points(landmarks, model_predictions)
        face_area = float(self._calculate_polygon_area(face_points))  # Ensure float
        
        # Get right eye area
        right_eye_points = self._get_eye_area_points(landmarks, 'right')
        right_eye_area = float(self._calculate_polygon_area(right_eye_points))  # Ensure float
        
        # Get left eye area
        left_eye_points = self._get_eye_area_points(landmarks, 'left')
        left_eye_area = float(self._calculate_polygon_area(left_eye_points))  # Ensure float
        
        # Calculate proportions
        right_eye_proportion = (right_eye_area / face_area * 100) if face_area > 0 else 0
        left_eye_proportion = (left_eye_area / face_area * 100) if face_area > 0 else 0
        
        return {
            'face_area': face_area,
            'right_eye_area': right_eye_area,
            'left_eye_area': left_eye_area,
            'right_eye_proportion': float(right_eye_proportion),  # Ensure float
            'left_eye_proportion': float(left_eye_proportion)     # Ensure float
        }

    def _calculate_inner_outer_face_proportions(self, landmarks, model_predictions):
        """Calculate inner face area proportion to outer face area - FIXED for JSON serialization"""
        # Get outer face area points (includes model points for better boundary)
        outer_points = self._get_outer_face_points(landmarks, model_predictions)
        outer_area = float(self._calculate_polygon_area(outer_points))  # Ensure float
        
        # Get inner face area points (eyebrows to mouth)
        inner_points = self._get_inner_face_points(landmarks)
        inner_area = float(self._calculate_polygon_area(inner_points))  # Ensure float
        
        # Calculate percentage
        inner_outer_percentage = float((inner_area / outer_area * 100) if outer_area > 0 else 0)  # Ensure float
        
        # Convert numpy arrays to lists for JSON serialization
        outer_points_list = outer_points.tolist() if isinstance(outer_points, np.ndarray) else outer_points
        inner_points_list = inner_points.tolist() if isinstance(inner_points, np.ndarray) else inner_points
        
        return {
            'outer_area': outer_area,
            'inner_area': inner_area,
            'inner_outer_percentage': inner_outer_percentage,
            'outer_points': outer_points_list,  # Converted to list
            'inner_points': inner_points_list   # Converted to list
        }

    def analyze_face(self, image, confidence_threshold=0.5):
        """
        Complete facial analysis with all new features
        
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
            extended_points, has_point_1, calculated_c1 = self._extend_landmarks_with_model(
                landmarks, img_processed.shape, model_predictions
            )
            
            # Calculate proportions and slopes
            proportions = self._calculate_proportions(extended_points)
            slopes = self._calculate_eyebrow_slopes(extended_points, model_predictions)

            # NEW FEATURES: Calculate additional analysis
            eyebrow_proportions = self._calculate_eyebrow_proportions(extended_points)
            eye_angles = self._calculate_eye_angles(extended_points, model_predictions)
            eyebrow_eyelid_distances = self._calculate_eyebrow_eyelid_distances(extended_points)
            mouth_measurements = self._calculate_mouth_measurements(extended_points)
            eye_face_proportions = self._calculate_eye_face_proportions(landmarks, model_predictions)
            inner_outer_proportions = self._calculate_inner_outer_face_proportions(landmarks, model_predictions)
            
            # Create summary
            summary = self._create_analysis_summary(
                proportions, slopes, model_predictions, eyebrow_proportions,
                eye_angles, eyebrow_eyelid_distances, mouth_measurements, eye_face_proportions, inner_outer_proportions, calculated_c1
            )
            
            return {
                "landmarks": landmarks,
                "model_predictions": model_predictions,
                "extended_points": extended_points,
                "proportions": proportions,
                "slopes": slopes,
                "eyebrow_proportions": eyebrow_proportions,
                "eye_angles": eye_angles,
                "eyebrow_eyelid_distances": eyebrow_eyelid_distances,
                "mouth_measurements": mouth_measurements,
                "eye_face_proportions": eye_face_proportions,
                "inner_outer_proportions": inner_outer_proportions,
                "summary": summary,
                "image_processed": img_processed,
                "calculated_c1": calculated_c1
            }
            
        except Exception as e:
            print(f"ERROR: Analysis error: {e}")
            return None

    def _create_analysis_summary(self, proportions, slopes, model_predictions,
                               eyebrow_proportions, eye_angles, eyebrow_eyelid_distances, mouth_measurements, eye_face_proportions,
                               inner_outer_proportions, calculated_c1=None):
        """Create comprehensive analysis summary with labels"""
        summary = {
            "facial_thirds": {
                "primer_tercio": self._label_proportion(proportions['distance_69_68_proportion'], 'primer tercio'),
                "segundo_tercio": self._label_proportion(proportions['distance_68_34_proportion'], 'segundo tercio'),
                "tercer_tercio": self._label_proportion(proportions['distance_34_9_proportion'], 'tercer tercio')
            },
            "eye_analysis": {
                "internal_proportion": self._label_proportion(proportions['eye_distance_proportion'], 'proporcion interna ojos'),
                "left_eye_angle": self._classify_eye_angle(eye_angles['left_eye_angle']),
                "left_eye_angle_degrees": f"{eye_angles['left_eye_angle']:.2f}°",
                "right_eye_angle": self._classify_eye_angle(eye_angles['right_eye_angle']),
                "right_eye_angle_degrees": f"{eye_angles['right_eye_angle']:.2f}°",
                "left_eye_face_proportion": f"{eye_face_proportions['left_eye_proportion']:.2f}%",
                "left_eye_size_classification": self._classify_eye_size(eye_face_proportions['left_eye_proportion']),
                "right_eye_face_proportion": f"{eye_face_proportions['right_eye_proportion']:.2f}%",
                "right_eye_size_classification": self._classify_eye_size(eye_face_proportions['right_eye_proportion']),
                "left_eyebrow_eyelid_proportion": f"{eyebrow_eyelid_distances['left_eyebrow_eyelid_proportion']:.4f}",
                "left_eyebrow_eyelid_classification": self._classify_eyebrow_eyelid_position(eyebrow_eyelid_distances['left_eyebrow_eyelid_proportion']),
                "right_eyebrow_eyelid_proportion": f"{eyebrow_eyelid_distances['right_eyebrow_eyelid_proportion']:.4f}",
                "right_eyebrow_eyelid_classification": self._classify_eyebrow_eyelid_position(eyebrow_eyelid_distances['right_eyebrow_eyelid_proportion']),
                "left_eyebrow_eyelid_distance": f"{eyebrow_eyelid_distances['left_eyebrow_eyelid_distance']:.1f}px",
                "right_eyebrow_eyelid_distance": f"{eyebrow_eyelid_distances['right_eyebrow_eyelid_distance']:.1f}px"
            },
            "eyebrow_analysis": {
                "left_eyebrow_classification": self._classify_eyebrow_length(eyebrow_proportions['left_eyebrow_proportion']),
                "right_eyebrow_classification": self._classify_eyebrow_length(eyebrow_proportions['right_eyebrow_proportion']),
                "left_eyebrow_proportion": eyebrow_proportions['left_eyebrow_proportion'],
                "right_eyebrow_proportion": eyebrow_proportions['right_eyebrow_proportion']
            },
            "eyebrow_slope_analysis": {
                "right_eyebrow_slopes": {
                    "portion_1_angle": f"{slopes['right_eyebrow']['portion_1']:.2f}°",
                    "portion_1_classification": self._label_proportion(slopes['right_eyebrow']['portion_1'], 'portion_1'),
                    "portion_2_angle": f"{slopes['right_eyebrow']['portion_2']:.2f}°",
                    "portion_2_classification": self._label_proportion(slopes['right_eyebrow']['portion_2'], 'portion_2'),
                    "portion_3_angle": f"{slopes['right_eyebrow']['portion_3']:.2f}°",
                    "portion_3_classification": self._label_proportion(slopes['right_eyebrow']['portion_3'], 'portion_3')
                },
                "left_eyebrow_slopes": {
                    "portion_1_angle": f"{slopes['left_eyebrow']['portion_1']:.2f}°",
                    "portion_1_classification": self._label_proportion(slopes['left_eyebrow']['portion_1'], 'portion_1'),
                    "portion_2_angle": f"{slopes['left_eyebrow']['portion_2']:.2f}°",
                    "portion_2_classification": self._label_proportion(slopes['left_eyebrow']['portion_2'], 'portion_2'),
                    "portion_3_angle": f"{slopes['left_eyebrow']['portion_3']:.2f}°",
                    "portion_3_classification": self._label_proportion(slopes['left_eyebrow']['portion_3'], 'portion_3')
                },
                "vertical_reference_used": slopes.get('vertical_reference_used', False),
                "vertical_reference_angle": f"{slopes.get('vertical_reference_angle_deg', 0):.2f}°"
            },
            "mouth_analysis": {
                "mouth_to_eye_relation": self._label_proportion(proportions['mouth_to_eye_proportion'], 'relacion boca - pupilas'),
                "mouth_length_proportion": f"{proportions['mouth_length_proportion']:.4f}",
                "mouth_length_percentage": f"{proportions['mouth_length_proportion'] * 100:.2f}%",
                "mouth_length_classification": self._classify_mouth_length(proportions['mouth_length_proportion']),
                "integral_diagnosis": self._integral_mouth_diagnosis(
                    self._label_proportion(proportions['mouth_to_eye_proportion'], 'relacion boca - pupilas'),
                    self._classify_mouth_length(proportions['mouth_length_proportion'])
                ),
                "left_cupid_arch_proportion": f"{mouth_measurements['left_cupid_arch_proportion']:.4f}",
                "left_cupid_arch_classification": self._classify_cupid_arch(mouth_measurements['left_cupid_arch_proportion']),
                "right_cupid_arch_proportion": f"{mouth_measurements['right_cupid_arch_proportion']:.4f}",
                "right_cupid_arch_classification": self._classify_cupid_arch(mouth_measurements['right_cupid_arch_proportion']),
                "left_cupid_arch_distance": f"{mouth_measurements['left_cupid_arch_distance']:.1f}px",
                "right_cupid_arch_distance": f"{mouth_measurements['right_cupid_arch_distance']:.1f}px",
                "lip_thickness_proportion": f"{mouth_measurements['lip_thickness_proportion']:.4f}",
                "lip_thickness_classification": self._classify_lip_thickness(mouth_measurements['lip_thickness_proportion']),
                "lip_thickness_distance": f"{mouth_measurements['lip_thickness_distance']:.1f}px",
                "lips_ratio": f"{mouth_measurements['lips_ratio']:.4f}",
                "upper_lip_thickness_classification": self._classify_upper_lip_thickness(mouth_measurements['lips_ratio']),
                "upper_lip_distance": f"{mouth_measurements['upper_lip_distance']:.1f}px",
                "lower_lip_distance": f"{mouth_measurements['lower_lip_distance']:.1f}px"
            },
            "face_area_analysis": {
                "inner_outer_percentage": f"{inner_outer_proportions['inner_outer_percentage']:.2f}%",
                "inner_face_size_classification": self._classify_inner_face_size(inner_outer_proportions['inner_outer_percentage']),
                "total_face_area": inner_outer_proportions['outer_area'],
                "inner_face_area": inner_outer_proportions['inner_area']
            },
            "model_integration": {
                "point_2_used": 2 in model_predictions,
                "point_3_used": 3 in model_predictions,
                "point_1_detected": 1 in model_predictions,
                "total_model_points": len(model_predictions),
                "calculated_c1_used": calculated_c1 is not None,
                "c1_calculation": f"X from M2, Y from M9" if calculated_c1 is not None else None
            }
        }
        return summary

    def _classify_eyebrow_length(self, proportion):
        """Classify eyebrow length based on proportion to eye length"""
        if proportion >= 1.4:
            return 'ceja larga'
        elif 1.0 <= proportion < 1.4:
            return 'ceja normal'
        else:
            return 'ceja corta'

    def _classify_mouth_length(self, proportion):
        """Classify mouth length based on proportion to head width (as percentage)"""
        percentage = proportion * 100
        if percentage > 37:
            return 'boca ancha'
        elif 34 <= percentage <= 37:
            return 'boca promedio'
        else:  # percentage < 34
            return 'boca angosta'

    def _integral_mouth_diagnosis(self, mouth_to_eye_classification, mouth_length_classification):
        """
        Integral mouth diagnosis combining mouth_to_eye_proportion and mouth_length_proportion

        Args:
            mouth_to_eye_classification: Classification from mouth_to_eye_proportion
            mouth_length_classification: Classification from mouth_length_proportion

        Returns:
            'boca grande', 'boca pequeña', or 'boca estandar'
        """
        # Check for 'boca grande': both must indicate large mouth
        is_large_relative_to_eyes = 'grande' in mouth_to_eye_classification.lower()
        is_wide_mouth = mouth_length_classification == 'boca ancha'

        if is_large_relative_to_eyes and is_wide_mouth:
            return 'boca grande'

        # Check for 'boca pequeña': both must indicate small mouth
        is_small_relative_to_eyes = 'pequeña' in mouth_to_eye_classification.lower()
        is_narrow_mouth = mouth_length_classification == 'boca angosta'

        if is_small_relative_to_eyes and is_narrow_mouth:
            return 'boca pequeña'

        # Otherwise: standard mouth
        return 'boca estandar'

    def _classify_eye_angle(self, angle):
        """Classify eye angle based on degrees"""
        if -5 <= angle <= 5:
            return 'angulo normal'
        elif angle > 5:
            return 'angulo hacia arriba'
        else:  # angle < -5
            return 'angulo hacia abajo'

    def _classify_eye_size(self, percentage):
        """Classify eye size based on eye-to-face proportion percentage"""
        if percentage < 0.74:
            return 'ojo pequeño'
        elif 0.74 <= percentage <= 0.85:
            return 'ojo mediano'
        else:  # percentage > 0.85
            return 'ojo grande'

    def _classify_inner_face_size(self, percentage):
        """Classify inner face size based on inner-to-outer face proportion percentage"""
        if percentage < 38:
            return 'cara interna pequeña'
        elif 38 <= percentage <= 42.3:
            return 'cara interna promedio'
        else:  # percentage > 43.3
            return 'cara interna grande'

    def _classify_eyebrow_eyelid_position(self, proportion):
        """Classify eyebrow position based on eyebrow-eyelid proportion"""
        if proportion > 0.31:
            return 'high_eyebrows'
        elif 0.225 <= proportion <= 0.31:
            return 'normal_eyebrows'
        else:  # proportion < 0.225
            return 'low_eyebrows'

    def _classify_lip_thickness(self, proportion):
        """Classify lip thickness based on lip thickness proportion (as percentage)"""
        percentage = proportion * 100
        if percentage > 30:
            return 'thick_lips'
        elif 18 <= percentage <= 30:
            return 'normal_lips'
        else:  # percentage < 18
            return 'thin_lips'

    def _classify_upper_lip_thickness(self, ratio):
        """Classify upper lip thickness based on lips ratio (as percentage)"""
        percentage = ratio * 100
        if percentage > 67:
            return 'thick_upper_lip'
        elif 49 <= percentage <= 67:
            return 'normal_upper_lip'
        else:  # percentage < 49
            return 'thin_upper_lip'

    def _classify_cupid_arch(self, proportion):
        """Classify cupid's arch presence based on proportion"""
        if proportion > 1:
            return 'cupid_arch'
        else:  # proportion <= 1
            return 'no_cupid_arch'

    def _convert_slope_to_degrees(self, slope):
        """Convert slope to degrees"""
        if slope == float('inf'):
            return 90
        elif slope == float('-inf'):
            return 270
        else:
            angle_rad = math.atan2(1, slope)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
            return angle_deg

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
            if proportion > 0.76:
                return 'boca grande en relación a las pupilas'
            elif proportion < 0.73:
                return 'boca pequeña en relación a las pupilas'
            else:
                return 'relación boca-pupilas estándar'
        elif section_name == "proporcion interna ojos":
            if proportion < 0.40:
                return 'cercanos'
            elif 0.40 <= proportion <= 0.435:
                return 'standard'
            elif proportion > 0.435:
                return 'lejanos'
        elif section_name in ["portion_1", "portion_2"]:
            if proportion > 5:
                return f'{section_name} - Acendente'
            elif -1 <= proportion <= 5:
                return f'{section_name} - Recto'
            else:  # proportion < -1
                return f'{section_name} - Decendente'
        elif section_name == "portion_3":
            if proportion > 10:
                return f'{section_name} - Acendente'
            elif -20 <= proportion <= 10:
                return f'{section_name} - Normal'
            elif proportion < -20:
                return f'{section_name} - Decendente'
            else:
                return f'{section_name} - Unknown angle'
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

    def get_detailed_analysis_report(self, analysis_results):
        """Generate a detailed text report of all analysis results"""
        if not analysis_results:
            return "No analysis results available"
        
        proportions = analysis_results.get('proportions', {})
        slopes = analysis_results.get('slopes', {})
        eyebrow_props = analysis_results.get('eyebrow_proportions', {})
        eye_angles = analysis_results.get('eye_angles', {})
        eye_face_props = analysis_results.get('eye_face_proportions', {})
        inner_outer_props = analysis_results.get('inner_outer_proportions', {})
        model_preds = analysis_results.get('model_predictions', {})
        summary = analysis_results.get('summary', {})
        
        report = []
        report.append("=== ANÁLISIS ANTROPOMÉTRICO COMPLETO ===\n")
        
        # Facial thirds
        report.append("TERCIOS FACIALES:")
        report.append(f"• Primer tercio: {proportions.get('distance_69_68_proportion', 0):.4f} - {summary.get('facial_thirds', {}).get('primer_tercio', 'N/A')}")
        report.append(f"• Segundo tercio: {proportions.get('distance_68_34_proportion', 0):.4f} - {summary.get('facial_thirds', {}).get('segundo_tercio', 'N/A')}")
        report.append(f"• Tercer tercio: {proportions.get('distance_34_9_proportion', 0):.4f} - {summary.get('facial_thirds', {}).get('tercer_tercio', 'N/A')}")
        report.append("")
        
        # Eye analysis
        eyebrow_eyelid_dists = analysis_results.get('eyebrow_eyelid_distances', {})
        report.append("ANÁLISIS OCULAR:")
        report.append(f"• Proporción interna ojos: {proportions.get('eye_distance_proportion', 0):.4f} - {summary.get('eye_analysis', {}).get('internal_proportion', 'N/A')}")
        report.append(f"• Proporción externa ojos: {proportions.get('outter_eye_distance_proportion', 0):.4f}")
        report.append(f"• Ángulo ojo izquierdo: {eye_angles.get('left_eye_angle', 0):.2f}° - {summary.get('eye_analysis', {}).get('left_eye_angle', 'N/A')}")
        report.append(f"• Ángulo ojo derecho: {eye_angles.get('right_eye_angle', 0):.2f}° - {summary.get('eye_analysis', {}).get('right_eye_angle', 'N/A')}")
        report.append(f"• Proporción ojo izquierdo/cara: {summary.get('eye_analysis', {}).get('left_eye_face_proportion', 'N/A')} - {summary.get('eye_analysis', {}).get('left_eye_size_classification', 'N/A')}")
        report.append(f"• Proporción ojo derecho/cara: {summary.get('eye_analysis', {}).get('right_eye_face_proportion', 'N/A')} - {summary.get('eye_analysis', {}).get('right_eye_size_classification', 'N/A')}")
        report.append(f"• Distancia ceja-párpado izquierdo: {summary.get('eye_analysis', {}).get('left_eyebrow_eyelid_distance', 'N/A')} (proporción: {summary.get('eye_analysis', {}).get('left_eyebrow_eyelid_proportion', 'N/A')})")
        report.append(f"• Distancia ceja-párpado derecho: {summary.get('eye_analysis', {}).get('right_eyebrow_eyelid_distance', 'N/A')} (proporción: {summary.get('eye_analysis', {}).get('right_eyebrow_eyelid_proportion', 'N/A')})")
        report.append("")
        
        # Eyebrow analysis
        report.append("ANÁLISIS DE CEJAS:")
        report.append(f"• Ceja izquierda: {eyebrow_props.get('left_eyebrow_proportion', 0):.4f} - {summary.get('eyebrow_analysis', {}).get('left_eyebrow_classification', 'N/A')}")
        report.append(f"• Ceja derecha: {eyebrow_props.get('right_eyebrow_proportion', 0):.4f} - {summary.get('eyebrow_analysis', {}).get('right_eyebrow_classification', 'N/A')}")
        report.append("")
        
        # Face measurements
        mouth_measurements = analysis_results.get('mouth_measurements', {})
        report.append("MEDIDAS FACIALES:")
        report.append(f"• Ancho facial: {proportions.get('head_width_proportion', 0):.4f}")
        report.append(f"• Longitud boca: {proportions.get('mouth_length_proportion', 0):.4f}")
        report.append(f"• Relación boca-pupila: {proportions.get('mouth_to_eye_proportion', 0):.4f} - {summary.get('mouth_analysis', {}).get('mouth_to_eye_relation', 'N/A')}")
        report.append(f"• Relación mentón-ancho cara: {proportions.get('chin_to_face_width_proportion', 0):.4f}")
        report.append("")

        # Mouth analysis
        report.append("ANÁLISIS BUCAL:")
        report.append(f"• Diagnóstico integral: {summary.get('mouth_analysis', {}).get('integral_diagnosis', 'N/A').upper()}")
        report.append(f"• Longitud boca: {summary.get('mouth_analysis', {}).get('mouth_length_percentage', 'N/A')} - {summary.get('mouth_analysis', {}).get('mouth_length_classification', 'N/A')}")
        report.append(f"• Arco cupido derecho (51-62): {summary.get('mouth_analysis', {}).get('right_cupid_arch_distance', 'N/A')} (proporción: {summary.get('mouth_analysis', {}).get('right_cupid_arch_proportion', 'N/A')})")
        report.append(f"• Arco cupido izquierdo (53-64): {summary.get('mouth_analysis', {}).get('left_cupid_arch_distance', 'N/A')} (proporción: {summary.get('mouth_analysis', {}).get('left_cupid_arch_proportion', 'N/A')})")
        report.append(f"• Grosor de labios (52-58): {summary.get('mouth_analysis', {}).get('lip_thickness_distance', 'N/A')} (proporción: {summary.get('mouth_analysis', {}).get('lip_thickness_proportion', 'N/A')})")
        report.append(f"• Labio superior (52-63): {summary.get('mouth_analysis', {}).get('upper_lip_distance', 'N/A')}")
        report.append(f"• Labio inferior (67-58): {summary.get('mouth_analysis', {}).get('lower_lip_distance', 'N/A')}")
        report.append(f"• Proporción labios (superior/inferior): {summary.get('mouth_analysis', {}).get('lips_ratio', 'N/A')}")
        report.append("")
        
        # Area analysis
        report.append("ANÁLISIS DE ÁREAS:")
        report.append(f"• Área total cara: {inner_outer_props.get('outer_area', 0):.2f} píxeles²")
        report.append(f"• Área interna cara: {inner_outer_props.get('inner_area', 0):.2f} píxeles²")
        report.append(f"• Porcentaje interno/externo: {summary.get('face_area_analysis', {}).get('inner_outer_percentage', 'N/A')} - {summary.get('face_area_analysis', {}).get('inner_face_size_classification', 'N/A')}")
        report.append("")
        
        # Eyebrow angles (now relative to vertical midline)
        if slopes:
            report.append("ÁNGULOS DE CEJAS (relativos a línea media vertical):")
            report.append(f"• Referencia vertical usada: {'M3' if slopes.get('vertical_reference_used') else 'punto 69'}")
            if 'vertical_reference_angle_deg' in slopes:
                report.append(f"• Ángulo de referencia vertical (9-M3): {slopes.get('vertical_reference_angle_deg', 0):.2f}°")

            right_angles = slopes.get('right_eyebrow', {})
            left_angles = slopes.get('left_eyebrow', {})

            for portion in ['portion_1', 'portion_2', 'portion_3']:
                if portion in right_angles:
                    angle = right_angles[portion]
                    label = self._label_proportion(angle, portion)
                    report.append(f"• Ceja derecha {portion}: {angle:.2f}° - {label}")

            for portion in ['portion_1', 'portion_2', 'portion_3']:
                if portion in left_angles:
                    angle = left_angles[portion]
                    label = self._label_proportion(angle, portion)
                    report.append(f"• Ceja izquierda {portion}: {angle:.2f}° - {label}")
            report.append("")
        
        # Model integration
        report.append("INTEGRACIÓN DEL MODELO:")
        report.append(f"• Puntos del modelo detectados: {summary.get('model_integration', {}).get('total_model_points', 0)}")
        report.append(f"• Punto 1 detectado: {'YES' if summary.get('model_integration', {}).get('point_1_detected') else 'NO'}")
        report.append(f"• Punto 2 usado (entre cejas): {'YES' if summary.get('model_integration', {}).get('point_2_used') else 'NO'}")
        report.append(f"• Punto 3 usado (parte superior cabeza): {'YES' if summary.get('model_integration', {}).get('point_3_used') else 'NO'}")
        report.append(f"• Punto C1 calculado (X de M2, Y de M9): {'YES' if summary.get('model_integration', {}).get('calculated_c1_used') else 'NO'}")
        if summary.get('model_integration', {}).get('calculated_c1_used'):
            report.append("  → C1 evita interferencia de entradas en el cabello")
        
        if model_preds:
            report.append("")
            report.append("TODOS LOS PUNTOS DEL MODELO DETECTADOS:")
            for label, point in model_preds.items():
                report.append(f"• Punto modelo {label}: {point}")
        
        return "\n".join(report)
