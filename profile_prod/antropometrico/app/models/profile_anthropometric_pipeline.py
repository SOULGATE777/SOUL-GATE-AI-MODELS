import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import torchvision
from math import degrees, atan2
import base64
import io
from typing import Dict, List, Tuple, Optional

class MinimalPointModel(nn.Module):
    """Minimal point detection model (model 3 only)"""
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

class ProfileAnthropometricPipeline:
    """Profile-specific anthropometric analysis pipeline with comprehensive measurements"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the point detection model
        self.model = self._load_model(model_path)
        print("Profile anthropometric model loaded successfully!")
    
    def _load_model(self, model_path: str):
        """Load the point detection model"""
        print("Loading profile point detection model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        all_classes = checkpoint['all_classes']
        num_keypoints = checkpoint['num_keypoints']
        heatmap_size = checkpoint.get('heatmap_size', 112)
        
        # Create model
        model = MinimalPointModel(num_keypoints)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.point_classes = all_classes
        self.heatmap_size = heatmap_size
        print(f"Point model loaded with {len(all_classes)} point classes")
        return model
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 224) -> Tuple[np.ndarray, torch.Tensor]:
        """Preprocess image for analysis"""
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            original_image = image.copy()
        else:
            # Convert if needed
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize for model
        resized_image = cv2.resize(original_image, (target_size, target_size))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(resized_image.transpose((2, 0, 1))).float() / 255.0
        
        return original_image, image_tensor.unsqueeze(0).to(self.device)
    
    def detect_points(self, image_tensor: torch.Tensor) -> List[Dict]:
        """Detect anthropometric points"""
        with torch.no_grad():
            heatmaps, profile_logits = self.model(image_tensor)
        
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
                    'coordinates': point.tolist(),
                    'confidence': float(conf)
                })
        
        return detected_points
    
    def filter_spurious_predictions(self, detected_points: List[Dict]) -> Tuple[List[Dict], str]:
        """Filter out spurious predictions from the minority side"""
        # Count points by suffix
        left_count = sum(1 for p in detected_points if p['class'].endswith('_i'))
        right_count = sum(1 for p in detected_points if p['class'].endswith('_d'))
        
        print(f"Point count - Left (_i): {left_count}, Right (_d): {right_count}")
        
        # Determine the dominant side
        if left_count > right_count:
            dominant_suffix = '_i'
            actual_profile = 'left'
        elif right_count > left_count:
            dominant_suffix = '_d'
            actual_profile = 'right'
        else:
            # If equal, determine from vector analysis later
            dominant_suffix = None
            actual_profile = 'unknown'
        
        print(f"Dominant side: {actual_profile}")
        
        # Filter points to keep only the dominant side or points without suffix
        filtered_points = []
        removed_points = []
        
        for point in detected_points:
            class_name = point['class']
            
            if dominant_suffix is None:
                # If we can't determine dominant side, keep all points for now
                clean_class = class_name.replace('_i', '').replace('_d', '')
                point_copy = point.copy()
                point_copy['class'] = clean_class
                filtered_points.append(point_copy)
            else:
                # Keep points that either have the dominant suffix or no suffix at all
                if class_name.endswith(dominant_suffix) or not (class_name.endswith('_i') or class_name.endswith('_d')):
                    # Remove suffix for processing
                    clean_class = class_name.replace('_i', '').replace('_d', '')
                    point_copy = point.copy()
                    point_copy['class'] = clean_class
                    filtered_points.append(point_copy)
                else:
                    removed_points.append(point)
        
        if removed_points:
            print(f"Removed {len(removed_points)} spurious predictions from minority side:")
            for p in removed_points:
                print(f"  - {p['class']} (conf: {p['confidence']:.3f})")
        
        return filtered_points, actual_profile
    
    def create_points_dict(self, filtered_points: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """Create a dictionary mapping point names to coordinates"""
        points = {}
        for point in filtered_points:
            points[point['class']] = tuple(point['coordinates'])
        return points
    
    def dist(self, p1: Optional[Tuple[float, float]], p2: Optional[Tuple[float, float]]) -> Optional[float]:
        """Calculate Euclidean distance between two points"""
        if p1 and p2:
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return None
    
    def perform_anthropometric_analysis(self, points: Dict[str, Tuple[float, float]]) -> Dict:
        """Perform comprehensive anthropometric measurements including all new features"""
        print("\n=== COMPREHENSIVE PROFILE ANTHROPOMETRIC ANALYSIS ===")
        
        # Get all required points
        point_1 = points.get("1", None)
        point_2 = points.get("2", None)
        point_3 = points.get("3", None)
        point_4 = points.get("4", None)
        point_5 = points.get("5", None)
        point_6 = points.get("6", None)
        point_7 = points.get("7", None)
        point_8 = points.get("8", None)
        point_9 = points.get("9", None)
        point_10 = points.get("10", None)
        point_11 = points.get("11", None)
        point_16 = points.get("16", None)
        point_17 = points.get("17", None)
        point_18 = points.get("18", None)
        point_19 = points.get("19", None)
        point_22 = points.get("22", None)
        point_24 = points.get("24", None)
        point_26 = points.get("26", None)
        point_30 = points.get("30", None)
        point_33 = points.get("33", None)
        point_34 = points.get("34", None)
        
        measurements = {}
        
        print("\n--- BASIC MEASUREMENTS ---")
        
        # Calculate reference distance (24 to 10)
        if point_24 and point_10:
            reference_distance = self.dist(point_24, point_10)
            measurements['reference_distance'] = reference_distance
            print(f"Reference distance (24 to 10): {reference_distance:.2f}")
        else:
            print("Warning: Reference points 24 and 10 not found. Cannot perform normalized measurements.")
            return measurements
        
        # === NOSE ANALYSIS ===
        print("\n--- NOSE ANALYSIS ---")
        if point_18 and point_17:
            distance_18_17 = self.dist(point_18, point_17)
            normalized_distance_18_17 = distance_18_17 / reference_distance
            measurements['nose_distance'] = distance_18_17
            measurements['nose_normalized'] = normalized_distance_18_17
            
            print(f"Distance between points 18 and 17: {distance_18_17:.2f}")
            print(f"Proportion between points 18 and 17: {normalized_distance_18_17:.3f}")
            
            # Classify nose
            if normalized_distance_18_17 > 0.2:
                nose_label = "nariz protruyente"
            elif 0.17 <= normalized_distance_18_17 <= 0.2:
                nose_label = "nariz normal"
            else:
                nose_label = "nariz corta"
            
            measurements['nose_classification'] = nose_label
            print(f"Nose classification: {nose_label}")
        
        # === FACIAL THIRDS ANALYSIS ===
        print("\n--- FACIAL THIRDS ANALYSIS ---")
        
        # Upper third (34 to 22)
        if point_34 and point_22:
            distance_34_22 = self.dist(point_34, point_22)
            normalized_distance_34_22 = distance_34_22 / reference_distance
            measurements['tercio_superior_distance'] = distance_34_22
            measurements['tercio_superior_normalized'] = normalized_distance_34_22
            
            print(f"Upper third distance (34-22): {distance_34_22:.2f}")
            print(f"Upper third proportion: {normalized_distance_34_22:.3f}")
            print(f"Upper third label: tercio superior")
        
        # Middle third (22 to 18)
        if point_22 and point_18:
            distance_22_18 = self.dist(point_22, point_18)
            normalized_distance_22_18 = distance_22_18 / reference_distance
            measurements['tercio_medio_distance'] = distance_22_18
            measurements['tercio_medio_normalized'] = normalized_distance_22_18
            
            print(f"Middle third distance (22-18): {distance_22_18:.2f}")
            print(f"Middle third proportion: {normalized_distance_22_18:.3f}")
            print(f"Middle third label: tercio medio")
        
        # Lower third (18 to 10)
        if point_18 and point_10:
            distance_18_10 = self.dist(point_18, point_10)
            normalized_distance_18_10 = distance_18_10 / reference_distance
            measurements['tercio_inferior_distance'] = distance_18_10
            measurements['tercio_inferior_normalized'] = normalized_distance_18_10
            
            print(f"Lower third distance (18-10): {distance_18_10:.2f}")
            print(f"Lower third proportion: {normalized_distance_18_10:.3f}")
            print(f"Lower third label: tercio inferior")
            
            # === MANDIBLE ANALYSIS ===
            print("\n--- MANDIBLE ANALYSIS ---")
            if point_3 and point_9:
                distance_3_9 = self.dist(point_3, point_9)
                normalized_distance_3_9 = distance_3_9 / distance_18_10
                measurements['mandibula_distance'] = distance_3_9
                measurements['mandibula_normalized'] = normalized_distance_3_9
                
                print(f"Mandible width (3-9): {distance_3_9:.2f}")
                print(f"Mandible proportion to lower third: {normalized_distance_3_9:.3f}")
                
                # Classify mandibula
                if normalized_distance_3_9 >= 0.8:
                    mandibula_label = "Mandibula Sanguinea"
                elif 0.75 <= normalized_distance_3_9 < 0.8:
                    mandibula_label = "Mandibula intermedia sanguineo/bilosa"
                elif 0.40 <= normalized_distance_3_9 < 0.75:
                    mandibula_label = "Mandibula Bilosa"
                elif 0.35 <= normalized_distance_3_9 < 0.40:
                    mandibula_label = "Mandibula intermedia bilosa/nerviosa"
                elif normalized_distance_3_9 < 0.35:
                    mandibula_label = "Mandibula Nerviosa"
                else:
                    mandibula_label = "Mandibula Intermedia"
                
                measurements['mandibula_classification'] = mandibula_label
                print(f"Mandible classification: {mandibula_label}")
        
        # === EAR ANALYSIS ===
        print("\n--- EAR ANALYSIS ---")
        
        # Basic ear width
        if point_2 and point_6:
            ear_width = self.dist(point_2, point_6)
            measurements['ear_width'] = ear_width
            print(f"Ear width (2-6): {ear_width:.2f}")
        
        # === NEW COMPREHENSIVE MEASUREMENTS ===
        print("\n--- ADDITIONAL MEASUREMENTS ---")
        
        # 1) Ear length proportion calculation
        if point_4 and point_5 and point_10 and point_24:
            ear_length = self.dist(point_4, point_5)
            face_length = self.dist(point_10, point_24)  # Same as reference_distance
            
            ear_length_proportion = ear_length / face_length
            measurements['ear_length'] = ear_length
            measurements['ear_length_proportion'] = ear_length_proportion
            
            print(f"Ear length (4-5): {ear_length:.2f}")
            print(f"Face length (10-24): {face_length:.2f}")
            print(f"Ear length proportion: {ear_length_proportion:.3f}")
            
            # Classify ear length
            if ear_length_proportion > 0.432:
                ear_length_label = "oreja larga"
            elif 0.38 <= ear_length_proportion <= 0.432:
                ear_length_label = "oreja normal"
            else:  # <= 0.38
                ear_length_label = "oreja corta"
            
            measurements['ear_length_classification'] = ear_length_label
            print(f"Ear length classification: {ear_length_label}")
        else:
            print("Missing points for ear length calculation (need points 4, 5, 10, 24)")
        
        # 2) Ear lobe proportion calculation
        if point_3 and point_5 and point_4:
            lobe_length = self.dist(point_3, point_5)
            if 'ear_length' not in measurements:
                ear_length = self.dist(point_4, point_5)
            else:
                ear_length = measurements['ear_length']
            
            ear_lobe_proportion = lobe_length / ear_length
            measurements['ear_lobe_length'] = lobe_length
            measurements['ear_lobe_proportion'] = ear_lobe_proportion
            
            print(f"Ear lobe length (3-5): {lobe_length:.2f}")
            print(f"Ear lobe proportion: {ear_lobe_proportion:.3f}")
            
            # Classify ear lobe
            if ear_lobe_proportion > 0.31:
                ear_lobe_label = "lobulo grande"
            elif 0.28 <= ear_lobe_proportion <= 0.31:
                ear_lobe_label = "lobulo normal"
            else:  # < 0.28
                ear_lobe_label = "lobulo chico"
            
            measurements['ear_lobe_classification'] = ear_lobe_label
            print(f"Ear lobe classification: {ear_lobe_label}")
        else:
            print("Missing points for ear lobe calculation (need points 3, 4, 5)")
        
        # 3) Nasal orifice triangulation calculation
        if point_26 and point_17 and point_18 and point_30:
            nasal_orifice_distance = self.dist(point_26, point_17)
            nose_reference_distance = self.dist(point_18, point_30)
            
            nasal_orifice_proportion = nasal_orifice_distance / nose_reference_distance
            measurements['nasal_orifice_distance'] = nasal_orifice_distance
            measurements['nose_reference_distance'] = nose_reference_distance
            measurements['nasal_orifice_proportion'] = nasal_orifice_proportion
            
            print(f"Nasal orifice distance (26-17): {nasal_orifice_distance:.2f}")
            print(f"Nose reference distance (18-30): {nose_reference_distance:.2f}")
            print(f"Nasal orifice proportion: {nasal_orifice_proportion:.3f}")
            
            # Classify nasal orifice triangulation
            if nasal_orifice_proportion > 0.27:
                nasal_triangulation_label = "triangulacion de fosa"
            else:  # <= 0.27
                nasal_triangulation_label = "sin triangulacion de fosa"
            
            measurements['nasal_triangulation_classification'] = nasal_triangulation_label
            print(f"Nasal triangulation classification: {nasal_triangulation_label}")
        else:
            print("Missing points for nasal orifice calculation (need points 26, 17, 18, 30)")
        
        # 4) Enhanced tragus-antitragus proportion calculation
        if point_7 and point_8 and point_2 and point_6:
            distance_7_8 = self.dist(point_7, point_8)
            if 'ear_width' not in measurements:
                ear_width = self.dist(point_2, point_6)
            else:
                ear_width = measurements['ear_width']
            
            tragus_antitragus_proportion = distance_7_8 / ear_width
            measurements['tragus_antitragus_distance'] = distance_7_8
            measurements['tragus_antitragus_proportion'] = tragus_antitragus_proportion
            
            print(f"Tragus-antitragus distance (7-8): {distance_7_8:.2f}")
            print(f"Ear width (2-6): {ear_width:.2f}")
            print(f"Tragus-antitragus proportion: {tragus_antitragus_proportion:.3f}")
            
            # Classify tragus-antitragus
            if tragus_antitragus_proportion >= 0.255:
                tragus_antitragus_label = "grande"
            elif 0.22 <= tragus_antitragus_proportion < 0.255:
                tragus_antitragus_label = "normal"
            else:  # < 0.22
                tragus_antitragus_label = "corta"
            
            measurements['tragus_antitragus_classification'] = tragus_antitragus_label
            print(f"Tragus-antitragus classification: {tragus_antitragus_label}")
        else:
            print("Missing points for tragus-antitragus calculation (need points 7, 8, 2, 6)")
        
        # Angular measurements
        self.calculate_angular_measurements(points, measurements)
        
        return measurements
    
    def calculate_angular_measurements(self, points: Dict[str, Tuple[float, float]], measurements: Dict):
        """Calculate all angular measurements with proper left/right profile handling via vector analysis"""
        print("\n--- ANGULAR ANALYSIS ---")
        
        point_22 = points.get("22", None)
        point_18 = points.get("18", None)
        point_17 = points.get("17", None)
        point_16 = points.get("16", None)
        point_19 = points.get("19", None)
        point_24 = points.get("24", None)
        point_11 = points.get("11", None)
        point_1 = points.get("1", None)
        point_3 = points.get("3", None)
        point_4 = points.get("4", None)
        point_5 = points.get("5", None)
        point_33 = points.get("33", None)
        point_9 = points.get("9", None)
        point_37 = points.get("37", None)
        point_38 = points.get("38", None)
        point_39 = points.get("39", None)
        
        if not (point_24 and point_18):
            print("Warning: Cannot calculate angular measurements without points 24 and 18")
            return
        
        # Reference vector (24 to 18)
        vector_24_18 = np.array([point_18[0] - point_24[0], point_18[1] - point_24[1]])
        
        # Determine head direction based on the reference vector (infallible method)
        head_direction = "right" if vector_24_18[0] > 0 else "left"
        print(f"Head direction determined via vector analysis: {head_direction}")
        measurements['head_direction'] = head_direction
        
        # Nose tip angle (18 to 17) - using your original logic
        if point_17 and point_18:
            vector_18_17 = np.array([point_17[0] - point_18[0], point_17[1] - point_18[1]])
            
            # Calculate the perpendicular slope (negative reciprocal)
            ref_slope = vector_24_18[1] / vector_24_18[0] if vector_24_18[0] != 0 else float('inf')
            perp_slope = -1/ref_slope if ref_slope != 0 else float('inf')
            
            # Create a perpendicular vector at point 18
            perp_vector = np.array([1, perp_slope])
            
            # Normalize both vectors for dot product calculation
            v1_u = vector_18_17 / np.linalg.norm(vector_18_17)
            v2_u = perp_vector / np.linalg.norm(perp_vector)
            
            # Calculate angle using dot product formula
            cos_angle = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            
            # Fix the angle sign based on the direction (from your original code)
            if point_17[0] < point_18[0]:  # Head is turning right
                angle_degrees = -angle_degrees
            elif point_17[0] > point_18[0]:  # Head is turning left
                angle_degrees = abs(angle_degrees)
            
            # Normalize the angle to be between -90 and +90 degrees
            if angle_degrees > 90:
                angle_degrees = -(180 - angle_degrees)
            elif angle_degrees < -90:
                angle_degrees = 180 + angle_degrees
            
            measurements['nose_tip_angle'] = angle_degrees
            print(f"Reference line slope (24-18): {ref_slope:.2f}")
            print(f"Perpendicular line slope: {perp_slope:.2f}")
            print(f"Nose tip angle: {angle_degrees:.2f} degrees")
            
            # Classify nose tip
            if angle_degrees >= 27:
                nose_tip_label = "punta de nariz hacia arriba"
            elif 12 <= angle_degrees < 27:
                nose_tip_label = "punta de nariz promedio"
            else:  # < 12
                nose_tip_label = "punta hacia abajo"
            
            measurements['nose_tip_classification'] = nose_tip_label
            print(f"Nose tip classification: {nose_tip_label}")
        
        # Calculate slope for 'angulo de nariz' (19 to 17)
        if point_17 and point_19:
            def calculate_slope(p1, p2):
                if p1 and p2:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    return dy / dx if dx != 0 else float('inf')
                return None
            
            slope_19_17 = calculate_slope(point_17, point_19)
            if slope_19_17 is not None:
                measurements['nose_slope_19_17'] = slope_19_17
                print(f"Nose angle slope (19-17): {slope_19_17:.2f}")
                angle_19_17 = degrees(atan2(slope_19_17, 1))
                measurements['nose_angle_19_17'] = angle_19_17
                print(f"Nose angle (19-17): {angle_19_17:.2f} degrees")
        
        # Forehead angle (24 to 33) - with proper left/right handling
        if point_24 and point_33:
            vector_24_33 = np.array([point_33[0] - point_24[0], point_33[1] - point_24[1]])
            
            # Normalize both vectors for dot product calculation
            v1_u = vector_24_18 / np.linalg.norm(vector_24_18)
            v2_u = vector_24_33 / np.linalg.norm(vector_24_33)
            
            # Calculate angle using dot product formula
            cos_angle = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            
            # Adjust angle based on head direction (from your original logic)
            if head_direction == "left":
                if vector_24_33[0] < vector_24_18[0]:  # Turning right
                    angle_degrees = abs(angle_degrees)
                else:  # Turning left
                    angle_degrees = -angle_degrees
            else:  # Right side of face
                if vector_24_33[0] < vector_24_18[0]:  # Turning right
                    angle_degrees = -angle_degrees
                else:  # Turning left
                    angle_degrees = abs(angle_degrees)
            
            # Normalize angle to be between 0 and 90 degrees
            if angle_degrees < 0:
                angle_degrees = abs(angle_degrees)
            if angle_degrees > 90:
                angle_degrees = 180 - angle_degrees
            
            measurements['forehead_angle'] = angle_degrees
            print(f"Forehead angle (24-33): {angle_degrees:.2f} degrees")
            
            # Classify forehead using np.select logic from original
            if angle_degrees > 15:
                forehead_label = 'frente inclinada hacia atras'
            elif 11 <= angle_degrees <= 15:
                forehead_label = 'frente neutra'
            else:  # < 11
                forehead_label = 'frente vertical'
            
            measurements['forehead_classification'] = forehead_label
            print(f"Forehead classification: {forehead_label}")
        
        # Chin angle (18 to 11) - with proper face side detection
        if point_18 and point_11:
            vector_18_11 = np.array([point_11[0] - point_18[0], point_11[1] - point_18[1]])
            
            # Normalize both vectors for dot product calculation
            v1_u = vector_24_18 / np.linalg.norm(vector_24_18)
            v2_u = vector_18_11 / np.linalg.norm(vector_18_11)
            
            # Calculate angle using dot product formula
            cos_angle = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            
            # Determine which side of face and adjust angle accordingly (from your original logic)
            if point_17 and point_17[0] < point_18[0]:  # Left side of face
                # On left side, positive angles mean turning right
                if vector_18_11[0] < vector_24_18[0]:  # Turning right
                    angle_degrees = abs(angle_degrees)
                else:  # Turning left
                    angle_degrees = -angle_degrees
            else:  # Right side of face
                # On right side, positive angles mean turning left
                if vector_18_11[0] < vector_24_18[0]:  # Turning right
                    angle_degrees = -angle_degrees
                else:  # Turning left
                    angle_degrees = abs(angle_degrees)
            
            # Normalize angle to be between -90 and +90 degrees
            if angle_degrees > 90:
                angle_degrees = -(180 - angle_degrees)
            elif angle_degrees < -90:
                angle_degrees = 180 + angle_degrees
            
            measurements['chin_angle'] = angle_degrees
            print(f"Chin angle (18-11): {angle_degrees:.2f} degrees")
            
            # Classify chin using the exact logic from your original code
            if angle_degrees <= -5:
                chin_label = 'menton nervioso'
            elif -5 < angle_degrees <= 5.5:
                chin_label = 'menton biloso/linfatico'
            else:  # > 5.5
                chin_label = 'menton sanguineo'
            
            measurements['chin_classification'] = chin_label
            print(f"Chin classification: {chin_label}")
        
        # Mandible angular analysis (24-18 vector vs 3-9 vector intersection)
        if all([point_24, point_18, point_3, point_9]):
            mandible_results = self.calculate_mandible_angular_analysis(
                point_24, point_18, point_3, point_9, head_direction
            )
            
            if mandible_results[0] is not None:
                mandible_angle, ref_slope, mandible_slope = mandible_results
                measurements['mandible_intersection_angle'] = mandible_angle
                measurements['reference_vector_slope'] = ref_slope
                measurements['mandible_vector_slope'] = mandible_slope
                print(f"Mandible intersection angle (24_18 vs 3_9): {mandible_angle:.2f} degrees")
                print(f"Reference slope: {ref_slope:.3f}, Mandible slope: {mandible_slope:.3f}")
                
                # Optional: Add classification based on angle ranges
                if mandible_angle < 70:
                    mandible_angle_class = "acute mandible angle"
                elif 70 <= mandible_angle <= 110:
                    mandible_angle_class = "normal mandible angle"  
                else:
                    mandible_angle_class = "obtuse mandible angle"
                
                measurements['mandible_angle_classification'] = mandible_angle_class
                print(f"Mandible angle classification: {mandible_angle_class}")
        
        # Ear implantation angular analysis (24-18 vector vs 1-3 vector intersection)
        if all([point_24, point_18, point_1, point_3]):
            ear_implantation_results = self.calculate_ear_implantation_angular_analysis(
                point_24, point_18, point_1, point_3, head_direction
            )
            
            if ear_implantation_results[0] is not None:
                ear_implantation_angle, ref_slope, ear_slope = ear_implantation_results
                measurements['ear_implantation_intersection_angle'] = ear_implantation_angle
                measurements['ear_implantation_vector_slope'] = ear_slope
                print(f"Ear implantation intersection angle (24_18 vs 1_3): {ear_implantation_angle:.2f} degrees")
                print(f"Reference slope: {ref_slope:.3f}, Ear implantation slope: {ear_slope:.3f}")
                
                # Optional: Add classification based on angle ranges
                if ear_implantation_angle < 60:
                    ear_implantation_class = "acute ear implantation"
                elif 60 <= ear_implantation_angle <= 120:
                    ear_implantation_class = "normal ear implantation"  
                else:
                    ear_implantation_class = "obtuse ear implantation"
                
                measurements['ear_implantation_angle_classification'] = ear_implantation_class
                print(f"Ear implantation classification: {ear_implantation_class}")
        
        # Eye protrusion analysis (orbital plane method)
        if all([point_37, point_38, point_39]):
            # Angular analysis for reference (slopes and angle)
            eye_protrusion_results = self.calculate_eye_protrusion_angular_analysis(
                point_39, point_37, point_38, point_37, head_direction
            )

            if eye_protrusion_results[0] is not None:
                eye_protrusion_angle, vector_39_37_slope, vector_38_37_slope = eye_protrusion_results
                measurements['eye_protrusion_intersection_angle'] = eye_protrusion_angle
                measurements['vector_39_37_slope'] = vector_39_37_slope
                measurements['vector_38_37_slope'] = vector_38_37_slope
                print(f"Eye protrusion intersection angle (39_37 vs 38_37): {eye_protrusion_angle:.2f} degrees")
                print(f"Vector 39-37 slope: {vector_39_37_slope:.3f}, Vector 38-37 slope: {vector_38_37_slope:.3f}")

            # Main classification using orbital plane crossing method
            eye_protrusion_classification_results = self.calculate_eye_protrusion_classification(
                point_37, point_39, point_38, head_direction
            )

            if eye_protrusion_classification_results[0] is not None:
                eye_protrusion_distance, eye_protrusion_classification = eye_protrusion_classification_results
                measurements['eye_protrusion_distance'] = eye_protrusion_distance
                measurements['eye_protrusion_classification'] = eye_protrusion_classification
                print(f"Eye protrusion distance from orbital plane: {eye_protrusion_distance:.2f} pixels")
                print(f"Eye protrusion classification: {eye_protrusion_classification}")
        
        # Implantation angles with head direction consideration
        if all([point_24, point_18, point_4, point_5, point_1, point_3]):
            implantation_results = self.calculate_implantation_angles(
                point_24, point_18, point_4, point_5, point_1, point_3, head_direction
            )
            
            if implantation_results:
                (angle_superior, classification_superior,
                 angle_inferior, classification_inferior,
                 angle_intersection, classification_intersection) = implantation_results
                
                measurements['implantation_superior_angle'] = angle_superior
                measurements['implantation_superior_classification'] = classification_superior
                measurements['implantation_inferior_angle'] = angle_inferior
                measurements['implantation_inferior_classification'] = classification_inferior
                measurements['intersection_angle'] = angle_intersection
                measurements['intersection_classification'] = classification_intersection
                
                print(f"Superior implantation angle: {angle_superior:.2f} degrees")
                print(f"Superior implantation classification: {classification_superior}")
                print(f"Inferior implantation angle: {angle_inferior:.2f} degrees")
                print(f"Inferior implantation classification: {classification_inferior}")
                print(f"Vector intersection angle (24_18 and 1_3): {angle_intersection:.2f} degrees")
    
    def calculate_implantation_angles(self, point_24, point_18, point_4, point_5, point_1, point_3, head_direction=None):
        """
        Calculate both superior and inferior implantation angles relative to the line from 24 to 18.
        Also calculates intersection angle between vectors 24_18 and 1_3.
        Takes into account head direction for proper angle adjustment.
        Returns angles and classifications for both superior and inferior positions.
        """
        if not all([point_24, point_18, point_4, point_5, point_1, point_3]):
            return None
        
        # Calculate vectors
        vector_24_18 = np.array([point_18[0] - point_24[0], point_18[1] - point_24[1]])
        vector_24_4 = np.array([point_4[0] - point_24[0], point_4[1] - point_24[1]])
        vector_18_5 = np.array([point_5[0] - point_18[0], point_5[1] - point_18[1]])
        vector_1_3 = np.array([point_1[0] - point_3[0], point_1[1] - point_3[1]])
        
        # Calculate slopes
        ref_slope = vector_24_18[1] / vector_24_18[0] if vector_24_18[0] != 0 else float('inf')
        perp_slope = -1/ref_slope if ref_slope != 0 else float('inf')
        
        # Create perpendicular vector at point 18
        perp_vector = np.array([1, perp_slope])
        
        # Calculate superior angle (point 4) - using 0-360° system
        v1_u_superior = vector_24_4 / np.linalg.norm(vector_24_4)
        v2_u = perp_vector / np.linalg.norm(perp_vector)
        angle_degrees_superior = np.degrees(np.arctan2(np.cross(v1_u_superior, v2_u), np.dot(v1_u_superior, v2_u)))
        if angle_degrees_superior < 0:
            angle_degrees_superior += 360
        
        # Calculate inferior angle (point 5) - using 0-360° system
        v1_u_inferior = vector_18_5 / np.linalg.norm(vector_18_5)
        angle_degrees_inferior = np.degrees(np.arctan2(np.cross(v1_u_inferior, v2_u), np.dot(v1_u_inferior, v2_u)))
        if angle_degrees_inferior < 0:
            angle_degrees_inferior += 360
        
        # Calculate intersection angle between vectors 24_18 and 1_3
        v1_u_intersection = vector_24_18 / np.linalg.norm(vector_24_18)
        v2_u_intersection = vector_1_3 / np.linalg.norm(vector_1_3)
        cos_angle_intersection = np.clip(np.dot(v1_u_intersection, v2_u_intersection), -1.0, 1.0)
        angle_radians_intersection = np.arccos(abs(cos_angle_intersection))  # Use abs() to always get acute angle
        angle_degrees_intersection = np.degrees(angle_radians_intersection)
        
        # Detect face direction if not provided
        if head_direction is None:
            head_direction = "right" if vector_24_18[0] > 0 else "left"
        
        # Adjust angles based on head direction (from your original logic)
        if head_direction in ["left", "right"]:
            angle_degrees_intersection = abs(angle_degrees_intersection)
        
        # Normalize intersection angle to 0 to 180 degrees (keep implantacion angles in 0-360°)
        if angle_degrees_intersection > 90:
            angle_degrees_intersection = 180 - angle_degrees_intersection
        elif angle_degrees_intersection < 0:
            angle_degrees_intersection = abs(angle_degrees_intersection)
        
        # Classify angles (converted from negative thresholds to 0-360° system)
        classification_superior = "implantacion alta" if angle_degrees_superior >= 351 else "implantacion estandard"
        classification_inferior = "implantacion baja" if angle_degrees_inferior >= 350 else "implantacion estandard"
        
        # Classify intersection angle
        if angle_degrees_intersection > 90:
            classification_intersection = "wide intersection"
        elif angle_degrees_intersection == 90:
            classification_intersection = "right intersection"
        else:
            classification_intersection = "acute intersection"
        
        return (angle_degrees_superior, classification_superior, 
                angle_degrees_inferior, classification_inferior,
                angle_degrees_intersection, classification_intersection)
    
    def calculate_mandible_angular_analysis(self, point_24, point_18, point_3, point_9, head_direction=None):
        """
        Calculate the angle and slopes between reference vector (24-18) and mandible vector (3-9).
        This represents the intersection angle if both vectors were extended.
        
        Args:
            point_24: Reference start point (24)
            point_18: Reference end point (18)
            point_3: Mandible start point (3)
            point_9: Mandible end point (9)
            head_direction: Left/right profile direction
            
        Returns:
            tuple: (angle_degrees, reference_slope, mandible_slope)
        """
        if not all([point_24, point_18, point_3, point_9]):
            return None, None, None
        
        # Create the two vectors
        vector_24_18 = np.array([point_18[0] - point_24[0], point_18[1] - point_24[1]])
        vector_3_9 = np.array([point_9[0] - point_3[0], point_9[1] - point_3[1]])
        
        # Calculate slopes (rise/run)
        reference_slope = vector_24_18[1] / vector_24_18[0] if vector_24_18[0] != 0 else float('inf')
        mandible_slope = vector_3_9[1] / vector_3_9[0] if vector_3_9[0] != 0 else float('inf')
        
        # Normalize both vectors
        v1_u = vector_24_18 / np.linalg.norm(vector_24_18)
        v2_u = vector_3_9 / np.linalg.norm(vector_3_9)
        
        # Calculate angle using dot product
        cos_angle = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle_radians = np.arccos(abs(cos_angle))  # Use abs to always get acute angle
        angle_degrees = np.degrees(angle_radians)
        
        # Ensure angle is between 0 and 180 degrees (intersection angle)
        if angle_degrees > 90:
            angle_degrees = 180 - angle_degrees
        
        print(f"Reference vector (24-18): [{vector_24_18[0]:.2f}, {vector_24_18[1]:.2f}], slope: {reference_slope:.3f}")
        print(f"Mandible vector (3-9): [{vector_3_9[0]:.2f}, {vector_3_9[1]:.2f}], slope: {mandible_slope:.3f}")
        print(f"Mandible intersection angle: {angle_degrees:.2f} degrees")
        
        return angle_degrees, reference_slope, mandible_slope
    
    def calculate_ear_implantation_angular_analysis(self, point_24, point_18, point_1, point_3, head_direction=None):
        """
        Calculate the angle and slopes between reference vector (24-18) and ear implantation vector (1-3).
        This represents the intersection angle if both vectors were extended downwards.
        
        Args:
            point_24: Reference start point (24)
            point_18: Reference end point (18)
            point_1: Ear implantation start point (1)
            point_3: Ear implantation end point (3)
            head_direction: Left/right profile direction
            
        Returns:
            tuple: (angle_degrees, reference_slope, ear_implantation_slope)
        """
        if not all([point_24, point_18, point_1, point_3]):
            return None, None, None
        
        # Create the two vectors
        vector_24_18 = np.array([point_18[0] - point_24[0], point_18[1] - point_24[1]])
        vector_1_3 = np.array([point_3[0] - point_1[0], point_3[1] - point_1[1]])
        
        # Calculate slopes (rise/run)
        reference_slope = vector_24_18[1] / vector_24_18[0] if vector_24_18[0] != 0 else float('inf')
        ear_implantation_slope = vector_1_3[1] / vector_1_3[0] if vector_1_3[0] != 0 else float('inf')
        
        # Normalize both vectors
        v1_u = vector_24_18 / np.linalg.norm(vector_24_18)
        v2_u = vector_1_3 / np.linalg.norm(vector_1_3)
        
        # Calculate angle using dot product
        cos_angle = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle_radians = np.arccos(abs(cos_angle))  # Use abs to always get acute angle
        angle_degrees = np.degrees(angle_radians)
        
        # Ensure angle is between 0 and 180 degrees (intersection angle)
        if angle_degrees > 90:
            angle_degrees = 180 - angle_degrees
        
        print(f"Reference vector (24-18): [{vector_24_18[0]:.2f}, {vector_24_18[1]:.2f}], slope: {reference_slope:.3f}")
        print(f"Ear implantation vector (1-3): [{vector_1_3[0]:.2f}, {vector_1_3[1]:.2f}], slope: {ear_implantation_slope:.3f}")
        print(f"Ear implantation intersection angle: {angle_degrees:.2f} degrees")
        
        return angle_degrees, reference_slope, ear_implantation_slope
    
    def calculate_eye_protrusion_angular_analysis(self, point_39, point_37_1, point_38, point_37_2, head_direction=None):
        """
        Calculate the angle and slopes between vectors 39-37 and 38-37.
        Both vectors point towards point 37, representing the eye opening angle.

        Args:
            point_39: First vector start point (39)
            point_37_1: First vector end point (37)
            point_38: Second vector start point (38)
            point_37_2: Second vector end point (37)
            head_direction: Left/right profile direction

        Returns:
            tuple: (angle_degrees, vector_39_37_slope, vector_38_37_slope)
        """
        if not all([point_39, point_37_1, point_38, point_37_2]):
            return None, None, None

        # Create the two vectors pointing towards point 37
        vector_39_37 = np.array([point_37_1[0] - point_39[0], point_37_1[1] - point_39[1]])
        vector_38_37 = np.array([point_37_2[0] - point_38[0], point_37_2[1] - point_38[1]])

        # Calculate slopes (rise/run)
        vector_39_37_slope = vector_39_37[1] / vector_39_37[0] if vector_39_37[0] != 0 else float('inf')
        vector_38_37_slope = vector_38_37[1] / vector_38_37[0] if vector_38_37[0] != 0 else float('inf')

        # Normalize both vectors
        v1_u = vector_39_37 / np.linalg.norm(vector_39_37)
        v2_u = vector_38_37 / np.linalg.norm(vector_38_37)

        # Calculate angle using dot product
        cos_angle = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle_radians = np.arccos(abs(cos_angle))  # Use abs to always get acute angle
        angle_degrees = np.degrees(angle_radians)

        # For eye protrusion, we want the opening angle
        if angle_degrees > 90:
            angle_degrees = 180 - angle_degrees

        print(f"Vector 39-37: [{vector_39_37[0]:.2f}, {vector_39_37[1]:.2f}], slope: {vector_39_37_slope:.3f}")
        print(f"Vector 38-37: [{vector_38_37[0]:.2f}, {vector_38_37[1]:.2f}], slope: {vector_38_37_slope:.3f}")
        print(f"Eye protrusion opening angle: {angle_degrees:.2f} degrees")

        return angle_degrees, vector_39_37_slope, vector_38_37_slope

    def calculate_eye_protrusion_classification(self, point_37, point_39, point_38, head_direction):
        """
        Calculate eye protrusion by checking if cornea (point 38) crosses the orbital plane.

        The orbital plane is defined by the vector from point 37 (orbital base) to point 39 (orbital top).
        Eye protrusion is detected when the cornea crosses beyond this plane in the profile direction.

        Args:
            point_37: Orbital base point (inner eye corner)
            point_39: Orbital top point (upper eye boundary)
            point_38: Cornea point (eye protrusion reference)
            head_direction: 'left' or 'right' profile direction

        Returns:
            tuple: (perpendicular_distance, classification)
                - perpendicular_distance: signed distance from point 38 to orbital plane
                  (positive = crosses in profile direction, negative = recessed)
                - classification: 'pronounced eye protrusion', 'normal eye protrusion', or 'minimal eye protrusion'
        """
        if not all([point_37, point_39, point_38]):
            return None, None

        if head_direction not in ['left', 'right']:
            print(f"Warning: Invalid head_direction '{head_direction}' for eye protrusion detection")
            return None, None

        # Create the orbital plane vector (37 -> 39)
        vector_37_39 = np.array([point_39[0] - point_37[0], point_39[1] - point_37[1]])

        # Create vector from point 37 to cornea (38)
        vector_37_38 = np.array([point_38[0] - point_37[0], point_38[1] - point_37[1]])

        # Calculate perpendicular distance using cross product
        # Cross product in 2D: (v1_x * v2_y) - (v1_y * v2_x)
        # This gives the signed area of the parallelogram formed by the two vectors
        cross_product = (vector_37_39[0] * vector_37_38[1]) - (vector_37_39[1] * vector_37_38[0])

        # Normalize by the length of the orbital plane vector to get perpendicular distance
        orbital_plane_length = np.linalg.norm(vector_37_39)
        perpendicular_distance = cross_product / orbital_plane_length if orbital_plane_length > 0 else 0

        # Determine if the cornea crosses the orbital plane based on profile direction
        # For LEFT profile: negative perpendicular distance = crossing to the left (protrusion)
        # For RIGHT profile: positive perpendicular distance = crossing to the right (protrusion)

        if head_direction == 'left':
            # For left profile, we want negative distance (crossing left)
            signed_distance = -perpendicular_distance
        else:  # right profile
            # For right profile, we want positive distance (crossing right)
            signed_distance = perpendicular_distance

        print(f"\n--- EYE PROTRUSION ANALYSIS (ORBITAL PLANE METHOD) ---")
        print(f"Head direction: {head_direction}")
        print(f"Orbital plane vector (37-39): [{vector_37_39[0]:.2f}, {vector_37_39[1]:.2f}]")
        print(f"Cornea vector (37-38): [{vector_37_38[0]:.2f}, {vector_37_38[1]:.2f}]")
        print(f"Raw perpendicular distance: {perpendicular_distance:.2f} pixels")
        print(f"Signed distance (profile-aware): {signed_distance:.2f} pixels")

        # Classification based on signed distance
        # ±3 pixels tolerance for "normal eye protrusion"
        TOLERANCE = 3.0

        if signed_distance > TOLERANCE:
            classification = "pronounced eye protrusion"
            print(f"Classification: PRONOUNCED EYE PROTRUSION (cornea crosses orbital plane by {signed_distance:.2f} pixels)")
        elif signed_distance < -TOLERANCE:
            classification = "minimal eye protrusion"
            print(f"Classification: MINIMAL EYE PROTRUSION (cornea recessed from orbital plane by {abs(signed_distance):.2f} pixels)")
        else:
            classification = "normal eye protrusion"
            print(f"Classification: NORMAL EYE PROTRUSION (cornea within ±{TOLERANCE} pixels of orbital plane)")

        return signed_distance, classification
    
    def create_visualization(self, original_image: np.ndarray, detected_points: List[Dict], 
                           actual_profile: str, measurements: Dict) -> str:
        """Create comprehensive visualization and return as base64 string"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Original image with detected points
        axes[0].imshow(original_image)
        axes[0].set_title(f"Profile Anthropometric Points - {actual_profile.title()} Profile", fontsize=14, fontweight='bold')
        
        image_h, image_w = original_image.shape[:2]
        scale_x = image_w / 224
        scale_y = image_h / 224
        
        # Draw points
        for i, point in enumerate(detected_points):
            x, y = point['coordinates']
            x_orig = x * scale_x
            y_orig = y * scale_y
            
            color = plt.cm.rainbow(i / max(1, len(detected_points)))
            axes[0].plot(x_orig, y_orig, 'o', color=color, markersize=8)
            axes[0].text(x_orig+3, y_orig-3, f"{point['class']}", 
                        color=color, fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
        
        axes[0].axis('off')
        
        # Right plot: Comprehensive measurements summary
        axes[1].axis('off')
        axes[1].set_title("Comprehensive Anthropometric Analysis", fontsize=14, fontweight='bold')
        
        # Create comprehensive text summary
        summary_text = self._create_comprehensive_measurement_summary(measurements)
        
        # Display text with smaller font to fit more content
        axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes, 
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_comprehensive_measurement_summary(self, measurements: Dict) -> str:
        """Create comprehensive formatted measurement summary text"""
        summary_lines = []
        
        # Basic info
        if 'reference_distance' in measurements:
            summary_lines.append("=== BASIC MEASUREMENTS ===")
            summary_lines.append(f"Reference Distance (24-10): {measurements['reference_distance']:.2f}")
            if 'head_direction' in measurements:
                summary_lines.append(f"Head Direction: {measurements['head_direction']}")
            summary_lines.append("")
        
        # Nose measurements
        if 'nose_classification' in measurements:
            summary_lines.append("=== NOSE ANALYSIS ===")
            summary_lines.append(f"Distance (18-17): {measurements.get('nose_distance', 0):.2f}")
            summary_lines.append(f"Normalized: {measurements.get('nose_normalized', 0):.3f}")
            summary_lines.append(f"Classification: {measurements['nose_classification']}")
            if 'nose_tip_classification' in measurements:
                summary_lines.append(f"Tip Angle: {measurements.get('nose_tip_angle', 0):.1f}°")
                summary_lines.append(f"Tip Classification: {measurements['nose_tip_classification']}")
            summary_lines.append("")
        
        # Facial thirds
        if 'tercio_superior_distance' in measurements:
            summary_lines.append("=== FACIAL THIRDS ===")
            summary_lines.append(f"Superior (34-22): {measurements['tercio_superior_distance']:.2f} ({measurements['tercio_superior_normalized']:.3f})")
            if 'tercio_medio_distance' in measurements:
                summary_lines.append(f"Middle (22-18): {measurements['tercio_medio_distance']:.2f} ({measurements['tercio_medio_normalized']:.3f})")
            if 'tercio_inferior_distance' in measurements:
                summary_lines.append(f"Inferior (18-10): {measurements['tercio_inferior_distance']:.2f} ({measurements['tercio_inferior_normalized']:.3f})")
            summary_lines.append("")
        
        # Mandible
        if 'mandibula_classification' in measurements:
            summary_lines.append("=== MANDIBLE ANALYSIS ===")
            if 'mandibula_distance' in measurements:
                summary_lines.append(f"Distance (3-9): {measurements['mandibula_distance']:.2f}")
                summary_lines.append(f"Proportion: {measurements['mandibula_normalized']:.3f}")
            summary_lines.append(f"Classification: {measurements['mandibula_classification']}")
            if 'mandible_intersection_angle' in measurements:
                summary_lines.append(f"Intersection Angle: {measurements['mandible_intersection_angle']:.1f}°")
                summary_lines.append(f"Angle Classification: {measurements.get('mandible_angle_classification', 'N/A')}")
                if 'reference_vector_slope' in measurements:
                    summary_lines.append(f"Reference Vector Slope: {measurements['reference_vector_slope']:.3f}")
                if 'mandible_vector_slope' in measurements:
                    summary_lines.append(f"Mandible Vector Slope: {measurements['mandible_vector_slope']:.3f}")
            summary_lines.append("")
        
        # Ear implantation angular analysis
        if 'ear_implantation_intersection_angle' in measurements:
            summary_lines.append("=== EAR IMPLANTATION ANGULAR ANALYSIS ===")
            summary_lines.append(f"Intersection Angle (24-18 vs 1-3): {measurements['ear_implantation_intersection_angle']:.1f}°")
            if 'ear_implantation_angle_classification' in measurements:
                summary_lines.append(f"Angle Classification: {measurements['ear_implantation_angle_classification']}")
            if 'ear_implantation_vector_slope' in measurements:
                summary_lines.append(f"Ear Implantation Vector Slope: {measurements['ear_implantation_vector_slope']:.3f}")
            summary_lines.append("")
        
        # Eye protrusion analysis
        if 'eye_protrusion_classification' in measurements:
            summary_lines.append("=== EYE PROTRUSION ANALYSIS ===")
            summary_lines.append(f"Classification: {measurements['eye_protrusion_classification']}")
            if 'eye_protrusion_distance' in measurements:
                summary_lines.append(f"Distance from Orbital Plane: {measurements['eye_protrusion_distance']:.2f} pixels")
            if 'eye_protrusion_intersection_angle' in measurements:
                summary_lines.append(f"Opening Angle (39-37 vs 38-37): {measurements['eye_protrusion_intersection_angle']:.1f}°")
            if 'vector_39_37_slope' in measurements:
                summary_lines.append(f"Vector 39-37 Slope: {measurements['vector_39_37_slope']:.3f}")
            if 'vector_38_37_slope' in measurements:
                summary_lines.append(f"Vector 38-37 Slope: {measurements['vector_38_37_slope']:.3f}")
            summary_lines.append("")
        
        # Angular measurements
        angular_measurements = []
        if 'forehead_classification' in measurements:
            angular_measurements.append(f"Forehead: {measurements['forehead_classification']} ({measurements.get('forehead_angle', 0):.1f}°)")
        if 'chin_classification' in measurements:
            angular_measurements.append(f"Chin: {measurements['chin_classification']} ({measurements.get('chin_angle', 0):.1f}°)")
        
        if angular_measurements:
            summary_lines.append("=== ANGULAR ANALYSIS ===")
            summary_lines.extend(angular_measurements)
            summary_lines.append("")
        
        # Implantation
        if 'implantation_superior_classification' in measurements:
            summary_lines.append("=== IMPLANTATION ===")
            summary_lines.append(f"Superior: {measurements['implantation_superior_classification']} ({measurements.get('implantation_superior_angle', 0):.1f}°)")
            summary_lines.append(f"Inferior: {measurements['implantation_inferior_classification']} ({measurements.get('implantation_inferior_angle', 0):.1f}°)")
            summary_lines.append("")
        
        # Comprehensive ear measurements
        ear_measurements = []
        if 'ear_width' in measurements:
            ear_measurements.append(f"Width (2-6): {measurements['ear_width']:.2f}")
        if 'ear_length_classification' in measurements:
            ear_measurements.append(f"Length: {measurements['ear_length_classification']} ({measurements.get('ear_length_proportion', 0):.3f})")
        if 'ear_lobe_classification' in measurements:
            ear_measurements.append(f"Lobe: {measurements['ear_lobe_classification']} ({measurements.get('ear_lobe_proportion', 0):.3f})")
        if 'tragus_antitragus_classification' in measurements:
            ear_measurements.append(f"Tragus-Antitragus: {measurements['tragus_antitragus_classification']} ({measurements.get('tragus_antitragus_proportion', 0):.3f})")
        
        if ear_measurements:
            summary_lines.append("=== EAR ANALYSIS ===")
            summary_lines.extend(ear_measurements)
            summary_lines.append("")
        
        # Nasal triangulation
        if 'nasal_triangulation_classification' in measurements:
            summary_lines.append("=== NASAL TRIANGULATION ===")
            summary_lines.append(f"Orifice Distance (26-17): {measurements.get('nasal_orifice_distance', 0):.2f}")
            summary_lines.append(f"Reference Distance (18-30): {measurements.get('nose_reference_distance', 0):.2f}")
            summary_lines.append(f"Proportion: {measurements.get('nasal_orifice_proportion', 0):.3f}")
            summary_lines.append(f"Classification: {measurements['nasal_triangulation_classification']}")
            summary_lines.append("")
        
        return '\n'.join(summary_lines)
    
    def print_final_summary(self, measurements: Dict):
        """Print a comprehensive final summary of all measurements"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANTHROPOMETRIC ANALYSIS SUMMARY")
        print("="*80)
        
        # Collect all classifications
        classifications = {}
        
        # Basic measurements
        if 'reference_distance' in measurements:
            classifications['Reference Distance (24-10)'] = f"{measurements['reference_distance']:.2f}"
        if 'head_direction' in measurements:
            classifications['Head Direction'] = measurements['head_direction']
        
        # Nose analysis
        if 'nose_classification' in measurements:
            classifications['Nose Type'] = measurements['nose_classification']
        if 'nose_tip_classification' in measurements:
            classifications['Nose Tip'] = measurements['nose_tip_classification']
        
        # Facial structure
        if 'mandibula_classification' in measurements:
            classifications['Mandible'] = measurements['mandibula_classification']
        if 'mandible_angle_classification' in measurements:
            classifications['Mandible Angle'] = measurements['mandible_angle_classification']
        if 'ear_implantation_angle_classification' in measurements:
            classifications['Ear Implantation Angle'] = measurements['ear_implantation_angle_classification']
        if 'eye_protrusion_classification' in measurements:
            classifications['Eye Protrusion'] = measurements['eye_protrusion_classification']
        if 'forehead_classification' in measurements:
            classifications['Forehead'] = measurements['forehead_classification']
        if 'chin_classification' in measurements:
            classifications['Chin'] = measurements['chin_classification']
        
        # Implantation
        if 'implantation_superior_classification' in measurements:
            classifications['Superior Implantation'] = measurements['implantation_superior_classification']
        if 'implantation_inferior_classification' in measurements:
            classifications['Inferior Implantation'] = measurements['implantation_inferior_classification']
        
        # New comprehensive measurements
        if 'ear_length_classification' in measurements:
            classifications['Ear Length'] = measurements['ear_length_classification']
        if 'ear_lobe_classification' in measurements:
            classifications['Ear Lobe'] = measurements['ear_lobe_classification']
        if 'nasal_triangulation_classification' in measurements:
            classifications['Nasal Triangulation'] = measurements['nasal_triangulation_classification']
        if 'tragus_antitragus_classification' in measurements:
            classifications['Tragus-Antitragus'] = measurements['tragus_antitragus_classification']
        
        # Print all classifications
        for feature, classification in classifications.items():
            print(f"{feature:<30}: {classification}")
        
        print("="*80)
        print("Analysis includes all traditional and enhanced anthropometric measurements!")
        print("="*80)
    
    def analyze_image(self, image: np.ndarray, include_visualization: bool = True) -> Dict:
        """Complete comprehensive analysis pipeline for profile images"""
        print("Analyzing profile image with comprehensive measurements...")
        
        # Preprocess image
        original_image, image_tensor = self.preprocess_image(image)
        
        # Detect anthropometric points
        print("Detecting anthropometric points...")
        detected_points = self.detect_points(image_tensor)
        
        # Filter spurious predictions
        print("Filtering spurious predictions...")
        filtered_points, actual_profile = self.filter_spurious_predictions(detected_points)
        
        # Create points dictionary for measurements
        points_dict = self.create_points_dict(filtered_points)
        
        # Perform comprehensive anthropometric analysis
        measurements = self.perform_anthropometric_analysis(points_dict)
        
        # Print final summary
        self.print_final_summary(measurements)
        
        # Create visualization if requested
        visualization_base64 = None
        if include_visualization and filtered_points:
            visualization_base64 = self.create_visualization(
                original_image, filtered_points, actual_profile, measurements
            )
        
        # Compile comprehensive results
        results = {
            'profile_side': actual_profile,
            'total_detected_points': len(detected_points),
            'filtered_points': len(filtered_points),
            'anthropometric_points': filtered_points,
            'measurements': measurements,
            'visualization': visualization_base64 if include_visualization else None
        }
        
        return results
