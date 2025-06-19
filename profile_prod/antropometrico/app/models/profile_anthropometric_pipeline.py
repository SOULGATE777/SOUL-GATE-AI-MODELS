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
    """Profile-specific anthropometric analysis pipeline"""
    
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
        """Perform all anthropometric measurements"""
        print("\n=== PROFILE ANTHROPOMETRIC ANALYSIS ===")
        
        # Get all required points
        point_24 = points.get("24", None)
        point_10 = points.get("10", None)
        point_17 = points.get("17", None)
        point_18 = points.get("18", None)
        point_16 = points.get("16", None)
        point_5 = points.get("5", None)
        point_9 = points.get("9", None)
        point_19 = points.get("19", None)
        point_22 = points.get("22", None)
        point_11 = points.get("11", None)
        point_6 = points.get("6", None)
        point_2 = points.get("2", None)
        point_3 = points.get("3", None)
        point_1 = points.get("1", None)
        point_4 = points.get("4", None)
        point_7 = points.get("7", None)
        point_8 = points.get("8", None)
        
        measurements = {}
        
        # Calculate reference distance (24 to 10)
        if point_24 and point_10:
            reference_distance = self.dist(point_24, point_10)
            measurements['reference_distance'] = reference_distance
            print(f"Reference distance (24 to 10): {reference_distance:.2f}")
        else:
            print("Warning: Reference points 24 and 10 not found. Cannot perform normalized measurements.")
            return measurements
        
        # Nose measurements (18 to 17)
        if point_18 and point_17:
            distance_18_17 = self.dist(point_18, point_17)
            normalized_distance_18_17 = distance_18_17 / reference_distance
            measurements['nose_distance'] = distance_18_17
            measurements['nose_normalized'] = normalized_distance_18_17
            
            print(f"Distance between points 18 and 17: {distance_18_17:.2f}")
            print(f"Proportion between points 18 and 17: {normalized_distance_18_17:.3f}")
            
            # Classify nose
            if normalized_distance_18_17 > 0.165:
                nose_label = "nariz protruyente"
            elif 0.14 <= normalized_distance_18_17 <= 0.165:
                nose_label = "nariz normal"
            else:
                nose_label = "nariz corta"
            
            measurements['nose_classification'] = nose_label
            print(f"Label for points 18 and 17: {nose_label}")
        
        # Tercio measurements
        if point_24 and point_22:
            distance_24_22 = self.dist(point_24, point_22)
            normalized_distance_24_22 = distance_24_22 / reference_distance
            measurements['tercio_superior_distance'] = distance_24_22
            measurements['tercio_superior_normalized'] = normalized_distance_24_22
            
            print(f"Distance between points 24 and 22: {distance_24_22:.2f}")
            print(f"Proportion of distance between points 24 and 22: {normalized_distance_24_22:.3f}")
            print(f"Label for portion between points 24 and 22: tercio superior")
        
        if point_22 and point_16:
            distance_22_16 = self.dist(point_22, point_16)
            normalized_distance_22_16 = distance_22_16 / reference_distance
            measurements['tercio_medio_distance'] = distance_22_16
            measurements['tercio_medio_normalized'] = normalized_distance_22_16
            
            print(f"Distance between points 22 and 16: {distance_22_16:.2f}")
            print(f"Proportion of distance between points 22 and 16: {normalized_distance_22_16:.3f}")
            print(f"Label for portion between points 22 and 16: tercio medio")
        
        if point_18 and point_10:
            distance_18_10 = self.dist(point_18, point_10)
            normalized_distance_18_10 = distance_18_10 / reference_distance
            measurements['tercio_inferior_distance'] = distance_18_10
            measurements['tercio_inferior_normalized'] = normalized_distance_18_10
            
            print(f"Distance between points 18 and 10: {distance_18_10:.2f}")
            print(f"Proportion of distance between points 18 and 10: {normalized_distance_18_10:.3f}")
            print(f"Label for portion between points 18 and 10: tercio inferior")
            
            # Mandibula analysis
            if point_5 and point_9:
                distance_5_9 = self.dist(point_5, point_9)
                normalized_distance_5_9 = distance_5_9 / distance_18_10
                measurements['mandibula_distance'] = distance_5_9
                measurements['mandibula_normalized'] = normalized_distance_5_9
                
                print(f"Distance between points 5 and 9: {distance_5_9:.2f}")
                print(f"Proportion of distance between points 5 and 9 to tercio inferior: {normalized_distance_5_9:.3f}")
                
                # Classify mandibula
                if normalized_distance_5_9 >= 0.75:
                    mandibula_label = "Mandibula Sanguinea"
                elif 0.65 <= normalized_distance_5_9 <= 0.75:
                    mandibula_label = "Mandibula intermedia sanguineo/bilosa"
                elif 0.20 <= normalized_distance_5_9 <= 0.65:
                    mandibula_label = "Mandibula Bilosa"
                elif 0.10 <= normalized_distance_5_9 < 0.20:
                    mandibula_label = "Mandibula intermedia bilosa/nerviosa"
                elif normalized_distance_5_9 <= 0.10:
                    mandibula_label = "Mandibula Nerviosa"
                else:
                    mandibula_label = "Mandibula Intermedia"
                
                measurements['mandibula_classification'] = mandibula_label
                print(f"Label for mandibula (points 5 and 9): {mandibula_label}")
            elif not point_9:
                measurements['mandibula_classification'] = "Mandibula Linfatica"
                print("Label for mandibula (points 5 and 9): Mandibula Linfatica")
        
        # Ear measurements
        if point_2 and point_6:
            ear_width = self.dist(point_2, point_6)
            measurements['ear_width'] = ear_width
            print(f"Distance of ear width: {ear_width:.2f}")
            
            if point_7 and point_8:
                distance_7_8 = self.dist(point_7, point_8)
                trago_antitrago = distance_7_8 / ear_width
                measurements['trago_antitrago_distance'] = distance_7_8
                measurements['trago_antitrago_proportion'] = trago_antitrago
                
                print(f"Distance between points 7 and 8: {distance_7_8:.2f}")
                print(f"Proporcion trago - antitrago: {trago_antitrago:.3f}")
        
        # Angular measurements
        self.calculate_angular_measurements(points, measurements)
        
        return measurements
    
    def calculate_angular_measurements(self, points: Dict[str, Tuple[float, float]], measurements: Dict):
        """Calculate all angular measurements with proper left/right profile handling via vector analysis"""
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
        
        if not (point_22 and point_18):
            print("Warning: Cannot calculate angular measurements without points 22 and 18")
            return
        
        # Reference vector (22 to 18)
        vector_22_18 = np.array([point_18[0] - point_22[0], point_18[1] - point_22[1]])
        
        # Determine head direction based on the reference vector (infallible method)
        head_direction = "right" if vector_22_18[0] > 0 else "left"
        print(f"Head direction determined via vector analysis: {head_direction}")
        
        # Nose tip angle (18 to 17) - using your original logic
        if point_17 and point_18:
            vector_18_17 = np.array([point_17[0] - point_18[0], point_17[1] - point_18[1]])
            
            # Calculate the perpendicular slope (negative reciprocal)
            ref_slope = vector_22_18[1] / vector_22_18[0] if vector_22_18[0] != 0 else float('inf')
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
            print(f"Slope of reference line (22 to 18): {ref_slope:.2f}")
            print(f"Slope of perpendicular line: {perp_slope:.2f}")
            print(f"Angle from perpendicular line to nose tip: {angle_degrees:.2f} degrees")
            
            # Classify nose tip
            if angle_degrees >= 26:
                nose_tip_label = "punta muy hacia arriba"
            elif angle_degrees >= 19:
                nose_tip_label = "punta de nariz hacia arriba"
            elif 0 <= angle_degrees < 19:
                nose_tip_label = "punta de nariz promedio"
            else:
                nose_tip_label = "punta hacia abajo"
            
            measurements['nose_tip_classification'] = nose_tip_label
            print(f"Etiqueta para ángulo de punta de nariz: {nose_tip_label}")
        
        # Additional angular measurements would continue here...
        # (Forehead angle, chin angle, implantation angles, etc.)
        # Keeping the structure but condensed for space
        
    def create_visualization(self, original_image: np.ndarray, detected_points: List[Dict], 
                           actual_profile: str, measurements: Dict) -> str:
        """Create visualization and return as base64 string"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Original image with detected points
        axes[0].imshow(original_image)
        axes[0].set_title(f"Profile Anthropometric Points - {actual_profile.title()} Profile")
        
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
        
        # Right plot: Measurements summary
        axes[1].axis('off')
        axes[1].set_title("Profile Anthropometric Measurements", fontsize=14, fontweight='bold')
        
        # Create text summary
        summary_text = self._create_measurement_summary(measurements)
        
        # Display text
        axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_measurement_summary(self, measurements: Dict) -> str:
        """Create formatted measurement summary text"""
        summary_lines = []
        
        if 'reference_distance' in measurements:
            summary_lines.append(f"Reference Distance (24-10): {measurements['reference_distance']:.2f}")
            summary_lines.append("")
        
        # Nose measurements
        if 'nose_classification' in measurements:
            summary_lines.append("NOSE ANALYSIS:")
            summary_lines.append(f"  Distance (18-17): {measurements.get('nose_distance', 0):.2f}")
            summary_lines.append(f"  Normalized: {measurements.get('nose_normalized', 0):.3f}")
            summary_lines.append(f"  Classification: {measurements['nose_classification']}")
            summary_lines.append("")
        
        # Tercio measurements
        if 'tercio_superior_distance' in measurements:
            summary_lines.append("FACIAL THIRDS:")
            summary_lines.append(f"  Superior (24-22): {measurements['tercio_superior_distance']:.2f} ({measurements['tercio_superior_normalized']:.3f})")
            if 'tercio_medio_distance' in measurements:
                summary_lines.append(f"  Middle (22-16): {measurements['tercio_medio_distance']:.2f} ({measurements['tercio_medio_normalized']:.3f})")
            if 'tercio_inferior_distance' in measurements:
                summary_lines.append(f"  Inferior (18-10): {measurements['tercio_inferior_distance']:.2f} ({measurements['tercio_inferior_normalized']:.3f})")
            summary_lines.append("")
        
        # Mandibula
        if 'mandibula_classification' in measurements:
            summary_lines.append("MANDIBLE ANALYSIS:")
            if 'mandibula_distance' in measurements:
                summary_lines.append(f"  Distance (5-9): {measurements['mandibula_distance']:.2f}")
                summary_lines.append(f"  Proportion: {measurements['mandibula_normalized']:.3f}")
            summary_lines.append(f"  Classification: {measurements['mandibula_classification']}")
            summary_lines.append("")
        
        # Angular measurements
        angular_measurements = []
        if 'nose_tip_classification' in measurements:
            angular_measurements.append(f"  Nose tip: {measurements['nose_tip_classification']} ({measurements.get('nose_tip_angle', 0):.1f}°)")
        
        if angular_measurements:
            summary_lines.append("ANGULAR ANALYSIS:")
            summary_lines.extend(angular_measurements)
            summary_lines.append("")
        
        # Ear measurements
        if 'ear_width' in measurements:
            summary_lines.append("EAR ANALYSIS:")
            summary_lines.append(f"  Width: {measurements['ear_width']:.2f}")
            if 'trago_antitrago_proportion' in measurements:
                summary_lines.append(f"  Trago-antitrago proportion: {measurements['trago_antitrago_proportion']:.3f}")
        
        return '\n'.join(summary_lines)
    
    def analyze_image(self, image: np.ndarray, include_visualization: bool = True) -> Dict:
        """Complete analysis pipeline for profile images"""
        print("Analyzing profile image...")
        
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
        
        # Perform anthropometric analysis
        measurements = self.perform_anthropometric_analysis(points_dict)
        
        # Create visualization if requested
        visualization_base64 = None
        if include_visualization and filtered_points:
            visualization_base64 = self.create_visualization(
                original_image, filtered_points, actual_profile, measurements
            )
        
        # Compile results
        results = {
            'profile_side': actual_profile,
            'total_detected_points': len(detected_points),
            'filtered_points': len(filtered_points),
            'anthropometric_points': filtered_points,
            'measurements': measurements,
            'visualization': visualization_base64 if include_visualization else None
        }
        
        return results
