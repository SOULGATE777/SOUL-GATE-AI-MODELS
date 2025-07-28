import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import base64
import io

class ProfileAnthropometricVisualizer:
    """Visualization utilities for profile anthropometric analysis"""
    
    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
        
    def create_point_visualization(self, 
                                 image: np.ndarray,
                                 points: List[Dict],
                                 title: str = "Profile Anthropometric Points") -> str:
        """
        Create visualization of detected anthropometric points
        
        Args:
            image: Original image (RGB)
            points: List of detected points with coordinates and classes
            title: Plot title
            
        Returns:
            Base64 encoded image string
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Scale coordinates from model size (224) to original image size
        image_h, image_w = image.shape[:2]
        scale_x = image_w / 224
        scale_y = image_h / 224
        
        # Plot points
        for i, point in enumerate(points):
            x, y = point['coordinates']
            x_orig = x * scale_x
            y_orig = y * scale_y
            
            # Use different colors for different point types
            color = self.colors[i % len(self.colors)]
            
            # Draw point
            ax.plot(x_orig, y_orig, 'o', color=color, markersize=10, 
                   markeredgecolor='white', markeredgewidth=2)
            
            # Add label
            ax.text(x_orig + 5, y_orig - 5, f"{point['class']}", 
                   color=color, fontsize=9, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=2))
        
        ax.axis('off')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def create_measurement_visualization(self,
                                       image: np.ndarray,
                                       points: List[Dict],
                                       measurements: Dict,
                                       profile_side: str) -> str:
        """
        Create comprehensive visualization with measurements
        
        Args:
            image: Original image (RGB)
            points: List of detected points
            measurements: Measurement results
            profile_side: Determined profile side
            
        Returns:
            Base64 encoded image string
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Image with points and measurement lines
        ax1.imshow(image)
        ax1.set_title(f"Profile Analysis - {profile_side.title()} Side", fontsize=14, fontweight='bold')
        
        image_h, image_w = image.shape[:2]
        scale_x = image_w / 224
        scale_y = image_h / 224
        
        # Create points lookup
        points_dict = {}
        for point in points:
            points_dict[point['class']] = (
                point['coordinates'][0] * scale_x,
                point['coordinates'][1] * scale_y
            )
        
        # Draw points
        for i, point in enumerate(points):
            x, y = point['coordinates']
            x_orig = x * scale_x
            y_orig = y * scale_y
            
            color = self.colors[i % len(self.colors)]
            ax1.plot(x_orig, y_orig, 'o', color=color, markersize=8,
                    markeredgecolor='white', markeredgewidth=1)
            ax1.text(x_orig + 3, y_orig - 3, f"{point['class']}", 
                    color=color, fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))
        
        # Draw measurement lines
        self._draw_measurement_lines(ax1, points_dict, measurements)
        
        ax1.axis('off')
        
        # Right plot: Measurement summary
        ax2.axis('off')
        ax2.set_title("Anthropometric Measurements", fontsize=14, fontweight='bold')
        
        # Create formatted text summary
        summary_text = self._format_measurements_text(measurements, profile_side)
        
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _draw_measurement_lines(self, ax, points_dict: Dict, measurements: Dict):
        """Draw measurement lines on the visualization"""
        # Reference line (24 to 10)
        if '24' in points_dict and '10' in points_dict:
            p1, p2 = points_dict['24'], points_dict['10']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, alpha=0.7, label='Reference (24-10)')
        
        # Nose measurement (18 to 17)
        if '18' in points_dict and '17' in points_dict:
            p1, p2 = points_dict['18'], points_dict['17']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2, alpha=0.7, label='Nose (18-17)')
        
        # Tercio lines
        if all(p in points_dict for p in ['24', '22', '16', '18']):
            # Superior tercio
            p1, p2 = points_dict['24'], points_dict['22']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2, alpha=0.7, label='Superior tercio')
            
            # Medio tercio
            p1, p2 = points_dict['22'], points_dict['16']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'orange', linewidth=2, alpha=0.7, label='Medio tercio')
            
            # Inferior tercio
            p1, p2 = points_dict['16'], points_dict['18']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'purple', linewidth=2, alpha=0.7, label='Inferior tercio')
        
        # Mandibula line (3 to 9)
        if '3' in points_dict and '9' in points_dict:
            p1, p2 = points_dict['3'], points_dict['9']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'cyan', linewidth=2, alpha=0.7, label='Mandibula (3-9)')
        
        # Ear measurements (2 to 6, 7 to 8)
        if '2' in points_dict and '6' in points_dict:
            p1, p2 = points_dict['2'], points_dict['6']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'brown', linewidth=2, alpha=0.7, label='Ear width')
        
        if '7' in points_dict and '8' in points_dict:
            p1, p2 = points_dict['7'], points_dict['8']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'pink', linewidth=2, alpha=0.7, label='Trago-antitrago')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    def _format_measurements_text(self, measurements: Dict, profile_side: str) -> str:
        """Format measurements into readable text"""
        lines = [f"PROFILE SIDE: {profile_side.upper()}", ""]
        
        # Reference distance
        if 'reference_distance' in measurements:
            lines.extend([
                "REFERENCE MEASUREMENT:",
                f"  Distance (24-10): {measurements['reference_distance']:.2f}px",
                ""
            ])
        
        # Nose analysis
        if 'nose_classification' in measurements:
            lines.extend([
                "NOSE ANALYSIS:",
                f"  Distance (18-17): {measurements.get('nose_distance', 0):.2f}px",
                f"  Normalized ratio: {measurements.get('nose_normalized', 0):.3f}",
                f"  Classification: {measurements['nose_classification']}",
                ""
            ])
        
        # Facial thirds
        tercio_data = []
        if 'tercio_superior_distance' in measurements:
            tercio_data.append(f"  Superior: {measurements['tercio_superior_distance']:.1f}px ({measurements['tercio_superior_normalized']:.3f})")
        if 'tercio_medio_distance' in measurements:
            tercio_data.append(f"  Medio: {measurements['tercio_medio_distance']:.1f}px ({measurements['tercio_medio_normalized']:.3f})")
        if 'tercio_inferior_distance' in measurements:
            tercio_data.append(f"  Inferior: {measurements['tercio_inferior_distance']:.1f}px ({measurements['tercio_inferior_normalized']:.3f})")
        
        if tercio_data:
            lines.extend(["FACIAL THIRDS:"] + tercio_data + [""])
        
        # Mandibula analysis
        if 'mandibula_classification' in measurements:
            lines.extend([
                "MANDIBULA ANALYSIS:",
                f"  Classification: {measurements['mandibula_classification']}"
            ])
            if 'mandibula_distance' in measurements:
                lines.extend([
                    f"  Distance (3-9): {measurements['mandibula_distance']:.2f}px",
                    f"  Proportion: {measurements['mandibula_normalized']:.3f}"
                ])
            lines.append("")
        
        # Angular measurements
        angular_data = []
        if 'nose_tip_classification' in measurements:
            angle = measurements.get('nose_tip_angle', 0)
            angular_data.append(f"  Nose tip: {measurements['nose_tip_classification']} ({angle:.1f}°)")
        if 'forehead_classification' in measurements:
            angle = measurements.get('forehead_angle', 0)
            angular_data.append(f"  Forehead: {measurements['forehead_classification']} ({angle:.1f}°)")
        if 'chin_classification' in measurements:
            angle = measurements.get('chin_angle', 0)
            angular_data.append(f"  Chin: {measurements['chin_classification']} ({angle:.1f}°)")
        
        if angular_data:
            lines.extend(["ANGULAR ANALYSIS:"] + angular_data + [""])
        
        # Implantation analysis
        if 'implantation_superior_classification' in measurements:
            lines.extend([
                "IMPLANTATION ANALYSIS:",
                f"  Superior: {measurements['implantation_superior_classification']}",
                f"  Inferior: {measurements['implantation_inferior_classification']}",
                ""
            ])
        
        # Ear measurements
        if 'ear_width' in measurements:
            lines.extend([
                "EAR ANALYSIS:",
                f"  Width: {measurements['ear_width']:.2f}px"
            ])
            if 'trago_antitrago_proportion' in measurements:
                lines.append(f"  Trago-antitrago: {measurements['trago_antitrago_proportion']:.3f}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def create_comparison_grid(self, images_and_results: List[Tuple[np.ndarray, Dict]]) -> str:
        """
        Create a grid comparison of multiple profile analyses
        
        Args:
            images_and_results: List of (image, results) tuples
            
        Returns:
            Base64 encoded image string
        """
        n_images = len(images_and_results)
        if n_images == 0:
            return ""
        
        # Calculate grid dimensions
        cols = min(n_images, 3)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (image, results) in enumerate(images_and_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            ax.imshow(image)
            
            # Add title with basic info
            profile_side = results.get('profile_side', 'unknown')
            num_points = results.get('filtered_points', 0)
            ax.set_title(f"Profile {i+1}: {profile_side.title()} ({num_points} points)", 
                        fontsize=10, fontweight='bold')
            
            # Draw points if available
            if 'anthropometric_points' in results:
                image_h, image_w = image.shape[:2]
                scale_x = image_w / 224
                scale_y = image_h / 224
                
                for j, point in enumerate(results['anthropometric_points']):
                    x, y = point['coordinates']
                    x_orig = x * scale_x
                    y_orig = y * scale_y
                    
                    color = self.colors[j % len(self.colors)]
                    ax.plot(x_orig, y_orig, 'o', color=color, markersize=6,
                           markeredgecolor='white', markeredgewidth=1)
            
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
