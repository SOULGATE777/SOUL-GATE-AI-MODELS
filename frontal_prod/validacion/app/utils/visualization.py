import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def create_validation_visualization(
    image_path: str,
    detections: List[Dict],
    class_names: Dict[int, str],
    output_path: str,
    confidence_threshold: float = 0.20
) -> str:
    """
    Create visualization of facial validation detections
    
    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        class_names: Mapping of class IDs to names
        output_path: Path to save visualization
        confidence_threshold: Confidence threshold used
        
    Returns:
        Path to saved visualization
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.imshow(image_rgb)
        
        # Set title
        title = f'Facial Feature Validation\n{os.path.basename(image_path)} | '
        title += f'Detections: {len(detections)} | Threshold: {confidence_threshold}'
        ax.set_title(title, fontsize=16, pad=20, weight='bold')
        ax.axis('off')
        
        # Color scheme for different feature categories
        colors = {
            'hair_coverage': '#FF6B6B',      # Red
            'facial_hair': '#4ECDC4',        # Teal
            'facial_expression': '#45B7D1',  # Blue
            'accessories': '#96CEB4',        # Green
            'body_modifications': '#FFEAA7', # Yellow
            'head_characteristics': '#DDA0DD', # Plum
            'eye_features': '#98D8C8',       # Mint
            'facial_points': '#F7DC6F',      # Light yellow
            'default': '#A8A8A8'             # Gray
        }
        
        # Feature categories mapping
        feature_categories = {
            'cabello_tapando_i': 'hair_coverage',
            'cabello_tapando_derecho': 'hair_coverage', 
            'cabello_tapando_central': 'hair_coverage',
            'barba': 'facial_hair',
            'bc_bigote': 'facial_hair',
            'bc_abierta': 'facial_expression',
            'bc_sonriendo': 'facial_expression',
            'piercing': 'accessories',
            'lentes': 'accessories',
            'objeto_frente': 'accessories',
            'tatuaje': 'body_modifications',
            'calvo': 'head_characteristics',
            'l_ej_i': 'eye_features',
            'l_ej_d': 'eye_features',
            'p_d_g_iz': 'facial_points',
            'p_d_g_d': 'facial_points',
            'p_d_v': 'facial_points'
        }
        
        # Draw detections
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Get color for this feature category
            category = feature_categories.get(class_name, 'default')
            color = colors.get(category, colors['default'])
            
            # Draw bounding box
            width = x2 - x1
            height = y2 - y1
            
            # Main rectangle
            rect = Rectangle(
                (x1, y1), width, height,
                linewidth=3,
                edgecolor=color,
                facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)
            
            # Confidence-based styling
            alpha = 0.8 if confidence > 0.7 else 0.6 if confidence > 0.4 else 0.4
            
            # Create label
            label = f'{class_name}\n{confidence:.3f}'
            
            # Label background
            bbox_props = dict(
                boxstyle="round,pad=0.3",
                facecolor=color,
                alpha=alpha,
                edgecolor='white',
                linewidth=1
            )
            
            # Position label above or below box to avoid overlap
            label_y = y1 - 15 if y1 > 50 else y2 + 5
            
            ax.text(
                x1, label_y, label,
                color='white',
                fontsize=10,
                weight='bold',
                ha='left',
                va='top' if y1 > 50 else 'bottom',
                bbox=bbox_props
            )
            
            # Add center point for reference
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.plot(center_x, center_y, 'o', color=color, markersize=4, alpha=0.8)
        
        # Create legend
        legend_elements = []
        used_categories = set()
        
        for detection in detections:
            category = feature_categories.get(detection['class_name'], 'default')
            if category not in used_categories:
                legend_elements.append(
                    patches.Patch(color=colors[category], label=category.replace('_', ' ').title())
                )
                used_categories.add(category)
        
        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc='upper right',
                bbox_to_anchor=(0.98, 0.98),
                framealpha=0.9,
                fontsize=10
            )
        
        # Add statistics text
        if detections:
            avg_conf = np.mean([d['confidence'] for d in detections])
            high_conf_count = len([d for d in detections if d['confidence'] > 0.7])
            
            stats_text = f'Statistics:\n'
            stats_text += f'Total Detections: {len(detections)}\n'
            stats_text += f'Average Confidence: {avg_conf:.3f}\n'
            stats_text += f'High Confidence (>0.7): {high_conf_count}\n'
            stats_text += f'Threshold: {confidence_threshold}'
            
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
            )
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✅ Visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise

def create_feature_summary_chart(detections: List[Dict], output_path: str) -> str:
    """Create a summary chart of detected features"""
    try:
        # Count detections by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if not class_counts:
            return None
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax.bar(classes, counts, color='steelblue', alpha=0.7)
        
        # Customize chart
        ax.set_title('Detected Facial Features Summary', fontsize=16, weight='bold')
        ax.set_xlabel('Feature Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', weight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating summary chart: {e}")
        return None
