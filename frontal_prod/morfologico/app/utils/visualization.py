import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import uuid
import os
from typing import List, Dict, Any

# Beautiful color palette
COLORS = {
    'primary': (74, 144, 226),     # Professional blue
    'secondary': (155, 89, 182),   # Purple
    'accent': (46, 204, 113),      # Green
    'warning': (241, 196, 15),     # Yellow
    'danger': (231, 76, 60),       # Red
    'dark': (52, 73, 94),          # Dark blue
    'light': (236, 240, 241),      # Light gray
    'white': (255, 255, 255)
}

def get_color_for_class(class_idx: int, total_classes: int = 50) -> tuple:
    """Get a distinctive color for each class"""
    # Use HSV colorspace for better color distribution
    hue = (class_idx * 137.5) % 360  # Golden angle for good distribution
    saturation = 0.8
    value = 0.9
    
    # Convert HSV to RGB
    import colorsys
    rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
    return tuple(int(c * 255) for c in rgb)

def draw_sleek_bbox(image: np.ndarray, bbox: List[float], label: str, 
                   score: float, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw a sleek bounding box with modern styling"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw main rectangle with thin border
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Create label background
    label_text = f"{label}: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 1
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, text_thickness
    )
    
    # Create rounded rectangle for label background
    label_bg_padding = 6
    label_x1 = x1
    label_y1 = y1 - text_height - label_bg_padding * 2
    label_x2 = x1 + text_width + label_bg_padding * 2
    label_y2 = y1
    
    # Ensure label doesn't go out of image bounds
    if label_y1 < 0:
        label_y1 = y2
        label_y2 = y2 + text_height + label_bg_padding * 2
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Draw text
    text_x = label_x1 + label_bg_padding
    text_y = label_y2 - label_bg_padding
    cv2.putText(image, label_text, (text_x, text_y), font, font_scale, 
                COLORS['white'], text_thickness, cv2.LINE_AA)
    
    return image

def draw_sleek_point(image: np.ndarray, point: tuple, label: str, 
                    score: float, color: tuple, radius: int = 4) -> np.ndarray:
    """Draw a sleek point with modern styling"""
    x, y = map(int, point)
    
    # Draw outer circle (border)
    cv2.circle(image, (x, y), radius + 1, COLORS['dark'], -1)
    
    # Draw inner circle (main color)
    cv2.circle(image, (x, y), radius, color, -1)
    
    # Draw small center dot
    cv2.circle(image, (x, y), 1, COLORS['white'], -1)
    
    # Add label if needed
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        
        # Position text slightly offset from point
        text_x = x + radius + 5
        text_y = y - radius
        
        # Draw text with shadow effect
        cv2.putText(image, f"{label}: {score:.2f}", (text_x + 1, text_y + 1), 
                   font, font_scale, COLORS['dark'], text_thickness, cv2.LINE_AA)
        cv2.putText(image, f"{label}: {score:.2f}", (text_x, text_y), 
                   font, font_scale, color, text_thickness, cv2.LINE_AA)
    
    return image

async def create_beautiful_visualization(
    image: np.ndarray, 
    facial_results: List[Dict[str, Any]], 
    point_results: List[Dict[str, Any]]
) -> str:
    """Create a beautiful combined visualization of all detection results"""
    
    # Create a copy of the image
    viz_image = image.copy()
    
    # Draw facial landmarks (bounding boxes)
    landmark_classes = set()
    for i, result in enumerate(facial_results):
        landmark_class = result['landmark_class']
        landmark_classes.add(landmark_class)
        
        # Get color for this landmark class
        class_idx = hash(landmark_class) % 20
        color = get_color_for_class(class_idx)
        
        # Draw bounding box
        bbox = result['box']
        label = f"{landmark_class}:{result['tag_name']}"
        score = result['score']
        
        viz_image = draw_sleek_bbox(
            viz_image, bbox, label, score, color, thickness=2
        )
    
    # Draw anthropometric points
    point_classes = set()
    for i, result in enumerate(point_results):
        point_class = result.get('point_class', f"Point_{i}")
        point_classes.add(point_class)
        
        # Get color for this point class  
        class_idx = hash(point_class) % 20 + 20  # Offset to avoid color collision
        color = get_color_for_class(class_idx)
        
        # Draw point
        if 'center_point' in result:
            point = result['center_point']
        elif 'point' in result:
            point = result['point']
        else:
            # Calculate center from bbox if available
            bbox = result.get('bbox', result.get('box'))
            if bbox:
                x1, y1, x2, y2 = bbox
                point = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                continue
        
        score = result.get('score', 1.0)
        viz_image = draw_sleek_point(
            viz_image, point, point_class, score, color, radius=5
        )
    
    # Add title and statistics overlay
    viz_image = add_info_overlay(
        viz_image, len(facial_results), len(point_results)
    )
    
    # Save visualization
    viz_filename = f"analysis_{uuid.uuid4().hex}.jpg"
    viz_path = f"/app/results/{viz_filename}"
    
    # Ensure results directory exists
    os.makedirs("/app/results", exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    viz_bgr = cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(viz_path, viz_bgr)
    
    return viz_path

def add_info_overlay(image: np.ndarray, num_landmarks: int, num_points: int) -> np.ndarray:
    """Add a sleek information overlay to the image"""
    h, w = image.shape[:2]
    
    # Create overlay
    overlay = image.copy()
    
    # Define overlay area (top-right corner)
    overlay_width = 280
    overlay_height = 100
    overlay_x = w - overlay_width - 20
    overlay_y = 20
    
    # Draw semi-transparent background
    cv2.rectangle(overlay, 
                 (overlay_x, overlay_y), 
                 (overlay_x + overlay_width, overlay_y + overlay_height), 
                 COLORS['dark'], -1)
    
    # Blend with original image
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = COLORS['white']
    
    # Title
    cv2.putText(image, "Facial Analysis Results", 
                (overlay_x + 10, overlay_y + 25), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Statistics
    cv2.putText(image, f"Landmarks: {num_landmarks}", 
                (overlay_x + 10, overlay_y + 50), 
                font, 0.5, color, 1, cv2.LINE_AA)
    
    cv2.putText(image, f"Points: {num_points}", 
                (overlay_x + 10, overlay_y + 70), 
                font, 0.5, color, 1, cv2.LINE_AA)
    
    return image

def create_matplotlib_visualization(
    image: np.ndarray, 
    facial_results: List[Dict[str, Any]], 
    point_results: List[Dict[str, Any]]
) -> str:
    """Create a publication-ready matplotlib visualization"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Display image
    ax.imshow(image)
    ax.set_title("Comprehensive Facial Analysis", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Color maps for different types
    landmark_colors = cm.Set3(np.linspace(0, 1, len(set(r['landmark_class'] for r in facial_results))))
    point_colors = cm.Set1(np.linspace(0, 1, max(len(point_results), 8)))
    
    # Draw facial landmarks
    landmark_patches = []
    for i, result in enumerate(facial_results):
        bbox = result['box']
        x1, y1, x2, y2 = bbox
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=landmark_colors[i % len(landmark_colors)], 
                               facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label
        label = f"{result['landmark_class']}: {result['tag_name']}"
        ax.text(x1, y1-5, label, fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Draw anthropometric points
    for i, result in enumerate(point_results):
        if 'center_point' in result:
            x, y = result['center_point']
        elif 'point' in result:
            x, y = result['point']
        else:
            continue
            
        ax.plot(x, y, 'o', color=point_colors[i % len(point_colors)], 
               markersize=8, markeredgecolor='white', markeredgewidth=1)
        
        # Add label
        point_class = result.get('point_class', f"Point_{i}")
        ax.text(x+5, y-5, point_class, fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    viz_filename = f"matplotlib_analysis_{uuid.uuid4().hex}.png"
    viz_path = f"/app/results/{viz_filename}"
    
    os.makedirs("/app/results", exist_ok=True)
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return viz_path
