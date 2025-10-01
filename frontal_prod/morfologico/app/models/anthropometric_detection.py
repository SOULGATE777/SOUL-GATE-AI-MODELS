import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as transforms
import uuid
import os
from typing import List, Dict, Any, Tuple

class AnthropometricPointDetector:
    def __init__(self, model_path: str, device=None):
        """
        Initialize the anthropometric point detector
        
        Args:
            model_path: Path to the trained model file
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Anthropometric detector using device: {self.device}")
        
        # Number of point classes (1-13) + background
        self.num_point_classes = 13
        self.point_radius = 5
        
        # Point class names for better labeling
        self.point_class_names = {
            1: "Point_1", 2: "Point_2", 3: "Point_3", 4: "Point_4", 
            5: "Point_5", 6: "Point_6", 7: "Point_7", 8: "Point_8",
            9: "Point_9", 10: "Point_10", 11: "Point_11", 12: "Point_12", 
            13: "Point_13"
        }
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def _load_model(self, model_path: str):
        """Load the trained Faster R-CNN model"""
        try:
            # Create model architecture
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            
            # Modify head for point detection
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.num_point_classes + 1  # +1 for background
            )
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            print("Anthropometric point detection model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading anthropometric model: {e}")
            raise e
    
    def detect_points(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect anthropometric points in an image
        
        Args:
            image: Input image as numpy array (RGB)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detection results
        """
        try:
            # Resize image for consistency
            orig_height, orig_width = image.shape[:2]
            image_resized = cv2.resize(image, (224, 224))
            
            # Scale factors for coordinate conversion
            scale_x = orig_width / 224
            scale_y = orig_height / 224
            
            # Prepare image tensor
            image_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)[0]
            
            # Filter predictions by confidence
            keep = predictions['scores'] > confidence_threshold
            boxes = predictions['boxes'][keep].cpu().numpy()
            labels = predictions['labels'][keep].cpu().numpy()
            scores = predictions['scores'][keep].cpu().numpy()
            
            # Process results
            results = []
            for box, label_idx, score in zip(boxes, labels, scores):
                # Convert box coordinates back to original image scale
                x1, y1, x2, y2 = box
                x1_orig = x1 * scale_x
                y1_orig = y1 * scale_y
                x2_orig = x2 * scale_x
                y2_orig = y2 * scale_y
                
                # Calculate center point
                center_x = (x1_orig + x2_orig) / 2
                center_y = (y1_orig + y2_orig) / 2
                
                # Get point class name
                point_class = self.point_class_names.get(label_idx, f"Point_{label_idx}")
                
                results.append({
                    'point_class': point_class,
                    'label_idx': int(label_idx),
                    'score': float(score),
                    'bbox': [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)],
                    'center_point': [float(center_x), float(center_y)],
                    'point': [float(center_x), float(center_y)]  # For compatibility
                })
            
            return results
            
        except Exception as e:
            print(f"Error in point detection: {e}")
            return []
    
    async def create_visualization(self, image: np.ndarray, results: List[Dict[str, Any]]) -> str:
        """Create a visualization of detected points"""
        from app.utils.visualization import draw_sleek_point, get_color_for_class
        
        viz_image = image.copy()
        
        # Draw each detected point
        for i, result in enumerate(results):
            center_point = result['center_point']
            point_class = result['point_class']
            score = result['score']
            
            # Get color for this point class
            color = get_color_for_class(result['label_idx'], self.num_point_classes)
            
            # Draw the point
            viz_image = draw_sleek_point(
                viz_image, center_point, point_class, score, color, radius=6
            )
        
        # Add summary overlay
        viz_image = self._add_point_summary(viz_image, len(results))
        
        # Save visualization
        viz_filename = f"points_{uuid.uuid4().hex}.jpg"
        viz_path = f"/app/results/{viz_filename}"
        
        # Ensure results directory exists
        os.makedirs("/app/results", exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        viz_bgr = cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(viz_path, viz_bgr)

        return viz_path
    
    def _add_point_summary(self, image: np.ndarray, num_points: int) -> np.ndarray:
        """Add a summary overlay for point detection"""
        h, w = image.shape[:2]

        # Create overlay
        overlay = image.copy()

        # Define overlay area (top-left corner)
        overlay_width = 220
        overlay_height = 60
        overlay_x = 20
        overlay_y = 20

        # Colors
        bg_color = (52, 73, 94)  # Dark blue
        text_color = (255, 255, 255)  # White

        # Draw semi-transparent background
        cv2.rectangle(overlay,
                     (overlay_x, overlay_y),
                     (overlay_x + overlay_width, overlay_y + overlay_height),
                     bg_color, -1)

        # Blend with original image
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        cv2.putText(image, "Anthropometric Points",
                    (overlay_x + 10, overlay_y + 25),
                    font, 0.6, text_color, 2, cv2.LINE_AA)

        # Count
        cv2.putText(image, f"Detected: {num_points} points",
                    (overlay_x + 10, overlay_y + 45),
                    font, 0.5, text_color, 1, cv2.LINE_AA)

        return image

    def get_point_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about detected points"""
        if not results:
            return {"total_points": 0, "point_classes": [], "average_confidence": 0.0}

        point_classes = [r['point_class'] for r in results]
        scores = [r['score'] for r in results]

        # Count points by class
        class_counts = {}
        for point_class in point_classes:
            class_counts[point_class] = class_counts.get(point_class, 0) + 1

        return {
            "total_points": len(results),
            "point_classes": list(set(point_classes)),
            "class_counts": class_counts,
            "average_confidence": float(np.mean(scores)),
            "confidence_range": [float(min(scores)), float(max(scores))]
        }
