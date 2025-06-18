import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as transforms
import os
import json
from typing import List, Dict, Any, Tuple

class FacialLandmarkClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(FacialLandmarkClassifier, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 2 * 2, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

class FacialAnalysisPipeline:
    def __init__(self, detection_model_path, classification_model_path, tag_mapping_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Facial analysis pipeline using device: {self.device}")
        
        # Define landmark classes
        self.landmark_classes = [
            'CjD', 'CjIz', 'CchD', 'CchIzq', 'OjD', 'OjIz', 'Nariz', 
            'N', 'F', 'Bc', 'PmlD', 'PmlIz', 'TrExCjDr', 'TrExCjIz', 
            'TrInCjDr', 'TrInCjIz', 'OD', 'OIz'
        ]
        
        # Load tag mapping if provided
        self.tags = []
        if tag_mapping_path and os.path.exists(tag_mapping_path):
            with open(tag_mapping_path, 'r') as f:
                self.tags = json.load(f)
            print(f"Loaded {len(self.tags)} tags from mapping file")
        
        # Load models
        self.detection_model = self._load_detection_model(detection_model_path)
        self.classification_model = self._load_classification_model(classification_model_path)
        
        # Define transforms
        self.detection_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.classification_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_detection_model(self, model_path: str):
        """Load the facial landmark detection model"""
        try:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.landmark_classes) + 1)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            print("Facial landmark detection model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading detection model: {e}")
            raise e
    
    def _load_classification_model(self, model_path: str):
        """Load the facial characteristic classification model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine number of classes from model checkpoint
            num_classes = None
            if isinstance(checkpoint, dict) and 'classifier.4.weight' in checkpoint:
                num_classes = checkpoint['classifier.4.weight'].shape[0]
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                for key in checkpoint['state_dict']:
                    if key.endswith('classifier.4.weight'):
                        num_classes = checkpoint['state_dict'][key].shape[0]
                        break
            
            if num_classes is None:
                if self.tags:
                    num_classes = len(self.tags)
                else:
                    print("Warning: Could not determine number of classes. Using default of 50.")
                    num_classes = 50
                    self.tags = [f"tag_{i}" for i in range(num_classes)]
            
            print(f"Using classification model with {num_classes} classes")
            model = FacialLandmarkClassifier(num_classes)
            
            # Load weights
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            print("Facial characteristic classification model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading classification model: {e}")
            raise e
    
    def process_image(self, image_path_or_array, confidence_threshold=0.5, output_path=None, display=False) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Process a single image and detect/classify facial landmarks"""
        try:
            # Load image
            if isinstance(image_path_or_array, str):
                image = cv2.imread(image_path_or_array)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path_or_array
                
            # Resize for consistent processing
            orig_height, orig_width = image.shape[:2]
            image_resized = cv2.resize(image, (224, 224))
            
            # Scale factors for bbox conversion
            scale_x = orig_width / 224
            scale_y = orig_height / 224
            
            # Prepare image for detection
            image_tensor = self.detection_transform(image_resized).unsqueeze(0).to(self.device)
            
            # Detect landmarks
            with torch.no_grad():
                detections = self.detection_model(image_tensor)[0]
            
            # Filter by confidence
            keep = detections['scores'] > confidence_threshold
            boxes = detections['boxes'][keep].cpu().numpy()
            labels = detections['labels'][keep].cpu().numpy()
            scores = detections['scores'][keep].cpu().numpy()
            
            # Create visualization copy
            image_viz = image_resized.copy()
            
            # Process each detection
            results = []
            
            for box, label_idx, score in zip(boxes, labels, scores):
                # Get landmark class
                landmark_class = self.landmark_classes[label_idx - 1]  # -1 because background is 0
                
                # Extract region
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, image_resized.shape[1]))
                y1 = max(0, min(y1, image_resized.shape[0]))
                x2 = max(0, min(x2, image_resized.shape[1]))
                y2 = max(0, min(y2, image_resized.shape[0]))
                
                region = image_resized[y1:y2, x1:x2]
                
                if region.size == 0:
                    continue
                    
                # Classify region
                try:
                    region_tensor = self.classification_transform(region).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.classification_model(region_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, pred = torch.max(probs, 1)
                        tag_idx = pred.item()
                        tag_confidence = confidence.item()
                        
                        # Get tag name if available, otherwise use index
                        if tag_idx < len(self.tags):
                            tag = self.tags[tag_idx]
                        else:
                            tag = f"tag_{tag_idx}"
                
                except Exception as e:
                    print(f"Warning: Classification failed for region: {e}")
                    tag = "unknown"
                    tag_confidence = 0.0
                
                # Add to results
                results.append({
                    'landmark_class': landmark_class,
                    'tag': tag,
                    'score': float(score),
                    'tag_confidence': float(tag_confidence),
                    'box': [float(x1*scale_x), float(y1*scale_y), float(x2*scale_x), float(y2*scale_y)]
                })
            
            return results, image_viz
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return [], image
    
    def process_batch(self, image_paths: List[str], confidence_threshold: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
        """Process multiple images and return aggregated results"""
        all_results = {}
        
        for img_path in image_paths:
            try:
                results, _ = self.process_image(img_path, confidence_threshold, display=False)
                all_results[img_path] = results
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                all_results[img_path] = []
                
        return all_results
    
    def get_landmark_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about detected landmarks"""
        if not results:
            return {"total_landmarks": 0, "landmark_classes": [], "tags": [], "average_confidence": 0.0}
        
        landmark_classes = [r['landmark_class'] for r in results]
        tags = [r['tag'] for r in results]
        scores = [r['score'] for r in results]
        
        # Count landmarks by class
        class_counts = {}
        for landmark_class in landmark_classes:
            class_counts[landmark_class] = class_counts.get(landmark_class, 0) + 1
        
        # Count tags
        tag_counts = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_landmarks": len(results),
            "landmark_classes": list(set(landmark_classes)),
            "tags": list(set(tags)),
            "class_counts": class_counts,
            "tag_counts": tag_counts,
            "average_confidence": float(np.mean(scores)) if scores else 0.0,
            "confidence_range": [float(min(scores)), float(max(scores))] if scores else [0.0, 0.0]
        }
