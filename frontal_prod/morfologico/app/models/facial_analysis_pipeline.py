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
        
        # Define landmark classes - updated to match 23-class model
        self.landmark_classes = [
            'cj_d', 'cj_i', 'cch_d', 'cch_i', 'oj_d', 'oj_i', 'nariz', 
            'n', 'f', 'bc', 'pml_d', 'pml_i', 'tr_ex_cj_dr', 'tr_ex_cj_i', 
            'tr_in_cj_d', 'tr_in_cj_i', 'o_d', 'o_i', 'ac_d', 'ac_i',
            'entrecejo', 'parpado_dr', 'parpado_i'
        ]
        
        # Use the exact tag mapping from classification model training (52 tags)
        self.tag_mapping = {
            "tag_0": "0", "tag_1": "1", "tag_2": "2", "tag_3": "3", "tag_4": "a_n", 
            "tag_5": "ab", "tag_6": "abierta", "tag_7": "adelgazamiento", "tag_8": "al", 
            "tag_9": "ap", "tag_10": "ar", "tag_11": "bigote", "tag_12": "carnosos", 
            "tag_13": "crl", "tag_14": "cv", "tag_15": "delgada", "tag_16": "el", 
            "tag_17": "fr", "tag_18": "g", "tag_19": "grueso", "tag_20": "h", 
            "tag_21": "hn", "tag_22": "i", "tag_23": "lineas_sonriza", "tag_24": "lineas_verticales", 
            "tag_25": "ll", "tag_26": "lunar", "tag_27": "md", "tag_28": "md_a", 
            "tag_29": "mercurial", "tag_30": "nd", "tag_31": "normal", "tag_32": "nrml", 
            "tag_33": "nt", "tag_34": "on", "tag_35": "pc", "tag_36": "pg", 
            "tag_37": "pl", "tag_38": "planos", "tag_39": "pliegue", "tag_40": "pm", 
            "tag_41": "pn", "tag_42": "ptosis", "tag_43": "pursed", "tag_44": "rc", 
            "tag_45": "rd", "tag_46": "salido", "tag_47": "sl", "tag_48": "solar", 
            "tag_49": "sonriendo", "tag_50": "sp_sl", "tag_51": "uniceja"
        }
        
        # Initialize tags list for all 52 classes
        self.tags = [f"tag_{i}" for i in range(52)]
        
        print(f"Initialized with {len(self.tags)} tags and {len(self.tag_mapping)} tag mappings")
        
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
            num_classes = 52  # We know it's 52 from your model info
            
            if isinstance(checkpoint, dict):
                if 'classifier.4.weight' in checkpoint:
                    detected_classes = checkpoint['classifier.4.weight'].shape[0]
                    print(f"Detected {detected_classes} classes from model checkpoint")
                    if detected_classes != 52:
                        print(f"Warning: Expected 52 classes but model has {detected_classes}")
                    num_classes = detected_classes
                elif 'state_dict' in checkpoint:
                    for key in checkpoint['state_dict']:
                        if key.endswith('classifier.4.weight'):
                            detected_classes = checkpoint['state_dict'][key].shape[0]
                            print(f"Detected {detected_classes} classes from model checkpoint")
                            if detected_classes != 52:
                                print(f"Warning: Expected 52 classes but model has {detected_classes}")
                            num_classes = detected_classes
                            break
            
            print(f"Loading classification model with {num_classes} classes")
            
            # Ensure our mapping matches the model
            if num_classes != len(self.tags):
                print(f"Adjusting tags from {len(self.tags)} to {num_classes} to match model")
                if num_classes < len(self.tags):
                    self.tags = self.tags[:num_classes]
                else:
                    while len(self.tags) < num_classes:
                        self.tags.append(f"tag_{len(self.tags)}")
            
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

                        # Get top 3 predictions instead of just top 1
                        top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)

                        # Extract top 3 predictions for top_tags format
                        top_tags = []
                        for i in range(min(3, top3_probs.shape[1])):
                            tag_idx = top3_indices[0][i].item()
                            tag_confidence = top3_probs[0][i].item()

                            # Get tag using correct index
                            if tag_idx < len(self.tags):
                                tag_id = self.tags[tag_idx]  # This gives us "tag_X"
                            else:
                                tag_id = f"tag_{tag_idx}"
                                print(f"Warning: Model predicted index {tag_idx} but we only have {len(self.tags)} tags")

                            # Get human-readable tag name
                            tag_name_mapped = self.tag_mapping.get(tag_id, "Unknown")

                            top_tags.append({
                                'tag': tag_name_mapped,  # Human-readable name
                                'confidence': float(tag_confidence),
                                'rank': i + 1
                            })

                        # For backward compatibility, keep the original fields for the top prediction
                        tag_idx = top3_indices[0][0].item()
                        tag = self.tags[tag_idx] if tag_idx < len(self.tags) else f"tag_{tag_idx}"
                        tag_name = self.tag_mapping.get(tag, "Unknown")
                        tag_confidence = top3_probs[0][0].item()
                
                except Exception as e:
                    print(f"Warning: Classification failed for region: {e}")
                    tag = "unknown"
                    tag_name = "Unknown"
                    tag_confidence = 0.0
                    top_tags = [{'tag': tag_name, 'confidence': tag_confidence, 'rank': 1}]

                # Create result
                result = {
                    'landmark_class': landmark_class,
                    'tag': tag,
                    'tag_name': tag_name,
                    'score': float(score),
                    'tag_confidence': float(tag_confidence),
                    'top_tags': top_tags,  # Add top 3 predictions in new format
                    'box': [float(x1*scale_x), float(y1*scale_y), float(x2*scale_x), float(y2*scale_y)]
                }
                
                results.append(result)
            
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
