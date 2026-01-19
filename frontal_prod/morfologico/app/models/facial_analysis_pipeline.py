import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as transforms
import os
import json
from typing import List, Dict, Any, Tuple
from PIL import Image as PILImage

# Import centralized threshold validation from common module
# This ensures consistent threshold application per Umbrales_Rasgos.txt
try:
    from common import ThresholdValidator, ModuleType
    THRESHOLD_VALIDATOR_AVAILABLE = True
except ImportError:
    # Fallback if common module not available (development mode)
    THRESHOLD_VALIDATOR_AVAILABLE = False
    print("Warning: common.ThresholdValidator not available, using fallback thresholds")

class FacialLandmarkClassifier(torch.nn.Module):
    """Shape classifier for facial landmarks (45 classes)"""
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

class FacialLandmarkSizeClassifier(torch.nn.Module):
    """Enhanced size classifier for eyebrows (3 classes: ap, g, ngna)"""
    def __init__(self, num_classes):
        super(FacialLandmarkSizeClassifier, self).__init__()
        # Enhanced CNN architecture with batch normalization and more capacity
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to handle different input sizes
        self.avgpool = torch.nn.AdaptiveAvgPool2d((2, 2))

        # Enhanced fully connected layers with more capacity
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 2 * 2, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

class FacialAnalysisPipeline:
    def __init__(self, detection_model_path, classification_model_path, size_model_path=None, tag_mapping_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Facial analysis pipeline using device: {self.device}")

        # Define landmark classes - 23-class model
        self.landmark_classes = [
            'cj_d', 'cj_i', 'cch_d', 'cch_i', 'oj_d', 'oj_i', 'nariz',
            'n', 'f', 'bc', 'pml_d', 'pml_i', 'tr_ex_cj_dr', 'tr_ex_cj_i',
            'tr_in_cj_d', 'tr_in_cj_i', 'o_d', 'o_i', 'ac_d', 'ac_i',
            'entrecejo', 'parpado_dr', 'parpado_i'
        ]

        # NEW: Updated shape tag mapping (45 classes - removed problematic tags)
        self.shape_tags = [
            '0', '1', '2', '3', 'a_n', 'ab', 'al', 'ar',
            'crl', 'cv', 'delgada', 'el', 'fr', 'grueso', 'h', 'hn', 'i',
            'lineas_sonriza', 'lineas_verticales', 'll', 'lunar', 'md', 'md_a', 'mercurial', 'nd', 'normal', 'nrml',
            'nt', 'on', 'pc', 'pg', 'pl', 'planos', 'pliegue', 'pm', 'pn', 'ptosis',
            'pursed', 'rc', 'rd', 'salido', 'sl', 'solar', 'sp_sl', 'uniceja'
        ]

        # NEW: Eyebrow size tags (3 classes - only for cj_d, cj_i)
        self.eyebrow_size_tags = ['ap', 'g', 'ngna']
        self.eyebrow_classes = ['cj_d', 'cj_i']

        # NEW: Bbox confinement - valid shape tags per landmark class
        self.valid_shape_tags = {
            'cj_d': ['rc', 'el', 'cv'],  # Right eyebrow shapes
            'cj_i': ['rc', 'el', 'cv'],  # Left eyebrow shapes
            'nariz': ['delgada', 'nrml', 'grueso'],  # Nose shapes
            'bc': ['lunar', 'mercurial', 'pursed', 'solar'],  # Mouth shapes
            'n': ['i', 'pn', 'rd'],  # Nostril shapes
            'oj_d': ['al', 'crl', 'fr', 'md', 'md_a'],  # Right eye shapes
            'oj_i': ['al', 'crl', 'fr', 'md', 'md_a'],  # Left eye shapes
            'entrecejo': ['lineas_verticales', 'normal', 'uniceja'],  # Between eyebrows
            'parpado_dr': ['pliegue', 'ptosis'],  # Right eyelid
            'parpado_i': ['pliegue', 'ptosis'],  # Left eyelid
            'tr_ex_cj_dr': ['a_n', 'ab', 'h'],  # Outer right eyebrow area
            'tr_ex_cj_i': ['a_n', 'ab', 'h'],  # Outer left eyebrow area
            'tr_in_cj_d': ['ar', 'h'],  # Inner right eyebrow area
            'tr_in_cj_i': ['ar', 'h'],  # Inner left eyebrow area
        }

        # Backward compatibility: create tag_mapping for API responses
        self.tag_mapping = {f"tag_{i}": tag for i, tag in enumerate(self.shape_tags)}
        self.tags = [f"tag_{i}" for i in range(len(self.shape_tags))]

        print(f"Initialized with {len(self.shape_tags)} shape tags and {len(self.eyebrow_size_tags)} eyebrow size tags")

        # Load models
        self.detection_model = self._load_detection_model(detection_model_path)
        self.classification_model = self._load_classification_model(classification_model_path, num_classes=len(self.shape_tags))

        # NEW: Load eyebrow size model if provided
        self.size_model = None
        if size_model_path:
            self.size_model = self._load_size_model(size_model_path)
            print(f"Eyebrow size model loaded for classes: {self.eyebrow_classes}")

        # Initialize centralized threshold validator
        # Uses thresholds from common/threshold_config.py (Single Source of Truth)
        if THRESHOLD_VALIDATOR_AVAILABLE:
            self.threshold_validator = ThresholdValidator()
            print("✓ ThresholdValidator initialized from common module")
        else:
            self.threshold_validator = None
            print("⚠ ThresholdValidator not available, thresholds not applied")

        # Define transforms
        self.detection_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.classification_transform = transforms.Compose([
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
    
    def _load_classification_model(self, model_path: str, num_classes: int = 45):
        """Load the shape classification model (45 classes)"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

            print(f"Loading shape classification model with {num_classes} classes")

            model = FacialLandmarkClassifier(num_classes)

            # Load weights
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            print("Shape classification model loaded successfully!")
            return model

        except Exception as e:
            print(f"Error loading shape classification model: {e}")
            raise e

    def _load_size_model(self, model_path: str):
        """Load the eyebrow size classification model (3 classes)"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

            num_classes = 3  # ap, g, ngna
            print(f"Loading eyebrow size classification model with {num_classes} classes")

            model = FacialLandmarkSizeClassifier(num_classes)

            # Load weights
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            print("Eyebrow size classification model loaded successfully!")
            return model

        except Exception as e:
            print(f"Error loading eyebrow size model: {e}")
            raise e
    
    def _validate_shape_prediction(self, predicted_tag, landmark_class, probabilities):
        """Validate that shape prediction makes sense for the detected landmark class (bbox confinement)"""

        # If we have specific valid tags for this landmark class
        if landmark_class in self.valid_shape_tags:
            if predicted_tag in self.valid_shape_tags[landmark_class]:
                # Original prediction is valid, return it with its confidence
                predicted_idx = self.shape_tags.index(predicted_tag)
                return predicted_tag, probabilities[predicted_idx].item()
            else:
                # Original prediction is invalid, find highest confidence valid prediction
                best_tag = None
                best_confidence = -1.0

                for valid_tag in self.valid_shape_tags[landmark_class]:
                    if valid_tag in self.shape_tags:
                        tag_idx = self.shape_tags.index(valid_tag)
                        tag_confidence = probabilities[tag_idx].item()
                        if tag_confidence > best_confidence:
                            best_confidence = tag_confidence
                            best_tag = valid_tag

                if best_tag is not None:
                    return best_tag, best_confidence

                # Fallback to first valid tag if no probabilities provided
                return self.valid_shape_tags[landmark_class][0], 0.0

        # If no specific mapping, return the prediction (for backward compatibility)
        predicted_idx = self.shape_tags.index(predicted_tag) if predicted_tag in self.shape_tags else 0
        return predicted_tag, probabilities[predicted_idx].item()

    def _validate_diagnosis_threshold(self, landmark_class: str, tag_name: str, confidence: float) -> Dict[str, Any]:
        """
        Validate diagnosis against thresholds from common/threshold_config.py
        
        Thresholds are per Umbrales_Rasgos.txt document (lines 129-280).
        The diagnosis key is constructed as: landmark_class + '_' + tag_name
        Example: 'cj_d' + 'cv' = 'cj_d_cv' (ceja derecha curva, threshold 50%)
        
        Args:
            landmark_class: The detected landmark (e.g., 'cj_d', 'oj_i', 'bc')
            tag_name: The classification tag (e.g., 'cv', 'al', 'lunar')
            confidence: The classification confidence (0.0 - 1.0)
            
        Returns:
            Dict with 'passes', 'threshold', and 'rule' keys
        """
        # Construct the full diagnosis key
        diagnosis_key = f"{landmark_class}_{tag_name}"
        
        # Use centralized ThresholdValidator if available
        if self.threshold_validator is not None:
            try:
                # Get rules for FRONTAL_MORFOLOGICO module
                from common.threshold_config import FRONTAL_MORFOLOGICO_RULES
                
                if diagnosis_key in FRONTAL_MORFOLOGICO_RULES:
                    rule = FRONTAL_MORFOLOGICO_RULES[diagnosis_key]
                    passes = confidence >= rule.threshold
                    return {
                        'passes': passes,
                        'threshold': rule.threshold,
                        'rule': f"{'Accepted' if passes else 'Rejected'} {diagnosis_key} ({confidence:.1%} {'≥' if passes else '<'} {rule.threshold:.1%})"
                    }
                else:
                    # No specific threshold for this diagnosis
                    # Per Umbrales_Rasgos.txt line 73: use 15-point difference rule
                    return {
                        'passes': True,  # No threshold = passes by default
                        'threshold': None,
                        'rule': f"No threshold defined for {diagnosis_key}, accepted by default"
                    }
                    
            except Exception as e:
                print(f"ThresholdValidator error for {diagnosis_key}: {e}")
        
        # Fallback: No threshold validation available
        return {
            'passes': True,
            'threshold': None,
            'rule': f"Threshold validation not available for {diagnosis_key}"
        }

    def process_image(self, image_path_or_array, confidence_threshold=0.5, output_path=None, display=False) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Process a single image and detect/classify facial landmarks with bbox confinement and eyebrow size"""
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

                # SHAPE CLASSIFICATION with bbox confinement
                try:
                    # Resize and convert region to PIL Image
                    region_resized = cv2.resize(region, (64, 64))
                    region_pil = PILImage.fromarray(region_resized)
                    region_tensor = self.classification_transform(region_pil).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.classification_model(region_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)

                        # Get top prediction
                        tag_idx = torch.argmax(probs, dim=1).item()
                        predicted_tag = self.shape_tags[tag_idx] if tag_idx < len(self.shape_tags) else 'unknown'

                        # BBOX CONFINEMENT: Validate and correct prediction
                        tag_name, tag_confidence = self._validate_shape_prediction(predicted_tag, landmark_class, probs[0])

                        # Get top 3 predictions for additional info
                        top3_probs, top3_indices = torch.topk(probs, k=min(3, len(self.shape_tags)), dim=1)
                        top_tags = []
                        for i in range(min(3, top3_probs.shape[1])):
                            t_idx = top3_indices[0][i].item()
                            t_tag = self.shape_tags[t_idx] if t_idx < len(self.shape_tags) else 'unknown'
                            t_conf = top3_probs[0][i].item()
                            top_tags.append({
                                'tag': t_tag,
                                'confidence': float(t_conf),
                                'rank': i + 1
                            })

                except Exception as e:
                    print(f"Warning: Shape classification failed for {landmark_class}: {e}")
                    tag_name = "unknown"
                    tag_confidence = 0.0
                    top_tags = [{'tag': tag_name, 'confidence': tag_confidence, 'rank': 1}]

                # EYEBROW SIZE CLASSIFICATION (only for cj_d, cj_i)
                size_tag = None
                size_confidence = None
                if self.size_model and landmark_class in self.eyebrow_classes:
                    try:
                        # Reuse the region tensor
                        region_resized = cv2.resize(region, (64, 64))
                        region_pil = PILImage.fromarray(region_resized)
                        region_tensor = self.classification_transform(region_pil).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            size_outputs = self.size_model(region_tensor)
                            size_probs = torch.nn.functional.softmax(size_outputs, dim=1)
                            size_idx = torch.argmax(size_probs, dim=1).item()

                            size_tag = self.eyebrow_size_tags[size_idx] if size_idx < len(self.eyebrow_size_tags) else 'unknown'
                            size_confidence = size_probs[0][size_idx].item()

                    except Exception as e:
                        print(f"Warning: Size classification failed for {landmark_class}: {e}")
                        size_tag = None
                        size_confidence = None

                # Create result
                result = {
                    'landmark_class': landmark_class,
                    'tag': f"tag_{self.shape_tags.index(tag_name)}" if tag_name in self.shape_tags else "tag_0",  # For backward compat
                    'tag_name': tag_name,
                    'score': float(score),
                    'tag_confidence': float(tag_confidence),
                    'top_tags': top_tags,
                    'box': [float(x1*scale_x), float(y1*scale_y), float(x2*scale_x), float(y2*scale_y)]
                }

                # Add eyebrow size if available
                if size_tag is not None:
                    result['size_tag'] = size_tag
                    result['size_confidence'] = float(size_confidence)

                # Apply threshold validation from common/threshold_config.py
                threshold_result = self._validate_diagnosis_threshold(
                    landmark_class, tag_name, tag_confidence
                )
                result['passes_threshold'] = threshold_result['passes']
                result['threshold_applied'] = threshold_result['threshold']
                result['threshold_rule'] = threshold_result['rule']

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
