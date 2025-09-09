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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class MinimalPointModel(nn.Module):
    """Optimized point detection model (ensemble version)"""
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

class FacialLandmarkGNN(nn.Module):
    """GNN model for facial landmark refinement and false positive filtering"""
    def __init__(self, node_features=3, hidden_dim=128):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(3)
        ])
        
        self.coordinate_refiner = nn.Linear(hidden_dim, 2)
        self.confidence_refiner = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_features, adjacency_matrix):
        h = self.node_encoder(node_features)
        
        for layer in self.gnn_layers:
            messages = torch.matmul(adjacency_matrix, h)
            h = layer(torch.cat([h, messages], dim=-1)) + h
        
        refined_coords = self.coordinate_refiner(h)
        confidence_scores = torch.sigmoid(self.confidence_refiner(h))
        
        return refined_coords, confidence_scores

class IntegratedLandmarkPipelineV2:
    """Working pipeline class from the successful PDF script"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.point_model = None
        self.gnn_model = None
        self.adjacency_matrix = None
        self.all_classes = []
        
    def load_models(self, point_model_path: str, gnn_model_path: str = None):
        """Load the ensemble models (point detection + GNN)"""
        try:
            # Load point detection model
            print("Loading optimized point detection model...")
            checkpoint = torch.load(point_model_path, map_location=self.device, weights_only=False)
            
            self.all_classes = checkpoint['all_classes']
            num_keypoints = checkpoint.get('num_keypoints', len(self.all_classes))
            
            self.point_model = MinimalPointModel(num_keypoints)
            self.point_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.point_model.to(self.device)
            self.point_model.eval()
            
            print(f"Point detection model loaded with {len(self.all_classes)} classes")
            
            # Load GNN model if provided
            if gnn_model_path and Path(gnn_model_path).exists():
                print("Loading GNN refinement model...")
                try:
                    gnn_checkpoint = torch.load(gnn_model_path, map_location=self.device, weights_only=False)
                    
                    self.gnn_model = FacialLandmarkGNN()
                    
                    # Try different possible keys for the model state dict
                    model_state_key = None
                    for key in ['model_state_dict', 'state_dict', 'model']:
                        if key in gnn_checkpoint:
                            model_state_key = key
                            break
                    
                    if model_state_key:
                        self.gnn_model.load_state_dict(gnn_checkpoint[model_state_key])
                        self.gnn_model.to(self.device)
                        self.gnn_model.eval()
                        
                        # Load adjacency matrix for GNN
                        self.adjacency_matrix = gnn_checkpoint.get('adjacency_matrix')
                        if self.adjacency_matrix is not None:
                            self.adjacency_matrix = self.adjacency_matrix.to(self.device)
                        else:
                            # Create default adjacency matrix if missing
                            num_points = len(self.all_classes)
                            self.adjacency_matrix = torch.ones(num_points, num_points, device=self.device)
                            print("Created default adjacency matrix for GNN")
                        
                        print("GNN model loaded successfully")
                    else:
                        print(f"Available keys in GNN checkpoint: {list(gnn_checkpoint.keys())}")
                        raise KeyError("No valid model state dict key found")
                        
                except Exception as e:
                    print(f"Warning: Could not load GNN model: {e}")
                    self.gnn_model = None
            else:
                print("GNN model not provided, using point detection only")
                self.gnn_model = None
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def _extract_keypoints_from_heatmaps(self, heatmaps, image_size):
        """Extract keypoints from heatmaps - WORKING VERSION"""
        batch_size, num_keypoints, hm_height, hm_width = heatmaps.shape
        
        # Enhanced smoothing for better point detection
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
        
        # Scale from heatmap to model input size (224x224)
        scale_x = image_size / hm_width
        scale_y = image_size / hm_height
        
        keypoints = torch.stack([max_x * scale_x, max_y * scale_y], dim=2)
        confidences = max_vals
        
        return keypoints, confidences

    def _remove_wrong_suffix_points(self, coords, confidences, profile_type):
        """Remove points with wrong suffix based on profile type - WORKING VERSION"""
        wrong_suffix = '_d' if profile_type == 'left' else '_i'
        
        for i, class_name in enumerate(self.all_classes):
            if wrong_suffix in class_name:
                confidences[i] *= 0.01  # Heavy penalty for wrong suffix
        
        return confidences
    
    def _filter_false_positives(self, coords, confidences):
        """Apply enhanced confidence-based filtering - WORKING VERSION"""
        threshold = 0.15
        low_conf_mask = confidences < threshold
        confidences[low_conf_mask] *= 0.1
        
        return confidences

    def predict(self, image_path):
        """Simple prediction method for compatibility with existing code"""
        # Load and process image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized.transpose((2, 0, 1))).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Point detection and profile classification
            heatmaps, profile_logits = self.point_model(image_tensor)
            
            # Determine profile type
            profile_probs = torch.softmax(profile_logits[0], dim=0)
            profile_idx = profile_probs.argmax().item()
            profile_type = 'left' if profile_idx == 0 else 'right'
            profile_confidence = profile_probs.max().item()
            
            # Extract coordinates from heatmaps
            coords, confidences = self._extract_keypoints_from_heatmaps(heatmaps, 224)
            coords = coords[0]  # Remove batch dimension
            confidences = confidences[0]
            
            # Apply profile-aware filtering
            confidences = self._remove_wrong_suffix_points(coords, confidences, profile_type)
            
            # Apply confidence filtering
            confidences = self._filter_false_positives(coords, confidences)
            
            # Scale coordinates to original image size
            orig_height, orig_width = image_rgb.shape[:2]
            scale_x = orig_width / 224.0
            scale_y = orig_height / 224.0
            
            final_coords_scaled = coords.clone()
            final_coords_scaled[:, 0] *= scale_x
            final_coords_scaled[:, 1] *= scale_y
            
            # GNN-based filtering (if available)
            if self.gnn_model is not None and self.adjacency_matrix is not None:
                # Normalize coordinates for GNN
                norm_coords = coords / 224.0
                node_features = torch.cat([norm_coords, confidences.unsqueeze(1)], dim=1)
                
                # Get GNN confidence scores
                _, gnn_confidence = self.gnn_model(node_features, self.adjacency_matrix)
                
                # Apply GNN filtering
                final_confidences = confidences * gnn_confidence.squeeze()
            else:
                final_confidences = confidences
        
        # Prepare results
        results = {
            'profile_type': profile_type,
            'profile_confidence': profile_confidence,
            'landmarks': {},
            'raw_predictions': {},
            'image_size': (orig_width, orig_height),
            'filtered_count': 0
        }
        
        # Extract high-confidence landmarks
        confidence_threshold = 0.15
        for i, class_name in enumerate(self.all_classes):
            conf = final_confidences[i].item()
            x, y = final_coords_scaled[i].cpu().numpy()
            
            # Store raw predictions
            results['raw_predictions'][class_name] = {
                'x': float(x),
                'y': float(y),
                'confidence': float(conf)
            }
            
            # Store high-confidence landmarks
            if conf > confidence_threshold:
                results['landmarks'][class_name] = {
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(conf)
                }
            else:
                results['filtered_count'] += 1
        
        return results