import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from scipy.spatial.distance import euclidean
from scipy import ndimage

logger = logging.getLogger(__name__)

class AnthropometricAnalysisPipeline:
    """Production pipeline for body anthropometric analysis using YOLO pose detection"""
    
    def __init__(self, model_path: str = "/app/models/yolov8n-pose.pt"):
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        
        # YOLO keypoint names (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Body part groupings
        self.body_parts = {
            'head': [0, 1, 2, 3, 4],  # nose, eyes, ears
            'torso': [5, 6, 11, 12],  # shoulders, hips
            'left_arm': [5, 7, 9],    # left shoulder, elbow, wrist
            'right_arm': [6, 8, 10],  # right shoulder, elbow, wrist
            'left_leg': [11, 13, 15], # left hip, knee, ankle
            'right_leg': [12, 14, 16] # right hip, knee, ankle
        }
        
        logger.info(f"Pipeline initialized for device: {self.device}")
    
    def _get_device(self) -> str:
        """Get the best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU")
        except ImportError:
            device = "cpu"
            logger.info("PyTorch not available, using CPU")
        return device
    
    def load_model(self) -> bool:
        """Load the YOLO pose detection model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            logger.info(f"✅ YOLO model loaded successfully from: {self.model_path}")
            logger.info(f"Model type: YOLOv8n-pose")
            logger.info(f"Keypoints supported: {len(self.keypoint_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_pose_only(self, image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Basic pose detection and keypoint extraction"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run YOLO inference
            results = self.model(image)
            detections = []
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    
                    for person_keypoints in keypoints:
                        person_data = {
                            'keypoints': {},
                            'body_parts': {},
                            'confidence_scores': {}
                        }
                        
                        # Extract keypoints
                        for i, (x, y, conf) in enumerate(person_keypoints):
                            if conf > confidence_threshold:
                                person_data['keypoints'][self.keypoint_names[i]] = (int(x), int(y))
                                person_data['confidence_scores'][self.keypoint_names[i]] = float(conf)
                        
                        # Group keypoints into body parts
                        for part_name, keypoint_indices in self.body_parts.items():
                            part_keypoints = {}
                            for idx in keypoint_indices:
                                keypoint_name = self.keypoint_names[idx]
                                if keypoint_name in person_data['keypoints']:
                                    part_keypoints[keypoint_name] = person_data['keypoints'][keypoint_name]
                            
                            if part_keypoints:
                                person_data['body_parts'][part_name] = part_keypoints
                        
                        detections.append(person_data)
            
            return {
                'image_path': image_path,
                'image_shape': image.shape,
                'detections': detections,
                'num_persons': len(detections),
                'processing_successful': True,
                'timestamp': datetime.now().isoformat(),
                'confidence_threshold_used': confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in pose detection: {e}")
            return {
                'image_path': image_path,
                'processing_successful': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def estimate_head_orientation(self, head_keypoints: Dict[str, Tuple[int, int]]) -> Tuple[float, str]:
        """Estimate head orientation and tilt from facial keypoints"""
        if 'left_eye' not in head_keypoints or 'right_eye' not in head_keypoints:
            return 0, "frontal"  # Default assumption
        
        left_eye = head_keypoints['left_eye']
        right_eye = head_keypoints['right_eye']
        
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        # Determine orientation
        if abs(angle) < 15:
            orientation = "frontal"
        elif angle > 15:
            orientation = "tilted_left"
        else:
            orientation = "tilted_right"
        
        return angle, orientation
    
    def get_anatomical_skull_bbox(self, head_keypoints: Dict[str, Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        """Calculate skull bounding box using anatomical proportions"""
        if not head_keypoints:
            return None, "no_keypoints"
        
        # Get head orientation
        angle, orientation = self.estimate_head_orientation(head_keypoints)
        
        # Method 1: Nose-centered with anatomical proportions
        if 'nose' in head_keypoints:
            nose_x, nose_y = head_keypoints['nose']
            
            # Calculate eye distance for scale reference
            eye_distance = 50  # default
            if 'left_eye' in head_keypoints and 'right_eye' in head_keypoints:
                left_eye = head_keypoints['left_eye']
                right_eye = head_keypoints['right_eye']
                eye_distance = euclidean(left_eye, right_eye)
            
            # Anatomical skull proportions
            skull_width = int(eye_distance * 2.4)
            skull_height = int(skull_width * 1.4)
            
            # Adjust for head orientation
            if orientation == "tilted_left" or orientation == "tilted_right":
                skull_width = int(skull_width * 1.1)
            
            # Position skull relative to nose
            skull_center_x = nose_x
            skull_center_y = nose_y - int(skull_height * 0.15)
            
            # Calculate bounding box
            x_min = max(0, skull_center_x - skull_width // 2)
            x_max = skull_center_x + skull_width // 2
            y_min = max(0, skull_center_y - skull_height // 2)
            y_max = skull_center_y + skull_height // 2
            
            return (x_min, y_min, x_max, y_max), "nose_anatomical"
        
        # Method 2: Eye-centered approach
        elif 'left_eye' in head_keypoints and 'right_eye' in head_keypoints:
            left_eye = head_keypoints['left_eye']
            right_eye = head_keypoints['right_eye']
            
            eye_center_x = (left_eye[0] + right_eye[0]) // 2
            eye_center_y = (left_eye[1] + right_eye[1]) // 2
            eye_distance = euclidean(left_eye, right_eye)
            
            # Skull dimensions based on eye distance
            skull_width = int(eye_distance * 2.3)
            skull_height = int(skull_width * 1.6)
            
            # Eyes are approximately 45% down from top of skull
            skull_top_y = eye_center_y - int(skull_height * 0.45)
            skull_bottom_y = skull_top_y + skull_height
            
            x_min = max(0, eye_center_x - skull_width // 2)
            x_max = eye_center_x + skull_width // 2
            y_min = max(0, skull_top_y)
            y_max = skull_bottom_y
            
            return (x_min, y_min, x_max, y_max), "eye_anatomical"
        
        return None, "insufficient_keypoints"
    
    def get_contour_refined_skull(self, image_path: str, initial_bbox: Tuple[int, int, int, int]) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        """Refine skull detection using image processing and contour detection"""
        if not initial_bbox:
            return None, "no_initial_bbox"
        
        image = cv2.imread(image_path)
        if image is None:
            return None, "image_load_error"
        
        x_min, y_min, x_max, y_max = initial_bbox
        
        # Extract head region with padding
        padding = 20
        head_x1 = max(0, x_min - padding)
        head_y1 = max(0, y_min - padding)
        head_x2 = min(image.shape[1], x_max + padding)
        head_y2 = min(image.shape[0], y_max + padding)
        
        head_region = image[head_y1:head_y2, head_x1:head_x2]
        
        if head_region.size == 0:
            return initial_bbox, "region_too_small"
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges1 = cv2.Canny(blurred, 30, 100)
            edges2 = cv2.Canny(blurred, 50, 150)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return initial_bbox, "no_contours"
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box of the largest contour
            cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(largest_contour)
            
            # Convert back to original image coordinates
            refined_x_min = head_x1 + cont_x
            refined_y_min = head_y1 + cont_y
            refined_x_max = refined_x_min + cont_w
            refined_y_max = refined_y_min + cont_h
            
            # Validate refined bbox
            refined_area = cont_w * cont_h
            initial_area = (x_max - x_min) * (y_max - y_min)
            
            # If refined bbox is too different, stick with initial
            if refined_area < initial_area * 0.3 or refined_area > initial_area * 3:
                return initial_bbox, "contour_unreasonable"
            
            return (refined_x_min, refined_y_min, refined_x_max, refined_y_max), "contour_refined"
            
        except Exception as e:
            logger.warning(f"Contour refinement failed: {e}")
            return initial_bbox, "contour_failed"
    
    def calculate_body_proportions(self, person_data: Dict[str, Any], image_path: str, 
                                 include_contour_refinement: bool = True) -> Dict[str, Any]:
        """Calculate precise body proportions using enhanced skull detection"""
        proportions = {
            'skull_height': None,
            'skull_width': None,
            'body_height': None,
            'skull_to_body_ratio': None,
            'measurements_available': False,
            'detection_method': None,
            'skull_bbox': None,
            'head_orientation': None,
            'anatomical_assessment': None,
            'skull_percentage': None
        }
        
        # Get skull bounding box
        if 'head' in person_data['body_parts']:
            head_keypoints = person_data['body_parts']['head']
            
            # Get head orientation info
            angle, orientation = self.estimate_head_orientation(head_keypoints)
            proportions['head_orientation'] = f"{orientation} ({angle:.1f}°)"
            
            # Get anatomical skull bbox
            skull_bbox, anat_method = self.get_anatomical_skull_bbox(head_keypoints)
            
            if skull_bbox and include_contour_refinement:
                # Refine using contour detection
                refined_bbox, refine_method = self.get_contour_refined_skull(image_path, skull_bbox)
                skull_bbox = refined_bbox
                method = f"{anat_method}+{refine_method}"
            else:
                method = anat_method
            
            if skull_bbox:
                x_min, y_min, x_max, y_max = skull_bbox
                proportions['skull_height'] = y_max - y_min
                proportions['skull_width'] = x_max - x_min
                proportions['detection_method'] = method
                proportions['skull_bbox'] = skull_bbox
        
        # Calculate body height
        keypoints = person_data['keypoints']
        if keypoints:
            y_coordinates = [y for x, y in keypoints.values()]
            proportions['body_height'] = max(y_coordinates) - min(y_coordinates)
        
        # Calculate ratio and assessment
        if proportions['skull_height'] and proportions['body_height']:
            proportions['skull_to_body_ratio'] = proportions['skull_height'] / proportions['body_height']
            proportions['measurements_available'] = True
            proportions['skull_percentage'] = proportions['skull_to_body_ratio'] * 100
            
            # Anatomical assessment
            ratio = proportions['skull_to_body_ratio']
            if 1/8 <= ratio <= 1/7:
                proportions['anatomical_assessment'] = "Normal adult skull proportions"
            elif ratio < 1/8:
                proportions['anatomical_assessment'] = "Skull smaller than typical adult"
            else:
                proportions['anatomical_assessment'] = "Skull larger than typical adult"
        
        return proportions
    
    def detect_skull_measurements_only(self, image_path: str, confidence_threshold: float = 0.5,
                                     include_contour_refinement: bool = True) -> Dict[str, Any]:
        """Skull detection and measurement analysis only"""
        # First get pose detection
        pose_results = self.detect_pose_only(image_path, confidence_threshold)
        
        if not pose_results.get('processing_successful', False):
            return pose_results
        
        try:
            skull_results = {
                'image_path': image_path,
                'skull_detections': [],
                'processing_successful': True,
                'timestamp': datetime.now().isoformat(),
                'confidence_threshold_used': confidence_threshold,
                'num_persons': pose_results['num_persons']
            }
            
            # Process each detected person
            for i, person_data in enumerate(pose_results['detections']):
                skull_data = self.calculate_body_proportions(
                    person_data, image_path, include_contour_refinement
                )
                
                # Add person-specific info
                skull_data['person_id'] = i + 1
                skull_data['head_keypoints_detected'] = len(person_data['body_parts'].get('head', {}))
                skull_data['total_keypoints_detected'] = len(person_data['keypoints'])
                
                # Add confidence scores for head keypoints
                head_confidences = {}
                if 'head' in person_data['body_parts']:
                    for kp_name in person_data['body_parts']['head'].keys():
                        if kp_name in person_data['confidence_scores']:
                            head_confidences[kp_name] = person_data['confidence_scores'][kp_name]
                
                skull_data['head_keypoint_confidences'] = head_confidences
                if head_confidences:
                    skull_data['average_head_confidence'] = np.mean(list(head_confidences.values()))
                
                skull_results['skull_detections'].append(skull_data)
            
            return skull_results
            
        except Exception as e:
            logger.error(f"Error in skull measurement detection: {e}")
            return {
                'image_path': image_path,
                'processing_successful': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_body_anthropometry(self, image_path: str, confidence_threshold: float = 0.5,
                                 detailed_analysis: bool = True) -> Dict[str, Any]:
        """Complete body anthropometric analysis"""
        # Get pose detection
        pose_results = self.detect_pose_only(image_path, confidence_threshold)
        
        if not pose_results.get('processing_successful', False):
            return pose_results
        
        try:
            analysis_results = {
                'image_path': image_path,
                'image_shape': pose_results['image_shape'],
                'num_persons': pose_results['num_persons'],
                'anthropometric_analysis': [],
                'processing_successful': True,
                'timestamp': datetime.now().isoformat(),
                'confidence_threshold_used': confidence_threshold,
                'analysis_type': 'complete_anthropometric'
            }
            
            # Process each detected person
            for i, person_data in enumerate(pose_results['detections']):
                person_analysis = {
                    'person_id': i + 1,
                    'keypoint_summary': self._get_keypoint_summary(person_data),
                    'body_proportions': self.calculate_body_proportions(person_data, image_path, True),
                    'pose_keypoints': person_data['keypoints'] if detailed_analysis else {},
                    'confidence_analysis': self._analyze_confidence_scores(person_data)
                }
                
                # Add detailed anatomical analysis if requested
                if detailed_analysis:
                    person_analysis['detailed_analysis'] = self._get_detailed_anatomical_analysis(
                        person_analysis['body_proportions'], person_data
                    )
                
                analysis_results['anthropometric_analysis'].append(person_analysis)
            
            # Add overall analysis summary
            analysis_results['analysis_summary'] = self._get_analysis_summary(
                analysis_results['anthropometric_analysis']
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in complete anthropometric analysis: {e}")
            return {
                'image_path': image_path,
                'processing_successful': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_keypoint_summary(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of detected keypoints"""
        total_keypoints = len(person_data['keypoints'])
        body_parts_detected = len(person_data['body_parts'])
        
        head_keypoints = len(person_data['body_parts'].get('head', {}))
        torso_keypoints = len(person_data['body_parts'].get('torso', {}))
        
        return {
            'total_keypoints': total_keypoints,
            'total_possible': len(self.keypoint_names),
            'detection_percentage': (total_keypoints / len(self.keypoint_names)) * 100,
            'body_parts_detected': body_parts_detected,
            'head_keypoints': head_keypoints,
            'torso_keypoints': torso_keypoints,
            'keypoint_completeness': 'excellent' if total_keypoints >= 15 else 
                                   'good' if total_keypoints >= 12 else
                                   'fair' if total_keypoints >= 8 else 'poor'
        }
    
    def _analyze_confidence_scores(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence scores for keypoints"""
        confidences = list(person_data['confidence_scores'].values())
        
        if not confidences:
            return {'analysis': 'no_confidence_data'}
        
        # Overall confidence statistics
        avg_confidence = np.mean(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        # Head-specific confidence
        head_confidences = []
        if 'head' in person_data['body_parts']:
            for kp_name in person_data['body_parts']['head'].keys():
                if kp_name in person_data['confidence_scores']:
                    head_confidences.append(person_data['confidence_scores'][kp_name])
        
        analysis = {
            'overall_average': float(avg_confidence),
            'confidence_range': {'min': float(min_confidence), 'max': float(max_confidence)},
            'head_average': float(np.mean(head_confidences)) if head_confidences else None,
            'reliability_assessment': self._assess_reliability(avg_confidence, head_confidences)
        }
        
        return analysis
    
    def _assess_reliability(self, overall_avg: float, head_confidences: List[float]) -> str:
        """Assess overall reliability of the detection"""
        if head_confidences:
            head_avg = np.mean(head_confidences)
            if head_avg >= 0.8 and overall_avg >= 0.7:
                return "excellent - very reliable measurements"
            elif head_avg >= 0.6 and overall_avg >= 0.6:
                return "good - reliable measurements"
            elif head_avg >= 0.4 and overall_avg >= 0.5:
                return "fair - moderate reliability"
            else:
                return "poor - low reliability, use with caution"
        else:
            return "insufficient - no head keypoints detected"
    
    def _get_detailed_anatomical_analysis(self, proportions: Dict[str, Any], 
                                        person_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed anatomical analysis and insights"""
        analysis = {
            'skull_analysis': {},
            'body_composition': {},
            'anthropometric_insights': {},
            'age_assessment': {},
            'measurement_quality': {}
        }
        
        # Skull-specific analysis
        if proportions['measurements_available']:
            skull_percentage = proportions['skull_percentage']
            
            analysis['skull_analysis'] = {
                'skull_to_body_percentage': skull_percentage,
                'anatomical_classification': proportions['anatomical_assessment'],
                'comparison_to_norms': {
                    'adult_range': '12.5-14.3%',
                    'child_range': '16-18%',
                    'measured_value': f"{skull_percentage:.1f}%"
                }
            }
            
            # Age assessment based on skull proportions
            if 12.5 <= skull_percentage <= 14.3:
                age_assessment = "adult_proportions"
                age_description = "Skull proportions consistent with adult anatomy"
            elif skull_percentage > 16:
                age_assessment = "child_like_proportions"
                age_description = "Skull proportions suggest younger individual or different body type"
            else:
                age_assessment = "intermediate_proportions"
                age_description = "Skull proportions between typical adult and child ranges"
            
            analysis['age_assessment'] = {
                'classification': age_assessment,
                'description': age_description,
                'confidence': 'high' if 12.0 <= skull_percentage <= 18.0 else 'moderate'
            }
        
        # Body composition insights
        torso_keypoints = person_data['body_parts'].get('torso', {})
        limb_keypoints = len(person_data['body_parts'].get('left_arm', {})) + len(person_data['body_parts'].get('right_arm', {}))
        
        analysis['body_composition'] = {
            'torso_detection_quality': 'good' if len(torso_keypoints) >= 3 else 'limited',
            'limb_detection_quality': 'good' if limb_keypoints >= 4 else 'limited',
            'pose_assessment': self._assess_pose_quality(person_data)
        }
        
        # Anthropometric insights
        if proportions['head_orientation']:
            orientation = proportions['head_orientation']
            analysis['anthropometric_insights'] = {
                'head_orientation': orientation,
                'measurement_reliability': 'high' if 'frontal' in orientation else 'moderate',
                'recommended_measurements': self._get_measurement_recommendations(proportions, person_data)
            }
        
        return analysis
    
    def _assess_pose_quality(self, person_data: Dict[str, Any]) -> str:
        """Assess the quality of the detected pose for anthropometric analysis"""
        total_keypoints = len(person_data['keypoints'])
        head_keypoints = len(person_data['body_parts'].get('head', {}))
        
        if total_keypoints >= 15 and head_keypoints >= 4:
            return "excellent - suitable for detailed anthropometric analysis"
        elif total_keypoints >= 12 and head_keypoints >= 3:
            return "good - suitable for basic anthropometric analysis"
        elif total_keypoints >= 8 and head_keypoints >= 2:
            return "fair - limited anthropometric analysis possible"
        else:
            return "poor - insufficient keypoints for reliable analysis"
    
    def _get_measurement_recommendations(self, proportions: Dict[str, Any], 
                                       person_data: Dict[str, Any]) -> List[str]:
        """Get recommendations for improving measurement accuracy"""
        recommendations = []
        
        # Head orientation recommendations
        if proportions['head_orientation'] and 'tilted' in proportions['head_orientation']:
            recommendations.append("Consider retaking image with subject facing camera directly for more accurate skull measurements")
        
        # Keypoint detection recommendations
        head_keypoints = len(person_data['body_parts'].get('head', {}))
        if head_keypoints < 4:
            recommendations.append("Ensure face is clearly visible and well-lit for better skull detection")
        
        # Overall pose recommendations
        total_keypoints = len(person_data['keypoints'])
        if total_keypoints < 12:
            recommendations.append("Full body should be visible in image for complete anthropometric analysis")
        
        if not recommendations:
            recommendations.append("Good detection quality - measurements are reliable")
        
        return recommendations
    
    def _get_analysis_summary(self, anthropometric_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get overall analysis summary"""
        if not anthropometric_analyses:
            return {'status': 'no_analyses_available'}
        
        # Count successful measurements
        successful_skull_measurements = sum(
            1 for analysis in anthropometric_analyses 
            if analysis['body_proportions']['measurements_available']
        )
        
        # Average confidence if available
        confidences = []
        for analysis in anthropometric_analyses:
            if 'overall_average' in analysis['confidence_analysis']:
                confidences.append(analysis['confidence_analysis']['overall_average'])
        
        avg_confidence = np.mean(confidences) if confidences else None
        
        return {
            'total_persons_detected': len(anthropometric_analyses),
            'successful_skull_measurements': successful_skull_measurements,
            'measurement_success_rate': (successful_skull_measurements / len(anthropometric_analyses)) * 100,
            'average_detection_confidence': float(avg_confidence) if avg_confidence else None,
            'overall_quality': 'excellent' if successful_skull_measurements == len(anthropometric_analyses) and (avg_confidence or 0) > 0.7
                              else 'good' if successful_skull_measurements > 0 and (avg_confidence or 0) > 0.5
                              else 'poor',
            'recommendations': self._get_overall_recommendations(anthropometric_analyses)
        }
    
    def _get_overall_recommendations(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Get overall recommendations for the analysis"""
        recommendations = []
        
        successful_measurements = sum(
            1 for analysis in analyses 
            if analysis['body_proportions']['measurements_available']
        )
        
        if successful_measurements == 0:
            recommendations.append("No skull measurements could be obtained - ensure face is clearly visible")
        elif successful_measurements < len(analyses):
            recommendations.append("Some persons could not be measured - check image quality and pose")
        
        # Check for low confidence
        low_confidence_count = sum(
            1 for analysis in analyses
            if analysis['confidence_analysis'].get('overall_average', 1.0) < 0.5
        )
        
        if low_confidence_count > 0:
            recommendations.append("Some detections have low confidence - consider better lighting or image quality")
        
        if not recommendations:
            recommendations.append("Analysis completed successfully with good quality measurements")
        
        return recommendations
