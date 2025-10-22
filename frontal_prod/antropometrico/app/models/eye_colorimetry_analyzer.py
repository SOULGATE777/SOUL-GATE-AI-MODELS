import cv2
import numpy as np
import dlib
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import colorsys
from collections import Counter
import os
import json


class EyeColorimetryAnalyzer:
    """
    A module for extracting eye regions and performing colorimetry analysis.
    Integrated into the antropometrico system.
    """
    
    def __init__(self, predictor_path="/app/models/shape_predictor_68_face_landmarks.dat"):
        """
        Initialize the eye colorimetry analyzer.
        
        Args:
            predictor_path (str): Path to the dlib shape predictor model
        """
        self.detector = dlib.get_frontal_face_detector()
        
        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
            print("✅ Eye colorimetry analyzer initialized with dlib predictor")
        else:
            print("❌ ERROR: Dlib predictor not found for eye colorimetry")
            self.predictor = None
        
        # Define eye color ranges based on RGB values
        # INCLUSIVE CRITERIA: If multiple matches, return all with equal percentages
        self.eye_color_ranges = {
            'color_de_ojo_negro/cafe_oscuro': {
                'r_range': (0, 120),
                'g_range': (0, 100),
                'b_range': (0, 60),
                'conditions': None
            },

            'cafe_claro/hazel': {
                'r_range': (120, 180),
                'g_range': (80, 140),
                'b_range': (30, 100),
                'conditions': 'r > g'
            },

            'amarillo': {
                'r_range': (120, 255),
                'g_range': (120, 255),
                'b_range': (0, 160),
                'conditions': 'rg_similar_and_dominant_v2'
            },

            'verde': {
                'r_range': (0, 150),
                'g_range': (70, 255),
                'b_range': (0, 110),
                'conditions': 'g >= r'
            },

            'azul_claro/gris': {
                'r_range': (0, 220),
                'g_range': (80, 255),
                'b_range': (70, 255),
                'conditions': 'gb_dominant_balanced_v2'
            },

            'gris': {
                'r_range': (0, 220),
                'g_range': (60, 255),
                'b_range': (70, 255),
                'conditions': 'gray_conditions'
            },

            'azul_oscuro': {
                'r_range': (0, 90),
                'g_range': (0, 120),
                'b_range': (70, 135),
                'conditions': 'g >= r and b >= g'
            },

            'azul_intenso/morado': {
                'r_range': (0, 140),
                'g_range': (0, 130),
                'b_range': (135, 255),
                'conditions': 'b > g'
            },

            'azul_verde': {
                'r_range': (60, 85),
                'g_range': (70, 170),
                'b_range': (69, 169),
                'conditions': 'turquoise_conditions'
            }
        }
    
    def detect_face_landmarks(self, img):
        """
        Detect face landmarks using dlib.
        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            list: List of detected landmarks for each face
        """
        if self.predictor is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        landmarks_list = []
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks_list.append(landmarks)
            
        return landmarks_list
    
    def get_eye_region_points(self, landmarks, eye_side='left'):
        """
        Extract eye region points from facial landmarks.
        
        Args:
            landmarks: dlib landmarks object
            eye_side (str): 'left' or 'right' eye
            
        Returns:
            numpy.ndarray: Array of eye contour points
        """
        if eye_side == 'left':
            # Left eye points (42-47 in dlib)
            eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                 for i in range(42, 48)])
        else:
            # Right eye points (36-41 in dlib)
            eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                                 for i in range(36, 42)])
        
        return eye_points
    
    def create_eye_mask(self, img_shape, eye_points):
        """
        Create a binary mask for the eye region.
        
        Args:
            img_shape (tuple): Shape of the image (height, width)
            eye_points (numpy.ndarray): Eye contour points
            
        Returns:
            numpy.ndarray: Binary mask for the eye region
        """
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        # Create convex hull of eye points
        try:
            hull = ConvexHull(eye_points)
            hull_points = eye_points[hull.vertices]
            cv2.fillPoly(mask, [hull_points.astype(np.int32)], 255)
        except:
            # Fallback: use the points directly
            cv2.fillPoly(mask, [eye_points.astype(np.int32)], 255)
            
        return mask
    
    def extract_eye_bitmap(self, img, landmarks, eye_side='left', padding=10):
        """
        Extract the eye region as a bitmap image.
        
        Args:
            img (numpy.ndarray): Input image
            landmarks: dlib landmarks object
            eye_side (str): 'left' or 'right' eye
            padding (int): Padding around the eye region
            
        Returns:
            tuple: (eye_bitmap, eye_mask, bounding_box)
        """
        # Get eye region points
        eye_points = self.get_eye_region_points(landmarks, eye_side)
        
        # Calculate bounding box
        x_min, y_min = np.min(eye_points, axis=0) - padding
        x_max, y_max = np.max(eye_points, axis=0) + padding
        
        # Ensure bounds are within image
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(img.shape[1], int(x_max))
        y_max = min(img.shape[0], int(y_max))
        
        # Extract eye region
        eye_bitmap = img[y_min:y_max, x_min:x_max]
        
        # Create mask for the extracted region
        adjusted_points = eye_points - np.array([x_min, y_min])
        mask_shape = (y_max - y_min, x_max - x_min)
        eye_mask = self.create_eye_mask(mask_shape, adjusted_points)
        
        bounding_box = (x_min, y_min, x_max, y_max)
        
        return eye_bitmap, eye_mask, bounding_box
    
    def is_likely_iris_color(self, r, g, b):
        """
        Check if RGB values are likely to be iris color (not skin or sclera).
        
        Args:
            r, g, b (int): RGB values
            
        Returns:
            bool: True if likely iris color
        """
        brightness = (r + g + b) / 3.0
        
        # Exclude very dark (pupil) and very bright (sclera/reflection)
        if brightness < 30 or brightness > 200:
            return False
        
        # Exclude obvious skin tones (red dominant)
        if r > g + 30 and r > b + 30 and r > 120:
            return False
        
        # Exclude obvious sclera (very white/gray)
        if r > 180 and g > 180 and b > 180:
            return False
        
        return True
    
    def is_skin_tone(self, r, g, b):
        """
        Check if RGB values represent skin tone.
        
        Args:
            r, g, b (int): RGB values
            
        Returns:
            bool: True if likely skin tone
        """
        # Skin tone typically has red dominant and higher brightness
        brightness = (r + g + b) / 3.0
        
        # Skin tone characteristics
        if brightness > 100 and r > g and r > b and r > 120:
            return True
        
        return False
    
    def extract_iris_region_simple(self, eye_bitmap, eye_mask):
        """
        Simple iris extraction focusing on center region with improved filtering.
        
        Args:
            eye_bitmap (numpy.ndarray): Eye region image
            eye_mask (numpy.ndarray): Eye region mask
            
        Returns:
            tuple: (iris_bitmap, iris_mask)
        """
        # Create iris mask starting from center region
        h, w = eye_mask.shape
        center_y, center_x = h // 2, w // 2
        
        # Start with a center circle (likely iris area)
        iris_mask = np.zeros_like(eye_mask)
        radius = min(h, w) // 4  # Conservative radius
        cv2.circle(iris_mask, (center_x, center_y), radius, 255, -1)
        
        # Combine with eye mask
        iris_mask = cv2.bitwise_and(iris_mask, eye_mask)
        
        # Filter pixels in this region
        refined_mask = np.zeros_like(eye_mask)
        
        # Get pixels in the center region
        center_pixels = eye_bitmap[iris_mask > 0]
        mask_coords = np.where(iris_mask > 0)
        
        if len(center_pixels) == 0:
            return eye_bitmap, iris_mask
        
        print(f"    🔍 Analyzing {len(center_pixels)} center pixels for iris")
        
        # Simple filtering: exclude very dark (pupil) and very light (sclera/skin)
        good_pixels = 0
        for i, pixel in enumerate(center_pixels):
            try:
                b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])  # BGR format
                brightness = (r + g + b) / 3
                
                # Simple rules: not too dark, not too bright
                if 40 < brightness < 180:
                    # Additional check: avoid obvious skin tones
                    if not (r > g + 20 and r > b + 20 and r > 120):
                        y, x = mask_coords[0][i], mask_coords[1][i]
                        refined_mask[y, x] = 255
                        good_pixels += 1
            except (ValueError, OverflowError, IndexError):
                continue
        
        print(f"    ✅ Selected {good_pixels} pixels as iris")
        
        # If we got very few pixels, be more lenient
        if good_pixels < len(center_pixels) * 0.3:
            print("    ⚠️ Too few pixels, being more lenient...")
            refined_mask = np.zeros_like(eye_mask)
            good_pixels = 0
            
            for i, pixel in enumerate(center_pixels):
                try:
                    b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
                    brightness = (r + g + b) / 3
                    
                    # Very simple: just exclude very dark and very bright
                    if 30 < brightness < 200:
                        y, x = mask_coords[0][i], mask_coords[1][i]
                        refined_mask[y, x] = 255
                        good_pixels += 1
                except (ValueError, OverflowError, IndexError):
                    continue
            
            print(f"    ✅ Lenient selection: {good_pixels} pixels")
        
        # If still no good pixels, just use the center circle
        if good_pixels == 0:
            print("    ⚠️ No good pixels found, using center circle as-is")
            refined_mask = iris_mask
        
        # Create the iris bitmap
        iris_bitmap = cv2.bitwise_and(eye_bitmap, eye_bitmap, mask=refined_mask)
        
        return iris_bitmap, refined_mask
    
    def analyze_colors(self, image, mask, n_clusters=5):
        """
        Perform color analysis on the masked region.
        
        Args:
            image (numpy.ndarray): Input image
            mask (numpy.ndarray): Binary mask
            n_clusters (int): Number of color clusters for K-means
            
        Returns:
            dict: Color analysis results
        """
        # Extract pixels within the mask
        masked_pixels = image[mask > 0]
        
        if len(masked_pixels) == 0:
            return {"error": "No pixels found in masked region"}
        
        # Convert BGR to RGB for analysis
        masked_pixels_rgb = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB)
        masked_pixels_rgb = masked_pixels_rgb.reshape(-1, 3)
        
        # K-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=min(n_clusters, len(masked_pixels_rgb)), random_state=42, n_init=10)
        kmeans.fit(masked_pixels_rgb)
        
        # Get dominant colors and their percentages
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        color_percentages = []
        for i, color in enumerate(colors):
            percentage = (label_counts[i] / total_pixels) * 100
            color_percentages.append((color.tolist(), percentage))
        
        # Sort by percentage
        color_percentages.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate average color
        avg_color = np.mean(masked_pixels_rgb, axis=0).astype(int)
        
        # Convert to HSV for additional analysis
        avg_color_hsv = colorsys.rgb_to_hsv(avg_color[0]/255, avg_color[1]/255, avg_color[2]/255)
        
        # Determine dominant hue
        dominant_color_rgb = color_percentages[0][0]
        dominant_hue = colorsys.rgb_to_hsv(dominant_color_rgb[0]/255, dominant_color_rgb[1]/255, dominant_color_rgb[2]/255)[0]
        
        return {
            "average_color_rgb": avg_color.tolist(),
            "average_color_hsv": [float(avg_color_hsv[0] * 360), float(avg_color_hsv[1] * 100), float(avg_color_hsv[2] * 100)],
            "dominant_colors": color_percentages,
            "dominant_hue_degrees": float(dominant_hue * 360),
            "total_pixels_analyzed": int(total_pixels)
        }
    
    def check_color_conditions(self, r, g, b, conditions):
        """
        Check specific color conditions for eye color classification.

        Args:
            r, g, b (int): RGB values
            conditions (str): Condition type to check

        Returns:
            bool: Whether conditions are met
        """
        if conditions is None:
            return True

        if conditions == 'r > g':
            return r > g

        elif conditions == 'g > r':
            return g > r

        elif conditions == 'g >= r':
            return g >= r

        elif conditions == 'b > g':
            return b > g

        elif conditions == 'g > r and b > g':
            return g > r and b > g

        elif conditions == 'g >= r and b >= g':
            return g >= r and b >= g

        elif conditions == 'rg_similar_and_dominant':
            # OLD: R y G dentro del 20% de cada uno, R o G mas del doble que B
            rg_similar = abs(r - g) / max(r, g, 1) <= 0.2
            dominant_over_b = (r >= 2 * b) or (g >= 2 * b)
            return rg_similar and dominant_over_b

        elif conditions == 'rg_similar_and_dominant_v2':
            # NEW: R y G dentro del 30% de cada uno, B menos del 60% que R o G
            rg_similar = abs(r - g) / max(r, g, 1) <= 0.3
            b_less_than_rg = (b < 0.6 * r) or (b < 0.6 * g)
            return rg_similar and b_less_than_rg

        elif conditions == 'gb_dominant_balanced':
            # OLD: G y B mayor que R, G no mayor a B mas del 40%
            gb_dominant = g > r and b > r
            g_not_too_dominant = abs(g - b) / max(g, b, 1) <= 0.4
            return gb_dominant and g_not_too_dominant

        elif conditions == 'gb_dominant_balanced_v2':
            # NEW: G y B mayor o igual a R, G no mas del 40% mayor a B
            gb_dominant = g >= r and b >= r
            g_not_too_dominant = abs(g - b) / max(g, b, 1) <= 0.4
            return gb_dominant and g_not_too_dominant

        elif conditions == 'gray_conditions':
            # NEW: G no mas del 40% mayor que B
            g_not_too_dominant = abs(g - b) / max(g, b, 1) <= 0.4
            return g_not_too_dominant

        elif conditions == 'turquoise_conditions':
            # G mayor a B, R menor que 50% de G, B mayor que R
            return g > b and r < (0.5 * g) and b > r

        return False
    
    def classify_eye_color_new_system(self, color_analysis, use_dominant=False):
        """
        Classify eye color based on the new RGB range system with INCLUSIVE criteria.
        If multiple categories match, returns all matches with equal percentages.

        Args:
            color_analysis (dict): Results from analyze_colors
            use_dominant (bool): If True, use dominant color; if False, use average color

        Returns:
            dict: {
                'classifications': [list of matching color names],
                'percentages': [list of percentages for each match],
                'primary_classification': 'string of all matches separated by /'
            }
        """
        if "error" in color_analysis:
            return {
                'classifications': ['unknown'],
                'percentages': [100.0],
                'primary_classification': 'unknown'
            }

        if use_dominant:
            # Use the most dominant color from clustering
            if "dominant_colors" in color_analysis and len(color_analysis["dominant_colors"]) > 0:
                dominant_color_rgb = color_analysis["dominant_colors"][0][0]
                r, g, b = dominant_color_rgb
            else:
                return {
                    'classifications': ['unknown'],
                    'percentages': [100.0],
                    'primary_classification': 'unknown'
                }
        else:
            # Use average color
            avg_color = color_analysis["average_color_rgb"]
            r, g, b = avg_color

        # Collect ALL matching categories (INCLUSIVE approach)
        matching_colors = []

        for color_name, color_data in self.eye_color_ranges.items():
            r_range = color_data['r_range']
            g_range = color_data['g_range']
            b_range = color_data['b_range']
            conditions = color_data['conditions']

            # Check if RGB values are within ranges
            r_in_range = r_range[0] <= r <= r_range[1]
            g_in_range = g_range[0] <= g <= g_range[1]
            b_in_range = b_range[0] <= b <= b_range[1]

            if r_in_range and g_in_range and b_in_range:
                # Check additional conditions if any
                if self.check_color_conditions(r, g, b, conditions):
                    matching_colors.append(color_name)

        # If no matches, return unclassified
        if len(matching_colors) == 0:
            return {
                'classifications': ['no_clasificado'],
                'percentages': [100.0],
                'primary_classification': 'no_clasificado'
            }

        # Calculate equal percentages for all matches
        percentage_per_match = 100.0 / len(matching_colors)
        percentages = [percentage_per_match] * len(matching_colors)

        # Create combined classification string
        primary_classification = ' / '.join(matching_colors)

        return {
            'classifications': matching_colors,
            'percentages': percentages,
            'primary_classification': primary_classification
        }
    
    def classify_eye_color_hsv(self, color_analysis):
        """
        Classify eye color based on HSV color analysis (original system).
        
        Args:
            color_analysis (dict): Results from analyze_colors
            
        Returns:
            str: Classified eye color
        """
        if "error" in color_analysis:
            return "unknown"
        
        dominant_hue = color_analysis["dominant_hue_degrees"]
        avg_color = color_analysis["average_color_rgb"]
        
        # Convert to HSV for classification
        r, g, b = avg_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h_deg = h * 360
        s_percent = s * 100
        v_percent = v * 100
        
        # Eye color classification based on HSV values
        if v_percent < 25:
            return "very_dark_brown"
        elif v_percent < 40 and s_percent < 30:
            return "dark_brown"
        elif 15 <= h_deg <= 45 and s_percent > 30:
            if v_percent > 60:
                return "amber"
            else:
                return "brown"
        elif 45 <= h_deg <= 85:
            if s_percent > 40:
                return "hazel"
            else:
                return "light_brown"
        elif 85 <= h_deg <= 165:
            if s_percent > 30:
                return "green"
            else:
                return "hazel_green"
        elif 165 <= h_deg <= 260:
            if s_percent > 40 and v_percent > 50:
                return "blue"
            elif s_percent < 25:
                return "gray"
            else:
                return "blue_gray"
        else:
            return "mixed"
    
    def analyze_eye_colorimetry(self, image, confidence_threshold=0.5):
        """
        Complete eye colorimetry analysis for an image.
        
        Args:
            image (numpy.ndarray): Input image
            confidence_threshold (float): Not used in this analyzer but kept for consistency
            
        Returns:
            dict: Complete eye colorimetry analysis results
        """
        try:
            # Detect landmarks
            landmarks_list = self.detect_face_landmarks(image)
            
            if not landmarks_list:
                return {"error": "No faces detected"}
            
            results = []
            
            for face_idx, landmarks in enumerate(landmarks_list):
                face_results = {"face_index": face_idx}
                
                # Analyze both eyes
                for eye_side in ['left', 'right']:
                    try:
                        print(f"🔍 Analyzing {eye_side} eye for face {face_idx}")
                        
                        # Extract eye bitmap
                        eye_bitmap, eye_mask, bbox = self.extract_eye_bitmap(image, landmarks, eye_side)
                        
                        # Extract iris region
                        iris_bitmap, iris_mask = self.extract_iris_region_simple(eye_bitmap, eye_mask)
                        
                        # Analyze colors
                        eye_color_analysis = self.analyze_colors(eye_bitmap, eye_mask)
                        iris_color_analysis = self.analyze_colors(iris_bitmap, iris_mask)
                        
                        # Classify colors using both systems
                        eye_classification_hsv = self.classify_eye_color_hsv(eye_color_analysis)
                        iris_classification_hsv = self.classify_eye_color_hsv(iris_color_analysis)
                        
                        eye_classification_rgb_avg = self.classify_eye_color_new_system(eye_color_analysis, use_dominant=False)
                        iris_classification_rgb_avg = self.classify_eye_color_new_system(iris_color_analysis, use_dominant=False)
                        
                        eye_classification_rgb_dom = self.classify_eye_color_new_system(eye_color_analysis, use_dominant=True)
                        iris_classification_rgb_dom = self.classify_eye_color_new_system(iris_color_analysis, use_dominant=True)
                        
                        # Store results
                        face_results[f"{eye_side}_eye"] = {
                            "bounding_box": bbox,
                            "eye_color_analysis": eye_color_analysis,
                            "iris_color_analysis": iris_color_analysis,
                            "classifications": {
                                "eye_hsv": eye_classification_hsv,
                                "iris_hsv": iris_classification_hsv,
                                "eye_rgb_average": eye_classification_rgb_avg,
                                "iris_rgb_average": iris_classification_rgb_avg,
                                "eye_rgb_dominant": eye_classification_rgb_dom,
                                "iris_rgb_dominant": iris_classification_rgb_dom
                            }
                        }

                        # Get primary classification for logging
                        iris_primary = iris_classification_rgb_avg.get('primary_classification', 'unknown') if isinstance(iris_classification_rgb_avg, dict) else iris_classification_rgb_avg
                        print(f"✅ {eye_side} eye analysis complete: {iris_primary}")
                        
                    except Exception as e:
                        print(f"❌ Error analyzing {eye_side} eye: {str(e)}")
                        face_results[f"{eye_side}_eye"] = {"error": str(e)}
                
                results.append(face_results)
            
            return {
                "faces": results,
                "analysis_type": "eye_colorimetry",
                "total_faces": len(landmarks_list)
            }
            
        except Exception as e:
            print(f"❌ Eye colorimetry analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}