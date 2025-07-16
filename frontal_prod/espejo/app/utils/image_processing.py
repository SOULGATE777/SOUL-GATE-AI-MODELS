import cv2
import numpy as np
from fastapi import UploadFile, HTTPException
import io
from PIL import Image
import base64

async def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """
    Process uploaded image file and convert to numpy array
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        numpy.ndarray: Image as numpy array in BGR format
        
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Validate image dimensions
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            raise HTTPException(status_code=400, detail="Image too small (minimum 100x100)")
        
        if height > 4000 or width > 4000:
            # Resize large images
            scale = min(4000/height, 4000/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

def validate_image_quality(image: np.ndarray) -> dict:
    """
    Validate image quality for facial analysis
    
    Args:
        image: Input image as numpy array
        
    Returns:
        dict: Quality assessment results
    """
    height, width = image.shape[:2]
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate image sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = gray.std()
    
    # Assess quality
    quality_score = 0
    warnings = []
    
    # Sharpness check
    if laplacian_var > 100:
        quality_score += 25
    elif laplacian_var < 50:
        warnings.append("Image may be blurry")
    else:
        quality_score += 15
    
    # Brightness check
    if 50 <= brightness <= 200:
        quality_score += 25
    elif brightness < 30:
        warnings.append("Image is too dark")
    elif brightness > 220:
        warnings.append("Image is too bright")
    else:
        quality_score += 15
    
    # Contrast check
    if contrast > 40:
        quality_score += 25
    elif contrast < 20:
        warnings.append("Image has low contrast")
    else:
        quality_score += 15
    
    # Size check
    if width >= 400 and height >= 400:
        quality_score += 25
    elif width >= 200 and height >= 200:
        quality_score += 15
        warnings.append("Image resolution is low")
    else:
        warnings.append("Image resolution is very low")
    
    return {
        "quality_score": quality_score,
        "sharpness": laplacian_var,
        "brightness": brightness,
        "contrast": contrast,
        "resolution": f"{width}x{height}",
        "warnings": warnings,
        "suitable_for_analysis": quality_score >= 60
    }

def preprocess_for_analysis(image: np.ndarray, target_size: tuple = None) -> tuple:
    """
    Preprocess image for facial analysis
    
    Args:
        image: Input image
        target_size: Optional target size (width, height)
        
    Returns:
        tuple: (processed_image, grayscale_image, scale_factor)
    """
    original_height, original_width = image.shape[:2]
    
    # Calculate scale factor if target size is specified
    scale_factor = 1.0
    processed_image = image.copy()
    
    if target_size:
        target_width, target_height = target_size
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale_factor = min(scale_x, scale_y)
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        processed_image = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    gray_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    
    return processed_image, gray_enhanced, scale_factor

def enhance_image_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better face detection
    
    Args:
        image: Input image
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def resize_image_for_model(image: np.ndarray, target_size: int = 224) -> tuple:
    """
    Resize image for model input while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target size for the largest dimension
        
    Returns:
        tuple: (resized_image, scale_factor)
    """
    height, width = image.shape[:2]
    
    # Calculate scale factor
    scale = target_size / max(height, width)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Pad to square if needed
    if new_width != new_height:
        # Create square canvas
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_size - new_width) // 2
        pad_y = (target_size - new_height) // 2
        
        # Place resized image on canvas
        canvas[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        
        return canvas, scale
    
    return resized, scale

def save_image_result(image: np.ndarray, filepath: str, quality: int = 95) -> bool:
    """
    Save image result with error handling
    
    Args:
        image: Image to save
        filepath: Output file path
        quality: JPEG quality (0-100)
        
    Returns:
        bool: Success status
    """
    try:
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save image
        if filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg'):
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(filepath, image)
        
        return True
        
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def create_thumbnail(image: np.ndarray, size: tuple = (150, 150)) -> np.ndarray:
    """
    Create thumbnail of the image
    
    Args:
        image: Input image
        size: Thumbnail size (width, height)
        
    Returns:
        numpy.ndarray: Thumbnail image
    """
    return cv2.resize(image, size)

def convert_to_base64(image: np.ndarray) -> str:
    """
    Convert image to base64 string
    
    Args:
        image: Input image
        
    Returns:
        str: Base64 encoded image
    """
    # Encode image
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert to base64
    base64_string = base64.b64encode(buffer).decode('utf-8')
    
    return base64_string

def prepare_mirror_image_for_classification(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Prepare mirror image for classification models
    
    Args:
        image: Mirror image array
        target_size: Target size for classification
        
    Returns:
        numpy.ndarray: Processed image ready for classification
    """
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Resize while maintaining aspect ratio
    height, width = image_rgb.shape[:2]
    aspect_ratio = width / height
    
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    # Resize image
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Create square canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2
    
    # Place resized image on canvas
    canvas[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
    
    return canvas

def draw_analysis_overlay(image: np.ndarray, landmarks: np.ndarray, custom_points: dict, 
                         proportions: dict, intersection_points: dict = None) -> np.ndarray:
    """
    Draw analysis overlay on image with landmarks, custom points, and proportion lines
    
    Args:
        image: Input image
        landmarks: Facial landmarks array
        custom_points: Custom model points dict
        proportions: Proportions dict
        intersection_points: Intersection points dict
        
    Returns:
        numpy.ndarray: Image with overlay
    """
    overlay = image.copy()
    
    # Draw facial landmarks
    for point in landmarks:
        cv2.circle(overlay, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
    
    # Draw custom model points
    for point_id, point in custom_points.items():
        cv2.circle(overlay, point, 4, (255, 0, 0), -1)
        cv2.putText(overlay, f"M{point_id}", (point[0] + 5, point[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # Draw proportion lines
    if intersection_points:
        # Face proportion lines
        if 'face_intersection' in intersection_points and intersection_points['face_intersection'] is not None:
            if 2 in custom_points:
                point_m2 = custom_points[2]
                point_dlib_8 = tuple(landmarks[8])
                point_dlib_1 = tuple(landmarks[1])
                point_dlib_15 = tuple(landmarks[15])
                
                cv2.line(overlay, point_m2, point_dlib_8, (0, 255, 0), 2)
                cv2.line(overlay, point_dlib_1, point_dlib_15, (255, 0, 0), 2)
                cv2.circle(overlay, intersection_points['face_intersection'], 4, (0, 0, 255), -1)
        
        # Temporal proportion lines
        if 'temporal_intersection' in intersection_points and intersection_points['temporal_intersection'] is not None:
            if 2 in custom_points:
                point_dlib_0 = tuple(landmarks[0])
                point_dlib_16 = tuple(landmarks[16])
                
                cv2.line(overlay, point_dlib_0, point_dlib_16, (0, 255, 255), 2)
                cv2.circle(overlay, intersection_points['temporal_intersection'], 4, (255, 165, 0), -1)
    
    # Draw forehead lines
    if 3 in custom_points and 2 in custom_points:
        point_m3 = custom_points[3]
        point_m2 = custom_points[2]
        
        cv2.line(overlay, point_m3, point_m2, (128, 0, 128), 2)
        
        if 13 in custom_points:
            point_m13 = custom_points[13]
            cv2.line(overlay, point_m13, point_m2, (0, 165, 255), 1)
        
        if 8 in custom_points:
            point_m8 = custom_points[8]
            cv2.line(overlay, point_m2, point_m8, (0, 255, 255), 1)
    
    return overlay

def draw_region_bboxes(image: np.ndarray, region_type: str = "mirror") -> np.ndarray:
    """
    Draw region bounding boxes on image
    
    Args:
        image: Input image
        region_type: Type of regions to draw
        
    Returns:
        numpy.ndarray: Image with bounding boxes
    """
    overlay = image.copy()
    height, width = image.shape[:2]
    
    if region_type == "mirror":
        # FRENTE region (forehead area) - upper 40% of image
        frente_bbox = {
            'x1': int(width * 0.1),
            'y1': int(height * 0.05),
            'x2': int(width * 0.9),
            'y2': int(height * 0.45),
            'label': 'FRENTE',
            'color': (0, 255, 0)  # Green
        }
        
        # rostro_menton region (chin/jaw area) - lower 40% of image
        rostro_bbox = {
            'x1': int(width * 0.15),
            'y1': int(height * 0.55),
            'x2': int(width * 0.85),
            'y2': int(height * 0.95),
            'label': 'rostro_menton',
            'color': (0, 0, 255)  # Red
        }
        
        bboxes = [frente_bbox, rostro_bbox]
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            color = bbox['color']
            label = bbox['label']
            
            # Draw rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(overlay, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return overlay

def create_side_by_side_comparison(left_image: np.ndarray, right_image: np.ndarray, 
                                   labels: tuple = ("Left", "Right")) -> np.ndarray:
    """
    Create side-by-side comparison of two images
    
    Args:
        left_image: Left image
        right_image: Right image  
        labels: Tuple of labels for images
        
    Returns:
        numpy.ndarray: Side-by-side comparison image
    """
    # Ensure images have same height
    h1, w1 = left_image.shape[:2]
    h2, w2 = right_image.shape[:2]
    
    target_height = min(h1, h2)
    
    # Resize images to same height
    left_resized = cv2.resize(left_image, (int(w1 * target_height / h1), target_height))
    right_resized = cv2.resize(right_image, (int(w2 * target_height / h2), target_height))
    
    # Create combined image
    combined = np.hstack((left_resized, right_resized))
    
    # Add labels
    cv2.putText(combined, labels[0], (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, labels[1], (left_resized.shape[1] + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined

def add_text_overlay(image: np.ndarray, text: str, position: tuple = (10, 30), 
                     font_scale: float = 0.7, color: tuple = (255, 255, 255), 
                     thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to image
    
    Args:
        image: Input image
        text: Text to add
        position: Text position (x, y)
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness
        
    Returns:
        numpy.ndarray: Image with text overlay
    """
    overlay = image.copy()
    
    # Split text into lines if it contains newlines
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        y_offset = position[1] + (i * 25)  # 25 pixels between lines
        cv2.putText(overlay, line, (position[0], y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return overlay

def create_analysis_grid(images: list, labels: list, grid_size: tuple = (2, 2)) -> np.ndarray:
    """
    Create grid layout of analysis images
    
    Args:
        images: List of images
        labels: List of labels
        grid_size: Grid size (rows, cols)
        
    Returns:
        numpy.ndarray: Grid image
    """
    rows, cols = grid_size
    
    if len(images) > rows * cols:
        images = images[:rows * cols]
        labels = labels[:rows * cols]
    
    # Find target size for all images
    target_height = 400
    target_width = 400
    
    # Resize all images to same size
    resized_images = []
    for img in images:
        resized = cv2.resize(img, (target_width, target_height))
        resized_images.append(resized)
    
    # Create grid
    grid_rows = []
    for row in range(rows):
        row_images = []
        for col in range(cols):
            idx = row * cols + col
            if idx < len(resized_images):
                img = resized_images[idx].copy()
                # Add label
                if idx < len(labels):
                    cv2.putText(img, labels[idx], (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                row_images.append(img)
            else:
                # Create empty placeholder
                placeholder = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                row_images.append(placeholder)
        
        # Combine images in row
        if row_images:
            grid_rows.append(np.hstack(row_images))
    
    # Combine rows
    if grid_rows:
        grid = np.vstack(grid_rows)
        return grid
    
    return np.zeros((target_height, target_width, 3), dtype=np.uint8)