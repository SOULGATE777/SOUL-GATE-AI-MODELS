import cv2
import numpy as np
import math

def create_visualization(image, landmarks, model_predictions, proportions, slopes):
    """
    Create a comprehensive visualization of the anthropometric analysis
    
    Args:
        image: Original image
        landmarks: Dlib landmarks object
        model_predictions: Dictionary of model predictions
        proportions: Dictionary of calculated proportions
        slopes: Dictionary of calculated slopes
        
    Returns:
        numpy array: Annotated image
    """
    img_vis = image.copy()
    
    # Get extended points
    extended_points = _get_extended_points_from_landmarks(landmarks, image.shape, model_predictions)
    
    # Draw original landmarks (points 0-67) with same size as model points
    for i, point in enumerate(extended_points[:68]):
        cv2.circle(img_vis, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)  # Same size as model points
    
    # Draw extended landmarks (points 68-71) with same size
    for i, point in enumerate(extended_points[68:72], 68):
        cv2.circle(img_vis, (int(point[0]), int(point[1])), 4, (255, 0, 0), -1)  # Same size
        cv2.putText(img_vis, str(i), (int(point[0]) + 6, int(point[1]) - 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Highlight model predictions with same size but different colors
    if 2 in model_predictions:  # Between eyebrows (point 68)
        cv2.circle(img_vis, (int(extended_points[68][0]), int(extended_points[68][1])), 4, (0, 255, 255), -1)
        cv2.putText(img_vis, "M2", (int(extended_points[68][0]) + 6, int(extended_points[68][1]) - 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    if 3 in model_predictions:  # Top of head (point 69)
        cv2.circle(img_vis, (int(extended_points[69][0]), int(extended_points[69][1])), 4, (255, 255, 0), -1)
        cv2.putText(img_vis, "M3", (int(extended_points[69][0]) + 6, int(extended_points[69][1]) - 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    if 1 in model_predictions and len(extended_points) > 72:  # Model point 1
        cv2.circle(img_vis, (int(extended_points[72][0]), int(extended_points[72][1])), 4, (0, 255, 0), -1)
        cv2.putText(img_vis, "M1", (int(extended_points[72][0]) + 6, int(extended_points[72][1]) - 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw facial thirds lines ONLY (no distance measurement lines)
    _draw_facial_thirds(img_vis, extended_points)
    
    # Add text overlay with key measurements
    _add_text_overlay(img_vis, proportions, model_predictions)
    
    return img_vis

def _get_extended_points_from_landmarks(landmarks, img_shape, model_predictions):
    """Convert landmarks to extended points array"""
    # Convert landmarks to numpy array
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                       for i in range(68)])
    
    # Calculate extended points
    face_height = max(points[:, 1]) - min(points[:, 1])
    
    # Find highest points of eyebrows
    left_eyebrow = points[17:22]
    right_eyebrow = points[22:27]
    left_highest = left_eyebrow[np.argmin(left_eyebrow[:, 1])]
    right_highest = right_eyebrow[np.argmin(right_eyebrow[:, 1])]
    
    # Point 68: Between eyebrows
    if 2 in model_predictions:
        between_eyebrows = model_predictions[2]
    else:
        between_eyebrows = (
            (left_highest[0] + right_highest[0]) // 2,
            (left_highest[1] + right_highest[1]) // 2
        )
    
    # Point 69: Top of head
    if 3 in model_predictions:
        top_of_head = model_predictions[3]
    else:
        top_of_head = (
            between_eyebrows[0],
            between_eyebrows[1] - (face_height * 0.4)
        )
    
    # Pupil points
    left_pupil = (
        (points[37][0] + points[40][0]) // 2,
        (points[37][1] + points[40][1]) // 2
    )
    right_pupil = (
        (points[43][0] + points[46][0]) // 2,
        (points[43][1] + points[46][1]) // 2
    )
    
    # Combine all points
    extended_points = np.vstack([
        points,  # Original 68 dlib points (0-67)
        [between_eyebrows],  # Point 68
        [top_of_head],  # Point 69
        [left_pupil],  # Point 70
        [right_pupil]  # Point 71
    ])
    
    # Add model point 1 if available
    if 1 in model_predictions:
        extended_points = np.vstack([extended_points, [model_predictions[1]]])
    
    return extended_points

def _draw_facial_thirds(img, extended_points):
    """Draw facial thirds reference lines ONLY"""
    if len(extended_points) < 70:
        return
    
    # Get key points
    top_head = extended_points[69]  # Point 69
    between_eyebrows = extended_points[68]  # Point 68
    nose_base = extended_points[33]  # Point 34 (index 33)
    chin = extended_points[8]  # Point 9 (index 8)
    
    # Draw horizontal lines for facial thirds - thinner and more subtle
    line_color = (100, 200, 100)  # Light green
    line_thickness = 1
    
    # Line 1: Top of head
    cv2.line(img, (int(top_head[0] - 40), int(top_head[1])), 
             (int(top_head[0] + 40), int(top_head[1])), line_color, line_thickness)
    
    # Line 2: Between eyebrows
    cv2.line(img, (int(between_eyebrows[0] - 40), int(between_eyebrows[1])), 
             (int(between_eyebrows[0] + 40), int(between_eyebrows[1])), line_color, line_thickness)
    
    # Line 3: Nose base
    cv2.line(img, (int(nose_base[0] - 40), int(nose_base[1])), 
             (int(nose_base[0] + 40), int(nose_base[1])), line_color, line_thickness)
    
    # Line 4: Chin
    cv2.line(img, (int(chin[0] - 40), int(chin[1])), 
             (int(chin[0] + 40), int(chin[1])), line_color, line_thickness)

def _add_text_overlay(img, proportions, model_predictions):
    """Add text overlay with key measurements"""
    height, width = img.shape[:2]
    
    # Create text background
    overlay = img.copy()
    
    # Text content - more compact
    texts = [
        f"Primer tercio: {proportions.get('distance_69_68_proportion', 0):.3f}",
        f"Segundo tercio: {proportions.get('distance_68_34_proportion', 0):.3f}",
        f"Tercer tercio: {proportions.get('distance_34_9_proportion', 0):.3f}",
        f"Proporción ojos: {proportions.get('eye_distance_proportion', 0):.3f}",
        "",
        "Modelo:",
        f"P2: {'✓' if 2 in model_predictions else '✗'}",
        f"P3: {'✓' if 3 in model_predictions else '✗'}",
        f"P1: {'✓' if 1 in model_predictions else '✗'}"
    ]
    
    # Calculate text area
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Position text on the right side
    x_start = width - 250
    y_start = 30
    line_height = 25
    
    # Draw background rectangle
    cv2.rectangle(overlay, (x_start - 10, y_start - 20), 
                  (width - 10, y_start + len(texts) * line_height + 10), 
                  (0, 0, 0), -1)
    
    # Add transparency
    alpha = 0.7
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Draw text
    for i, text in enumerate(texts):
        if text:  # Skip empty lines
            y_pos = y_start + i * line_height
            cv2.putText(img_new, text, (x_start, y_pos), font, font_scale, 
                       (255, 255, 255), thickness)
    
    return img_new

def convert_slope_to_degrees(slope):
    """Convert slope to degrees"""
    if slope == float('inf'):
        return 90
    elif slope == float('-inf'):
        return 270
    else:
        angle_rad = math.atan2(1, slope)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        return angle_deg

def create_detailed_report_image(image, analysis_results):
    """Create a detailed report image with all measurements"""
    img_report = image.copy()
    height, width = img_report.shape[:2]
    
    # Create a larger canvas for the report
    report_width = width + 400
    report_canvas = np.zeros((height, report_width, 3), dtype=np.uint8)
    
    # Place original image on the left
    report_canvas[:height, :width] = img_report
    
    # Add detailed measurements on the right
    text_area = report_canvas[:, width:]
    text_area.fill(50)  # Dark gray background
    
    # Add comprehensive text report
    proportions = analysis_results.get('proportions', {})
    slopes = analysis_results.get('slopes', {})
    model_preds = analysis_results.get('model_predictions', {})
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_height = 20
    
    report_lines = [
        "=== ANÁLISIS ANTROPOMÉTRICO ===",
        "",
        "TERCIOS FACIALES:",
        f"• Primer tercio: {proportions.get('distance_69_68_proportion', 0):.4f}",
        f"• Segundo tercio: {proportions.get('distance_68_34_proportion', 0):.4f}",
        f"• Tercer tercio: {proportions.get('distance_34_9_proportion', 0):.4f}",
        "",
        "PROPORCIONES OCULARES:",
        f"• Distancia interna: {proportions.get('eye_distance_proportion', 0):.4f}",
        f"• Distancia externa: {proportions.get('outter_eye_distance_proportion', 0):.4f}",
        "",
        "MEDIDAS FACIALES:",
        f"• Ancho facial: {proportions.get('head_width_proportion', 0):.4f}",
        f"• Longitud boca: {proportions.get('mouth_length_proportion', 0):.4f}",
        f"• Relación boca-pupila: {proportions.get('mouth_to_eye_proportion', 0):.4f}",
        "",
        "MODELO INTEGRADO:",
        f"• Punto 1 detectado: {'✓' if 1 in model_preds else '✗'}",
        f"• Punto 2 detectado: {'✓' if 2 in model_preds else '✗'}",
        f"• Punto 3 detectado: {'✓' if 3 in model_preds else '✗'}",
    ]
    
    # Draw the report text
    for i, line in enumerate(report_lines):
        y_pos = 30 + i * line_height
        if y_pos < height - 20:  # Make sure text fits
            color = (255, 255, 255) if not line.startswith("===") else (0, 255, 255)
            cv2.putText(report_canvas, line, (width + 20, y_pos), 
                       font, font_scale, color, thickness)
    
    return report_canvas
