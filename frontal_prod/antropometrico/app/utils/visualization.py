import cv2
import numpy as np
import math
from scipy.spatial import ConvexHull

def create_visualization(image, landmarks, model_predictions, proportions, slopes, calculated_c1=None):
    """
    Create a comprehensive visualization of the anthropometric analysis
    
    Args:
        image: Original image
        landmarks: Dlib landmarks object
        model_predictions: Dictionary of model predictions
        proportions: Dictionary of calculated proportions
        slopes: Dictionary of calculated slopes
        calculated_c1: Calculated point C1 (X from M2, Y from M9) or None
        
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
    
    # Highlight ALL model predictions with different colors
    model_colors = {
        1: (0, 255, 0),      # Green
        2: (0, 255, 255),    # Yellow
        3: (255, 255, 0),    # Cyan
        4: (255, 0, 255),    # Magenta
        5: (128, 0, 128),    # Purple
        6: (255, 165, 0),    # Orange
        7: (0, 128, 255),    # Light Blue
        8: (255, 192, 203),  # Pink
        9: (128, 128, 0),    # Olive
        10: (0, 128, 128),   # Teal
        11: (128, 128, 128), # Gray
        12: (255, 69, 0),    # Red Orange
        13: (50, 205, 50)    # Lime Green
    }
    
    for label, point in model_predictions.items():
        color = model_colors.get(label, (255, 255, 255))  # Default white if label not found
        cv2.circle(img_vis, (int(point[0]), int(point[1])), 5, color, -1)
        cv2.putText(img_vis, f"M{label}", (int(point[0]) + 7, int(point[1]) - 7), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Special highlighting for key points used in calculations
    if 2 in model_predictions:  # Between eyebrows (point 68)
        cv2.circle(img_vis, (int(extended_points[68][0]), int(extended_points[68][1])), 6, (0, 255, 255), 2)
        cv2.putText(img_vis, "68", (int(extended_points[68][0]) + 8, int(extended_points[68][1]) - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    if 3 in model_predictions:  # Top of head (point 69)
        cv2.circle(img_vis, (int(extended_points[69][0]), int(extended_points[69][1])), 6, (255, 255, 0), 2)
        cv2.putText(img_vis, "69", (int(extended_points[69][0]) + 8, int(extended_points[69][1]) - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    if 1 in model_predictions and len(extended_points) > 72:  # Model point 1
        cv2.circle(img_vis, (int(extended_points[72][0]), int(extended_points[72][1])), 6, (0, 255, 0), 2)
        cv2.putText(img_vis, "72", (int(extended_points[72][0]) + 8, int(extended_points[72][1]) - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Highlight calculated point C1 if it exists
    if calculated_c1 is not None:
        cv2.circle(img_vis, (int(calculated_c1[0]), int(calculated_c1[1])), 7, (255, 215, 0), 3)  # Gold color, larger
        cv2.putText(img_vis, "C1", (int(calculated_c1[0]) + 10, int(calculated_c1[1]) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        # Add lines to show the calculation components
        if 2 in model_predictions and 9 in model_predictions:
            # Draw line from M2 to C1 (horizontal component)
            cv2.line(img_vis, (int(model_predictions[2][0]), int(model_predictions[2][1])), 
                    (int(calculated_c1[0]), int(calculated_c1[1])), (255, 215, 0), 1)
            # Draw line from M9 to C1 (vertical component)  
            cv2.line(img_vis, (int(model_predictions[9][0]), int(model_predictions[9][1])), 
                    (int(calculated_c1[0]), int(calculated_c1[1])), (255, 215, 0), 1)
    
    # Draw facial thirds lines
    _draw_facial_thirds(img_vis, extended_points)
    
    # Draw eye angle indicators
    _draw_eye_angles(img_vis, extended_points)

    # Draw eyebrow-to-eyelid distance measurements
    _draw_eyebrow_eyelid_distances(img_vis, extended_points)

    # Draw mouth measurements (cupid's bow arches and lips)
    _draw_mouth_measurements(img_vis, extended_points)

    # Add enhanced text overlay with new features
    _add_enhanced_text_overlay(img_vis, proportions, model_predictions)
    
    return img_vis

def create_detailed_report_image(image, analysis_results):
    """Create a detailed report image with all measurements and new features"""
    img_report = image.copy()
    height, width = img_report.shape[:2]
    
    # Create a larger canvas for the report
    report_width = width + 500  # Increased width for more content
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
    eyebrow_props = analysis_results.get('eyebrow_proportions', {})
    eye_angles = analysis_results.get('eye_angles', {})
    eyebrow_eyelid_dists = analysis_results.get('eyebrow_eyelid_distances', {})
    mouth_measurements = analysis_results.get('mouth_measurements', {})
    eye_face_props = analysis_results.get('eye_face_proportions', {})
    inner_outer_props = analysis_results.get('inner_outer_proportions', {})
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    line_height = 18
    
    report_lines = [
        "=== ANÁLISIS ANTROPOMÉTRICO COMPLETO ===",
        "",
        "TERCIOS FACIALES:",
        f"• Primer tercio: {proportions.get('distance_69_68_proportion', 0):.4f}",
        f"• Segundo tercio: {proportions.get('distance_68_34_proportion', 0):.4f}",
        f"• Tercer tercio: {proportions.get('distance_34_9_proportion', 0):.4f}",
        "",
        "ANÁLISIS OCULAR:",
        f"• Distancia interna: {proportions.get('eye_distance_proportion', 0):.4f}",
        f"• Distancia externa: {proportions.get('outter_eye_distance_proportion', 0):.4f}",
        f"• Ángulo ojo izq: {eye_angles.get('left_eye_angle', 0):.2f}°",
        f"• Ángulo ojo der: {eye_angles.get('right_eye_angle', 0):.2f}°",
        f"• Dist ceja-párpado izq: {eyebrow_eyelid_dists.get('left_eyebrow_eyelid_proportion', 0):.4f}",
        f"• Dist ceja-párpado der: {eyebrow_eyelid_dists.get('right_eyebrow_eyelid_proportion', 0):.4f}",
        f"• Área ojo izq/cara: {eye_face_props.get('left_eye_proportion', 0):.3f}%",
        f"• Área ojo der/cara: {eye_face_props.get('right_eye_proportion', 0):.3f}%",
        "",
        "ANÁLISIS DE CEJAS:",
        f"• Proporción ceja izq: {eyebrow_props.get('left_eyebrow_proportion', 0):.3f}",
        f"• Proporción ceja der: {eyebrow_props.get('right_eyebrow_proportion', 0):.3f}",
        f"• Longitud ceja izq: {eyebrow_props.get('left_eyebrow_length', 0):.1f}px",
        f"• Longitud ceja der: {eyebrow_props.get('right_eyebrow_length', 0):.1f}px",
        "",
        "ANÁLISIS BUCAL:",
        f"• Arco cupido izq: {mouth_measurements.get('left_cupid_arch_proportion', 0):.4f}",
        f"• Arco cupido der: {mouth_measurements.get('right_cupid_arch_proportion', 0):.4f}",
        f"• Proporción labios: {mouth_measurements.get('lips_ratio', 0):.4f}",
        f"• Labio superior: {mouth_measurements.get('upper_lip_distance', 0):.1f}px",
        f"• Labio inferior: {mouth_measurements.get('lower_lip_distance', 0):.1f}px",
        "",
        "MEDIDAS FACIALES:",
        f"• Ancho facial: {proportions.get('head_width_proportion', 0):.4f}",
        f"• Longitud boca: {proportions.get('mouth_length_proportion', 0):.4f}",
        f"• Relación boca-pupila: {proportions.get('mouth_to_eye_proportion', 0):.4f}",
        f"• Relación mentón-cara: {proportions.get('chin_to_face_width_proportion', 0):.4f}",
        "",
        "ANÁLISIS DE ÁREAS:",
        f"• Área total cara: {inner_outer_props.get('outer_area', 0):.0f}px²",
        f"• Área interna: {inner_outer_props.get('inner_area', 0):.0f}px²",
        f"• Proporción int/ext: {inner_outer_props.get('inner_outer_percentage', 0):.2f}%",
        "",
        "MODELO INTEGRADO:",
        f"• Puntos detectados: {len(model_preds)}",
        f"• Punto 1: {'✓' if 1 in model_preds else '✗'}",
        f"• Punto 2 (cejas): {'✓' if 2 in model_preds else '✗'}",
        f"• Punto 3 (cabeza): {'✓' if 3 in model_preds else '✗'}",
    ]
    
    # Add more model points if detected
    if len(model_preds) > 3:
        report_lines.append("")
        report_lines.append("PUNTOS ADICIONALES:")
        for label in sorted(model_preds.keys()):
            if label not in [1, 2, 3]:
                point = model_preds[label]
                report_lines.append(f"• Punto {label}: ({point[0]}, {point[1]})")
    
    # Draw the report text
    for i, line in enumerate(report_lines):
        y_pos = 30 + i * line_height
        if y_pos < height - 20:  # Make sure text fits
            if line.startswith("==="):
                color = (0, 255, 255)  # Cyan for headers
            elif line.startswith("•"):
                color = (255, 255, 255)  # White for items
            elif line.endswith(":"):
                color = (100, 255, 100)  # Light green for sections
            else:
                color = (200, 200, 200)  # Light gray for other text
            
            cv2.putText(report_canvas, line, (width + 15, y_pos), 
                       font, font_scale, color, thickness)
    
    return report_canvas

def create_area_visualization(image, landmarks, model_predictions, inner_outer_props):
    """Create visualization showing face area analysis"""
    img_vis = image.copy()
    
    # Draw outer face boundary
    outer_points = inner_outer_props['outer_points']
    try:
        outer_hull = ConvexHull(outer_points)
        outer_polygon = outer_points[outer_hull.vertices]
        
        # Draw outer boundary
        for i in range(len(outer_polygon)):
            start_point = tuple(map(int, outer_polygon[i]))
            end_point = tuple(map(int, outer_polygon[(i + 1) % len(outer_polygon)]))
            cv2.line(img_vis, start_point, end_point, (0, 255, 0), 2)
    except:
        pass
    
    # Draw inner face boundary
    inner_points = inner_outer_props['inner_points']
    try:
        inner_hull = ConvexHull(inner_points)
        inner_polygon = inner_points[inner_hull.vertices]
        
        # Draw inner boundary
        for i in range(len(inner_polygon)):
            start_point = tuple(map(int, inner_polygon[i]))
            end_point = tuple(map(int, inner_polygon[(i + 1) % len(inner_polygon)]))
            cv2.line(img_vis, start_point, end_point, (0, 0, 255), 2)
    except:
        pass
    
    # Add area information
    text = f"Outer: {inner_outer_props['outer_area']:.0f}px² | Inner: {inner_outer_props['inner_area']:.0f}px²"
    text2 = f"Ratio: {inner_outer_props['inner_outer_percentage']:.2f}%"
    
    cv2.putText(img_vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img_vis, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_vis

def create_eyebrow_analysis_visualization(image, landmarks, extended_points, eyebrow_props):
    """Create visualization focusing on eyebrow analysis"""
    img_vis = image.copy()
    
    # Draw eyebrow points with enhanced visibility
    right_eyebrow_points = extended_points[17:22]
    left_eyebrow_points = extended_points[22:27]
    
    # Draw right eyebrow
    for i, point in enumerate(right_eyebrow_points):
        cv2.circle(img_vis, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
        cv2.putText(img_vis, f"R{i+17}", (int(point[0]) + 5, int(point[1]) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # Draw left eyebrow
    for i, point in enumerate(left_eyebrow_points):
        cv2.circle(img_vis, (int(point[0]), int(point[1])), 4, (255, 0, 0), -1)
        cv2.putText(img_vis, f"L{i+22}", (int(point[0]) + 5, int(point[1]) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    # Draw eyebrow lines
    cv2.line(img_vis, tuple(map(int, right_eyebrow_points[0])), tuple(map(int, right_eyebrow_points[-1])), (0, 255, 0), 2)
    cv2.line(img_vis, tuple(map(int, left_eyebrow_points[0])), tuple(map(int, left_eyebrow_points[-1])), (255, 0, 0), 2)
    
    # Draw eye corners for reference
    right_eye_inner = extended_points[39]
    right_eye_outer = extended_points[36]
    left_eye_inner = extended_points[42]
    left_eye_outer = extended_points[45]
    
    cv2.circle(img_vis, (int(right_eye_inner[0]), int(right_eye_inner[1])), 3, (0, 255, 255), -1)
    cv2.circle(img_vis, (int(right_eye_outer[0]), int(right_eye_outer[1])), 3, (0, 255, 255), -1)
    cv2.circle(img_vis, (int(left_eye_inner[0]), int(left_eye_inner[1])), 3, (255, 255, 0), -1)
    cv2.circle(img_vis, (int(left_eye_outer[0]), int(left_eye_outer[1])), 3, (255, 255, 0), -1)
    
    # Draw eye lines for comparison
    cv2.line(img_vis, tuple(map(int, right_eye_outer)), tuple(map(int, right_eye_inner)), (0, 255, 255), 2)
    cv2.line(img_vis, tuple(map(int, left_eye_outer)), tuple(map(int, left_eye_inner)), (255, 255, 0), 2)
    
    # Add measurements
    texts = [
        f"Right Eyebrow: {eyebrow_props['right_eyebrow_proportion']:.3f}",
        f"Left Eyebrow: {eyebrow_props['left_eyebrow_proportion']:.3f}",
        f"Right Eye Length: {eyebrow_props['right_eye_length']:.1f}px",
        f"Left Eye Length: {eyebrow_props['left_eye_length']:.1f}px"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(img_vis, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_vis

def create_eye_angle_visualization(image, extended_points, eye_angles):
    """Create visualization showing eye angles"""
    img_vis = image.copy()
    
    # Get eye points
    right_inner = extended_points[39]
    right_outer = extended_points[36]
    left_inner = extended_points[42]
    left_outer = extended_points[45]
    
    # Draw eye lines
    cv2.line(img_vis, tuple(map(int, right_outer)), tuple(map(int, right_inner)), (0, 255, 0), 3)
    cv2.line(img_vis, tuple(map(int, left_outer)), tuple(map(int, left_inner)), (255, 0, 0), 3)
    
    # Draw angle indicators
    right_center = ((right_inner[0] + right_outer[0]) // 2, (right_inner[1] + right_outer[1]) // 2)
    left_center = ((left_inner[0] + left_outer[0]) // 2, (left_inner[1] + left_outer[1]) // 2)
    
    # Draw horizontal reference lines
    line_length = 30
    cv2.line(img_vis, (int(right_center[0] - line_length), int(right_center[1])), 
             (int(right_center[0] + line_length), int(right_center[1])), (100, 100, 100), 1)
    cv2.line(img_vis, (int(left_center[0] - line_length), int(left_center[1])), 
             (int(left_center[0] + line_length), int(left_center[1])), (100, 100, 100), 1)
    
    # Add angle measurements
    cv2.putText(img_vis, f"Right: {eye_angles['right_eye_angle']:.1f}°", 
                (int(right_center[0]) + 10, int(right_center[1]) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img_vis, f"Left: {eye_angles['left_eye_angle']:.1f}°", 
                (int(left_center[0]) + 10, int(left_center[1]) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img_vis

def _get_extended_points_from_landmarks(landmarks, img_shape, model_predictions):
    """Convert landmarks to extended points array"""
    # Convert landmarks to numpy array
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                       for i in range(68)])
    
    # Calculate extended points
    face_height = max(points[:, 1]) - min(points[:, 1])
    
    # Find highest points of eyebrows
    right_eyebrow = points[17:22]  # Dlib points 18-22 = RIGHT eyebrow
    left_eyebrow = points[22:27]   # Dlib points 23-27 = LEFT eyebrow
    right_highest = right_eyebrow[np.argmin(right_eyebrow[:, 1])]
    left_highest = left_eyebrow[np.argmin(left_eyebrow[:, 1])]
    
    # Point 68: Between eyebrows
    if 2 in model_predictions:
        between_eyebrows = model_predictions[2]
    else:
        between_eyebrows = (
            (left_highest[0] + right_highest[0]) // 2,
            (left_highest[1] + right_highest[1]) // 2
        )
    
    # Point 69: Top of head - use calculated point C1 (M9 Y + M2 X) to match main pipeline
    if 9 in model_predictions and 2 in model_predictions:
        # Create calculated point C1: X from model point 2, Y from model point 9
        top_of_head = (
            int(model_predictions[2][0]),  # X from point 2 (between eyebrows)
            int(model_predictions[9][1])   # Y from model point 9 (M9)
        )
    elif 3 in model_predictions:
        # Fallback to model point 3
        top_of_head = model_predictions[3]
    else:
        # Final fallback to calculated estimate
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
    """Draw facial thirds reference lines"""
    if len(extended_points) < 70:
        return
    
    # Get key points
    top_head = extended_points[69]  # Point 69
    between_eyebrows = extended_points[68]  # Point 68
    nose_base = extended_points[33]  # Point 34 (index 33)
    chin = extended_points[8]  # Point 9 (index 8)
    
    # Draw horizontal lines for facial thirds
    line_color = (100, 200, 100)  # Light green
    line_thickness = 2
    
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

def _draw_eye_angles(img, extended_points):
    """Draw eye angle indicators"""
    # Get eye points
    right_inner = extended_points[39]
    right_outer = extended_points[36]
    left_inner = extended_points[42]
    left_outer = extended_points[45]

    # Draw subtle angle lines
    cv2.line(img, tuple(map(int, right_outer)), tuple(map(int, right_inner)), (200, 200, 0), 1)
    cv2.line(img, tuple(map(int, left_outer)), tuple(map(int, left_inner)), (200, 200, 0), 1)

def _draw_eyebrow_eyelid_distances(img, extended_points):
    """Draw eyebrow-to-eyelid distance measurement lines"""
    # RIGHT eye: point 20 (eyebrow) to point 38 (eyelid) - dlib points 20 and 38
    right_eyebrow_point = extended_points[19]  # dlib point 20, Python index 19
    right_eyelid_point = extended_points[37]   # dlib point 38, Python index 37

    # LEFT eye: point 25 (eyebrow) to point 45 (eyelid) - dlib points 25 and 45
    left_eyebrow_point = extended_points[24]  # dlib point 25, Python index 24
    left_eyelid_point = extended_points[44]   # dlib point 45, Python index 44

    # Draw measurement lines with distinct colors
    line_color_left = (255, 128, 0)   # Orange for left eye
    line_color_right = (0, 128, 255)  # Blue for right eye
    line_thickness = 2

    # Draw left eyebrow-to-eyelid line
    cv2.line(img, tuple(map(int, left_eyebrow_point)), tuple(map(int, left_eyelid_point)),
             line_color_left, line_thickness)

    # Draw right eyebrow-to-eyelid line
    cv2.line(img, tuple(map(int, right_eyebrow_point)), tuple(map(int, right_eyelid_point)),
             line_color_right, line_thickness)

    # Highlight the measurement points with circles
    cv2.circle(img, tuple(map(int, left_eyebrow_point)), 3, line_color_left, -1)
    cv2.circle(img, tuple(map(int, left_eyelid_point)), 3, line_color_left, -1)
    cv2.circle(img, tuple(map(int, right_eyebrow_point)), 3, line_color_right, -1)
    cv2.circle(img, tuple(map(int, right_eyelid_point)), 3, line_color_right, -1)

    # Add point labels (dlib point numbers)
    cv2.putText(img, "25", tuple(map(int, left_eyebrow_point + np.array([5, -5]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color_left, 1)
    cv2.putText(img, "45", tuple(map(int, left_eyelid_point + np.array([5, 5]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color_left, 1)
    cv2.putText(img, "20", tuple(map(int, right_eyebrow_point + np.array([-15, -5]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color_right, 1)
    cv2.putText(img, "38", tuple(map(int, right_eyelid_point + np.array([-15, 5]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color_right, 1)

def _draw_mouth_measurements(img, extended_points):
    """Draw mouth measurement lines for cupid's bow arches and lips ratio"""
    # Cupid's bow arch measurements
    # RIGHT cupid's arch base: dlib points 51 to 62 (Python index 50 to 61) - subject's right side
    right_cupid_base_51 = extended_points[50]  # dlib point 51
    right_cupid_base_62 = extended_points[61]  # dlib point 62

    # LEFT cupid's arch base: dlib points 53 to 64 (Python index 52 to 63) - subject's left side
    left_cupid_base_53 = extended_points[52]  # dlib point 53
    left_cupid_base_64 = extended_points[63]  # dlib point 64

    # Lips ratio measurements (vertical thickness at center)
    # Upper lip: dlib point 52 to 63 (Python index 51 to 62) - center of lips
    upper_lip_point_52 = extended_points[51]  # dlib point 52
    upper_lip_point_63 = extended_points[62]  # dlib point 63

    # Lower lip: dlib point 67 to 58 (Python index 66 to 57) - center of lips
    lower_lip_point_67 = extended_points[66]  # dlib point 67
    lower_lip_point_58 = extended_points[57]  # dlib point 58

    # Define colors for different measurements
    cupid_left_color = (255, 0, 128)    # Pink for LEFT cupid's arch (subject's left)
    cupid_right_color = (128, 0, 255)   # Purple for RIGHT cupid's arch (subject's right)
    upper_lip_color = (0, 255, 128)     # Green for upper lip
    lower_lip_color = (255, 128, 0)     # Orange for lower lip
    line_thickness = 2

    # Draw cupid's bow arch lines
    cv2.line(img, tuple(map(int, right_cupid_base_51)), tuple(map(int, right_cupid_base_62)),
             cupid_right_color, line_thickness)
    cv2.line(img, tuple(map(int, left_cupid_base_53)), tuple(map(int, left_cupid_base_64)),
             cupid_left_color, line_thickness)

    # Draw lips ratio lines
    cv2.line(img, tuple(map(int, upper_lip_point_52)), tuple(map(int, upper_lip_point_63)),
             upper_lip_color, line_thickness)
    cv2.line(img, tuple(map(int, lower_lip_point_67)), tuple(map(int, lower_lip_point_58)),
             lower_lip_color, line_thickness)

    # Highlight measurement points with circles
    # Cupid's bow points
    cv2.circle(img, tuple(map(int, right_cupid_base_51)), 2, cupid_right_color, -1)
    cv2.circle(img, tuple(map(int, right_cupid_base_62)), 2, cupid_right_color, -1)
    cv2.circle(img, tuple(map(int, left_cupid_base_53)), 2, cupid_left_color, -1)
    cv2.circle(img, tuple(map(int, left_cupid_base_64)), 2, cupid_left_color, -1)

    # Lips ratio points
    cv2.circle(img, tuple(map(int, upper_lip_point_52)), 2, upper_lip_color, -1)
    cv2.circle(img, tuple(map(int, upper_lip_point_63)), 2, upper_lip_color, -1)
    cv2.circle(img, tuple(map(int, lower_lip_point_67)), 2, lower_lip_color, -1)
    cv2.circle(img, tuple(map(int, lower_lip_point_58)), 2, lower_lip_color, -1)

    # Add point labels
    cv2.putText(img, "51", tuple(map(int, right_cupid_base_51 + np.array([-5, -8]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, cupid_right_color, 1)
    cv2.putText(img, "62", tuple(map(int, right_cupid_base_62 + np.array([3, -5]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, cupid_right_color, 1)
    cv2.putText(img, "53", tuple(map(int, left_cupid_base_53 + np.array([3, -8]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, cupid_left_color, 1)
    cv2.putText(img, "64", tuple(map(int, left_cupid_base_64 + np.array([-15, -5]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, cupid_left_color, 1)

    cv2.putText(img, "52", tuple(map(int, upper_lip_point_52 + np.array([-5, 8]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, upper_lip_color, 1)
    cv2.putText(img, "63", tuple(map(int, upper_lip_point_63 + np.array([3, 8]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, upper_lip_color, 1)
    cv2.putText(img, "67", tuple(map(int, lower_lip_point_67 + np.array([-5, 12]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, lower_lip_color, 1)
    cv2.putText(img, "58", tuple(map(int, lower_lip_point_58 + np.array([3, 12]))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, lower_lip_color, 1)

def _add_enhanced_text_overlay(img, proportions, model_predictions):
    """Add enhanced text overlay with new features"""
    height, width = img.shape[:2]
    
    # Create text background
    overlay = img.copy()
    
    # Enhanced text content
    texts = [
        f"Primer tercio: {proportions.get('distance_69_68_proportion', 0):.3f}",
        f"Segundo tercio: {proportions.get('distance_68_34_proportion', 0):.3f}",
        f"Tercer tercio: {proportions.get('distance_34_9_proportion', 0):.3f}",
        f"Proporción ojos: {proportions.get('eye_distance_proportion', 0):.3f}",
        f"Ancho facial: {proportions.get('head_width_proportion', 0):.3f}",
        f"Boca-pupila: {proportions.get('mouth_to_eye_proportion', 0):.3f}",
        "",
        "Modelo Enhanced:",
        f"Puntos detectados: {len(model_predictions)}",
        f"P1: {'✓' if 1 in model_predictions else '✗'}",
        f"P2: {'✓' if 2 in model_predictions else '✗'}",
        f"P3: {'✓' if 3 in model_predictions else '✗'}"
    ]
    
    # Calculate text area
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    
    # Position text on the right side
    x_start = width - 280
    y_start = 30
    line_height = 22
    
    # Draw background rectangle
    cv2.rectangle(overlay, (x_start - 10, y_start - 20), 
                  (width - 10, y_start + len(texts) * line_height + 10), 
                  (0, 0, 0), -1)
    
    # Add transparency
    alpha = 0.8
    img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Draw text with colors
    for i, text in enumerate(texts):
        if text:  # Skip empty lines
            y_pos = y_start + i * line_height
            if text.startswith("Modelo"):
                color = (0, 255, 255)  # Cyan for header
            elif "✓" in text:
                color = (0, 255, 0)    # Green for success
            elif "✗" in text:
                color = (0, 0, 255)    # Red for missing
            else:
                color = (255, 255, 255)  # White for normal text
            
            cv2.putText(img_new, text, (x_start, y_pos), font, font_scale, color, thickness)
    
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

def create_comprehensive_analysis_grid(image, analysis_results):
    """Create a comprehensive grid showing all analysis aspects"""
    # This would create a 2x3 grid showing:
    # 1. Original with landmarks
    # 2. Facial thirds
    # 3. Eye analysis
    # 4. Eyebrow analysis  
    # 5. Area analysis
    # 6. Model points
    
    # For now, return the detailed report as this is the most comprehensive
    return create_detailed_report_image(image, analysis_results)

def create_eye_colorimetry_visualization(image, eye_colorimetry_results):
    """
    Create visualization for eye colorimetry analysis results
    
    Args:
        image: Original image
        eye_colorimetry_results: Results from eye colorimetry analysis
        
    Returns:
        numpy array: Annotated image with eye colorimetry information
    """
    img_vis = image.copy()
    
    if "error" in eye_colorimetry_results:
        # Draw error message
        cv2.putText(img_vis, f"Error: {eye_colorimetry_results['error']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img_vis
    
    # Process each face
    for face_idx, face_data in enumerate(eye_colorimetry_results.get("faces", [])):
        face_y_offset = face_idx * 150
        
        # Draw face index
        cv2.putText(img_vis, f"Face {face_data.get('face_index', face_idx)}", 
                   (10, 30 + face_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Process each eye
        for eye_idx, eye_side in enumerate(['left', 'right']):
            eye_key = f"{eye_side}_eye"
            if eye_key in face_data and "error" not in face_data[eye_key]:
                eye_data = face_data[eye_key]
                
                # Get bounding box
                bbox = eye_data.get("bounding_box", (0, 0, 0, 0))
                x_min, y_min, x_max, y_max = bbox
                
                # Draw eye bounding box
                cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Get classification results
                classifications = eye_data.get("classifications", {})
                iris_rgb_avg = classifications.get("iris_rgb_average", "unknown")
                iris_rgb_dom = classifications.get("iris_rgb_dominant", "unknown")
                iris_hsv = classifications.get("iris_hsv", "unknown")
                
                # Draw labels
                label_x = x_min
                label_y = y_min - 10 if y_min > 30 else y_max + 20
                
                cv2.putText(img_vis, f"{eye_side.title()} Eye", 
                           (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img_vis, f"RGB: {iris_rgb_avg}", 
                           (label_x, label_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(img_vis, f"HSV: {iris_hsv}", 
                           (label_x, label_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw color sample if available
                iris_analysis = eye_data.get("iris_color_analysis", {})
                if "average_color_rgb" in iris_analysis:
                    avg_color = iris_analysis["average_color_rgb"]
                    # Draw color rectangle
                    color_rect_x = x_max + 5
                    color_rect_y = y_min
                    cv2.rectangle(img_vis, (color_rect_x, color_rect_y), 
                                 (color_rect_x + 20, color_rect_y + 20), 
                                 (avg_color[2], avg_color[1], avg_color[0]), -1)  # BGR format
                    cv2.rectangle(img_vis, (color_rect_x, color_rect_y), 
                                 (color_rect_x + 20, color_rect_y + 20), 
                                 (255, 255, 255), 1)
    
    return img_vis

def create_eye_color_comparison_chart(image, eye_colorimetry_results):
    """
    Create a comparison chart showing different classification methods
    
    Args:
        image: Original image
        eye_colorimetry_results: Results from eye colorimetry analysis
        
    Returns:
        numpy array: Chart image
    """
    # Create a larger canvas for the comparison
    height, width = image.shape[:2]
    chart_width = width + 400
    chart_canvas = np.zeros((height, chart_width, 3), dtype=np.uint8)
    
    # Place original image on the left
    chart_canvas[:height, :width] = image
    
    # Add comparison chart on the right
    chart_area = chart_canvas[:, width:]
    chart_area.fill(40)  # Dark background
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 25
    
    # Title
    cv2.putText(chart_canvas, "EYE COLOR CLASSIFICATION COMPARISON", 
                (width + 10, 30), font, 0.6, (255, 255, 255), 2)
    
    y_pos = 60
    
    if "error" in eye_colorimetry_results:
        cv2.putText(chart_canvas, f"Error: {eye_colorimetry_results['error']}", 
                   (width + 10, y_pos), font, font_scale, (0, 0, 255), thickness)
        return chart_canvas
    
    # Process each face
    for face_idx, face_data in enumerate(eye_colorimetry_results.get("faces", [])):
        # Face header
        cv2.putText(chart_canvas, f"Face {face_data.get('face_index', face_idx)}:", 
                   (width + 10, y_pos), font, 0.6, (0, 255, 255), thickness)
        y_pos += line_height + 5
        
        # Process each eye
        for eye_side in ['left', 'right']:
            eye_key = f"{eye_side}_eye"
            if eye_key in face_data and "error" not in face_data[eye_key]:
                eye_data = face_data[eye_key]
                classifications = eye_data.get("classifications", {})
                
                # Eye header
                cv2.putText(chart_canvas, f"  {eye_side.title()} Eye:", 
                           (width + 15, y_pos), font, font_scale, (255, 255, 255), thickness)
                y_pos += line_height
                
                # Classification methods
                methods = [
                    ("HSV System", classifications.get("iris_hsv", "unknown"), (255, 255, 0)),
                    ("RGB Average", classifications.get("iris_rgb_average", "unknown"), (0, 255, 255)),
                    ("RGB Dominant", classifications.get("iris_rgb_dominant", "unknown"), (255, 0, 255))
                ]
                
                for method_name, result, color in methods:
                    cv2.putText(chart_canvas, f"    {method_name}: {result}", 
                               (width + 20, y_pos), font, 0.4, color, thickness)
                    y_pos += line_height
                
                # Color information
                iris_analysis = eye_data.get("iris_color_analysis", {})
                if "average_color_rgb" in iris_analysis:
                    avg_color = iris_analysis["average_color_rgb"]
                    cv2.putText(chart_canvas, f"    Avg RGB: {avg_color}", 
                               (width + 20, y_pos), font, 0.4, (200, 200, 200), thickness)
                    y_pos += line_height
                
                if "total_pixels_analyzed" in iris_analysis:
                    pixels = iris_analysis["total_pixels_analyzed"]
                    cv2.putText(chart_canvas, f"    Pixels: {pixels}", 
                               (width + 20, y_pos), font, 0.4, (200, 200, 200), thickness)
                    y_pos += line_height
                
                y_pos += 10  # Extra space between eyes
            else:
                cv2.putText(chart_canvas, f"  {eye_side.title()} Eye: Error", 
                           (width + 15, y_pos), font, font_scale, (0, 0, 255), thickness)
                y_pos += line_height + 10
        
        y_pos += 15  # Extra space between faces
    
    return chart_canvas

def create_color_palette_visualization(eye_colorimetry_results):
    """
    Create a color palette visualization showing detected colors
    
    Args:
        eye_colorimetry_results: Results from eye colorimetry analysis
        
    Returns:
        numpy array: Color palette image
    """
    palette_height = 200
    palette_width = 600
    palette_canvas = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    
    if "error" in eye_colorimetry_results:
        cv2.putText(palette_canvas, f"Error: {eye_colorimetry_results['error']}", 
                   (10, palette_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return palette_canvas
    
    # Process each face and eye
    current_y = 0
    row_height = 40
    
    for face_idx, face_data in enumerate(eye_colorimetry_results.get("faces", [])):
        for eye_side in ['left', 'right']:
            eye_key = f"{eye_side}_eye"
            if eye_key in face_data and "error" not in face_data[eye_key]:
                eye_data = face_data[eye_key]
                iris_analysis = eye_data.get("iris_color_analysis", {})
                
                # Draw label
                label = f"Face {face_data.get('face_index', face_idx)} - {eye_side.title()}"
                cv2.putText(palette_canvas, label, (10, current_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw color palette
                if "dominant_colors" in iris_analysis:
                    dominant_colors = iris_analysis["dominant_colors"]
                    x_pos = 150
                    
                    for color_data, percentage in dominant_colors[:5]:  # Show top 5 colors
                        color_width = int(percentage * 3)  # Scale width by percentage
                        color_width = max(color_width, 10)  # Minimum width
                        
                        # Draw color rectangle (convert RGB to BGR for OpenCV)
                        color_bgr = (color_data[2], color_data[1], color_data[0])
                        cv2.rectangle(palette_canvas, (x_pos, current_y + 5), 
                                     (x_pos + color_width, current_y + 30), color_bgr, -1)
                        cv2.rectangle(palette_canvas, (x_pos, current_y + 5), 
                                     (x_pos + color_width, current_y + 30), (255, 255, 255), 1)
                        
                        # Add percentage text
                        cv2.putText(palette_canvas, f"{percentage:.1f}%", 
                                   (x_pos, current_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                                   (255, 255, 255), 1)
                        
                        x_pos += color_width + 5
                
                current_y += row_height
                
                if current_y >= palette_height - row_height:
                    break
            
            if current_y >= palette_height - row_height:
                break
        
        if current_y >= palette_height - row_height:
            break
    
    return palette_canvas
