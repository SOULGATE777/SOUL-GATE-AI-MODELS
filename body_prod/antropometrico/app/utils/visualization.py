import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
import logging

logger = logging.getLogger(__name__)

# Set matplotlib backend for server environments
plt.switch_backend('Agg')

# Configure style
plt.style.use('default')
sns.set_palette("husl")

# Color scheme for anthropometric visualization
COLORS = {
    'skull': (255, 255, 0),      # Yellow for skull
    'face': (255, 128, 0),       # Orange for face region
    'torso': (255, 0, 0),        # Red for torso
    'limbs': (128, 128, 128),    # Gray for limbs
    'keypoints': (0, 255, 0),    # Green for keypoints
    'connections': (255, 255, 255), # White for connections
}

def create_anthropometric_visualization(
    image_path: str,
    results: Dict[str, Any],
    output_path: str = "/app/results/anthropometric_analysis.png",
    figsize: Tuple[int, int] = (18, 12)
) -> Optional[str]:
    """
    Create comprehensive anthropometric analysis visualization
    
    Args:
        image_path: Path to input image
        results: Analysis results from pipeline
        output_path: Path to save visualization
        figsize: Figure size (width, height)
    
    Returns:
        Path to saved visualization or None if failed
    """
    try:
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Extract analysis data
        anthropometric_analysis = results.get('anthropometric_analysis', [])
        
        if not anthropometric_analysis:
            # No analysis data available
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No anthropometric analysis data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Anthropometric Analysis', fontsize=18, fontweight='bold')
            ax.axis('off')
        else:
            # 1. Original Image with Annotations (top-left, spanning 2 columns)
            ax1 = fig.add_subplot(gs[0, :2])
            annotated_image = _create_annotated_image(image_rgb, anthropometric_analysis)
            ax1.imshow(annotated_image)
            ax1.set_title('Anthropometric Analysis - Keypoints & Measurements', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # 2. Skull Analysis (top-right, spanning 2 columns)
            ax2 = fig.add_subplot(gs[0, 2:])
            _plot_skull_analysis(ax2, anthropometric_analysis)
            
            # 3. Body Proportions (middle-left)
            ax3 = fig.add_subplot(gs[1, 0])
            _plot_body_proportions(ax3, anthropometric_analysis)
            
            # 4. Confidence Analysis (middle-center)
            ax4 = fig.add_subplot(gs[1, 1])
            _plot_confidence_analysis(ax4, anthropometric_analysis)
            
            # 5. Keypoint Summary (middle-right, spanning 2 columns)
            ax5 = fig.add_subplot(gs[1, 2:])
            _plot_keypoint_summary(ax5, anthropometric_analysis)
            
            # 6. Detailed Analysis Summary (bottom, spanning all columns)
            ax6 = fig.add_subplot(gs[2, :])
            _plot_detailed_summary(ax6, anthropometric_analysis, results.get('analysis_summary', {}))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Anthropometric visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating anthropometric visualization: {e}")
        plt.close('all')
        return None

def _create_annotated_image(image: np.ndarray, analyses: List[Dict[str, Any]]) -> np.ndarray:
    """Create annotated image with keypoints and measurements"""
    annotated = image.copy()
    
    for analysis in analyses:
        person_id = analysis.get('person_id', 1)
        pose_keypoints = analysis.get('pose_keypoints', {})
        body_proportions = analysis.get('body_proportions', {})
        
        # Draw skull bounding box if available
        skull_bbox = body_proportions.get('skull_bbox')
        if skull_bbox:
            x_min, y_min, x_max, y_max = skull_bbox
            # Draw ultra-thin skull boundary
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), COLORS['skull'], 2)
            # Add label
            cv2.putText(annotated, f'Skull P{person_id}', (x_min, y_min-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['skull'], 2)
        
        # Draw keypoints
        for keypoint_name, (x, y) in pose_keypoints.items():
            # Color based on body part
            if keypoint_name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']:
                color = COLORS['face']
                radius = 6
            elif keypoint_name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
                color = COLORS['torso']
                radius = 5
            else:
                color = COLORS['limbs']
                radius = 4
            
            cv2.circle(annotated, (x, y), radius, color, -1)
            cv2.circle(annotated, (x, y), radius + 1, (0, 0, 0), 1)  # Black outline
        
        # Draw connections for head keypoints
        head_connections = [
            ('left_eye', 'right_eye'),
            ('nose', 'left_eye'),
            ('nose', 'right_eye'),
            ('left_ear', 'left_eye'),
            ('right_ear', 'right_eye')
        ]
        
        for start_kp, end_kp in head_connections:
            if start_kp in pose_keypoints and end_kp in pose_keypoints:
                start_pos = pose_keypoints[start_kp]
                end_pos = pose_keypoints[end_kp]
                cv2.line(annotated, start_pos, end_pos, COLORS['connections'], 2)
        
        # Add person label
        if pose_keypoints:
            # Find topmost keypoint for label placement
            y_coords = [y for x, y in pose_keypoints.values()]
            min_y = min(y_coords)
            x_coords = [x for x, y in pose_keypoints.values()]
            center_x = int(np.mean(x_coords))
            
            cv2.putText(annotated, f'Person {person_id}', (center_x-30, min_y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f'Person {person_id}', (center_x-30, min_y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    return annotated

def _plot_skull_analysis(ax, analyses: List[Dict[str, Any]]):
    """Plot skull analysis results"""
    ax.set_title('Skull Measurement Analysis', fontsize=14, fontweight='bold')
    
    if not analyses:
        ax.text(0.5, 0.5, 'No skull analysis data', ha='center', va='center')
        ax.axis('off')
        return
    
    # Collect skull data
    skull_data = []
    person_labels = []
    
    for analysis in analyses:
        proportions = analysis.get('body_proportions', {})
        if proportions.get('measurements_available', False):
            skull_data.append({
                'person': f"Person {analysis.get('person_id', 1)}",
                'skull_percentage': proportions.get('skull_percentage', 0),
                'skull_height': proportions.get('skull_height', 0),
                'skull_width': proportions.get('skull_width', 0),
                'method': proportions.get('detection_method', 'unknown')
            })
    
    if not skull_data:
        ax.text(0.5, 0.5, 'No successful skull measurements', ha='center', va='center')
        ax.axis('off')
        return
    
    # Create bar chart of skull percentages
    persons = [data['person'] for data in skull_data]
    percentages = [data['skull_percentage'] for data in skull_data]
    
    bars = ax.bar(persons, percentages, color='skyblue', alpha=0.7)
    
    # Add reference lines
    ax.axhline(y=12.5, color='green', linestyle='--', alpha=0.7, label='Adult min (12.5%)')
    ax.axhline(y=14.3, color='green', linestyle='--', alpha=0.7, label='Adult max (14.3%)')
    ax.axhline(y=16, color='orange', linestyle='--', alpha=0.7, label='Child range (16%+)')
    
    # Customize plot
    ax.set_ylabel('Skull to Body Ratio (%)', fontsize=12)
    ax.set_title('Skull-to-Body Percentage', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Color bars based on normal ranges
    for bar, percentage in zip(bars, percentages):
        if 12.5 <= percentage <= 14.3:
            bar.set_color('lightgreen')  # Normal adult
        elif percentage > 16:
            bar.set_color('lightsalmon')  # Child-like
        else:
            bar.set_color('lightblue')   # Intermediate

def _plot_body_proportions(ax, analyses: List[Dict[str, Any]]):
    """Plot body proportions comparison"""
    ax.set_title('Body Proportions', fontsize=12, fontweight='bold')
    
    if not analyses:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return
    
    # Collect proportion data
    skull_heights = []
    body_heights = []
    labels = []
    
    for analysis in analyses:
        proportions = analysis.get('body_proportions', {})
        if proportions.get('measurements_available', False):
            skull_heights.append(proportions.get('skull_height', 0))
            body_heights.append(proportions.get('body_height', 0))
            labels.append(f"P{analysis.get('person_id', 1)}")
    
    if not skull_heights:
        ax.text(0.5, 0.5, 'No measurements\navailable', ha='center', va='center')
        ax.axis('off')
        return
    
    # Create stacked bar chart
    x = np.arange(len(labels))
    width = 0.35
    
    # Normalize to show proportions
    total_heights = np.array(body_heights)
    skull_props = np.array(skull_heights) / total_heights * 100
    body_props = 100 - skull_props
    
    bars1 = ax.bar(x, skull_props, width, label='Skull', color='gold', alpha=0.8)
    bars2 = ax.bar(x, body_props, width, bottom=skull_props, label='Body', color='lightblue', alpha=0.8)
    
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Skull vs Body Proportions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add percentage labels
    for i, (skull_prop, bar) in enumerate(zip(skull_props, bars1)):
        ax.text(bar.get_x() + bar.get_width()/2., skull_prop/2,
               f'{skull_prop:.1f}%', ha='center', va='center', fontweight='bold')

def _plot_confidence_analysis(ax, analyses: List[Dict[str, Any]]):
    """Plot confidence analysis"""
    ax.set_title('Detection Confidence', fontsize=12, fontweight='bold')
    
    if not analyses:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return
    
    # Collect confidence data
    overall_confidences = []
    head_confidences = []
    labels = []
    
    for analysis in analyses:
        conf_analysis = analysis.get('confidence_analysis', {})
        overall_conf = conf_analysis.get('overall_average')
        head_conf = conf_analysis.get('head_average')
        
        if overall_conf is not None:
            overall_confidences.append(overall_conf)
            head_confidences.append(head_conf if head_conf is not None else 0)
            labels.append(f"P{analysis.get('person_id', 1)}")
    
    if not overall_confidences:
        ax.text(0.5, 0.5, 'No confidence\ndata available', ha='center', va='center')
        ax.axis('off')
        return
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, overall_confidences, width, label='Overall', alpha=0.8)
    bars2 = ax.bar(x + width/2, head_confidences, width, label='Head', alpha=0.8)
    
    ax.set_ylabel('Confidence Score')
    ax.set_xlabel('Person')
    ax.set_title('Detection Confidence Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add threshold line
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')

def _plot_keypoint_summary(ax, analyses: List[Dict[str, Any]]):
    """Plot keypoint detection summary"""
    ax.set_title('Keypoint Detection Summary', fontsize=12, fontweight='bold')
    
    if not analyses:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return
    
    # Collect keypoint data
    persons = []
    total_keypoints = []
    head_keypoints = []
    
    for analysis in analyses:
        keypoint_summary = analysis.get('keypoint_summary', {})
        persons.append(f"P{analysis.get('person_id', 1)}")
        total_keypoints.append(keypoint_summary.get('total_keypoints', 0))
        head_keypoints.append(keypoint_summary.get('head_keypoints', 0))
    
    if not total_keypoints:
        ax.text(0.5, 0.5, 'No keypoint\ndata available', ha='center', va='center')
        ax.axis('off')
        return
    
    x = np.arange(len(persons))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, total_keypoints, width, label='Total (17 max)', alpha=0.8, color='lightblue')
    bars2 = ax.bar(x + width/2, head_keypoints, width, label='Head (5 max)', alpha=0.8, color='orange')
    
    ax.set_ylabel('Keypoints Detected')
    ax.set_xlabel('Person')
    ax.set_title('Keypoint Detection Count')
    ax.set_xticks(x)
    ax.set_xticklabels(persons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)

def _plot_detailed_summary(ax, analyses: List[Dict[str, Any]], summary: Dict[str, Any]):
    """Plot detailed analysis summary as text"""
    ax.axis('off')
    
    # Create comprehensive text summary
    text_content = []
    
    # Overall summary
    text_content.append("📊 ANTHROPOMETRIC ANALYSIS SUMMARY")
    text_content.append("=" * 50)
    
    if summary:
        total_persons = summary.get('total_persons_detected', 0)
        successful_measurements = summary.get('successful_skull_measurements', 0)
        success_rate = summary.get('measurement_success_rate', 0)
        avg_confidence = summary.get('average_detection_confidence')
        overall_quality = summary.get('overall_quality', 'unknown')
        
        text_content.append(f"• Persons detected: {total_persons}")
        text_content.append(f"• Successful skull measurements: {successful_measurements}")
        text_content.append(f"• Success rate: {success_rate:.1f}%")
        if avg_confidence:
            text_content.append(f"• Average confidence: {avg_confidence:.3f}")
        text_content.append(f"• Overall quality: {overall_quality.upper()}")
        text_content.append("")
    
    # Individual person details
    for i, analysis in enumerate(analyses):
        person_id = analysis.get('person_id', i + 1)
        text_content.append(f"👤 PERSON {person_id} DETAILS:")
        
        # Keypoint summary
        keypoint_summary = analysis.get('keypoint_summary', {})
        total_kp = keypoint_summary.get('total_keypoints', 0)
        head_kp = keypoint_summary.get('head_keypoints', 0)
        completeness = keypoint_summary.get('keypoint_completeness', 'unknown')
        
        text_content.append(f"  • Keypoints: {total_kp}/17 total, {head_kp}/5 head ({completeness})")
        
        # Body proportions
        proportions = analysis.get('body_proportions', {})
        if proportions.get('measurements_available', False):
            skull_pct = proportions.get('skull_percentage', 0)
            skull_dims = f"{proportions.get('skull_width', 0):.0f}×{proportions.get('skull_height', 0):.0f}px"
            assessment = proportions.get('anatomical_assessment', 'unknown')
            head_orientation = proportions.get('head_orientation', 'unknown')
            
            text_content.append(f"  • Skull ratio: {skull_pct:.1f}% of body height")
            text_content.append(f"  • Skull dimensions: {skull_dims}")
            text_content.append(f"  • Assessment: {assessment}")
            text_content.append(f"  • Head orientation: {head_orientation}")
        else:
            text_content.append(f"  • Skull measurements: NOT AVAILABLE")
        
        # Confidence analysis
        conf_analysis = analysis.get('confidence_analysis', {})
        reliability = conf_analysis.get('reliability_assessment', 'unknown')
        text_content.append(f"  • Detection reliability: {reliability}")
        
        # Detailed analysis if available
        detailed = analysis.get('detailed_analysis', {})
        if detailed:
            age_assessment = detailed.get('age_assessment', {})
            if age_assessment:
                age_class = age_assessment.get('classification', 'unknown')
                age_desc = age_assessment.get('description', '')
                text_content.append(f"  • Age assessment: {age_class}")
                if len(age_desc) < 60:  # Only show if not too long
                    text_content.append(f"    {age_desc}")
        
        text_content.append("")
    
    # Recommendations
    if summary and 'recommendations' in summary:
        text_content.append("💡 RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            text_content.append(f"• {rec}")
        text_content.append("")
    
    # Technical details
    text_content.append("🔧 TECHNICAL DETAILS:")
    text_content.append("• Detection method: YOLOv8n pose estimation")
    text_content.append("• Skull detection: Anatomical proportions + contour refinement")
    text_content.append("• Measurements: Based on facial keypoint geometry")
    text_content.append("• Reference: Adult skull = 12.5-14.3% of body height")
    
    # Display text
    full_text = '\n'.join(text_content)
    ax.text(0.02, 0.98, full_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    ax.set_title('Detailed Analysis Report', fontsize=14, fontweight='bold')

def create_simple_anthropometric_visualization(
    image_path: str,
    results: Dict[str, Any],
    output_path: str = "/app/results/simple_anthropometric.png",
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[str]:
    """
    Create simple anthropometric visualization for quick analysis
    
    Args:
        image_path: Path to input image
        results: Analysis results from pipeline
        output_path: Path to save visualization
        figsize: Figure size (width, height)
    
    Returns:
        Path to saved visualization or None if failed
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Extract analysis data
        anthropometric_analysis = results.get('anthropometric_analysis', [])
        
        # 1. Annotated image
        if anthropometric_analysis:
            annotated_image = _create_annotated_image(image_rgb, anthropometric_analysis)
            axes[0].imshow(annotated_image)
            axes[0].set_title('Detected Keypoints & Skull', fontsize=12, fontweight='bold')
        else:
            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Skull measurements
        if anthropometric_analysis:
            _plot_skull_analysis(axes[1], anthropometric_analysis)
        else:
            axes[1].text(0.5, 0.5, 'No skull\nmeasurements', ha='center', va='center')
            axes[1].set_title('Skull Analysis', fontsize=12, fontweight='bold')
            axes[1].axis('off')
        
        # 3. Confidence scores
        if anthropometric_analysis:
            _plot_confidence_analysis(axes[2], anthropometric_analysis)
        else:
            axes[2].text(0.5, 0.5, 'No confidence\ndata', ha='center', va='center')
            axes[2].set_title('Detection Confidence', fontsize=12, fontweight='bold')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Simple anthropometric visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating simple anthropometric visualization: {e}")
        plt.close('all')
        return None

def create_skull_measurement_visualization(
    image_path: str,
    skull_results: Dict[str, Any],
    output_path: str = "/app/results/skull_measurements.png",
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[str]:
    """
    Create focused skull measurement visualization
    
    Args:
        image_path: Path to input image
        skull_results: Skull detection results
        output_path: Path to save visualization
        figsize: Figure size (width, height)
    
    Returns:
        Path to saved visualization or None if failed
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        skull_detections = skull_results.get('skull_detections', [])
        
        # 1. Original image with skull boxes (top-left)
        annotated = image_rgb.copy()
        for detection in skull_detections:
            skull_bbox = detection.get('skull_bbox')
            if skull_bbox:
                x_min, y_min, x_max, y_max = skull_bbox
                # Draw skull bounding box
                rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                               linewidth=2, edgecolor='yellow', facecolor='none')
                # Note: We can't add patches directly to the image array, so we'll use OpenCV
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                person_id = detection.get('person_id', 1)
                cv2.putText(annotated, f'Skull P{person_id}', (x_min, y_min-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        axes[0, 0].imshow(annotated)
        axes[0, 0].set_title('Skull Detection Results', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Skull dimensions (top-right)
        if skull_detections:
            persons = [f"P{d.get('person_id', 1)}" for d in skull_detections if d.get('measurements_available')]
            widths = [d.get('skull_width', 0) for d in skull_detections if d.get('measurements_available')]
            heights = [d.get('skull_height', 0) for d in skull_detections if d.get('measurements_available')]
            
            if persons:
                x = np.arange(len(persons))
                width = 0.35
                
                bars1 = axes[0, 1].bar(x - width/2, widths, width, label='Width (px)', alpha=0.8)
                bars2 = axes[0, 1].bar(x + width/2, heights, width, label='Height (px)', alpha=0.8)
                
                axes[0, 1].set_ylabel('Pixels')
                axes[0, 1].set_title('Skull Dimensions')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(persons)
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No skull\ndimensions\navailable', ha='center', va='center')
            axes[0, 1].set_title('Skull Dimensions', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
        
        # 3. Skull percentages (bottom-left)
        if skull_detections:
            persons = [f"P{d.get('person_id', 1)}" for d in skull_detections if d.get('measurements_available')]
            percentages = [d.get('skull_percentage', 0) for d in skull_detections if d.get('measurements_available')]
            
            if persons and percentages:
                bars = axes[1, 0].bar(persons, percentages, color='lightcoral', alpha=0.7)
                
                # Add reference lines
                axes[1, 0].axhline(y=12.5, color='green', linestyle='--', alpha=0.7, label='Adult min')
                axes[1, 0].axhline(y=14.3, color='green', linestyle='--', alpha=0.7, label='Adult max')
                axes[1, 0].axhline(y=16, color='orange', linestyle='--', alpha=0.7, label='Child range')
                
                axes[1, 0].set_ylabel('Skull Percentage (%)')
                axes[1, 0].set_title('Skull-to-Body Ratio')
                axes[1, 0].legend(fontsize=8)
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, pct in zip(bars, percentages):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No skull\npercentages\navailable', ha='center', va='center')
            axes[1, 0].set_title('Skull-to-Body Ratio', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
        
        # 4. Detection quality summary (bottom-right)
        axes[1, 1].axis('off')
        
        if skull_detections:
            summary_text = []
            summary_text.append("🔍 SKULL DETECTION SUMMARY")
            summary_text.append("=" * 25)
            
            for detection in skull_detections:
                person_id = detection.get('person_id', 1)
                summary_text.append(f"\n👤 Person {person_id}:")
                
                if detection.get('measurements_available'):
                    method = detection.get('detection_method', 'unknown')
                    skull_pct = detection.get('skull_percentage', 0)
                    assessment = detection.get('anatomical_assessment', 'unknown')
                    head_conf = detection.get('average_head_confidence')
                    
                    summary_text.append(f"  • Method: {method}")
                    summary_text.append(f"  • Skull ratio: {skull_pct:.1f}%")
                    summary_text.append(f"  • Assessment: {assessment}")
                    if head_conf:
                        summary_text.append(f"  • Confidence: {head_conf:.3f}")
                else:
                    summary_text.append(f"  • No measurements available")
            
            summary_text.append(f"\n📏 Reference ranges:")
            summary_text.append(f"  • Adult: 12.5-14.3%")
            summary_text.append(f"  • Child: 16-18%")
            
            full_text = '\n'.join(summary_text)
            axes[1, 1].text(0.05, 0.95, full_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        else:
            axes[1, 1].text(0.5, 0.5, 'No skull detection\nresults available', 
                           ha='center', va='center', fontsize=12)
        
        axes[1, 1].set_title('Detection Summary', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Skull measurement visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating skull measurement visualization: {e}")
        plt.close('all')
        return None
