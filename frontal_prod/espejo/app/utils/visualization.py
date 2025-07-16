import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns

def create_mirror_visualization(original_image: np.ndarray, mirror_images: Dict, 
                               classification_results: Dict, proportions: Dict) -> np.ndarray:
    """
    Create comprehensive mirror analysis visualization
    
    Args:
        original_image: Original input image
        mirror_images: Dict containing mirror images
        classification_results: Classification results
        proportions: Proportions analysis results
        
    Returns:
        numpy.ndarray: Visualization image
    """
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        right_mirror_rgb = cv2.cvtColor(mirror_images['right_mirrored'], cv2.COLOR_BGR2RGB)
        left_mirror_rgb = cv2.cvtColor(mirror_images['left_mirrored'], cv2.COLOR_BGR2RGB)
        
        # Plot original image
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image\nAnthropometric Analysis', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot right mirrored image
        axes[1].imshow(right_mirror_rgb)
        axes[1].set_title('Right Mirrored Face\nClassification Results', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Plot left mirrored image
        axes[2].imshow(left_mirror_rgb)
        axes[2].set_title('Left Mirrored Face\nClassification Results', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Add region bounding boxes to mirror images
        _add_region_bboxes_to_plot(axes[1], right_mirror_rgb.shape[:2])
        _add_region_bboxes_to_plot(axes[2], left_mirror_rgb.shape[:2])
        
        # Add proportion information to original image
        _add_proportion_text(axes[0], proportions)
        
        # Add classification results
        _add_classification_text(axes[1], classification_results['right_mirrored'], 'right')
        _add_classification_text(axes[2], classification_results['left_mirrored'], 'left')
        
        plt.tight_layout()
        
        # Convert matplotlib figure to opencv image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        visualization = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        return visualization
        
    except Exception as e:
        print(f"Error creating mirror visualization: {e}")
        # Return a simple error image
        error_img = np.zeros((400, 800, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Visualization Error: {str(e)}", (10, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_img

def create_analysis_dashboard(original_image: np.ndarray, analysis_results: Dict) -> np.ndarray:
    """
    Create comprehensive analysis dashboard
    
    Args:
        original_image: Original input image
        analysis_results: Complete analysis results
        
    Returns:
        numpy.ndarray: Dashboard image
    """
    try:
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Original image with analysis overlay
        ax1 = fig.add_subplot(gs[0, :])
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        ax1.imshow(original_rgb)
        ax1.set_title('Original Image - Anthropometric Analysis', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Mirror images
        ax2 = fig.add_subplot(gs[1, 0])
        if 'mirror_images' in analysis_results:
            right_mirror_rgb = cv2.cvtColor(analysis_results['mirror_images']['right_mirrored'], cv2.COLOR_BGR2RGB)
            ax2.imshow(right_mirror_rgb)
            _add_region_bboxes_to_plot(ax2, right_mirror_rgb.shape[:2])
        ax2.set_title('Right Mirrored Face', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[1, 1])
        if 'mirror_images' in analysis_results:
            left_mirror_rgb = cv2.cvtColor(analysis_results['mirror_images']['left_mirrored'], cv2.COLOR_BGR2RGB)
            ax3.imshow(left_mirror_rgb)
            _add_region_bboxes_to_plot(ax3, left_mirror_rgb.shape[:2])
        ax3.set_title('Left Mirrored Face', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Proportions chart
        ax4 = fig.add_subplot(gs[1, 2])
        _create_proportions_chart(ax4, analysis_results.get('proportions', {}))
        
        # Classification results tables
        ax5 = fig.add_subplot(gs[2, :])
        _create_classification_summary_table(ax5, analysis_results.get('classification_results', {}))
        
        plt.tight_layout()
        
        # Convert matplotlib figure to opencv image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        dashboard = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        return dashboard
        
    except Exception as e:
        print(f"Error creating analysis dashboard: {e}")
        # Return a simple error image
        error_img = np.zeros((800, 1200, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Dashboard Error: {str(e)}", (10, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_img

def _add_region_bboxes_to_plot(ax, image_shape: Tuple[int, int]):
    """Add region bounding boxes to matplotlib plot"""
    height, width = image_shape
    
    # FRENTE region (forehead area) - upper 40% of image
    frente_rect = Rectangle((width * 0.1, height * 0.05), 
                           width * 0.8, height * 0.4, 
                           linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(frente_rect)
    ax.text(width * 0.12, height * 0.02, 'FRENTE', 
           fontsize=10, color='green', fontweight='bold')
    
    # rostro_menton region (chin/jaw area) - lower 40% of image
    rostro_rect = Rectangle((width * 0.15, height * 0.55), 
                           width * 0.7, height * 0.4, 
                           linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rostro_rect)
    ax.text(width * 0.17, height * 0.52, 'ROSTRO_MENTON', 
           fontsize=10, color='red', fontweight='bold')

def _add_proportion_text(ax, proportions: Dict):
    """Add proportion information to plot"""
    prop_text = "ANTHROPOMETRIC PROPORTIONS:\n\n"
    
    if 'face_proportions' in proportions:
        face_props = proportions['face_proportions']
        prop_text += "FACE PROPORTIONS:\n"
        prop_text += f"Right: {face_props.get('right', 'N/A'):.4f}\n" if face_props.get('right') is not None else "Right: N/A\n"
        prop_text += f"Left: {face_props.get('left', 'N/A'):.4f}\n" if face_props.get('left') is not None else "Left: N/A\n"
    
    if 'forehead_proportions' in proportions:
        forehead_props = proportions['forehead_proportions']
        prop_text += "\nFOREHEAD PROPORTIONS:\n"
        prop_text += f"Right: {forehead_props.get('right', 'N/A'):.4f}\n" if forehead_props.get('right') is not None else "Right: N/A\n"
        prop_text += f"Left: {forehead_props.get('left', 'N/A'):.4f}\n" if forehead_props.get('left') is not None else "Left: N/A\n"
    
    if 'temporal_proportions' in proportions:
        temporal_props = proportions['temporal_proportions']
        prop_text += "\nTEMPORAL PROPORTIONS:\n"
        prop_text += f"Right: {temporal_props.get('right', 'N/A'):.4f}\n" if temporal_props.get('right') is not None else "Right: N/A\n"
        prop_text += f"Left: {temporal_props.get('left', 'N/A'):.4f}\n" if temporal_props.get('left') is not None else "Left: N/A\n"
    
    ax.text(0.02, 0.98, prop_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=8, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def _add_classification_text(ax, classification_results: Dict, side: str):
    """Add classification results to plot"""
    class_text = f"🎯 {side.upper()} SIDE FINAL DIAGNOSIS:\n\n"
    
    # Final split diagnosis
    class_text += "FINAL SPLIT DIAGNOSIS:\n"
    class_text += f"FRENTE: {classification_results.get('frente_split_diagnosis', 'N/A')}\n"
    class_text += f"ROSTRO: {classification_results.get('rostro_split_diagnosis', 'N/A')}\n\n"
    
    # Decision tree diagnosis
    class_text += "DECISION TREE DIAGNOSIS:\n"
    class_text += f"FRENTE: {classification_results.get('frente_final_diagnosis', 'N/A')}\n"
    class_text += f"ROSTRO: {classification_results.get('rostro_final_diagnosis', 'N/A')}\n\n"
    
    # Top predictions
    class_text += "TOP PREDICTIONS:\n"
    class_text += "FRENTE (Green Box):\n"
    frente_preds = classification_results.get('frente_predictions', [])
    frente_probs = classification_results.get('frente_probabilities', [])
    
    for i, (pred, prob) in enumerate(zip(frente_preds[:3], frente_probs[:3])):
        pred_short = pred[:15] + "..." if len(pred) > 15 else pred
        class_text += f"{i+1}. {pred_short} {prob:.1%}\n"
    
    class_text += "\nROSTRO (Red Box):\n"
    rostro_preds = classification_results.get('rostro_predictions', [])
    rostro_probs = classification_results.get('rostro_probabilities', [])
    
    for i, (pred, prob) in enumerate(zip(rostro_preds[:3], rostro_probs[:3])):
        pred_short = pred[:15] + "..." if len(pred) > 15 else pred
        class_text += f"{i+1}. {pred_short} {prob:.1%}\n"
    
    # Position text appropriately
    bbox_color = 'lightblue' if side == 'right' else 'lightgreen'
    ax.text(0.02, 0.02, class_text, transform=ax.transAxes, 
           verticalalignment='bottom', fontsize=7, 
           bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.9))

def _create_proportions_chart(ax, proportions: Dict):
    """Create proportions bar chart"""
    ax.set_title('Anthropometric Proportions', fontsize=12, fontweight='bold')
    
    # Prepare data for chart
    categories = []
    right_values = []
    left_values = []
    
    if 'face_proportions' in proportions:
        face_props = proportions['face_proportions']
        categories.append('Face')
        right_values.append(face_props.get('right', 0) if face_props.get('right') is not None else 0)
        left_values.append(face_props.get('left', 0) if face_props.get('left') is not None else 0)
    
    if 'forehead_proportions' in proportions:
        forehead_props = proportions['forehead_proportions']
        categories.append('Forehead')
        right_values.append(forehead_props.get('right', 0) if forehead_props.get('right') is not None else 0)
        left_values.append(forehead_props.get('left', 0) if forehead_props.get('left') is not None else 0)
    
    if 'temporal_proportions' in proportions:
        temporal_props = proportions['temporal_proportions']
        categories.append('Temporal')
        right_values.append(temporal_props.get('right', 0) if temporal_props.get('right') is not None else 0)
        left_values.append(temporal_props.get('left', 0) if temporal_props.get('left') is not None else 0)
    
    if categories:
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, right_values, width, label='Right', color='skyblue')
        ax.bar(x + width/2, left_values, width, label='Left', color='lightcoral')
        
        ax.set_xlabel('Measurement Type')
        ax.set_ylabel('Proportion Value')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No proportion data available', 
               transform=ax.transAxes, ha='center', va='center')

def _create_classification_summary_table(ax, classification_results: Dict):
    """Create classification summary table"""
    ax.set_title('Classification Summary with Decision Rules', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if not classification_results:
        ax.text(0.5, 0.5, 'No classification results available', 
               transform=ax.transAxes, ha='center', va='center')
        return
    
    # Create summary table
    table_data = []
    
    # Header
    table_data.append(['Side', 'Region', 'Final Diagnosis', 'Decision Tree', 'Top Prediction', 'Confidence'])
    
    # Right side data
    if 'right_mirrored' in classification_results:
        right_results = classification_results['right_mirrored']
        
        # FRENTE row
        frente_top_pred = right_results.get('frente_predictions', ['N/A'])[0]
        frente_top_conf = right_results.get('frente_probabilities', [0])[0]
        table_data.append([
            'Right',
            'FRENTE',
            right_results.get('frente_split_diagnosis', 'N/A'),
            right_results.get('frente_final_diagnosis', 'N/A'),
            frente_top_pred[:20] + "..." if len(frente_top_pred) > 20 else frente_top_pred,
            f"{frente_top_conf:.1%}" if frente_top_conf else "N/A"
        ])
        
        # ROSTRO row
        rostro_top_pred = right_results.get('rostro_predictions', ['N/A'])[0]
        rostro_top_conf = right_results.get('rostro_probabilities', [0])[0]
        table_data.append([
            '',
            'ROSTRO',
            right_results.get('rostro_split_diagnosis', 'N/A'),
            right_results.get('rostro_final_diagnosis', 'N/A'),
            rostro_top_pred[:20] + "..." if len(rostro_top_pred) > 20 else rostro_top_pred,
            f"{rostro_top_conf:.1%}" if rostro_top_conf else "N/A"
        ])
    
    # Left side data
    if 'left_mirrored' in classification_results:
        left_results = classification_results['left_mirrored']
        
        # FRENTE row
        frente_top_pred = left_results.get('frente_predictions', ['N/A'])[0]
        frente_top_conf = left_results.get('frente_probabilities', [0])[0]
        table_data.append([
            'Left',
            'FRENTE',
            left_results.get('frente_split_diagnosis', 'N/A'),
            left_results.get('frente_final_diagnosis', 'N/A'),
            frente_top_pred[:20] + "..." if len(frente_top_pred) > 20 else frente_top_pred,
            f"{frente_top_conf:.1%}" if frente_top_conf else "N/A"
        ])
        
        # ROSTRO row
        rostro_top_pred = left_results.get('rostro_predictions', ['N/A'])[0]
        rostro_top_conf = left_results.get('rostro_probabilities', [0])[0]
        table_data.append([
            '',
            'ROSTRO',
            left_results.get('rostro_split_diagnosis', 'N/A'),
            left_results.get('rostro_final_diagnosis', 'N/A'),
            rostro_top_pred[:20] + "..." if len(rostro_top_pred) > 20 else rostro_top_pred,
            f"{rostro_top_conf:.1%}" if rostro_top_conf else "N/A"
        ])
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 2)
    
    # Color header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

def create_detailed_report_visualization(analysis_results: Dict) -> np.ndarray:
    """
    Create detailed report visualization with all analysis components
    
    Args:
        analysis_results: Complete analysis results
        
    Returns:
        numpy.ndarray: Detailed report image
    """
    try:
        # Create figure with detailed layout
        fig = plt.figure(figsize=(20, 16))
        
        # Create detailed grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('Complete Espejo Analysis Report', fontsize=16, fontweight='bold', y=0.95)
        
        # Mirror images with region overlays
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        if 'mirror_images' in analysis_results:
            right_mirror_rgb = cv2.cvtColor(analysis_results['mirror_images']['right_mirrored'], cv2.COLOR_BGR2RGB)
            left_mirror_rgb = cv2.cvtColor(analysis_results['mirror_images']['left_mirrored'], cv2.COLOR_BGR2RGB)
            
            ax1.imshow(right_mirror_rgb)
            ax1.set_title('Right Mirror + Regions', fontsize=12, fontweight='bold')
            _add_region_bboxes_to_plot(ax1, right_mirror_rgb.shape[:2])
            
            ax2.imshow(left_mirror_rgb)
            ax2.set_title('Left Mirror + Regions', fontsize=12, fontweight='bold')
            _add_region_bboxes_to_plot(ax2, left_mirror_rgb.shape[:2])
        
        ax1.axis('off')
        ax2.axis('off')
        
        # Proportions visualization
        ax3 = fig.add_subplot(gs[0, 2:])
        _create_proportions_chart(ax3, analysis_results.get('proportions', {}))
        
        # Classification confidence charts
        ax4 = fig.add_subplot(gs[1, 0])
        _create_confidence_chart(ax4, analysis_results.get('classification_results', {}), 'right_mirrored', 'frente')
        
        ax5 = fig.add_subplot(gs[1, 1])
        _create_confidence_chart(ax5, analysis_results.get('classification_results', {}), 'right_mirrored', 'rostro')
        
        ax6 = fig.add_subplot(gs[1, 2])
        _create_confidence_chart(ax6, analysis_results.get('classification_results', {}), 'left_mirrored', 'frente')
        
        ax7 = fig.add_subplot(gs[1, 3])
        _create_confidence_chart(ax7, analysis_results.get('classification_results', {}), 'left_mirrored', 'rostro')
        
        # Decision rules summary
        ax8 = fig.add_subplot(gs[2, :])
        _create_decision_rules_summary(ax8, analysis_results.get('classification_results', {}))
        
        # Final diagnosis summary
        ax9 = fig.add_subplot(gs[3, :])
        _create_final_diagnosis_summary(ax9, analysis_results.get('classification_results', {}))
        
        plt.tight_layout()
        
        # Convert matplotlib figure to opencv image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        report = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        return report
        
    except Exception as e:
        print(f"Error creating detailed report: {e}")
        # Return a simple error image
        error_img = np.zeros((800, 1200, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Report Error: {str(e)}", (10, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_img

def _create_confidence_chart(ax, classification_results: Dict, side: str, region: str):
    """Create confidence chart for specific side and region"""
    ax.set_title(f'{side.split("_")[0].title()} Side - {region.upper()}', fontsize=10, fontweight='bold')
    
    if side in classification_results:
        results = classification_results[side]
        predictions = results.get(f'{region}_predictions', [])
        probabilities = results.get(f'{region}_probabilities', [])
        
        if predictions and probabilities:
            # Limit to top 5 predictions
            predictions = predictions[:5]
            probabilities = probabilities[:5]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(predictions))
            colors = plt.cm.viridis(np.linspace(0, 1, len(predictions)))
            
            bars = ax.barh(y_pos, probabilities, color=colors)
            ax.set_yticks(y_pos)
            
            # Truncate long labels
            labels = [pred[:15] + "..." if len(pred) > 15 else pred for pred in predictions]
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Confidence', fontsize=8)
            ax.set_xlim(0, 1)
            
            # Add percentage labels
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.1%}', ha='left', va='center', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   transform=ax.transAxes, ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'No data available', 
               transform=ax.transAxes, ha='center', va='center')

def _create_decision_rules_summary(ax, classification_results: Dict):
    """Create decision rules summary"""
    ax.set_title('Applied Decision Rules Summary', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    rules_text = ""
    
    for side in ['right_mirrored', 'left_mirrored']:
        if side in classification_results:
            side_name = side.split('_')[0].title()
            rules_text += f"\n{side_name} Side Rules:\n"
            rules_text += "=" * 20 + "\n"
            
            results = classification_results[side]
            
            # FRENTE rules
            rules_text += "FRENTE Applied Rules:\n"
            for rule in results.get('frente_applied_rules', []):
                rules_text += f"  • {rule}\n"
            
            # FRENTE split rules
            rules_text += "FRENTE Split Rules:\n"
            for rule in results.get('frente_split_rules', []):
                rules_text += f"  • {rule}\n"
            
            # ROSTRO rules
            rules_text += "ROSTRO Applied Rules:\n"
            for rule in results.get('rostro_applied_rules', []):
                rules_text += f"  • {rule}\n"
            
            # ROSTRO split rules
            rules_text += "ROSTRO Split Rules:\n"
            for rule in results.get('rostro_split_rules', []):
                rules_text += f"  • {rule}\n"
            
            rules_text += "\n"
    
    ax.text(0.02, 0.98, rules_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=8, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def _create_final_diagnosis_summary(ax, classification_results: Dict):
    """Create final diagnosis summary"""
    ax.set_title('Final Diagnosis Summary', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    diagnosis_text = "FINAL ESPEJO ANALYSIS RESULTS\n"
    diagnosis_text += "=" * 50 + "\n\n"
    
    for side in ['right_mirrored', 'left_mirrored']:
        if side in classification_results:
            side_name = side.split('_')[0].title()
            results = classification_results[side]
            
            diagnosis_text += f"{side_name} Side Final Diagnosis:\n"
            diagnosis_text += f"  FRENTE: {results.get('frente_split_diagnosis', 'N/A')}\n"
            diagnosis_text += f"  ROSTRO: {results.get('rostro_split_diagnosis', 'N/A')}\n"
            
            diagnosis_text += f"\n{side_name} Side Decision Tree Results:\n"
            diagnosis_text += f"  FRENTE: {results.get('frente_final_diagnosis', 'N/A')}\n"
            diagnosis_text += f"  ROSTRO: {results.get('rostro_final_diagnosis', 'N/A')}\n\n"
    
    ax.text(0.5, 0.5, diagnosis_text, transform=ax.transAxes, 
           ha='center', va='center', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

def create_simple_result_image(original_image: np.ndarray, final_diagnosis: Dict) -> np.ndarray:
    """
    Create simple result image with final diagnosis
    
    Args:
        original_image: Original input image
        final_diagnosis: Final diagnosis results
        
    Returns:
        numpy.ndarray: Simple result image
    """
    try:
        # Create a copy of the original image
        result_image = original_image.copy()
        
        # Add diagnosis text overlay
        diagnosis_text = "ESPEJO ANALYSIS RESULTS:\n\n"
        
        if 'right_side' in final_diagnosis:
            diagnosis_text += "RIGHT SIDE:\n"
            diagnosis_text += f"FRENTE: {final_diagnosis['right_side'].get('frente_diagnosis', 'N/A')}\n"
            diagnosis_text += f"ROSTRO: {final_diagnosis['right_side'].get('rostro_diagnosis', 'N/A')}\n\n"
        
        if 'left_side' in final_diagnosis:
            diagnosis_text += "LEFT SIDE:\n"
            diagnosis_text += f"FRENTE: {final_diagnosis['left_side'].get('frente_diagnosis', 'N/A')}\n"
            diagnosis_text += f"ROSTRO: {final_diagnosis['left_side'].get('rostro_diagnosis', 'N/A')}\n"
        
        # Add text overlay to image
        lines = diagnosis_text.split('\n')
        y_offset = 30
        
        for line in lines:
            if line.strip():
                cv2.putText(result_image, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(result_image, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25
        
        return result_image
        
    except Exception as e:
        print(f"Error creating simple result image: {e}")
        # Return original image with error message
        error_image = original_image.copy()
        cv2.putText(error_image, f"Error: {str(e)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_image