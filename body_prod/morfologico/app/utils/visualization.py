import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
from matplotlib.patches import Rectangle
import logging

logger = logging.getLogger(__name__)

# Set matplotlib backend for server environments
plt.switch_backend('Agg')

# Configure style
plt.style.use('default')
sns.set_palette("husl")

def create_body_analysis_visualization(
    image: np.ndarray,
    results: Dict[str, Any],
    bbox: Optional[List[int]] = None,
    output_path: str = "/app/results/body_analysis.png",
    figsize: Tuple[int, int] = (16, 12)
) -> Optional[str]:
    """
    Create comprehensive body analysis visualization
    
    Args:
        image: Input image as numpy array
        results: Analysis results from pipeline
        bbox: Optional bounding box [x1, y1, x2, y2]
        output_path: Path to save visualization
        figsize: Figure size (width, height)
    
    Returns:
        Path to saved visualization or None if failed
    """
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Extract analysis data
        body_analysis = results.get('body_type_analysis', {})
        metrics = results.get('analysis_metrics', {})
        anatomical_parts = results.get('anatomical_parts_analysis', {})
        
        # 1. Original Image (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Draw anatomical parts bounding boxes if available
        if anatomical_parts and 'part_predictions' in anatomical_parts:
            part_predictions = anatomical_parts['part_predictions']
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (part_name, part_data) in enumerate(part_predictions.items()):
                if 'bbox' in part_data:
                    x1, y1, x2, y2 = part_data['bbox']
                    color = colors[i % len(colors)]
                    
                    # Draw rectangle
                    rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
                    ax1.add_patch(rect)
                    
                    # Add label
                    ax1.text(x1, y1-5, f"{part_name}", color=color, fontsize=8, 
                           fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", 
                           facecolor='white', alpha=0.8))
        elif bbox is not None:
            # Fallback to original bbox if no anatomical parts
            x1, y1, x2, y2 = bbox
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x1, y1-10, 'Analysis Region', 
                    color='red', fontsize=10, fontweight='bold')
        
        # 2. Processed Image (top second)
        ax2 = fig.add_subplot(gs[0, 1])
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 > x1 and y2 > y1:
                cropped = image[y1:y2, x1:x2]
                ax2.imshow(cropped)
                ax2.set_title('Analyzed Region', fontsize=14, fontweight='bold')
            else:
                ax2.imshow(image)
                ax2.set_title('Full Image Analysis', fontsize=14, fontweight='bold')
        else:
            ax2.imshow(image)
            ax2.set_title('Full Image Analysis', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 3. Body Type Classification (top-right, spanning 2 columns)
        ax3 = fig.add_subplot(gs[0, 2:])
        _plot_body_type_analysis(ax3, body_analysis)
        
        
        # 4. Confidence Metrics (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        _plot_confidence_metrics(ax4, body_analysis, metrics)
        
        # 5. Final Prediction (middle-center, spanning 3 columns)
        ax5 = fig.add_subplot(gs[1, 1:])
        _plot_final_prediction(ax5, body_analysis, anatomical_parts)
        
        # 6. Anatomical Parts Summary (bottom, spanning all columns)
        ax6 = fig.add_subplot(gs[2, :])
        _plot_anatomical_parts_summary(ax6, anatomical_parts)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Body analysis visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating body analysis visualization: {e}")
        plt.close('all')  # Clean up any open figures
        return None

def _plot_body_type_analysis(ax, body_analysis: Dict[str, Any]):
    """Plot body type classification results"""
    if not body_analysis:
        ax.text(0.5, 0.5, 'No body type analysis available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Body Type Analysis', fontsize=14, fontweight='bold')
        ax.axis('off')
        return
    
    # Get probabilities and sort them
    probs = body_analysis.get('all_probabilities', {})
    if not probs:
        ax.text(0.5, 0.5, 'No probabilities available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Body Type Analysis', fontsize=14, fontweight='bold')
        ax.axis('off')
        return
    
    # Sort by probability
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0].split('/')[-1] if '/' in item[0] else item[0] for item, _ in sorted_items]
    probabilities = [prob for _, prob in sorted_items]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(classes)), probabilities, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
    
    # Customize plot
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_title('Body Type Classification', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.01, i, f'{prob:.3f}', 
               va='center', fontsize=9, fontweight='bold')
    
    # Highlight top prediction
    if probabilities:
        bars[0].set_color('#FF6B6B')
        bars[0].set_alpha(0.8)


def _plot_confidence_metrics(ax, body_analysis: Dict[str, Any], 
                           metrics: Dict[str, Any]):
    """Plot confidence metrics"""
    # Prepare data
    body_conf = body_analysis.get('confidence', 0)
    overall_conf = metrics.get('overall_confidence', 0)
    
    categories = ['Body Type', 'Overall']
    confidences = [body_conf, overall_conf]
    colors = ['#FF9999', '#99FF99']
    
    # Create bar plot
    bars = ax.bar(categories, confidences, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Confidence Metrics', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add confidence threshold line
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()

def _plot_final_prediction(ax, body_analysis: Dict[str, Any], anatomical_parts: Dict[str, Any]):
    """Plot final weighted prediction result"""
    ax.axis('off')
    
    predicted_class = body_analysis.get('predicted_class', 'Unknown')
    confidence = body_analysis.get('confidence', 0.0)
    parts_count = anatomical_parts.get('total_parts', 0)
    voting_strategy = anatomical_parts.get('voting_strategy', 'unknown')
    
    # Create summary text
    text_content = [
        "🎯 FINAL PREDICTION",
        f"Body Type: {predicted_class}",
        f"Confidence: {confidence:.3f}",
        f"Parts Used: {parts_count}/5",
        f"Method: {voting_strategy}"
    ]
    
    # Display text
    full_text = '\n'.join(text_content)
    ax.text(0.5, 0.5, full_text, transform=ax.transAxes, 
           fontsize=14, verticalalignment='center', horizontalalignment='center',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    ax.set_title('Weighted Average Result', fontsize=12, fontweight='bold')

def _plot_anatomical_parts_summary(ax, anatomical_parts: Dict[str, Any]):
    """Plot anatomical parts analysis summary"""
    ax.axis('off')
    
    # Create text summary
    text_content = []
    
    # Add anatomical parts summary if available
    if anatomical_parts and anatomical_parts.get('parts_detected'):
        text_content.append("🦴 ANATOMICAL PARTS ANALYSIS")
        parts_detected = anatomical_parts.get('parts_detected', [])
        total_parts = anatomical_parts.get('total_parts', 0)
        text_content.append(f"• Parts Detected: {total_parts}/5 ({', '.join(parts_detected)})")
        
        voting_strategy = anatomical_parts.get('voting_strategy', 'unknown')
        text_content.append(f"• Voting Strategy: {voting_strategy}")
        
        # Show individual part predictions if available
        part_predictions = anatomical_parts.get('part_predictions', {})
        if part_predictions:
            text_content.append("• Individual Parts:")
            for part_name, pred in part_predictions.items():
                body_type = pred.get('predicted_body_type', 'Unknown')
                confidence = pred.get('confidence', 0.0)
                text_content.append(f"  - {part_name}: {body_type} ({confidence:.2f})")
    
    # Display text
    if text_content:
        full_text = '\n'.join(text_content)
        ax.text(0.05, 0.95, full_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No anatomical parts data available', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, style='italic')
    
    ax.set_title('Anatomical Parts Analysis', fontsize=14, fontweight='bold')

def create_simple_body_visualization(
    image: np.ndarray,
    results: Dict[str, Any],
    output_path: str = "/app/results/simple_body_analysis.png",
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[str]:
    """
    Create simple body analysis visualization for quick classification
    
    Args:
        image: Input image as numpy array
        results: Analysis results from pipeline
        output_path: Path to save visualization
        figsize: Figure size (width, height)
    
    Returns:
        Path to saved visualization or None if failed
    """
    try:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Extract data
        body_analysis = results.get('body_type_analysis', {})
        
        # 1. Original image
        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Body type probabilities
        if body_analysis and body_analysis.get('all_probabilities'):
            probs = body_analysis['all_probabilities']
            sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            classes = [item[0].split('/')[-1] if '/' in item[0] else item[0] for item, _ in sorted_items]
            probabilities = [prob for _, prob in sorted_items]
            
            axes[1].barh(range(len(classes)), probabilities, 
                        color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
            axes[1].set_yticks(range(len(classes)))
            axes[1].set_yticklabels(classes, fontsize=10)
            axes[1].set_xlabel('Confidence')
            axes[1].set_title('Body Type Classification', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No body type data', ha='center', va='center')
            axes[1].set_title('Body Type Classification', fontsize=14, fontweight='bold')
        
        # 3. Analysis Summary
        axes[2].axis('off')
        predicted_class = body_analysis.get('predicted_class', 'Unknown')
        confidence = body_analysis.get('confidence', 0.0)
        
        summary_text = f"Predicted Body Type:\n{predicted_class}\n\nConfidence: {confidence:.3f}"
        axes[2].text(0.5, 0.5, summary_text, ha='center', va='center', 
                    transform=axes[2].transAxes, fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        axes[2].set_title('Analysis Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Simple body visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating simple body visualization: {e}")
        plt.close('all')
        return None
