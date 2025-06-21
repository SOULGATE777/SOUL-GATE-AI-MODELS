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
        gender_analysis = results.get('gender_analysis', {})
        metrics = results.get('analysis_metrics', {})
        insights = results.get('morphological_insights', {})
        
        # 1. Original Image (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Draw bounding box if provided
        if bbox is not None:
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
        
        # 4. Gender Analysis (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        _plot_gender_analysis(ax4, gender_analysis)
        
        # 5. Confidence Metrics (middle-center)
        ax5 = fig.add_subplot(gs[1, 1])
        _plot_confidence_metrics(ax5, body_analysis, gender_analysis, metrics)
        
        # 6. Top Predictions (middle-right, spanning 2 columns)
        ax6 = fig.add_subplot(gs[1, 2:])
        _plot_top_predictions(ax6, body_analysis)
        
        # 7. Morphological Insights (bottom, spanning all columns)
        ax7 = fig.add_subplot(gs[2, :])
        _plot_morphological_insights(ax7, insights, results.get('classification_summary', {}))
        
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

def _plot_gender_analysis(ax, gender_analysis: Dict[str, Any]):
    """Plot gender classification results"""
    if not gender_analysis:
        ax.text(0.5, 0.5, 'No gender analysis available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Gender Analysis', fontsize=12, fontweight='bold')
        ax.axis('off')
        return
    
    probs = gender_analysis.get('all_probabilities', {})
    if not probs:
        ax.text(0.5, 0.5, 'No probabilities available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Gender Analysis', fontsize=12, fontweight='bold')
        ax.axis('off')
        return
    
    # Create pie chart
    labels = list(probs.keys())
    values = list(probs.values())
    colors = ['#FFB6C1', '#87CEEB']  # Light pink and light blue
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                     colors=colors, startangle=90)
    
    ax.set_title('Gender Classification', fontsize=12, fontweight='bold')
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

def _plot_confidence_metrics(ax, body_analysis: Dict[str, Any], 
                           gender_analysis: Dict[str, Any], 
                           metrics: Dict[str, Any]):
    """Plot confidence metrics"""
    # Prepare data
    body_conf = body_analysis.get('confidence', 0)
    gender_conf = gender_analysis.get('confidence', 0)
    overall_conf = metrics.get('overall_confidence', 0)
    
    categories = ['Body Type', 'Gender', 'Overall']
    confidences = [body_conf, gender_conf, overall_conf]
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
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

def _plot_top_predictions(ax, body_analysis: Dict[str, Any]):
    """Plot top 3 body type predictions"""
    top_3 = body_analysis.get('top_3_predictions', [])
    
    if not top_3:
        ax.text(0.5, 0.5, 'No top predictions available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Top 3 Predictions', fontsize=12, fontweight='bold')
        ax.axis('off')
        return
    
    # Extract data
    classes = [pred['class'].split('/')[-1] if '/' in pred['class'] else pred['class'] 
              for pred in top_3]
    confidences = [pred['confidence'] for pred in top_3]
    ranks = [f"#{pred['rank']}" for pred in top_3]
    
    # Create horizontal bar plot
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    bars = ax.barh(range(len(classes)), confidences, color=colors[:len(classes)])
    
    # Customize plot
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{rank} {cls}" for rank, cls in zip(ranks, classes)], fontsize=10)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_title('Top 3 Body Type Predictions', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(confidences) * 1.1 if confidences else 1)
    
    # Add value labels
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax.text(conf + 0.01, i, f'{conf:.3f}', 
               va='center', fontsize=9, fontweight='bold')

def _plot_morphological_insights(ax, insights: Dict[str, Any], 
                               summary: Dict[str, Any]):
    """Plot morphological insights as text summary"""
    ax.axis('off')
    
    # Create text summary
    text_content = []
    
    # Classification summary
    if summary:
        primary = summary.get('primary_classification', 'Unknown')
        gender = summary.get('gender', 'Unknown')
        confidence_level = summary.get('confidence_level', 'unknown')
        
        text_content.append(f"🎯 CLASSIFICATION SUMMARY")
        text_content.append(f"Primary Type: {primary} | Gender: {gender} | Confidence: {confidence_level.title()}")
        text_content.append("")
    
    # Morphological insights
    if insights:
        text_content.append("🔬 MORPHOLOGICAL INSIGHTS")
        
        body_comp = insights.get('body_composition', '')
        if body_comp:
            text_content.append(f"• Body Composition: {body_comp}")
        
        metabolic = insights.get('metabolic_tendency', '')
        if metabolic:
            text_content.append(f"• Metabolic Tendency: {metabolic}")
        
        physical = insights.get('physical_characteristics', '')
        if physical:
            text_content.append(f"• Physical Characteristics: {physical}")
        
        health = insights.get('health_considerations', '')
        if health:
            text_content.append(f"• Health Considerations: {health}")
        
        analysis_note = insights.get('analysis_note', '')
        if analysis_note:
            text_content.append("")
            text_content.append(f"📝 Note: {analysis_note}")
    
    # Display text
    if text_content:
        full_text = '\n'.join(text_content)
        ax.text(0.05, 0.95, full_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No morphological insights available', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, style='italic')
    
    ax.set_title('Morphological Analysis Summary', fontsize=14, fontweight='bold')

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
        gender_analysis = results.get('gender_analysis', {})
        
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
        
        # 3. Gender classification
        if gender_analysis and gender_analysis.get('all_probabilities'):
            probs = gender_analysis['all_probabilities']
            labels = list(probs.keys())
            values = list(probs.values())
            colors = ['#FFB6C1', '#87CEEB']
            
            axes[2].pie(values, labels=labels, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
            axes[2].set_title('Gender Classification', fontsize=14, fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, 'No gender data', ha='center', va='center')
            axes[2].set_title('Gender Classification', fontsize=14, fontweight='bold')
        
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
