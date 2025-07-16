import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def create_hand_analysis_visualization(image_path: str, results: Dict[Any, Any], 
                                     output_path: str) -> Optional[str]:
    """
    Create comprehensive visualization of hand analysis results
    
    Args:
        image_path: Path to the original image
        results: Analysis results dictionary
        output_path: Path to save the visualization
        
    Returns:
        str: Path to saved visualization or None if failed
    """
    try:
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Original image with bounding box (if provided)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Draw bounding box if provided
        if results.get('bbox'):
            x_min, y_min, x_max, y_max = results['bbox']
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x_min, y_min-10, 'Hand Region', color='red', fontweight='bold')
        
        # 2. Hand region (cropped if bbox available)
        ax2 = fig.add_subplot(gs[0, 1])
        if results.get('bbox'):
            x_min, y_min, x_max, y_max = results['bbox']
            hand_region = image_rgb[int(y_min):int(y_max), int(x_min):int(x_max)]
        else:
            hand_region = image_rgb
        ax2.imshow(hand_region)
        ax2.set_title('Hand Region', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. CNN Classification Results
        ax3 = fig.add_subplot(gs[0, 2])
        cnn_text = "CNN CLASSIFICATION:\\n\\n"
        
        if results.get('cnn_prediction'):
            cnn = results['cnn_prediction']
            cnn_text += f"Predicted: {cnn['predicted_class']}\\n"
            cnn_text += f"Confidence: {cnn['confidence']:.1%}\\n\\n"
            cnn_text += f"Probabilities:\\n"
            for class_name, prob in cnn['probabilities'].items():
                cnn_text += f"  {class_name}: {prob:.1%}\\n"
            
            # Color based on confidence
            if cnn['confidence'] > 0.8:
                text_color = 'green'
            elif cnn['confidence'] > 0.6:
                text_color = 'orange'
            else:
                text_color = 'red'
        else:
            cnn_text += "CNN model not available\\nor prediction failed"
            text_color = 'gray'
        
        ax3.text(0.05, 0.95, cnn_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', color=text_color)
        ax3.set_title('CNN Results', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. Color Analysis Info
        ax4 = fig.add_subplot(gs[0, 3])
        color_text = "COLORIMETRY:\\n\\n"
        
        if results.get('colorimetry'):
            colorimetry = results['colorimetry']
            avg_color = colorimetry['average_color_rgb']
            main_color = colorimetry['dominant_colors'][0][0]
            
            color_text += f"Average RGB: {avg_color}\\n"
            color_text += f"Main RGB: {main_color}\\n"
            color_text += f"Hue: {colorimetry['hue_mean']:.1f}°\\n"
            color_text += f"Pixels: {colorimetry['total_pixels']:,}\\n\\n"
            
            # Color classification
            if results.get('color_classification'):
                cc = results['color_classification']
                color_text += "Average Color Type:\\n"
                for color_type, percentage in cc['average_color'].items():
                    color_text += f"  {color_type}: {percentage:.1f}%\\n"
        else:
            color_text += "Colorimetry analysis\\nnot available"
        
        ax4.text(0.05, 0.95, color_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Colorimetry Info', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. Skin mask (if colorimetry was performed)
        ax5 = fig.add_subplot(gs[1, 0])
        if results.get('colorimetry'):
            # Recreate skin mask for visualization
            from app.models.hand_analysis_pipeline import PalmColorimetryAnalyzer
            analyzer = PalmColorimetryAnalyzer()
            hand_bgr = cv2.cvtColor(hand_region, cv2.COLOR_RGB2BGR)
            skin_mask = analyzer.create_skin_mask(hand_bgr)
            ax5.imshow(skin_mask, cmap='gray')
            ax5.set_title('Skin Mask', fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'Skin mask\\nnot available', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Skin Mask', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # 6. Color palette
        ax6 = fig.add_subplot(gs[1, 1])
        if results.get('colorimetry'):
            dominant_colors = results['colorimetry']['dominant_colors']
            palette_height = 100
            palette_width = 200
            color_bar = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            x_pos = 0
            
            for color_data in dominant_colors:
                color, percentage = color_data[0], color_data[1]
                width = int(palette_width * percentage / 100)
                if width > 0 and x_pos + width <= palette_width:
                    color_bar[:, x_pos:x_pos+width] = color
                    x_pos += width
            
            ax6.imshow(color_bar)
            ax6.set_title('Dominant Colors', fontsize=12, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'Color palette\\nnot available', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Dominant Colors', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # 7. Color comparison patches
        ax7 = fig.add_subplot(gs[1, 2])
        if results.get('colorimetry'):
            avg_color = results['colorimetry']['average_color_rgb']
            main_color = results['colorimetry']['dominant_colors'][0][0]
            
            # Create color patches
            avg_patch = np.full((50, 100, 3), avg_color, dtype=np.uint8)
            main_patch = np.full((50, 100, 3), main_color, dtype=np.uint8)
            comparison = np.vstack([avg_patch, main_patch])
            
            ax7.imshow(comparison)
            ax7.set_title('Color Comparison', fontsize=12, fontweight='bold')
            
            # Add labels
            ax7.text(50, 25, 'Average', ha='center', va='center', fontweight='bold', color='white')
            ax7.text(50, 75, 'Dominant', ha='center', va='center', fontweight='bold', color='white')
        else:
            ax7.text(0.5, 0.5, 'Color comparison\\nnot available', ha='center', va='center',
                    transform=ax7.transAxes, fontsize=12)
            ax7.set_title('Color Comparison', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # 8. Color classification details
        ax8 = fig.add_subplot(gs[1, 3])
        if results.get('color_classification'):
            cc = results['color_classification']
            class_text = "COLOR CLASSIFICATION:\\n\\n"
            class_text += "Main Color:\\n"
            for color_type, percentage in cc['main_color'].items():
                class_text += f"  {color_type}: {percentage:.1f}%\\n"
            
            ax8.text(0.05, 0.95, class_text, transform=ax8.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace')
        else:
            ax8.text(0.5, 0.5, 'Color classification\\nnot available', ha='center', va='center',
                    transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Color Classification', fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        # 9. Summary section (bottom row)
        ax9 = fig.add_subplot(gs[2, :])
        summary_text = "ANALYSIS SUMMARY:\\n\\n"
        
        # CNN summary
        if results.get('cnn_prediction'):
            cnn = results['cnn_prediction']
            summary_text += f"• Hand Side: {cnn['predicted_class']} (confidence: {cnn['confidence']:.1%})\\n"
        
        # Colorimetry summary
        if results.get('colorimetry'):
            colorimetry = results['colorimetry']
            summary_text += f"• Skin pixels analyzed: {colorimetry['total_pixels']:,}\\n"
            summary_text += f"• Average hue: {colorimetry['hue_mean']:.1f}°\\n"
        
        # Color classification summary
        if results.get('color_classification'):
            cc = results['color_classification']
            # Get most likely color type from average color
            avg_colors = cc['average_color']
            most_likely = max(avg_colors.items(), key=lambda x: x[1])
            summary_text += f"• Most likely color type: {most_likely[0]} ({most_likely[1]:.1f}%)\\n"
        
        # Analysis metadata
        summary_text += f"\\n• Analysis ID: {results.get('analysis_id', 'N/A')}\\n"
        summary_text += f"• Analysis type: {results.get('analysis_type', 'comprehensive_hand_analysis')}\\n"
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        ax9.set_title('Analysis Summary', fontsize=14, fontweight='bold')
        ax9.axis('off')
        
        # Add main title
        fig.suptitle('Hand Analysis Results', fontsize=16, fontweight='bold', y=0.98)
        
        # Save visualization
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None

def create_color_analysis_chart(colorimetry_data: Dict[Any, Any], output_path: str) -> Optional[str]:
    """
    Create detailed color analysis chart
    
    Args:
        colorimetry_data: Colorimetry analysis results
        output_path: Path to save the chart
        
    Returns:
        str: Path to saved chart or None if failed
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Dominant colors bar chart
        ax1 = axes[0, 0]
        colors = [color_data[0] for color_data in colorimetry_data['dominant_colors'][:5]]
        percentages = [color_data[1] for color_data in colorimetry_data['dominant_colors'][:5]]
        
        # Convert colors to matplotlib format
        colors_norm = [(c[0]/255, c[1]/255, c[2]/255) for c in colors]
        
        bars = ax1.bar(range(len(colors)), percentages, color=colors_norm)
        ax1.set_title('Dominant Colors', fontweight='bold')
        ax1.set_xlabel('Color Rank')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_xticks(range(len(colors)))
        ax1.set_xticklabels([f'#{i+1}' for i in range(len(colors))])
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # 2. Color wheel/hue distribution
        ax2 = axes[0, 1]
        ax2.set_title('Average Color Info', fontweight='bold')
        
        avg_color = colorimetry_data['average_color_rgb']
        avg_hsv = colorimetry_data['average_color_hsv']
        
        info_text = f"RGB: ({avg_color[0]}, {avg_color[1]}, {avg_color[2]})\\n"
        info_text += f"HSV: ({avg_hsv[0]:.1f}°, {avg_hsv[1]:.1f}%, {avg_hsv[2]:.1f}%)\\n\\n"
        info_text += f"Hue Statistics:\\n"
        info_text += f"Mean: {colorimetry_data['hue_mean']:.1f}°\\n"
        info_text += f"Std Dev: {colorimetry_data['hue_std']:.1f}°\\n\\n"
        info_text += f"Total Pixels: {colorimetry_data['total_pixels']:,}"
        
        # Show average color patch
        color_patch = np.full((100, 100, 3), avg_color, dtype=np.uint8)
        ax2.imshow(color_patch)
        ax2.text(110, 50, info_text, fontsize=10, va='center')
        ax2.set_xlim(0, 300)
        ax2.axis('off')
        
        # 3. RGB channel values
        ax3 = axes[1, 0]
        channels = ['Red', 'Green', 'Blue']
        rgb_values = avg_color
        colors_rgb = ['red', 'green', 'blue']
        
        bars = ax3.bar(channels, rgb_values, color=colors_rgb, alpha=0.7)
        ax3.set_title('RGB Channel Values', fontweight='bold')
        ax3.set_ylabel('Value (0-255)')
        ax3.set_ylim(0, 255)
        
        # Add value labels
        for bar, val in zip(bars, rgb_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{val}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Color distribution pie chart
        ax4 = axes[1, 1]
        
        # Use top 4 colors for pie chart
        top_colors = colorimetry_data['dominant_colors'][:4]
        colors_pie = [color_data[0] for color_data in top_colors]
        percentages_pie = [color_data[1] for color_data in top_colors]
        labels_pie = [f'Color {i+1}' for i in range(len(top_colors))]
        
        # Convert colors for matplotlib
        colors_pie_norm = [(c[0]/255, c[1]/255, c[2]/255) for c in colors_pie]
        
        wedges, texts, autotexts = ax4.pie(percentages_pie, labels=labels_pie, 
                                          colors=colors_pie_norm, autopct='%1.1f%%',
                                          startangle=90)
        ax4.set_title('Color Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Color analysis chart saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating color analysis chart: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None

def draw_bounding_box_on_image(image_path: str, bbox: tuple, output_path: str, 
                              label: str = "Hand") -> Optional[str]:
    """
    Draw bounding box on image and save
    
    Args:
        image_path: Path to input image
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
        output_path: Path to save annotated image
        label: Label for the bounding box
        
    Returns:
        str: Path to saved image or None if failed
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        
        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (x_min, y_min - label_size[1] - 10), 
                     (x_min + label_size[0], y_min), (0, 255, 0), -1)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 2)
        
        # Save annotated image
        cv2.imwrite(output_path, image)
        
        logger.info(f"Annotated image saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error drawing bounding box: {e}")
        return None