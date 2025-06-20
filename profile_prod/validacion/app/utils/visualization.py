import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ProfileValidationVisualizer:
    """
    Visualization utilities for profile validation results
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Colors for different elements
        self.colors = {
            'objeto': '#FF4444',          # Red for objects
            'cabello_tapando_oreja': '#FFD700',  # Gold for hair covering ear
            'cabello_tapando_frente': '#FF8C00', # Orange for hair covering forehead
            'bbox': '#00FF00',            # Green for general bounding boxes
            'text': '#FFFFFF',            # White for text
            'background': '#000000',      # Black for text background
            'grid': '#CCCCCC',           # Light gray for grid
            'success': '#4CAF50',        # Green for success
            'warning': '#FF9800',        # Orange for warning
            'error': '#F44336'           # Red for error
        }
        
        # Font settings
        self.font_scale = 0.7
        self.font_thickness = 2
        self.title_font_size = 16
        self.label_font_size = 12
    
    def create_validation_visualization(self, image: np.ndarray, results: Dict[str, Any], 
                                      save_path: Optional[str] = None) -> np.ndarray:
        """
        Create comprehensive validation visualization
        
        Args:
            image: Original image
            results: Validation results dictionary
            save_path: Optional path to save the visualization
            
        Returns:
            Visualization image as numpy array
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            
            # Main image with detections (larger subplot)
            ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
            self._plot_main_detection_view(ax1, image, results)
            
            # Quality metrics (top right)
            ax2 = plt.subplot2grid((3, 3), (0, 2))
            self._plot_quality_metrics(ax2, results)
            
            # Occlusion summary (middle right)
            ax3 = plt.subplot2grid((3, 3), (1, 2))
            self._plot_occlusion_summary(ax3, results)
            
            # Overall score gauge (bottom left)
            ax4 = plt.subplot2grid((3, 3), (2, 0))
            self._plot_overall_score(ax4, results)
            
            # Recommendations (bottom middle and right)
            ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=2)
            self._plot_recommendations(ax5, results)
            
            # Add main title
            validation_status = results['validation_status']
            status_text = "✅ SUITABLE" if validation_status['is_suitable'] else "⚠️ NEEDS IMPROVEMENT"
            score = validation_status['overall_score']
            
            fig.suptitle(f'Profile Validation Analysis - {status_text} (Score: {score:.1f}/100)', 
                        fontsize=self.title_font_size, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                logger.info(f"Validation visualization saved to {save_path}")
            
            # Convert to numpy array
            fig.canvas.draw()
            vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return vis_array
            
        except Exception as e:
            logger.error(f"Error creating validation visualization: {e}")
            # Return original image if visualization fails
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _plot_main_detection_view(self, ax, image: np.ndarray, results: Dict[str, Any]):
        """Plot main image with occlusion detections"""
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
        
        # Draw bounding boxes for detections
        if 'occlusion_analysis' in results and 'detections' in results['occlusion_analysis']:
            detections = results['occlusion_analysis']['detections']
            
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Get color for this class
                color = self.colors.get(class_name, self.colors['bbox'])
                
                # Create rectangle
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                label = f"{class_name}\n{confidence:.2f}"
                ax.text(x1, y1 - 10, label, fontsize=10, color=color, 
                       fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor='white', alpha=0.8))
        
        ax.set_title('Profile Image with Occlusion Detection', fontsize=self.label_font_size, 
                    fontweight='bold')
        ax.axis('off')
    
    def _plot_quality_metrics(self, ax, results: Dict[str, Any]):
        """Plot quality assessment metrics"""
        if 'quality_assessment' not in results:
            ax.text(0.5, 0.5, 'Quality data\nnot available', ha='center', va='center')
            ax.set_title('Quality Metrics')
            return
        
        quality = results['quality_assessment']
        
        # Prepare data
        metrics = {
            'Quality Score': quality.get('quality_score', 0),
            'Brightness': quality.get('brightness', 0),
            'Contrast': quality.get('contrast', 0),
            'Sharpness': quality.get('blur_score', 0) / 10  # Scale down for display
        }
        
        # Create bar plot
        bars = ax.bar(range(len(metrics)), list(metrics.values()), 
                     color=[self.colors['success'] if v > 70 else 
                           self.colors['warning'] if v > 40 else 
                           self.colors['error'] for v in metrics.values()])
        
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Quality Metrics', fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_occlusion_summary(self, ax, results: Dict[str, Any]):
        """Plot occlusion detection summary"""
        if 'occlusion_analysis' not in results:
            ax.text(0.5, 0.5, 'Occlusion data\nnot available', ha='center', va='center')
            ax.set_title('Occlusion Summary')
            return
        
        occlusion_data = results['occlusion_analysis']
        detections = occlusion_data.get('detections', [])
        
        if not detections:
            ax.text(0.5, 0.5, '✅ No Occlusions\nDetected', ha='center', va='center', 
                   fontsize=14, color=self.colors['success'], fontweight='bold')
            ax.set_title('Occlusion Summary', fontweight='bold')
            ax.axis('off')
            return
        
        # Count detections by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create pie chart
        colors = [self.colors.get(cls, self.colors['error']) for cls in class_counts.keys()]
        wedges, texts, autotexts = ax.pie(list(class_counts.values()), 
                                         labels=list(class_counts.keys()),
                                         colors=colors, autopct='%1.0f',
                                         startangle=90)
        
        ax.set_title('Detected Occlusions', fontweight='bold')
        
        # Adjust text properties
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_overall_score(self, ax, results: Dict[str, Any]):
        """Plot overall validation score as a gauge"""
        validation_status = results.get('validation_status', {})
        score = validation_status.get('overall_score', 0)
        
        # Create gauge visualization
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=8)
        
        # Score arc
        score_theta = np.linspace(0, np.pi * (score / 100), int(score))
        if len(score_theta) > 0:
            color = (self.colors['success'] if score >= 70 else 
                    self.colors['warning'] if score >= 40 else 
                    self.colors['error'])
            ax.plot(np.cos(score_theta), np.sin(score_theta), color, linewidth=8)
        
        # Add score text
        ax.text(0, -0.3, f'{score:.0f}', ha='center', va='center', 
               fontsize=24, fontweight='bold')
        ax.text(0, -0.5, 'Overall Score', ha='center', va='center', 
               fontsize=12)
        
        # Add status indicators
        status_text = "SUITABLE" if validation_status.get('is_suitable', False) else "NEEDS WORK"
        status_color = self.colors['success'] if validation_status.get('is_suitable', False) else self.colors['error']
        ax.text(0, -0.7, status_text, ha='center', va='center', 
               fontsize=10, fontweight='bold', color=status_color)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Validation Score', fontweight='bold')
    
    def _plot_recommendations(self, ax, results: Dict[str, Any]):
        """Plot recommendations list"""
        recommendations = results.get('recommendations', [])
        
        if not recommendations:
            ax.text(0.5, 0.5, '✅ No recommendations needed\nImage meets all quality standards', 
                   ha='center', va='center', fontsize=12, color=self.colors['success'],
                   fontweight='bold')
            ax.set_title('Recommendations', fontweight='bold')
            ax.axis('off')
            return
        
        # Display recommendations as a list
        ax.text(0.02, 0.95, 'Recommendations for improvement:', 
               transform=ax.transAxes, fontsize=12, fontweight='bold')
        
        for i, rec in enumerate(recommendations[:5]):  # Show max 5 recommendations
            y_pos = 0.85 - (i * 0.15)
            ax.text(0.05, y_pos, f'• {rec}', transform=ax.transAxes, 
                   fontsize=10, wrap=True, verticalalignment='top')
        
        if len(recommendations) > 5:
            ax.text(0.05, 0.1, f'... and {len(recommendations) - 5} more recommendations', 
                   transform=ax.transAxes, fontsize=9, style='italic')
        
        ax.set_title('Recommendations', fontweight='bold')
        ax.axis('off')
    
    def create_simple_detection_visualization(self, image: np.ndarray, detections: List[Dict], 
                                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Create simple visualization showing only detections
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            save_path: Optional path to save the visualization
            
        Returns:
            Visualization image as numpy array
        """
        try:
            # Create a copy of the image
            vis_image = image.copy()
            
            # Draw bounding boxes
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Get color (convert to BGR)
                color_hex = self.colors.get(class_name, self.colors['bbox'])
                color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))  # Convert hex to BGR
                
                # Draw rectangle
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 3)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           self.font_scale, self.font_thickness)[0]
                
                # Background for text
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color_bgr, -1)
                
                # Text
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                          (255, 255, 255), self.font_thickness)
            
            # Save if path provided
            if save_path:
                cv2.imwrite(save_path, vis_image)
                logger.info(f"Simple detection visualization saved to {save_path}")
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error creating simple detection visualization: {e}")
            return image
