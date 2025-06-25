import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VisualizationManager:
    """Manager for creating profile analysis visualizations"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Color schemes
        self.bbox_colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.point_colors = plt.cm.rainbow(np.linspace(0, 1, 30))
        self.feature_colors = {
            'nasal': '#FF6B6B',
            'frontal': '#4ECDC4', 
            'mandibular': '#45B7D1',
            'auricular': '#96CEB4',
            'default': '#FFEAA7'
        }
    
    def create_complete_visualization(
        self,
        original_image: np.ndarray,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_confidence: bool = True
    ) -> str:
        """
        Create complete visualization with all analysis results
        
        Args:
            original_image: Original RGB image
            results: Complete analysis results
            save_path: Optional save path
            show_confidence: Whether to show confidence scores
            
        Returns:
            Path to saved visualization
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            fig.suptitle('Profile Morphological Analysis Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Original image
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image", fontsize=14)
            axes[0].axis('off')
            
            # Plot 2: Detected objects and landmarks
            self._plot_objects_and_landmarks(axes[1], original_image, results, show_confidence)
            
            # Plot 3: Anthropometric points
            self._plot_anthropometric_points(axes[2], original_image, results, show_confidence)
            
            plt.tight_layout()
            
            # Save visualization
            if save_path is None:
                save_path = self.results_dir / "profile_morphological_analysis.png"
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Complete visualization saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to create complete visualization: {str(e)}")
            raise
    
    def _plot_objects_and_landmarks(
        self, 
        ax: plt.Axes, 
        image: np.ndarray, 
        results: Dict[str, Any],
        show_confidence: bool = True
    ):
        """Plot detected objects and landmark classifications"""
        ax.imshow(image)
        
        detected_objects = results.get('detected_objects', [])
        classifications = results.get('landmark_classifications', [])
        
        title = f"Objects & Landmarks ({len(detected_objects)} objects, {len(classifications)} classified)"
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        if not detected_objects:
            ax.text(image.shape[1]//2, image.shape[0]//2, 
                   "No objects detected", ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            return
        
        # Scale factor from model input to original image
        image_h, image_w = image.shape[:2]
        scale_x = image_w / 224
        scale_y = image_h / 224
        
        # Draw bounding boxes and classifications
        for i, obj in enumerate(detected_objects):
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            
            # Scale to original image coordinates
            x1_orig = x1 * scale_x
            y1_orig = y1 * scale_y
            x2_orig = x2 * scale_x
            y2_orig = y2 * scale_y
            
            width = x2_orig - x1_orig
            height = y2_orig - y1_orig
            
            # Find corresponding classification
            tag_info = ""
            tag_confidence = 0.0
            for cls in classifications:
                if np.allclose(cls['bbox'], bbox, rtol=1e-3):
                    # Handle new top_tags format
                    if 'top_tags' in cls and cls['top_tags']:
                        # Use the first (highest confidence) tag for display
                        tag_info = cls['top_tags'][0]['tag']
                        tag_confidence = cls['top_tags'][0]['confidence']
                        
                        # Optionally show both tags
                        if len(cls['top_tags']) > 1:
                            second_tag = cls['top_tags'][1]
                            tag_info += f" | {second_tag['tag']} ({second_tag['confidence']:.2f})"
                    else:
                        # Fallback to old format for backward compatibility
                        tag_info = cls.get('tag', '')
                        tag_confidence = cls.get('tag_confidence', 0.0)
                    break
            
            # Choose color
            color = self.bbox_colors[i % len(self.bbox_colors)]
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1_orig, y1_orig), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Create label
            label_parts = [obj['class']]
            if show_confidence:
                label_parts.append(f"{obj['confidence']:.2f}")
            
            if tag_info:
                if show_confidence and tag_confidence > 0:
                    label_parts.append(f"→ {tag_info}")
                elif tag_info:
                    label_parts.append(f"→ {tag_info}")
            
            label = "\n".join(label_parts)
            
            # Draw label
            ax.text(x1_orig, y1_orig - 5, label,
                   color=color, fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _plot_anthropometric_points(
        self, 
        ax: plt.Axes, 
        image: np.ndarray, 
        results: Dict[str, Any],
        show_confidence: bool = True
    ):
        """Plot anthropometric points"""
        ax.imshow(image)
        
        points = results.get('anthropometric_points', [])
        profile_side = results.get('profile_side', '')
        profile_confidence = results.get('profile_confidence', 0.0)
        
        title = f"Anthropometric Points ({len(points)}) - {profile_side.title()}"
        if show_confidence and profile_confidence > 0:
            title += f" ({profile_confidence:.2f})"
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        if not points:
            ax.text(image.shape[1]//2, image.shape[0]//2, 
                   "No anthropometric points\ndetected above threshold", 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            return
        
        # Scale factor from model input to original image
        image_h, image_w = image.shape[:2]
        scale_x = image_w / 224
        scale_y = image_h / 224
        
        # Plot points
        for i, point in enumerate(points):
            x, y = point['coordinates']
            x_orig = x * scale_x
            y_orig = y * scale_y
            
            # Choose color based on point type or index
            color = self.point_colors[i % len(self.point_colors)]
            
            # Draw point
            ax.plot(x_orig, y_orig, 'o', color=color, markersize=8, 
                   markeredgecolor='white', markeredgewidth=1)
            
            # Create label
            label = point['class']
            if show_confidence:
                label += f"\n({point['confidence']:.2f})"
            
            # Draw label
            ax.text(x_orig + 5, y_orig - 5, label,
                   color=color, fontsize=7, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    def create_detailed_analysis_plot(
        self,
        original_image: np.ndarray,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create detailed analysis plot with feature groupings
        
        Args:
            original_image: Original RGB image
            results: Analysis results
            save_path: Optional save path
            
        Returns:
            Path to saved visualization
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Detailed Profile Morphological Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Original with all detections
            axes[0,0].imshow(original_image)
            axes[0,0].set_title("Complete Detection Results")
            self._overlay_all_detections(axes[0,0], original_image, results)
            
            # Plot 2: Feature groups
            axes[0,1].imshow(original_image)
            axes[0,1].set_title("Anatomical Feature Groups")
            self._plot_feature_groups(axes[0,1], original_image, results)
            
            # Plot 3: Analysis summary
            self._plot_analysis_summary(axes[1,0], results)
            
            # Plot 4: Quality metrics
            self._plot_quality_metrics(axes[1,1], results)
            
            plt.tight_layout()
            
            # Save visualization
            if save_path is None:
                save_path = self.results_dir / "detailed_profile_analysis.png"
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Detailed analysis plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to create detailed analysis plot: {str(e)}")
            raise
    
    def _overlay_all_detections(self, ax: plt.Axes, image: np.ndarray, results: Dict[str, Any]):
        """Overlay all detection results on image"""
        ax.axis('off')
        
        # Scale factors
        image_h, image_w = image.shape[:2]
        scale_x = image_w / 224
        scale_y = image_h / 224
        
        # Draw bounding boxes
        for i, obj in enumerate(results.get('detected_objects', [])):
            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            x1_orig, y1_orig = x1 * scale_x, y1 * scale_y
            x2_orig, y2_orig = x2 * scale_x, y2 * scale_y
            
            color = self.bbox_colors[i % len(self.bbox_colors)]
            rect = patches.Rectangle(
                (x1_orig, y1_orig), x2_orig - x1_orig, y2_orig - y1_orig,
                linewidth=1, edgecolor=color, facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)
        
        # Draw anthropometric points
        for i, point in enumerate(results.get('anthropometric_points', [])):
            x, y = point['coordinates']
            x_orig, y_orig = x * scale_x, y * scale_y
            
            color = self.point_colors[i % len(self.point_colors)]
            ax.plot(x_orig, y_orig, 'o', color=color, markersize=6,
                   markeredgecolor='white', markeredgewidth=1)
    
    def _plot_feature_groups(self, ax: plt.Axes, image: np.ndarray, results: Dict[str, Any]):
        """Plot points grouped by anatomical features"""
        ax.axis('off')
        
        points = results.get('anthropometric_points', [])
        if not points:
            return
        
        # Group points by feature type
        feature_groups = {
            'nasal': [],
            'frontal': [],
            'mandibular': [],
            'auricular': [],
            'other': []
        }
        
        for point in points:
            class_name = point.get('class', '').lower()
            
            # Categorize point
            if any(term in class_name for term in ['nariz', 'nose', 'nasal']):
                feature_groups['nasal'].append(point)
            elif any(term in class_name for term in ['frente', 'front', 'forehead']):
                feature_groups['frontal'].append(point)
            elif any(term in class_name for term in ['menton', 'chin', 'mandib']):
                feature_groups['mandibular'].append(point)
            elif any(term in class_name for term in ['oreja', 'ear', 'auricular']):
                feature_groups['auricular'].append(point)
            else:
                feature_groups['other'].append(point)
        
        # Scale factors
        image_h, image_w = image.shape[:2]
        scale_x = image_w / 224
        scale_y = image_h / 224
        
        # Plot each feature group
        legend_elements = []
        for feature_name, feature_points in feature_groups.items():
            if not feature_points:
                continue
                
            color = self.feature_colors.get(feature_name, self.feature_colors['default'])
            
            for point in feature_points:
                x, y = point['coordinates']
                x_orig, y_orig = x * scale_x, y * scale_y
                
                ax.plot(x_orig, y_orig, 'o', color=color, markersize=8,
                       markeredgecolor='white', markeredgewidth=1)
            
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8,
                                            label=f"{feature_name.title()} ({len(feature_points)})"))
        
        # Add legend
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def _plot_analysis_summary(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot analysis summary statistics"""
        ax.axis('off')
        
        # Collect summary data
        summary_data = {
            'Profile Type': results.get('profile_side', ''),
            'Profile Confidence': f"{results.get('profile_confidence', 0):.3f}",
            'Objects Detected': len(results.get('detected_objects', [])),
            'Landmarks Classified': len(results.get('landmark_classifications', [])),
            'Anthropometric Points': len(results.get('anthropometric_points', []))
        }
        
        # Create summary text
        y_pos = 0.9
        ax.text(0.05, y_pos, 'Analysis Summary', fontsize=14, fontweight='bold')
        y_pos -= 0.15
        
        for key, value in summary_data.items():
            ax.text(0.05, y_pos, f'{key}:', fontsize=10, fontweight='bold')
            ax.text(0.5, y_pos, str(value), fontsize=10)
            y_pos -= 0.1
        
        # Add classification details if available
        if results.get('landmark_classifications'):
            y_pos -= 0.05
            ax.text(0.05, y_pos, 'Landmark Tags:', fontsize=12, fontweight='bold')
            y_pos -= 0.1
            
            for cls in results['landmark_classifications'][:5]:  # Show first 5
                # Handle both old and new format
                if 'top_tags' in cls and cls['top_tags']:
                    tag = cls['top_tags'][0]['tag']
                    conf = cls['top_tags'][0]['confidence']
                else:
                    tag = cls.get('tag', cls.get('classified_tag', ''))
                    conf = cls.get('tag_confidence', 0)
                
                ax.text(0.1, y_pos, f'• {tag} ({conf:.2f})', fontsize=9)
                y_pos -= 0.08
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _plot_quality_metrics(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot quality assessment metrics"""
        ax.axis('off')
        
        # Calculate quality metrics
        points = results.get('anthropometric_points', [])
        objects = results.get('detected_objects', [])
        classifications = results.get('landmark_classifications', [])
        
        metrics = {
            'Point Detection': len(points) / 20.0,  # Assume max 20 points
            'Object Detection': min(len(objects) / 10.0, 1.0),  # Max 10 objects
            'Classification Rate': len(classifications) / max(len(objects), 1),
            'Average Point Confidence': np.mean([p['confidence'] for p in points]) if points else 0,
            'Profile Confidence': results.get('profile_confidence', 0)
        }
        
        # Create bar chart
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        bars = ax.barh(range(len(metric_names)), metric_values, color=colors[:len(metric_names)])
        
        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels(metric_names, fontsize=10)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_title('Quality Metrics', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.2f}', va='center', fontsize=9)
    
    def create_comparison_plot(
        self,
        images_and_results: List[Tuple[np.ndarray, Dict[str, Any]]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comparison plot for multiple profile analyses
        
        Args:
            images_and_results: List of (image, results) tuples
            save_path: Optional save path
            
        Returns:
            Path to saved visualization
        """
        try:
            n_images = len(images_and_results)
            fig, axes = plt.subplots(2, n_images, figsize=(6*n_images, 12))
            
            if n_images == 1:
                axes = axes.reshape(2, 1)
            
            fig.suptitle('Profile Analysis Comparison', fontsize=16, fontweight='bold')
            
            for i, (image, results) in enumerate(images_and_results):
                # Top row: original with overlays
                axes[0, i].imshow(image)
                axes[0, i].set_title(f'Profile {i+1} - {results.get("profile_side", "")}')
                self._overlay_all_detections(axes[0, i], image, results)
                
                # Bottom row: summary metrics
                self._plot_quality_metrics(axes[1, i], results)
            
            plt.tight_layout()
            
            # Save visualization
            if save_path is None:
                save_path = self.results_dir / "profile_comparison.png"
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Comparison plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to create comparison plot: {str(e)}")
            raise
    
    def save_results_summary(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """
        Save analysis results summary as JSON
        
        Args:
            results: Analysis results
            save_path: Optional save path
            
        Returns:
            Path to saved summary
        """
        import json
        
        if save_path is None:
            save_path = self.results_dir / "analysis_summary.json"
        
        # Prepare serializable results
        serializable_results = self._make_serializable(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results summary saved to: {save_path}")
        return str(save_path)
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
